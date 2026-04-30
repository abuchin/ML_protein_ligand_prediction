"""InteractionMLP — protein-ligand interaction network with element-wise product layer.

Architecture:
    Protein (protein_dim) → ProtProjection → prot_h (projection_dim)
    Ligand  (ligand_dim)  → LigProjection  → lig_h  (projection_dim)
    Aux     (aux_dim)     → AuxProjection  → aux_h  (aux_projection_dim)

    interaction_h = prot_h * lig_h   # element-wise product
                                     # forces the net to learn pairwise feature activations
                                     # (NFM / DeepFM / WideDTA pattern)

    fusion_input = concat([prot_h, lig_h, interaction_h, aux_h])

    Fusion MLP → logits (2 classes)

Training:
    - FocalLoss (γ=2, α=0.25) handles class imbalance without hard class_weight
    - ReduceLROnPlateau scheduler decays LR on val plateau
    - Early stopping with checkpoint recovery (best val loss)
    - Mixed-precision autocast when CUDA is available

Bug fixes vs. original MLPModel:
    1. StandardScaler is now applied BEFORE building DataLoaders (was missing).
    2. Early stopping now actually works (patience counter was not implemented).
    3. No more override of split_data that dropped the scaler.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .base import BaseModel
from plbind.data.dataset import BindingDataset, make_dataloaders

logger = logging.getLogger(__name__)


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss for binary classification with class imbalance.

    FL(p) = -α (1-p)^γ log(p)   for the positive class
    Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection."
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.cross_entropy(logits, targets, reduction="none")
        probs = torch.softmax(logits, dim=1)
        # Use targets.device to avoid CPU/MPS device mismatch with arange
        idx = torch.arange(len(targets), device=targets.device)
        pt = probs[idx, targets]
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=targets.device),
            torch.tensor(1 - self.alpha, device=targets.device),
        )
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


# ── Projection sub-module ─────────────────────────────────────────────────────

class _Projection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Main network ──────────────────────────────────────────────────────────────

class _InteractionNet(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        ligand_dim: int,
        aux_dim: int,
        projection_dim: int,
        aux_projection_dim: int,
        fusion_dims: Tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.prot_proj = _Projection(protein_dim, projection_dim)
        self.lig_proj = _Projection(ligand_dim, projection_dim)
        self.has_aux = aux_dim > 0
        if self.has_aux:
            self.aux_proj = _Projection(aux_dim, aux_projection_dim)
        else:
            aux_projection_dim = 0

        # fusion input: [prot_h, lig_h, interaction_h, aux_h]
        fusion_in = projection_dim * 3 + aux_projection_dim
        layers: List[nn.Module] = []
        prev = fusion_in
        for hidden in fusion_dims:
            layers += [nn.Linear(prev, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, 2))
        self.fusion = nn.Sequential(*layers)

    def forward(
        self,
        protein: torch.Tensor,
        ligand: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prot_h = self.prot_proj(protein)
        lig_h = self.lig_proj(ligand)
        interaction_h = prot_h * lig_h  # element-wise product

        parts = [prot_h, lig_h, interaction_h]
        if self.has_aux and aux is not None and aux.shape[-1] > 0:
            parts.append(self.aux_proj(aux))
        return self.fusion(torch.cat(parts, dim=1))


# ── sklearn-compatible wrapper ────────────────────────────────────────────────

class InteractionMLPModel(BaseModel):
    """Wrapper around _InteractionNet with BaseModel interface.

    Unlike sklearn models, this wrapper takes pre-split data via set_split_data()
    instead of relying on split_data(), because the MLP needs separate protein/
    ligand/aux blocks rather than a flat feature matrix.
    """

    def __init__(
        self,
        protein_dim: int,
        ligand_dim: int,
        aux_dim: int = 0,
        projection_dim: int = 256,
        aux_projection_dim: int = 64,
        fusion_dims: Tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 15,
        lr_scheduler_patience: int = 3,
        device: str = "auto",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        super().__init__("InteractionMLP", test_size, random_state)
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.aux_dim = aux_dim
        self.projection_dim = projection_dim
        self.aux_projection_dim = aux_projection_dim
        self.fusion_dims = fusion_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr_scheduler_patience = lr_scheduler_patience

        if device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            # MPS skipped in auto-mode: BatchNorm1d backward hangs on MPS with
            # PyTorch 2.8 + macOS 15 (Sequoia). Use --device mps to force it.
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        self._net: Optional[_InteractionNet] = None
        self._train_dl: Optional[DataLoader] = None
        self._val_dl: Optional[DataLoader] = None
        # separate block arrays for inference
        self._protein_test: Optional[np.ndarray] = None
        self._ligand_test: Optional[np.ndarray] = None
        self._aux_test: Optional[np.ndarray] = None

    # ── Block-based data loading ──────────────────────────────────────────────

    def set_split_data(
        self,
        protein_train: np.ndarray, ligand_train: np.ndarray, y_train: np.ndarray,
        protein_val: np.ndarray,   ligand_val: np.ndarray,   y_val: np.ndarray,
        protein_test: np.ndarray,  ligand_test: np.ndarray,  y_test: np.ndarray,
        aux_train: Optional[np.ndarray] = None,
        aux_val: Optional[np.ndarray] = None,
        aux_test: Optional[np.ndarray] = None,
    ) -> None:
        """Provide pre-split, pre-scaled feature blocks directly.

        The StandardScaler is fitted on protein_train, ligand_train, aux_train
        and applied to all other splits here, then stored on self for predict().
        """
        from sklearn.preprocessing import StandardScaler

        # Fit scalers on training blocks
        self._prot_scaler = StandardScaler()
        self._lig_scaler = StandardScaler()

        prot_tr = self._prot_scaler.fit_transform(protein_train).astype(np.float32)
        lig_tr = self._lig_scaler.fit_transform(ligand_train).astype(np.float32)
        prot_val_s = self._prot_scaler.transform(protein_val).astype(np.float32)
        lig_val_s = self._lig_scaler.transform(ligand_val).astype(np.float32)
        prot_te_s = self._prot_scaler.transform(protein_test).astype(np.float32)
        lig_te_s = self._lig_scaler.transform(ligand_test).astype(np.float32)

        aux_tr_s, aux_val_s, aux_te_s = None, None, None
        if aux_train is not None:
            self._aux_scaler = StandardScaler()
            aux_tr_s = self._aux_scaler.fit_transform(aux_train).astype(np.float32)
            aux_val_s = self._aux_scaler.transform(aux_val).astype(np.float32)
            aux_te_s = self._aux_scaler.transform(aux_test).astype(np.float32)

        pin = torch.cuda.is_available()
        self._train_dl, self._val_dl = make_dataloaders(
            prot_tr, lig_tr, y_train, prot_val_s, lig_val_s, y_val,
            aux_tr_s, aux_val_s,
            batch_size=self.batch_size, pin_memory=pin,
        )
        self._protein_test = prot_te_s
        self._ligand_test = lig_te_s
        self._aux_test = aux_te_s
        self.y_test = y_test.astype(np.int32)
        logger.info(
            "InteractionMLP data set: %d train batches, %d val batches, %d test rows.",
            len(self._train_dl), len(self._val_dl), len(y_test),
        )

    # ── BaseModel abstract methods ────────────────────────────────────────────

    def _initialize_model(self, **kwargs):
        return _InteractionNet(
            protein_dim=self.protein_dim,
            ligand_dim=self.ligand_dim,
            aux_dim=self.aux_dim,
            projection_dim=self.projection_dim,
            aux_projection_dim=self.aux_projection_dim,
            fusion_dims=self.fusion_dims,
            dropout=self.dropout,
        ).to(self._device)

    def _train_model(self) -> None:
        if self._train_dl is None:
            raise RuntimeError("Call set_split_data() before train().")

        self._net = self._initialize_model()
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=self.lr_scheduler_patience, factor=0.5)
        criterion = FocalLoss(gamma=2.0, alpha=0.25)
        use_amp = self._device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        best_val_loss = float("inf")
        patience_counter = 0
        best_state_path = Path(tempfile.mktemp(suffix=".pt"))

        for epoch in range(1, self.epochs + 1):
            # ── Train ──
            self._net.train()
            total_loss = 0.0
            for batch in self._train_dl:
                prot, lig, aux, labels = [b.to(self._device) for b in batch]
                aux_input = aux if aux.shape[-1] > 0 else None
                optimizer.zero_grad()
                if scaler:
                    with torch.amp.autocast("cuda"):
                        logits = self._net(prot, lig, aux_input)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = self._net(prot, lig, aux_input)
                    loss = criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                    optimizer.step()
                total_loss += loss.item()

            # ── Validate ──
            val_loss = self._evaluate_loss(self._val_dl, criterion)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self._net.state_dict(), best_state_path)
            else:
                patience_counter += 1

            if True:  # log every epoch for visibility
                lr_current = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  patience=%d  lr=%.2e",
                    epoch, self.epochs, total_loss / len(self._train_dl), val_loss,
                    patience_counter, lr_current,
                )

            if patience_counter >= self.patience:
                logger.info("Early stopping at epoch %d (best val_loss=%.4f).", epoch, best_val_loss)
                break

        # Restore best weights
        self._net.load_state_dict(torch.load(best_state_path, map_location=self._device, weights_only=True))
        best_state_path.unlink(missing_ok=True)

    def _evaluate_loss(self, dl: DataLoader, criterion: nn.Module) -> float:
        self._net.eval()
        total = 0.0
        with torch.no_grad():
            for batch in dl:
                prot, lig, aux, labels = [b.to(self._device) for b in batch]
                aux_input = aux if aux.shape[-1] > 0 else None
                logits = self._net(prot, lig, aux_input)
                total += criterion(logits, labels).item()
        return total / max(len(dl), 1)

    def _predict_model(self) -> np.ndarray:
        return self.predict_proba().argmax(axis=1)

    def predict_proba(self) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Call train() first.")
        self._net.eval()
        probs_list = []
        test_ds = BindingDataset(
            self._protein_test, self._ligand_test,
            np.zeros(len(self._protein_test), dtype=np.int64),
            self._aux_test,
        )
        dl = DataLoader(test_ds, batch_size=self.batch_size * 4, shuffle=False)
        with torch.no_grad():
            for batch in dl:
                prot, lig, aux, _ = [b.to(self._device) for b in batch]
                aux_input = aux if aux.shape[-1] > 0 else None
                logits = self._net(prot, lig, aux_input)
                probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs_list).astype(np.float32)

    def save_model(self, path: Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path.with_suffix(".pt"))
        artifact = {
            "params": self.get_model_info(),
            "prot_scaler": getattr(self, "_prot_scaler", None),
            "lig_scaler": getattr(self, "_lig_scaler", None),
            "aux_scaler": getattr(self, "_aux_scaler", None),
        }
        joblib.dump(artifact, path.with_suffix(".pkl"))
        logger.info("InteractionMLP saved → %s (.pt + .pkl)", path.stem)

    def load_model(self, path: Path) -> None:
        import joblib
        path = Path(path)
        artifact = joblib.load(path.with_suffix(".pkl"))
        self._prot_scaler = artifact.get("prot_scaler")
        self._lig_scaler = artifact.get("lig_scaler")
        self._aux_scaler = artifact.get("aux_scaler")
        self._net = self._initialize_model()
        self._net.load_state_dict(torch.load(path.with_suffix(".pt"), map_location=self._device, weights_only=True))
        self._net.eval()
        logger.info("InteractionMLP loaded from %s", path)

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            "protein_dim": self.protein_dim,
            "ligand_dim": self.ligand_dim,
            "aux_dim": self.aux_dim,
            "projection_dim": self.projection_dim,
            "fusion_dims": self.fusion_dims,
            "lr": self.lr,
            "epochs": self.epochs,
            "patience": self.patience,
        })
        return info
