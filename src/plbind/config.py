from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────────────────────
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def raw_data_path(self) -> Path:
        return self.data_dir / "raw" / "Drug_Discovery_dataset.csv"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.base_dir / "outputs"

    # ── Science: data & labels ───────────────────────────────────────────────
    # KIBA scores: lower = stronger binding. Literature threshold: < 12.1 is "bound".
    # Original paper: He et al. (2017); benchmark: Tang et al. (2020).
    kiba_threshold: float = 12.1
    kiba_binder_is_below: bool = True  # bound=1 when kiba_score < kiba_threshold

    # kiba_score_estimated==True means the score was *imputed* (noisy).
    # Set keep_only_measured=True to use only rows where estimated==False.
    # Verify against dataset README before changing — the filter may need to be flipped.
    keep_only_measured: bool = False  # set True after auditing the column semantics

    negative_ratio: float = 1.0  # negatives per positive (1.0 = balanced)
    use_property_matched_decoys: bool = True  # DUD-E style; falls back to random shuffle

    # ── Protein encoder ──────────────────────────────────────────────────────
    # ESM-2 (Meta, 2022) outperforms ProtBERT on FLIP benchmarks by 4-8%.
    # Swap to "facebook/esm2_t12_35M_UR50D" for CPU-only environments.
    protein_encoder: str = "facebook/esm2_t12_35M_UR50D"
    # "mean_max" → concat(mean, max) over residues → 2×embed_dim features.
    # "mean"     → mean only.
    protein_pooling: str = "mean_max"
    protein_max_length: int = 1022  # ESM-2 positional encoding limit

    # ── Ligand encoder ───────────────────────────────────────────────────────
    morgan_radius: int = 2
    morgan_bits: int = 1024
    morgan_use_counts: bool = True  # count fingerprint; more informative than binary
    use_maccs: bool = True          # 166-bit pharmacophore keys
    use_atompair: bool = True       # 1024-bit atom-pair fingerprint

    # ── Split strategy ───────────────────────────────────────────────────────
    # Options: "random" | "cold_protein" | "cold_ligand" | "scaffold" | "cold_both"
    # cold_protein is the recommended default for honest DTI benchmarking.
    split_strategy: str = "cold_protein"
    test_size: float = 0.2
    val_size: float = 0.1
    cv_folds: int = 5

    # ── Training ─────────────────────────────────────────────────────────────
    random_seed: int = 42
    device: str = "auto"       # "auto" = GPU if available, else CPU
    n_samples: Optional[int] = None  # None = use full dataset; set int for fast dev runs

    # ── MLP (InteractionMLP) ─────────────────────────────────────────────────
    projection_dim: int = 256
    aux_projection_dim: int = 64
    fusion_dims: Tuple[int, ...] = (512, 256)
    dropout: float = 0.3
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    patience: int = 10          # early stopping patience (epochs)

    # ── Evaluation ───────────────────────────────────────────────────────────
    shap_background_samples: int = 200
    ef_cutoffs: Tuple[float, ...] = (0.01, 0.05)  # EF@1% and EF@5%


# Singleton — import and use directly: from plbind.config import CFG
CFG = Config()
