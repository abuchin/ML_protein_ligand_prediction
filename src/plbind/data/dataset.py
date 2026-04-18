"""PyTorch Dataset for protein-ligand binding — keeps three feature blocks separate.

The InteractionMLP needs protein, ligand, and auxiliary embeddings as separate tensors
so it can apply dedicated projection layers before computing the interaction term.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BindingDataset(Dataset):
    """Dataset returning (protein, ligand, aux, label) tensor tuples.

    Args:
        protein:  float32 array (N, protein_dim)
        ligand:   float32 array (N, ligand_dim)
        labels:   int array (N,)
        aux:      optional float32 array (N, aux_dim)
    """

    def __init__(
        self,
        protein: np.ndarray,
        ligand: np.ndarray,
        labels: np.ndarray,
        aux: Optional[np.ndarray] = None,
    ) -> None:
        self.protein = torch.from_numpy(protein.astype(np.float32))
        self.ligand = torch.from_numpy(ligand.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.aux = torch.from_numpy(aux.astype(np.float32)) if aux is not None else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple:
        if self.aux is not None:
            return self.protein[idx], self.ligand[idx], self.aux[idx], self.labels[idx]
        dummy_aux = torch.zeros(0)
        return self.protein[idx], self.ligand[idx], dummy_aux, self.labels[idx]


def make_dataloaders(
    protein_train: np.ndarray,
    ligand_train: np.ndarray,
    y_train: np.ndarray,
    protein_val: np.ndarray,
    ligand_val: np.ndarray,
    y_val: np.ndarray,
    aux_train: Optional[np.ndarray] = None,
    aux_val: Optional[np.ndarray] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    train_ds = BindingDataset(protein_train, ligand_train, y_train, aux_train)
    val_ds = BindingDataset(protein_val, ligand_val, y_val, aux_val)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_dl, val_dl
