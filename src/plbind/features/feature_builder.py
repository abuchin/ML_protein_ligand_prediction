"""Assemble protein, ligand, and auxiliary features into model-ready matrices.

Flat feature layout (total dims depend on pooling strategy):
    protein embedding  | morgan counts | MACCS | atom-pair | descriptors | auxiliary
    2560 (mean_max)    |  1024         |  166  |  1024     |  15         |  95
    1280 (mean only)   |               |       |           |             |
    ──────────────────────────────────────────────────────────────────────────────
    Total (mean_max):  4884
    Total (mean only): 3604

The FeatureBuilder also exposes build_blocks() which returns the three blocks
separately — required by InteractionMLP so each block can have its own
projection layer rather than being treated as one undifferentiated vector.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from plbind.data.ligand_encoder import LigandEncoder, TOTAL_FINGERPRINT_BITS, DESCRIPTOR_NAMES

logger = logging.getLogger(__name__)

# Block names for feature importance aggregation
BLOCK_PROTEIN = "protein"
BLOCK_LIGAND = "ligand"
BLOCK_AUX = "aux"


class FeatureBuilder:
    """Build model input features from pre-computed embeddings and encodings.

    Args:
        protein_embeddings: dict UniProt_ID → np.ndarray (protein_dim,)
        cid_to_row:         dict pubchem_cid → row index in fp_matrix / desc_matrix
        fp_matrix:          scipy.sparse.csr_matrix (N_ligs, TOTAL_FINGERPRINT_BITS)
        desc_matrix:        np.ndarray (N_ligs, 15)
        aux_features:       pd.DataFrame indexed by UniProt_ID (aux feature columns)
    """

    def __init__(
        self,
        protein_embeddings: Dict[str, np.ndarray],
        cid_to_row: Dict[int, int],
        fp_matrix: sp.csr_matrix,
        desc_matrix: np.ndarray,
        aux_features: Optional[pd.DataFrame] = None,
    ) -> None:
        self.protein_embeddings = protein_embeddings
        self.cid_to_row = cid_to_row
        self.fp_matrix = fp_matrix
        self.desc_matrix = desc_matrix
        self.aux_features = aux_features

        # Infer dims
        sample_emb = next(iter(protein_embeddings.values()))
        self.protein_dim = sample_emb.shape[0]
        self.ligand_fp_dim = fp_matrix.shape[1]
        self.ligand_desc_dim = desc_matrix.shape[1]
        self.ligand_dim = self.ligand_fp_dim + self.ligand_desc_dim
        self.aux_dim = aux_features.shape[1] if aux_features is not None else 0
        self.total_dim = self.protein_dim + self.ligand_dim + self.aux_dim

    # ── Block index map ───────────────────────────────────────────────────────

    @property
    def block_map(self) -> Dict[str, slice]:
        """Mapping from block name → column slice in the flat feature matrix."""
        p_end = self.protein_dim
        l_end = p_end + self.ligand_dim
        a_end = l_end + self.aux_dim
        return {
            BLOCK_PROTEIN: slice(0, p_end),
            BLOCK_LIGAND: slice(p_end, l_end),
            BLOCK_AUX: slice(l_end, a_end),
        }

    @property
    def feature_names(self) -> List[str]:
        protein_names = [f"esm2_{i}" for i in range(self.protein_dim)]
        ligand_enc = LigandEncoder()
        ligand_names = ligand_enc.feature_names
        aux_names = list(self.aux_features.columns) if self.aux_features is not None else []
        return protein_names + ligand_names + aux_names

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        df: pd.DataFrame,
        log_attrition: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, slice]]:
        """Build flat feature matrix and binary label vector.

        Args:
            df:             DataFrame with UniProt_ID, pubchem_cid, bound columns.
            log_attrition:  Whether to log how many rows are dropped for missing features.

        Returns:
            X:          np.ndarray float32 (N, total_dim)
            y:          np.ndarray int32 (N,)
            block_map:  Dict mapping block name → column slice
        """
        protein_block, ligand_fp_block, desc_block, aux_block, mask = self._extract_blocks(df)

        n_dropped = (~mask).sum()
        if log_attrition and n_dropped > 0:
            logger.info(
                "Attrition: dropped %d / %d rows (%.1f%%) — missing protein embedding or SMILES.",
                n_dropped, len(df), 100 * n_dropped / len(df),
            )

        parts = [protein_block, ligand_fp_block.toarray().astype(np.float32), desc_block]
        if aux_block is not None:
            parts.append(aux_block)

        X = np.concatenate(parts, axis=1).astype(np.float32)
        y = df.loc[mask, "bound"].values.astype(np.int32)
        df_filtered = df.loc[mask].reset_index(drop=True)
        return X, y, self.block_map, df_filtered

    def build_blocks(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Return separate (protein, ligand, aux, y) blocks for InteractionMLP.

        Returns:
            protein_block:  float32 (N, protein_dim)
            ligand_block:   float32 (N, ligand_dim)
            aux_block:      float32 (N, aux_dim) or None
            y:              int32 (N,)
        """
        protein_block, ligand_fp_block, desc_block, aux_block, mask = self._extract_blocks(df)

        ligand_block = np.concatenate(
            [ligand_fp_block.toarray().astype(np.float32), desc_block], axis=1
        )
        y = df.loc[mask, "bound"].values.astype(np.int32)
        return protein_block, ligand_block, aux_block, y

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract_blocks(self, df: pd.DataFrame):
        """Extract aligned feature blocks, returning a boolean mask for valid rows."""
        valid_rows = []
        prot_rows = []
        lig_fp_rows = []
        lig_desc_rows = []
        aux_rows = []

        for idx, row in df.iterrows():
            uid = row["UniProt_ID"]
            cid = int(row["pubchem_cid"])

            prot_emb = self.protein_embeddings.get(uid)
            lig_row = self.cid_to_row.get(cid)

            if prot_emb is None or lig_row is None:
                valid_rows.append(False)
                continue

            valid_rows.append(True)
            prot_rows.append(prot_emb)
            lig_fp_rows.append(lig_row)
            lig_desc_rows.append(self.desc_matrix[lig_row])

            if self.aux_features is not None and uid in self.aux_features.index:
                aux_rows.append(self.aux_features.loc[uid].values.astype(np.float32))
            elif self.aux_features is not None:
                aux_rows.append(np.zeros(self.aux_dim, dtype=np.float32))

        mask = pd.Series(valid_rows, index=df.index)

        protein_block = np.stack(prot_rows).astype(np.float32)
        lig_fp_block = self.fp_matrix[lig_fp_rows]  # sparse slice
        desc_block = np.stack(lig_desc_rows).astype(np.float32)
        aux_block = np.stack(aux_rows).astype(np.float32) if aux_rows else None

        return protein_block, lig_fp_block, desc_block, aux_block, mask
