"""Unit tests for FeatureBuilder."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from plbind.features.feature_builder import FeatureBuilder

PROT_DIM = 4
FP_DIM = 8
DESC_DIM = 3
N_LIGS = 10
N_ROWS = 10


def _make_builder(n=N_ROWS, with_aux=False):
    rng = np.random.RandomState(42)
    protein_embeddings = {
        f"P{i:04d}": rng.rand(PROT_DIM).astype(np.float32) for i in range(n)
    }
    fp_matrix = sp.csr_matrix(rng.randint(0, 2, (N_LIGS, FP_DIM)).astype(np.float32))
    desc_matrix = rng.rand(N_LIGS, DESC_DIM).astype(np.float32)
    cid_to_row = {i + 1: i for i in range(N_LIGS)}
    aux_features = None
    if with_aux:
        aux_features = pd.DataFrame(
            rng.rand(n, 2),
            index=[f"P{i:04d}" for i in range(n)],
            columns=["aux_a", "aux_b"],
        )
    return FeatureBuilder(protein_embeddings, cid_to_row, fp_matrix, desc_matrix, aux_features)


def _make_df(n=N_ROWS):
    return pd.DataFrame({
        "UniProt_ID": [f"P{i:04d}" for i in range(n)],
        "pubchem_cid": list(range(1, n + 1)),
        "bound": [i % 2 for i in range(n)],
    })


class TestFeatureBuilderInit:
    def test_protein_dim(self):
        assert _make_builder().protein_dim == PROT_DIM

    def test_ligand_dim(self):
        assert _make_builder().ligand_dim == FP_DIM + DESC_DIM

    def test_total_dim_without_aux(self):
        b = _make_builder(with_aux=False)
        assert b.total_dim == PROT_DIM + FP_DIM + DESC_DIM

    def test_aux_dim_zero_without_aux(self):
        assert _make_builder(with_aux=False).aux_dim == 0

    def test_aux_dim_with_aux(self):
        assert _make_builder(with_aux=True).aux_dim == 2


class TestBuild:
    def test_output_types(self):
        X, y, block_map, _ = _make_builder().build(_make_df())
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(block_map, dict)

    def test_feature_width(self):
        X, y, _, _ = _make_builder().build(_make_df())
        assert X.shape[1] == PROT_DIM + FP_DIM + DESC_DIM

    def test_row_count_matches(self):
        X, y, _, _ = _make_builder().build(_make_df())
        assert len(X) == len(y) == N_ROWS

    def test_missing_protein_skipped(self):
        df = _make_df()
        df.loc[0, "UniProt_ID"] = "MISSING"
        X, y, _, _ = _make_builder().build(df, log_attrition=False)
        assert len(X) == N_ROWS - 1

    def test_missing_ligand_skipped(self):
        df = _make_df()
        df.loc[0, "pubchem_cid"] = 9999
        X, y, _, _ = _make_builder().build(df, log_attrition=False)
        assert len(X) == N_ROWS - 1

    def test_binary_labels_are_0_or_1(self):
        _, y, _, _ = _make_builder().build(_make_df())
        assert set(np.unique(y)).issubset({0, 1})

    def test_feature_width_with_aux(self):
        X, _, _, _ = _make_builder(with_aux=True).build(_make_df())
        assert X.shape[1] == PROT_DIM + FP_DIM + DESC_DIM + 2


class TestBuildBlocks:
    def test_returns_four_items(self):
        result = _make_builder().build_blocks(_make_df())
        assert len(result) == 4

    def test_protein_block_shape(self):
        protein_block, _, _, _ = _make_builder().build_blocks(_make_df())
        assert protein_block.shape == (N_ROWS, PROT_DIM)

    def test_ligand_block_shape(self):
        _, ligand_block, _, _ = _make_builder().build_blocks(_make_df())
        assert ligand_block.shape == (N_ROWS, FP_DIM + DESC_DIM)

    def test_aux_block_none_without_aux(self):
        _, _, aux_block, _ = _make_builder(with_aux=False).build_blocks(_make_df())
        assert aux_block is None

    def test_aux_block_shape_with_aux(self):
        _, _, aux_block, _ = _make_builder(with_aux=True).build_blocks(_make_df())
        assert aux_block is not None
        assert aux_block.shape == (N_ROWS, 2)

    def test_labels_shape(self):
        _, _, _, y = _make_builder().build_blocks(_make_df())
        assert len(y) == N_ROWS


class TestBlockMap:
    def test_keys_present(self):
        bm = _make_builder().block_map
        for key in ("protein", "ligand", "aux"):
            assert key in bm

    def test_protein_slice_length(self):
        bm = _make_builder().block_map
        s = bm["protein"]
        assert s.stop - s.start == PROT_DIM

    def test_ligand_slice_length(self):
        bm = _make_builder().block_map
        s = bm["ligand"]
        assert s.stop - s.start == FP_DIM + DESC_DIM

    def test_slices_are_contiguous(self):
        bm = _make_builder().block_map
        assert bm["protein"].stop == bm["ligand"].start
        assert bm["ligand"].stop == bm["aux"].start
