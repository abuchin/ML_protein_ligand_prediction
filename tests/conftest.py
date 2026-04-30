"""Shared fixtures for all unit tests."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def synthetic_data():
    """200 samples, 20 features, balanced binary labels."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(200, 20), columns=[f"f{i}" for i in range(20)])
    y = pd.Series(np.tile([0, 1], 100))
    return X, y


@pytest.fixture
def tiny_feature_data():
    """Minimal data for MLP tests (keeps training fast)."""
    rng = np.random.RandomState(0)
    n, d = 120, 10
    X = pd.DataFrame(rng.randn(n, d), columns=[f"f{i}" for i in range(d)])
    y = pd.Series(np.tile([0, 1], n // 2))
    return X, y


@pytest.fixture
def realistic_feature_data():
    """Feature matrix mirroring real pipeline dimensions (small subset).

    Protein: 480-dim (ESM2 35M mean-only), ligand-fp: 2214, desc: 15, aux: 0.
    Total: 2709 features per sample.
    """
    rng = np.random.RandomState(99)
    PROT_DIM = 480
    FP_DIM = 2214
    DESC_DIM = 15
    N = 300
    protein_block = rng.randn(N, PROT_DIM).astype(np.float32)
    fp_block = sp.random(N, FP_DIM, density=0.05, format="csr", dtype=np.float32,
                         random_state=rng)
    desc_block = rng.randn(N, DESC_DIM).astype(np.float32)
    X = np.concatenate([protein_block, fp_block.toarray(), desc_block], axis=1)
    y = rng.randint(0, 2, N).astype(np.int32)
    return X, y, protein_block, fp_block, desc_block


@pytest.fixture
def sparse_fingerprint_matrix():
    """Sparse fingerprint matrix matching real ligand encoder output."""
    rng = np.random.RandomState(7)
    return sp.random(500, 2214, density=0.04, format="csr", dtype=np.int32,
                     random_state=rng)
