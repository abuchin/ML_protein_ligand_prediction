"""Shared fixtures for all unit tests."""

import numpy as np
import pandas as pd
import pytest


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
