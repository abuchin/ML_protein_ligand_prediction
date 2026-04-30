"""Tests for SHAPAnalyzer — covers aggregation, top-features, and plot helpers."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from plbind.evaluation.interpretability import SHAPAnalyzer


@pytest.fixture
def block_map():
    return {
        "protein": slice(0, 4),
        "ligand": slice(4, 9),
        "aux": slice(9, 10),
    }


@pytest.fixture
def feature_names():
    return [f"feat_{i}" for i in range(10)]


@pytest.fixture
def analyzer(block_map, feature_names):
    return SHAPAnalyzer(feature_names=feature_names, block_map=block_map)


@pytest.fixture
def shap_array():
    return np.abs(np.random.RandomState(0).randn(50, 10))


class TestFeatureGroupImportance:
    def test_returns_dataframe(self, analyzer, shap_array):
        result = analyzer.feature_group_importance(shap_array)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_blocks(self, analyzer, shap_array):
        result = analyzer.feature_group_importance(shap_array)
        assert set(result["block"]) == {"protein", "ligand", "aux"}

    def test_fractions_sum_to_one(self, analyzer, shap_array):
        result = analyzer.feature_group_importance(shap_array)
        assert abs(result["fraction"].sum() - 1.0) < 1e-5

    def test_mean_abs_shap_positive(self, analyzer, shap_array):
        result = analyzer.feature_group_importance(shap_array)
        assert (result["mean_abs_shap"] >= 0).all()

    def test_3d_input_uses_positive_class(self, analyzer):
        """3-D SHAP (N, F, C) should select index 1 (positive class)."""
        rng = np.random.RandomState(1)
        shap_3d = rng.randn(20, 10, 2)
        result = analyzer.feature_group_importance(shap_3d)
        assert set(result["block"]) == {"protein", "ligand", "aux"}

    def test_shap_explanation_object(self, analyzer, shap_array):
        """Should accept objects with a .values attribute (shap.Explanation)."""
        mock_explanation = MagicMock()
        mock_explanation.values = shap_array
        result = analyzer.feature_group_importance(mock_explanation)
        assert abs(result["fraction"].sum() - 1.0) < 1e-5

    def test_missing_block_map_raises(self):
        a = SHAPAnalyzer()  # no block_map
        with pytest.raises(ValueError, match="block_map"):
            a.feature_group_importance(np.ones((10, 5)))


class TestTopFeatures:
    def test_returns_n_rows(self, analyzer, shap_array):
        result = analyzer.top_features(shap_array, n=5)
        assert len(result) == 5

    def test_has_expected_columns(self, analyzer, shap_array):
        result = analyzer.top_features(shap_array, n=3)
        assert "feature" in result.columns
        assert "mean_abs_shap" in result.columns

    def test_sorted_descending(self, analyzer, shap_array):
        result = analyzer.top_features(shap_array, n=10)
        vals = result["mean_abs_shap"].values
        assert (vals[:-1] >= vals[1:]).all()

    def test_n_larger_than_features_clamps(self, analyzer, shap_array):
        result = analyzer.top_features(shap_array, n=100)
        assert len(result) == shap_array.shape[1]

    def test_uses_provided_feature_names(self, analyzer, shap_array):
        result = analyzer.top_features(shap_array, n=5)
        assert all(name.startswith("feat_") for name in result["feature"])

    def test_fallback_feature_names(self, block_map, shap_array):
        a = SHAPAnalyzer(block_map=block_map)  # no feature_names
        result = a.top_features(shap_array, n=3)
        assert all(name.startswith("feat_") for name in result["feature"])


class TestPlotSummary:
    def test_no_show_by_default(self, analyzer, shap_array, tmp_path):
        """plot_summary must not call plt.show() when show=False (default)."""
        X = np.random.randn(50, 10).astype(np.float32)
        with patch("shap.summary_plot"), \
             patch("matplotlib.pyplot.show") as mock_show, \
             patch("matplotlib.pyplot.savefig"), \
             patch("matplotlib.pyplot.close"):
            analyzer.plot_summary(shap_array, X, show=False)
        mock_show.assert_not_called()

    def test_show_true_calls_show(self, analyzer, shap_array):
        X = np.random.randn(50, 10).astype(np.float32)
        with patch("shap.summary_plot"), \
             patch("matplotlib.pyplot.show") as mock_show, \
             patch("matplotlib.pyplot.close"):
            analyzer.plot_summary(shap_array, X, show=True)
        mock_show.assert_called_once()

    def test_saves_file_when_path_provided(self, analyzer, shap_array, tmp_path):
        X = np.random.randn(50, 10).astype(np.float32)
        save_path = tmp_path / "shap.png"
        with patch("shap.summary_plot"), \
             patch("matplotlib.pyplot.savefig") as mock_save, \
             patch("matplotlib.pyplot.close"):
            analyzer.plot_summary(shap_array, X, save_path=save_path)
        mock_save.assert_called_once()


class TestPlotGroupImportance:
    def test_no_show_by_default(self, analyzer, shap_array):
        with patch("matplotlib.pyplot.show") as mock_show, \
             patch("matplotlib.pyplot.close"), \
             patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock())), \
             patch("matplotlib.pyplot.tight_layout"):
            analyzer.plot_group_importance(shap_array, show=False)
        mock_show.assert_not_called()

    def test_show_true_calls_show(self, analyzer, shap_array):
        with patch("matplotlib.pyplot.show") as mock_show, \
             patch("matplotlib.pyplot.close"), \
             patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock())), \
             patch("matplotlib.pyplot.tight_layout"):
            analyzer.plot_group_importance(shap_array, show=True)
        mock_show.assert_called_once()
