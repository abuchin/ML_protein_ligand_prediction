"""Unit tests for ModelEvaluator."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from plbind.evaluation.evaluator import ModelEvaluator


@pytest.fixture
def evaluator():
    return ModelEvaluator()


@pytest.fixture
def perfect_preds():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    return y_true, y_pred


@pytest.fixture
def binary_preds_with_proba():
    rng = np.random.RandomState(42)
    y_true = np.tile([0, 1], 50)
    y_pred = np.tile([0, 1], 50)
    proba = np.column_stack([1 - rng.rand(100) * 0.3, rng.rand(100) * 0.3 + 0.7])
    proba[y_true == 0] = proba[y_true == 0][:, ::-1]
    return y_true, y_pred, proba


class TestComputeMetrics:
    def test_perfect_accuracy(self, evaluator, perfect_preds):
        y_true, y_pred = perfect_preds
        m = evaluator.compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0

    def test_keys_present_without_proba(self, evaluator, perfect_preds):
        y_true, y_pred = perfect_preds
        m = evaluator.compute_metrics(y_true, y_pred)
        for key in ("accuracy", "precision_binary", "recall_binary", "f1_binary",
                    "f1_macro", "confusion_matrix"):
            assert key in m

    def test_roc_auc_present_with_proba(self, evaluator, binary_preds_with_proba):
        y_true, y_pred, proba = binary_preds_with_proba
        m = evaluator.compute_metrics(y_true, y_pred, y_proba=proba)
        assert "roc_auc" in m
        assert m["roc_auc"] is not None

    def test_pr_auc_present_with_proba(self, evaluator, binary_preds_with_proba):
        y_true, y_pred, proba = binary_preds_with_proba
        m = evaluator.compute_metrics(y_true, y_pred, y_proba=proba)
        assert "pr_auc" in m

    def test_roc_auc_absent_without_proba(self, evaluator, perfect_preds):
        y_true, y_pred = perfect_preds
        m = evaluator.compute_metrics(y_true, y_pred)
        assert "roc_auc" not in m

    def test_confusion_matrix_shape(self, evaluator, perfect_preds):
        y_true, y_pred = perfect_preds
        m = evaluator.compute_metrics(y_true, y_pred)
        assert np.array(m["confusion_matrix"]).shape == (2, 2)

    def test_f1_macro_range(self, evaluator, perfect_preds):
        y_true, y_pred = perfect_preds
        m = evaluator.compute_metrics(y_true, y_pred)
        assert 0.0 <= m["f1_macro"] <= 1.0


class TestCompareModels:
    def _make_results(self, names):
        return {
            n: {
                "accuracy": 0.8,
                "f1_binary": 0.79,
                "f1_macro": 0.78,
                "roc_auc": 0.85,
                "pr_auc": 0.75,
            }
            for n in names
        }

    def test_returns_dataframe(self, evaluator):
        results = self._make_results(["LR", "RF"])
        df = evaluator.compare_models(results)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, evaluator):
        results = self._make_results(["LR", "RF", "XGB"])
        df = evaluator.compare_models(results)
        assert len(df) == 3

    def test_columns_present(self, evaluator):
        results = self._make_results(["LR"])
        df = evaluator.compare_models(results)
        for col in ("accuracy", "f1_macro", "roc_auc"):
            assert col in df.columns


class TestGetBestModel:
    def _make_results(self):
        return {
            "LR":  {"f1_macro": 0.60},
            "RF":  {"f1_macro": 0.80},
            "XGB": {"f1_macro": 0.75},
        }

    def test_returns_best_by_f1_macro(self, evaluator):
        results = self._make_results()
        name, _ = evaluator.get_best_model(results, metric="f1_macro")
        assert name == "RF"

    def test_empty_results_raises(self, evaluator):
        with pytest.raises((ValueError, TypeError)):
            evaluator.get_best_model({}, metric="f1_macro")


class TestCrossValidate:
    def test_returns_scoring_keys(self, evaluator):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 5)
        y = np.tile([0, 1], 50)
        estimator = LogisticRegression(max_iter=200)
        cv_results = evaluator.cross_validate(estimator, X, y, cv=3)
        for scoring in ("f1_macro", "roc_auc", "average_precision"):
            assert scoring in cv_results
            assert "mean" in cv_results[scoring]
            assert "std" in cv_results[scoring]

    def test_mean_in_range(self, evaluator):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 5)
        y = np.tile([0, 1], 50)
        estimator = LogisticRegression(max_iter=200)
        cv_results = evaluator.cross_validate(estimator, X, y, cv=3)
        assert 0.0 <= cv_results["f1_macro"]["mean"] <= 1.0


class TestEvaluateModel:
    def test_evaluate_trained_model(self):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 5)
        y = np.tile([0, 1], 50)
        clf = LogisticRegression(max_iter=200).fit(X, y)

        class FakeModel:
            model = clf

        result = ModelEvaluator.evaluate_model(FakeModel(), X, y)
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0
