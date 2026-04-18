"""Unit tests for model implementations (LR, RF, XGBoost)."""

import pickle
import tempfile
import numpy as np
import pandas as pd
import pytest

from plbind.models.logistic import LogisticRegressionModel
from plbind.models.random_forest import RandomForestModel
from plbind.models.xgboost_model import XGBoostModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n=200, d=20, seed=42):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, d), columns=[f"f{i}" for i in range(d)])
    y = pd.Series(np.tile([0, 1], n // 2))
    return X, y


def _train_sklearn_model(model, X, y):
    model.split_data(X, y)
    model.train()
    model.predict()
    return model


# ---------------------------------------------------------------------------
# BaseModel.split_data (tested via concrete subclass)
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_shapes_consistent(self, synthetic_data):
        X, y = synthetic_data
        m = LogisticRegressionModel()
        m.split_data(X, y)
        n_train = m.X_train.shape[0]
        n_test = m.X_test.shape[0]
        assert n_train + n_test == len(X)

    def test_test_size_respected(self, synthetic_data):
        X, y = synthetic_data
        m = LogisticRegressionModel(test_size=0.25)
        m.split_data(X, y)
        expected_test = int(len(X) * 0.25)
        assert abs(m.X_test.shape[0] - expected_test) <= 2

    def test_scaler_fitted_on_train(self, synthetic_data):
        X, y = synthetic_data
        m = LogisticRegressionModel()
        m.split_data(X, y)
        assert m.scaler is not None
        assert m.scaler.mean_.shape[0] == X.shape[1]

    def test_stratification_preserves_ratio(self, synthetic_data):
        X, y = synthetic_data
        m = LogisticRegressionModel()
        m.split_data(X, y)
        train_ratio = m.y_train.mean()
        test_ratio = m.y_test.mean()
        assert abs(train_ratio - 0.5) < 0.1
        assert abs(test_ratio - 0.5) < 0.1


# ---------------------------------------------------------------------------
# BaseModel.train / predict timing
# ---------------------------------------------------------------------------

class TestTrainPredict:
    def test_training_time_recorded(self, synthetic_data):
        X, y = synthetic_data
        m = LogisticRegressionModel()
        m.split_data(X, y)
        m.train()
        assert m.training_time is not None and m.training_time >= 0

    def test_prediction_time_recorded(self, synthetic_data):
        X, y = synthetic_data
        m = LogisticRegressionModel()
        m.split_data(X, y)
        m.train()
        m.predict()
        assert m.prediction_time is not None and m.prediction_time >= 0

    def test_y_pred_shape(self, synthetic_data):
        X, y = synthetic_data
        m = LogisticRegressionModel()
        m.split_data(X, y)
        m.train()
        preds = m.predict()
        assert len(preds) == m.X_test.shape[0]

    def test_get_predictions_raises_before_train(self, synthetic_data):
        X, y = synthetic_data
        m = LogisticRegressionModel()
        m.split_data(X, y)
        with pytest.raises(ValueError):
            m.get_predictions()

    def test_get_predictions_after_train(self, synthetic_data):
        X, y = synthetic_data
        m = _train_sklearn_model(LogisticRegressionModel(), *synthetic_data)
        y_pred, y_true = m.get_predictions()
        assert len(y_pred) == len(y_true)


# ---------------------------------------------------------------------------
# LogisticRegressionModel
# ---------------------------------------------------------------------------

class TestLogisticRegressionModel:
    def test_predict_proba_shape(self, synthetic_data):
        X, y = synthetic_data
        m = _train_sklearn_model(LogisticRegressionModel(), X, y)
        proba = m.predict_proba()
        assert proba.shape == (m.X_test.shape[0], 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_feature_importance_length(self, synthetic_data):
        X, y = synthetic_data
        m = _train_sklearn_model(LogisticRegressionModel(), X, y)
        fi = m.get_feature_importance()
        assert len(fi) == X.shape[1]

    def test_get_model_info_keys(self, synthetic_data):
        X, y = synthetic_data
        m = _train_sklearn_model(LogisticRegressionModel(), X, y)
        info = m.get_model_info()
        for key in ("model_name", "penalty", "C"):
            assert key in info

    def test_save_load_roundtrip(self, synthetic_data):
        X, y = synthetic_data
        m = _train_sklearn_model(LogisticRegressionModel(), X, y)
        original_preds = m.y_pred.copy()
        X_test = m.X_test
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            m.save_model(f.name)
            m2 = LogisticRegressionModel()
            m2.load_model(f.name)
        assert np.array_equal(m2.model.predict(X_test), original_preds)

    def test_tune_returns_best_params(self, synthetic_data):
        X, y = synthetic_data
        m = LogisticRegressionModel()
        m.split_data(X, y)
        best = m.tune(param_grid={"C": [0.1, 1.0], "penalty": ["l2"]}, cv=2)
        assert "C" in best


# ---------------------------------------------------------------------------
# RandomForestModel
# ---------------------------------------------------------------------------

class TestRandomForestModel:
    def test_predictions_are_binary(self, synthetic_data):
        X, y = synthetic_data
        m = _train_sklearn_model(RandomForestModel(n_estimators=5), X, y)
        assert set(m.y_pred).issubset({0, 1})

    def test_predict_proba_shape(self, synthetic_data):
        X, y = synthetic_data
        m = _train_sklearn_model(RandomForestModel(n_estimators=5), X, y)
        proba = m.predict_proba()
        assert proba.shape[1] == 2

    def test_feature_importance_sums_to_one(self, synthetic_data):
        X, y = synthetic_data
        m = _train_sklearn_model(RandomForestModel(n_estimators=5), X, y)
        # use_permutation=False → MDI importances, which sum to 1
        fi = m.get_feature_importance(use_permutation=False)
        assert abs(fi.sum() - 1.0) < 1e-5

    def test_feature_importance_df_columns(self, synthetic_data):
        X, y = synthetic_data
        m = _train_sklearn_model(RandomForestModel(n_estimators=5), X, y)
        df = m.get_feature_importance_df(feature_names=X.columns.tolist(), use_permutation=False)
        assert list(df.columns) == ["feature", "importance"]
        assert len(df) == X.shape[1]

    def test_save_load_roundtrip(self, synthetic_data):
        X, y = synthetic_data
        m = _train_sklearn_model(RandomForestModel(n_estimators=5), X, y)
        original_preds = m.y_pred.copy()
        X_test = m.X_test
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            m.save_model(f.name)
            m2 = RandomForestModel()
            m2.load_model(f.name)
        assert np.array_equal(m2.model.predict(X_test), original_preds)


# ---------------------------------------------------------------------------
# XGBoostModel
# ---------------------------------------------------------------------------

class TestXGBoostModel:
    def test_train_predict_end_to_end(self, synthetic_data):
        X, y = synthetic_data
        m = XGBoostModel(n_estimators=10, max_depth=3, early_stopping_rounds=2)
        m = _train_sklearn_model(m, X, y)
        assert len(m.y_pred) == m.X_test.shape[0]

    def test_predict_proba_shape(self, synthetic_data):
        X, y = synthetic_data
        m = _train_sklearn_model(
            XGBoostModel(n_estimators=10, max_depth=3, early_stopping_rounds=2), X, y
        )
        proba = m.predict_proba()
        assert proba.shape[1] == 2

    def test_get_model_info_keys(self, synthetic_data):
        X, y = synthetic_data
        m = _train_sklearn_model(
            XGBoostModel(n_estimators=10, max_depth=3, early_stopping_rounds=2), X, y
        )
        info = m.get_model_info()
        for key in ("n_estimators", "max_depth", "learning_rate"):
            assert key in info

    def test_save_load_roundtrip(self, synthetic_data):
        X, y = synthetic_data
        m = _train_sklearn_model(
            XGBoostModel(n_estimators=10, max_depth=3, early_stopping_rounds=2), X, y
        )
        original_preds = m.y_pred.copy()
        X_test = m.X_test
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            m.save_model(f.name)
            m2 = XGBoostModel(n_estimators=10)
            m2.load_model(f.name)
        assert np.array_equal(m2.model.predict(X_test), original_preds)
