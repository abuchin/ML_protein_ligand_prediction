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
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, stratify=y_arr, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    model.scaler = scaler
    model.X_train = scaler.fit_transform(X_train).astype(np.float32)
    model.X_test = scaler.transform(X_test).astype(np.float32)
    model.y_train = y_train.astype(np.int32)
    model.y_test = y_test.astype(np.int32)
    model.train()
    model.predict()
    return model


# ---------------------------------------------------------------------------
# BaseModel.split_data is deprecated — verify it raises RuntimeError
# ---------------------------------------------------------------------------

class TestSplitDataDeprecated:
    def test_raises_runtime_error(self, synthetic_data):
        X, y = synthetic_data
        m = LogisticRegressionModel()
        with pytest.raises(RuntimeError, match="deprecated"):
            m.split_data(X, y)


# ---------------------------------------------------------------------------
# BaseModel.train / predict timing
# ---------------------------------------------------------------------------

class TestTrainPredict:
    def test_training_time_recorded(self, synthetic_data):
        X, y = synthetic_data
        m = _train_sklearn_model(LogisticRegressionModel(), X, y)
        assert m.training_time is not None and m.training_time >= 0

    def test_prediction_time_recorded(self, synthetic_data):
        X, y = synthetic_data
        m = _train_sklearn_model(LogisticRegressionModel(), X, y)
        assert m.prediction_time is not None and m.prediction_time >= 0

    def test_y_pred_shape(self, synthetic_data):
        X, y = synthetic_data
        m = _train_sklearn_model(LogisticRegressionModel(), X, y)
        assert len(m.y_pred) == m.X_test.shape[0]

    def test_get_predictions_raises_before_train(self, synthetic_data):
        X, y = synthetic_data
        m = LogisticRegressionModel()
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
        _train_sklearn_model(m, X, y)
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
