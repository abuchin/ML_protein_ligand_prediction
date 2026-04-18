"""Unit tests for ModelFactory."""

import pytest

from plbind.models.factory import ModelFactory
from plbind.models.logistic import LogisticRegressionModel
from plbind.models.random_forest import RandomForestModel
from plbind.models.xgboost_model import XGBoostModel
from plbind.models.lightgbm_model import LightGBMModel


class TestCreateModel:
    def test_creates_logistic_regression(self):
        m = ModelFactory.create("logistic_regression")
        assert isinstance(m, LogisticRegressionModel)

    def test_creates_random_forest(self):
        m = ModelFactory.create("random_forest")
        assert isinstance(m, RandomForestModel)

    def test_creates_xgboost(self):
        m = ModelFactory.create("xgboost")
        assert isinstance(m, XGBoostModel)

    def test_creates_lightgbm(self):
        m = ModelFactory.create("lightgbm")
        assert isinstance(m, LightGBMModel)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelFactory.create("svm")

    def test_kwargs_forwarded(self):
        m = ModelFactory.create("logistic_regression", C=0.5)
        assert m.C == 0.5


class TestGetAvailableModels:
    def test_returns_list(self):
        models = ModelFactory.available()
        assert isinstance(models, list)

    def test_contains_all_types(self):
        models = ModelFactory.available()
        for t in ("logistic_regression", "random_forest", "xgboost", "lightgbm"):
            assert t in models


class TestCreateAllModels:
    def test_creates_all_model_types(self):
        models = ModelFactory.create_all()
        for t in ("logistic_regression", "random_forest", "xgboost", "lightgbm"):
            assert t in models

    def test_returns_base_model_instances(self):
        from plbind.models.base import BaseModel
        models = ModelFactory.create_all()
        for m in models.values():
            assert isinstance(m, BaseModel)

    def test_kwargs_forwarded_to_all(self):
        models = ModelFactory.create_all(random_state=7)
        for m in models.values():
            assert m.random_state == 7
