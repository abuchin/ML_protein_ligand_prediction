"""
Model factory for creating different types of models.
"""

import logging
from typing import Dict, Any
from .logistic_regression_model import LogisticRegressionModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel
from .mlp_model import MLPModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating model instances."""

    @staticmethod
    def create_model(model_type: str, **kwargs) -> Any:
        model_type = model_type.lower()
        if model_type == 'logistic_regression':
            return LogisticRegressionModel(**kwargs)
        elif model_type == 'random_forest':
            return RandomForestModel(**kwargs)
        elif model_type == 'xgboost':
            return XGBoostModel(**kwargs)
        elif model_type == 'mlp':
            return MLPModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_default_params(model_type: str) -> Dict[str, Any]:
        model_type = model_type.lower()
        if model_type == 'logistic_regression':
            return {'penalty': 'l2', 'C': 1.0, 'class_weight': 'balanced',
                    'max_iter': 1000, 'test_size': 0.2, 'random_state': 42}
        elif model_type == 'random_forest':
            return {'n_estimators': 100, 'max_depth': None, 'class_weight': 'balanced',
                    'test_size': 0.2, 'random_state': 42}
        elif model_type == 'xgboost':
            return {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
                    'scale_pos_weight': 1.0, 'early_stopping_rounds': 10,
                    'test_size': 0.2, 'random_state': 42}
        elif model_type == 'mlp':
            return {'hidden_size': 256, 'output_size': 2, 'batch_size': 32,
                    'learning_rate': 0.001, 'epochs': 50, 'patience': 5,
                    'test_size': 0.2, 'random_state': 42}
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_available_models() -> list:
        return ['logistic_regression', 'random_forest', 'xgboost', 'mlp']

    @staticmethod
    def create_all_models(input_size: int = None, **kwargs) -> Dict[str, Any]:
        models = {}
        for mtype in ModelFactory.get_available_models():
            try:
                if mtype == 'mlp' and input_size is None:
                    logger.warning("Skipping MLP: input_size required")
                    continue
                params = ModelFactory.get_default_params(mtype)
                params.update(kwargs)
                if mtype == 'mlp':
                    params['input_size'] = input_size
                models[mtype] = ModelFactory.create_model(mtype, **params)
            except Exception as e:
                logger.error(f"Failed to create {mtype}: {e}")
        return models
