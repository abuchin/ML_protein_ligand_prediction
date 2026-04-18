"""ModelFactory — creates and manages all classifier types."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from .base import BaseModel
from .logistic import LogisticRegressionModel
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel

logger = logging.getLogger(__name__)

_REGISTRY = {
    "logistic_regression": LogisticRegressionModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
}


class ModelFactory:
    """Create, store, and ensemble classifiers."""

    @staticmethod
    def create(model_type: str, **kwargs) -> BaseModel:
        if model_type not in _REGISTRY:
            raise ValueError(f"Unknown model type '{model_type}'. Available: {list(_REGISTRY)}")
        return _REGISTRY[model_type](**kwargs)

    @staticmethod
    def available() -> List[str]:
        return list(_REGISTRY.keys())

    @staticmethod
    def create_all(**kwargs) -> Dict[str, BaseModel]:
        return {name: _REGISTRY[name](**kwargs) for name in _REGISTRY}

    @staticmethod
    def create_ensemble(
        models: Dict[str, BaseModel],
        X_test: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Average predicted probabilities from all trained models.

        Args:
            models:  Dict of model_name → trained BaseModel with predict_proba().
            X_test:  Feature matrix (scaled, if models expect scaled input).
            weights: Optional per-model weights; defaults to uniform.

        Returns:
            np.ndarray of shape (N, 2) — averaged predicted class probabilities.
        """
        probas = []
        model_names = list(models.keys())
        if weights is None:
            weights = {name: 1.0 for name in model_names}

        total_weight = sum(weights.get(n, 1.0) for n in model_names)

        for name, model in models.items():
            w = weights.get(name, 1.0) / total_weight
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba()
            else:
                pred = model.predict()
                # Convert hard predictions to soft probabilities
                proba = np.stack([1 - pred, pred], axis=1).astype(np.float32)
            probas.append(w * proba)

        ensemble_proba = np.sum(probas, axis=0)
        logger.info(
            "Ensemble: averaged %d models (uniform weight=%.3f each)",
            len(models), 1.0 / len(models),
        )
        return ensemble_proba
