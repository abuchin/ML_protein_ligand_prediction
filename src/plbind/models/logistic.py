"""Logistic Regression classifier — interpretable baseline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from .base import BaseModel

logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseModel):
    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        class_weight: str = "balanced",
        max_iter: int = 1000,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        super().__init__("Logistic Regression", test_size, random_state)
        self.C = C
        self.penalty = penalty
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.model = self._initialize_model()

    def _initialize_model(self) -> LogisticRegression:
        return LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
            n_jobs=-1,
        )

    def _train_model(self) -> None:
        self.model.fit(self.X_train, self.y_train)

    def _predict_model(self) -> np.ndarray:
        return self.model.predict(self.X_test)

    def predict_proba(self) -> np.ndarray:
        return self.model.predict_proba(self.X_test)

    def tune(self, param_grid: Optional[dict] = None, cv: int = 3, scoring: str = "f1_macro") -> dict:
        if param_grid is None:
            param_grid = {"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l2"]}
        base = LogisticRegression(
            class_weight=self.class_weight, max_iter=self.max_iter,
            random_state=self.random_state, solver="lbfgs", n_jobs=-1,
        )
        gs = GridSearchCV(base, param_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
        gs.fit(self.X_train, self.y_train)
        self.model = gs.best_estimator_
        self.C = gs.best_params_.get("C", self.C)
        logger.info("LR best params: %s  score=%.4f", gs.best_params_, gs.best_score_)
        return gs.best_params_

    def get_feature_importance(self) -> np.ndarray:
        return np.abs(self.model.coef_[0])

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({"C": self.C, "penalty": self.penalty})
        return info
