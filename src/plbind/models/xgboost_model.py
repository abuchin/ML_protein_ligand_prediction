"""XGBoost classifier.

Fix from original: early_stopping_rounds must be passed to XGBClassifier constructor,
not to fit(), in xgboost >= 2.0.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

from .base import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        scale_pos_weight: float = 1.0,
        early_stopping_rounds: int = 20,
        test_size: float = 0.2,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__("XGBoost", test_size, random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs
        self.model = self._initialize_model()

    def _initialize_model(self) -> XGBClassifier:
        # early_stopping_rounds goes in the constructor (xgboost >= 2.0 requirement)
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=self.scale_pos_weight,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0,
        )

    def _train_model(self) -> None:
        # Carve internal val set for early stopping; does NOT bleed into self.X_test.
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_train,
            test_size=0.1, random_state=self.random_state, stratify=self.y_train,
        )
        self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        if hasattr(self.model, "best_iteration"):
            logger.info("XGBoost best iteration: %d", self.model.best_iteration)

    def _predict_model(self) -> np.ndarray:
        return self.model.predict(self.X_test)

    def predict_proba(self) -> np.ndarray:
        return self.model.predict_proba(self.X_test)

    def tune(self, param_grid: Optional[dict] = None, cv: int = 3, scoring: str = "f1_macro") -> dict:
        if param_grid is None:
            param_grid = {
                "max_depth": [3, 6],
                "learning_rate": [0.05, 0.1],
                "n_estimators": [100, 200],
            }
        # No early stopping during GridSearch (CV manages splits)
        base = XGBClassifier(
            scale_pos_weight=self.scale_pos_weight,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0,
        )
        gs = GridSearchCV(base, param_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
        gs.fit(self.X_train, self.y_train)
        self.model = gs.best_estimator_
        self.n_estimators = gs.best_params_.get("n_estimators", self.n_estimators)
        self.max_depth = gs.best_params_.get("max_depth", self.max_depth)
        self.learning_rate = gs.best_params_.get("learning_rate", self.learning_rate)
        logger.info("XGBoost best params: %s  score=%.4f", gs.best_params_, gs.best_score_)
        return gs.best_params_

    def get_feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_

    def get_feature_importance_df(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        importance = self.get_feature_importance()
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        return pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
            "importance", ascending=False
        )

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        })
        return info
