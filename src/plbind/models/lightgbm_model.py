"""LightGBM classifier — typically 3-5× faster than XGBoost at similar accuracy.

Accepts scipy sparse input natively (important for the fingerprint blocks).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

try:
    from lightgbm import LGBMClassifier
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False
    LGBMClassifier = None

from .base import BaseModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 500,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        class_weight: str = "balanced",
        early_stopping_rounds: int = 20,
        test_size: float = 0.2,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__("LightGBM", test_size, random_state)
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_samples = min_child_samples
        self.class_weight = class_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs
        self.model = self._initialize_model()

    def _initialize_model(self):
        if not _LIGHTGBM_AVAILABLE:
            raise ImportError("Install lightgbm: pip install lightgbm")
        return LGBMClassifier(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_samples=self.min_child_samples,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1,
        )

    def _train_model(self) -> None:
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_train,
            test_size=0.1, random_state=self.random_state, stratify=self.y_train,
        )
        callbacks = []
        try:
            from lightgbm import early_stopping, log_evaluation
            callbacks = [early_stopping(self.early_stopping_rounds, verbose=False), log_evaluation(-1)]
        except ImportError:
            pass

        self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks or None)
        if hasattr(self.model, "best_iteration_"):
            logger.info("LightGBM best iteration: %d", self.model.best_iteration_)

    def _predict_model(self) -> np.ndarray:
        return self.model.predict(self.X_test)

    def predict_proba(self) -> np.ndarray:
        return self.model.predict_proba(self.X_test)

    def tune(self, param_grid: Optional[dict] = None, cv: int = 3, scoring: str = "f1_macro") -> dict:
        if param_grid is None:
            param_grid = {
                "num_leaves": [31, 63, 127],
                "learning_rate": [0.05, 0.1],
                "n_estimators": [200, 500],
            }
        if not _LIGHTGBM_AVAILABLE:
            raise ImportError("Install lightgbm: pip install lightgbm")
        base = LGBMClassifier(
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1,
        )
        gs = GridSearchCV(base, param_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
        gs.fit(self.X_train, self.y_train)
        self.model = gs.best_estimator_
        self.n_estimators = gs.best_params_.get("n_estimators", self.n_estimators)
        self.num_leaves = gs.best_params_.get("num_leaves", self.num_leaves)
        self.learning_rate = gs.best_params_.get("learning_rate", self.learning_rate)
        logger.info("LightGBM best params: %s  score=%.4f", gs.best_params_, gs.best_score_)
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
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
        })
        return info
