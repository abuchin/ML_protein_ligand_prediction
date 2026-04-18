"""Random Forest classifier with permutation-based feature importance.

MDI (mean decrease in impurity) is biased toward high-cardinality features —
it systematically over-ranks the 2000+ fingerprint bits vs. the 95 auxiliary
features regardless of actual predictive value. Permutation importance on the
held-out test set is unbiased and model-agnostic.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV

from .base import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        class_weight: str = "balanced",
        test_size: float = 0.2,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__("Random Forest", test_size, random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.model = self._initialize_model()

    def _initialize_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def _train_model(self) -> None:
        self.model.fit(self.X_train, self.y_train)

    def _predict_model(self) -> np.ndarray:
        return self.model.predict(self.X_test)

    def predict_proba(self) -> np.ndarray:
        return self.model.predict_proba(self.X_test)

    def tune(self, param_grid: Optional[dict] = None, cv: int = 3, scoring: str = "f1_macro") -> dict:
        if param_grid is None:
            param_grid = {"n_estimators": [100, 200], "max_depth": [10, 20, None]}
        base = RandomForestClassifier(
            class_weight=self.class_weight, random_state=self.random_state, n_jobs=self.n_jobs
        )
        gs = GridSearchCV(base, param_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
        gs.fit(self.X_train, self.y_train)
        self.model = gs.best_estimator_
        self.n_estimators = gs.best_params_.get("n_estimators", self.n_estimators)
        self.max_depth = gs.best_params_.get("max_depth", self.max_depth)
        logger.info("RF best params: %s  score=%.4f", gs.best_params_, gs.best_score_)
        return gs.best_params_

    def get_feature_importance(
        self, n_repeats: int = 10, use_permutation: bool = True
    ) -> np.ndarray:
        """Return feature importance scores.

        Args:
            n_repeats:        Permutation repeats (higher = more stable estimates).
            use_permutation:  If True, use permutation importance (unbiased, recommended).
                              If False, fall back to fast MDI (biased toward high-cardinality).
        """
        if not use_permutation:
            return self.model.feature_importances_

        if self.X_test is None or self.y_test is None:
            logger.warning("Cannot compute permutation importance: test set not set. Using MDI.")
            return self.model.feature_importances_

        result = permutation_importance(
            self.model, self.X_test, self.y_test,
            n_repeats=n_repeats, random_state=self.random_state, n_jobs=self.n_jobs,
            scoring="average_precision",
        )
        return result.importances_mean

    def get_feature_importance_df(
        self, feature_names: Optional[List[str]] = None, use_permutation: bool = True
    ) -> pd.DataFrame:
        importance = self.get_feature_importance(use_permutation=use_permutation)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        return pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
            "importance", ascending=False
        )

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({"n_estimators": self.n_estimators, "max_depth": self.max_depth})
        return info
