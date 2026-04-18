"""
Random Forest model for protein-ligand binding prediction.
"""

import pickle
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest model implementation."""

    def __init__(self, n_estimators=100, max_depth=None, class_weight='balanced',
                 test_size=0.2, random_state=42):
        super().__init__("Random Forest", test_size, random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.model = self._initialize_model(
            n_estimators=n_estimators, max_depth=max_depth,
            class_weight=class_weight, random_state=random_state, n_jobs=-1
        )

    def _initialize_model(self, **kwargs) -> RandomForestClassifier:
        return RandomForestClassifier(**kwargs)

    def _train_model(self) -> None:
        self.model.fit(self.X_train, self.y_train)

    def _predict_model(self) -> np.ndarray:
        return self.model.predict(self.X_test)

    def predict_proba(self) -> np.ndarray:
        return self.model.predict_proba(self.X_test)

    def tune(self, param_grid: dict = None, cv: int = 3, scoring: str = 'f1_macro') -> dict:
        """Run GridSearchCV on X_train/y_train and replace self.model with the best estimator."""
        if param_grid is None:
            param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}

        base = RandomForestClassifier(
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )
        gs = GridSearchCV(base, param_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
        gs.fit(self.X_train, self.y_train)

        self.model = gs.best_estimator_
        self.n_estimators = gs.best_params_.get('n_estimators', self.n_estimators)
        self.max_depth = gs.best_params_.get('max_depth', self.max_depth)
        logger.info(f"RF best params: {gs.best_params_}  score={gs.best_score_:.4f}")
        return gs.best_params_

    def get_feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_

    def get_feature_importance_df(self, feature_names=None) -> 'pd.DataFrame':
        import pandas as pd
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.model.feature_importances_))]
        return pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_,
        }).sort_values('importance', ascending=False)

    def save_model(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Random Forest model saved to {path}")

    def load_model(self, path: str) -> None:
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        self.model = loaded.model
        self.scaler = loaded.scaler
        self.X_train = loaded.X_train
        self.X_test = loaded.X_test
        self.y_train = loaded.y_train
        self.y_test = loaded.y_test
        self.y_pred = loaded.y_pred
        self.training_time = loaded.training_time
        self.prediction_time = loaded.prediction_time
        logger.info(f"Random Forest model loaded from {path}")

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'class_weight': self.class_weight,
            'n_features': self.X_train.shape[1] if self.X_train is not None else None,
        })
        return info
