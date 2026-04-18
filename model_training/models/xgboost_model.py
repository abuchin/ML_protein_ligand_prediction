"""
XGBoost model for protein-ligand binding prediction.
"""

import pickle
import logging
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""

    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 scale_pos_weight=1.0, early_stopping_rounds=10,
                 test_size=0.2, random_state=42):
        super().__init__("XGBoost", test_size, random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.model = self._initialize_model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            n_jobs=-1,
        )

    def _initialize_model(self, **kwargs) -> XGBClassifier:
        return XGBClassifier(**kwargs)

    def _train_model(self) -> None:
        # Carve a small internal val set for early stopping.
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_train,
            test_size=0.1, random_state=self.random_state, stratify=self.y_train
        )
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        if self.early_stopping_rounds and hasattr(self.model, 'best_iteration'):
            try:
                logger.info(f"XGBoost best iteration: {self.model.best_iteration}")
            except AttributeError:
                pass

    def _predict_model(self) -> np.ndarray:
        return self.model.predict(self.X_test)

    def predict_proba(self) -> np.ndarray:
        return self.model.predict_proba(self.X_test)

    def tune(self, param_grid: dict = None, cv: int = 3, scoring: str = 'f1_macro') -> dict:
        """Run GridSearchCV on X_train/y_train and replace self.model with the best estimator."""
        if param_grid is None:
            param_grid = {
                'max_depth': [3, 6],
                'learning_rate': [0.05, 0.1],
                'n_estimators': [50, 100],
            }

        # No early_stopping_rounds here — GridSearchCV manages train/val splits.
        base = XGBClassifier(
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        gs = GridSearchCV(base, param_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
        gs.fit(self.X_train, self.y_train)

        self.model = gs.best_estimator_
        self.n_estimators = gs.best_params_.get('n_estimators', self.n_estimators)
        self.max_depth = gs.best_params_.get('max_depth', self.max_depth)
        self.learning_rate = gs.best_params_.get('learning_rate', self.learning_rate)
        logger.info(f"XGBoost best params: {gs.best_params_}  score={gs.best_score_:.4f}")
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
        logger.info(f"XGBoost model saved to {path}")

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
        logger.info(f"XGBoost model loaded from {path}")

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'scale_pos_weight': self.scale_pos_weight,
            'n_features': self.X_train.shape[1] if self.X_train is not None else None,
        })
        return info
