"""
Logistic Regression model for protein-ligand binding prediction.
"""

import pickle
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model implementation."""

    def __init__(self, penalty='l2', C=1.0, class_weight='balanced',
                 max_iter=1000, test_size=0.2, random_state=42):
        super().__init__("Logistic Regression", test_size, random_state)
        self.penalty = penalty
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.model = self._initialize_model(
            penalty=penalty, C=C, class_weight=class_weight,
            max_iter=max_iter, random_state=random_state
        )

    def _initialize_model(self, **kwargs) -> LogisticRegression:
        return LogisticRegression(**kwargs)

    def _train_model(self) -> None:
        self.model.fit(self.X_train, self.y_train)

    def _predict_model(self) -> np.ndarray:
        return self.model.predict(self.X_test)

    def predict_proba(self) -> np.ndarray:
        return self.model.predict_proba(self.X_test)

    def tune(self, param_grid: dict = None, cv: int = 3, scoring: str = 'f1_macro') -> dict:
        """Run GridSearchCV on X_train/y_train and replace self.model with the best estimator."""
        if param_grid is None:
            param_grid = {'C': [0.1, 1.0, 10.0], 'penalty': ['l2']}

        base = LogisticRegression(
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        gs = GridSearchCV(base, param_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
        gs.fit(self.X_train, self.y_train)

        self.model = gs.best_estimator_
        self.C = gs.best_params_.get('C', self.C)
        self.penalty = gs.best_params_.get('penalty', self.penalty)
        logger.info(f"LR best params: {gs.best_params_}  score={gs.best_score_:.4f}")
        return gs.best_params_

    def get_feature_importance(self) -> np.ndarray:
        return self.model.coef_[0]

    def save_model(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Logistic Regression model saved to {path}")

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
        logger.info(f"Logistic Regression model loaded from {path}")

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            'penalty': self.penalty,
            'C': self.C,
            'class_weight': self.class_weight,
            'n_features': self.X_train.shape[1] if self.X_train is not None else None,
        })
        return info
