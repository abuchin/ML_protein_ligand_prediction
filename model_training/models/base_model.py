"""
Base model class for protein-ligand binding prediction models.
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models in the pipeline."""

    def __init__(self, model_name: str, test_size: float = 0.2, random_state: int = 42):
        self.model_name = model_name
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.training_time = None
        self.prediction_time = None

    @abstractmethod
    def _initialize_model(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def _train_model(self) -> None:
        pass

    @abstractmethod
    def _predict_model(self) -> np.ndarray:
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        pass

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Split data into train/test sets and fit a StandardScaler on the train split."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=self.test_size, random_state=self.random_state
        )

        # Fit scaler on train only; transform both splits.
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train.values if hasattr(y_train, 'values') else y_train
        self.y_test = y_test.values if hasattr(y_test, 'values') else y_test

        logger.info(
            f"Data split: {self.X_train.shape[0]} train / {self.X_test.shape[0]} test — "
            f"StandardScaler applied"
        )

    def train(self) -> None:
        """Train the model and record wall-clock time."""
        import time

        logger.info(f"Training {self.model_name}...")
        start = time.time()
        self._train_model()
        self.training_time = time.time() - start
        logger.info(f"{self.model_name} trained in {self.training_time:.2f}s")

    def predict(self) -> np.ndarray:
        """Make predictions and record wall-clock time."""
        import time

        logger.info(f"Predicting with {self.model_name}...")
        start = time.time()
        self.y_pred = self._predict_model()
        self.prediction_time = time.time() - start
        return self.y_pred

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'train_samples': len(self.X_train) if self.X_train is not None else None,
            'test_samples': len(self.X_test) if self.X_test is not None else None,
        }

    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.y_pred is None or self.y_test is None:
            raise ValueError("Model must be trained and predictions made first")
        return self.y_pred, self.y_test
