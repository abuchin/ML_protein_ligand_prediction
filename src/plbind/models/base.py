"""Abstract base class for all protein-ligand binding classifiers."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base for all classifiers in the pipeline.

    The split_data() method fits a StandardScaler on the train split and
    applies it to both splits — preventing data leakage into the test set.

    Subclasses must implement _initialize_model, _train_model, _predict_model.
    """

    def __init__(self, model_name: str, test_size: float = 0.2, random_state: int = 42):
        self.model_name = model_name
        self.test_size = test_size
        self.random_state = random_state
        self.model: Any = None
        self.scaler: Optional[StandardScaler] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.y_pred: Optional[np.ndarray] = None
        self.training_time: Optional[float] = None
        self.prediction_time: Optional[float] = None

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def _initialize_model(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def _train_model(self) -> None:
        pass

    @abstractmethod
    def _predict_model(self) -> np.ndarray:
        pass

    # ── Concrete methods ──────────────────────────────────────────────────────

    def split_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Split into train/test and fit StandardScaler on train only."""
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=self.test_size, random_state=self.random_state
        )
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        self.X_test = self.scaler.transform(X_test).astype(np.float32)
        self.y_train = y_train.astype(np.int32) if hasattr(y_train, "astype") else np.array(y_train, dtype=np.int32)
        self.y_test = y_test.astype(np.int32) if hasattr(y_test, "astype") else np.array(y_test, dtype=np.int32)
        logger.info(
            "%s split: %d train / %d test  (pos_rate_train=%.3f)",
            self.model_name, len(self.X_train), len(self.X_test), self.y_train.mean(),
        )

    def train(self) -> None:
        start = time.time()
        logger.info("Training %s...", self.model_name)
        self._train_model()
        self.training_time = time.time() - start
        logger.info("%s trained in %.2fs", self.model_name, self.training_time)

    def predict(self) -> np.ndarray:
        start = time.time()
        self.y_pred = self._predict_model()
        self.prediction_time = time.time() - start
        return self.y_pred

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "training_time": self.training_time,
            "prediction_time": self.prediction_time,
            "train_samples": len(self.X_train) if self.X_train is not None else None,
            "test_samples": len(self.X_test) if self.X_test is not None else None,
        }

    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.y_pred is None or self.y_test is None:
            raise ValueError("Call train() and predict() first.")
        return self.y_pred, self.y_test

    def save_model(self, path: Path) -> None:
        """Save model + scaler + hyperparameters only — no training data."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self.model,
            "scaler": self.scaler,
            "params": self.get_model_info(),
        }
        joblib.dump(artifact, path)
        logger.info("Saved %s → %s", self.model_name, path)

    def load_model(self, path: Path) -> None:
        artifact = joblib.load(Path(path))
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        logger.info("Loaded %s from %s", self.model_name, path)
