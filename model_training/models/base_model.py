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
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.model_name = model_name
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.training_time = None
        self.prediction_time = None
    
    @abstractmethod
    def _initialize_model(self, **kwargs) -> Any:
        """
        Initialize the specific model implementation.
        
        Args:
            **kwargs: Model-specific parameters
            
        Returns:
            Initialized model
        """
        pass
    
    @abstractmethod
    def _train_model(self) -> None:
        """Train the model on the training data."""
        pass
    
    @abstractmethod
    def _predict_model(self) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Returns:
            Predictions array
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path where to save the model
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        pass
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature DataFrame
            y: Target Series
        """
        from sklearn.model_selection import train_test_split
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, stratify=y, test_size=self.test_size, random_state=self.random_state
        )
        
        logger.info(f"Data split: {self.X_train.shape[0]} train, {self.X_test.shape[0]} test")
    
    def train(self) -> None:
        """Train the model and record training time."""
        import time
        
        logger.info(f"Training {self.model_name}...")
        start_time = time.time()
        
        self._train_model()
        
        self.training_time = time.time() - start_time
        logger.info(f"{self.model_name} training completed in {self.training_time:.2f} seconds")
    
    def predict(self) -> np.ndarray:
        """
        Make predictions and record prediction time.
        
        Returns:
            Predictions array
        """
        import time
        
        logger.info(f"Making predictions with {self.model_name}...")
        start_time = time.time()
        
        self.y_pred = self._predict_model()
        
        self.prediction_time = time.time() - start_time
        logger.info(f"{self.model_name} predictions completed in {self.prediction_time:.2f} seconds")
        
        return self.y_pred
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'train_samples': len(self.X_train) if self.X_train is not None else None,
            'test_samples': len(self.X_test) if self.X_test is not None else None
        }
    
    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and true labels.
        
        Returns:
            Tuple of (predictions, true_labels)
        """
        if self.y_pred is None or self.y_test is None:
            raise ValueError("Model must be trained and predictions made first")
        
        return self.y_pred, self.y_test 