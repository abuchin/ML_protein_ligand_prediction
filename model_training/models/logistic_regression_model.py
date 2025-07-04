"""
Logistic Regression model for protein-ligand binding prediction.
"""

import pickle
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model implementation."""
    
    def __init__(self, penalty='l2', C=1.0, test_size=0.2, random_state=42):
        """
        Initialize the Logistic Regression model.
        
        Args:
            penalty: Regularization penalty ('l1' or 'l2')
            C: Inverse of regularization strength
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        super().__init__("Logistic Regression", test_size, random_state)
        self.penalty = penalty
        self.C = C
        self.model = self._initialize_model(penalty=penalty, C=C, random_state=random_state)
    
    def _initialize_model(self, **kwargs) -> LogisticRegression:
        """
        Initialize the Logistic Regression model.
        
        Args:
            **kwargs: Model parameters
            
        Returns:
            Initialized LogisticRegression model
        """
        return LogisticRegression(**kwargs)
    
    def _train_model(self) -> None:
        """Train the Logistic Regression model."""
        self.model.fit(self.X_train, self.y_train)
    
    def _predict_model(self) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Returns:
            Predictions array
        """
        return self.model.predict(self.X_test)
    
    def predict_proba(self) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Returns:
            Probability predictions array
        """
        return self.model.predict_proba(self.X_test)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance coefficients.
        
        Returns:
            Feature importance array
        """
        return self.model.coef_[0]
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path where to save the model
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Logistic Regression model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        with open(path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Copy attributes from loaded model
        self.model = loaded_model.model
        self.X_train = loaded_model.X_train
        self.X_test = loaded_model.X_test
        self.y_train = loaded_model.y_train
        self.y_test = loaded_model.y_test
        self.y_pred = loaded_model.y_pred
        self.training_time = loaded_model.training_time
        self.prediction_time = loaded_model.prediction_time
        
        logger.info(f"Logistic Regression model loaded from {path}")
    
    def get_model_info(self) -> dict:
        """
        Get extended model information including hyperparameters.
        
        Returns:
            Dictionary with model information
        """
        base_info = super().get_model_info()
        base_info.update({
            'penalty': self.penalty,
            'C': self.C,
            'n_features': self.X_train.shape[1] if self.X_train is not None else None
        })
        return base_info 