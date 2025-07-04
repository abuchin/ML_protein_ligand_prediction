"""
Random Forest model for protein-ligand binding prediction.
"""

import pickle
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest model implementation."""
    
    def __init__(self, n_estimators=100, max_depth=None, test_size=0.2, random_state=42):
        """
        Initialize the Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        super().__init__("Random Forest", test_size, random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = self._initialize_model(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state,
            n_jobs=-1
        )
    
    def _initialize_model(self, **kwargs) -> RandomForestClassifier:
        """
        Initialize the Random Forest model.
        
        Args:
            **kwargs: Model parameters
            
        Returns:
            Initialized RandomForestClassifier model
        """
        return RandomForestClassifier(**kwargs)
    
    def _train_model(self) -> None:
        """Train the Random Forest model."""
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
        Get feature importance scores.
        
        Returns:
            Feature importance array
        """
        return self.model.feature_importances_
    
    def get_feature_importance_df(self, feature_names=None) -> 'pd.DataFrame':
        """
        Get feature importance as a DataFrame.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        import pandas as pd
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.model.feature_importances_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path where to save the model
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Random Forest model saved to {path}")
    
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
        
        logger.info(f"Random Forest model loaded from {path}")
    
    def get_model_info(self) -> dict:
        """
        Get extended model information including hyperparameters.
        
        Returns:
            Dictionary with model information
        """
        base_info = super().get_model_info()
        base_info.update({
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'n_features': self.X_train.shape[1] if self.X_train is not None else None
        })
        return base_info 