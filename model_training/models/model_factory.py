"""
Model factory for creating different types of models.
"""

import logging
from typing import Dict, Any
from .logistic_regression_model import LogisticRegressionModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel
from .mlp_model import MLPModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating different types of models."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> Any:
        """
        Create a model instance based on the model type.
        
        Args:
            model_type: Type of model to create ('logistic_regression', 'random_forest', 'xgboost', 'mlp')
            **kwargs: Model-specific parameters
            
        Returns:
            Model instance
        """
        model_type = model_type.lower()
        
        if model_type == 'logistic_regression':
            return LogisticRegressionModel(**kwargs)
        elif model_type == 'random_forest':
            return RandomForestModel(**kwargs)
        elif model_type == 'xgboost':
            return XGBoostModel(**kwargs)
        elif model_type == 'mlp':
            return MLPModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_default_params(model_type: str) -> Dict[str, Any]:
        """
        Get default parameters for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of default parameters
        """
        model_type = model_type.lower()
        
        if model_type == 'logistic_regression':
            return {
                'penalty': 'l2',
                'C': 1.0,
                'test_size': 0.2,
                'random_state': 42
            }
        elif model_type == 'random_forest':
            return {
                'n_estimators': 100,
                'max_depth': None,
                'test_size': 0.2,
                'random_state': 42
            }
        elif model_type == 'xgboost':
            return {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'test_size': 0.2,
                'random_state': 42
            }
        elif model_type == 'mlp':
            return {
                'hidden_size': 256,
                'output_size': 2,
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 10,
                'test_size': 0.2,
                'random_state': 42
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_available_models() -> list:
        """
        Get list of available model types.
        
        Returns:
            List of available model types
        """
        return ['logistic_regression', 'random_forest', 'xgboost', 'mlp']
    
    @staticmethod
    def create_all_models(input_size: int = None, **kwargs) -> Dict[str, Any]:
        """
        Create all available models.
        
        Args:
            input_size: Input size for MLP (required for MLP model)
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Dictionary mapping model names to model instances
        """
        models = {}
        
        for model_type in ModelFactory.get_available_models():
            try:
                if model_type == 'mlp' and input_size is None:
                    logger.warning("Skipping MLP model: input_size is required")
                    continue
                
                # Get default parameters
                params = ModelFactory.get_default_params(model_type)
                
                # Override with provided parameters
                params.update(kwargs)
                
                # Add input_size for MLP
                if model_type == 'mlp':
                    params['input_size'] = input_size
                
                # Create model
                models[model_type] = ModelFactory.create_model(model_type, **params)
                logger.info(f"Created {model_type} model")
                
            except Exception as e:
                logger.error(f"Failed to create {model_type} model: {e}")
        
        return models 