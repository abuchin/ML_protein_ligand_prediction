"""
Main pipeline for protein-ligand binding model training.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from model_training.config.config import *
from model_training.data_loader.feature_engineer import FeatureEngineer
from model_training.models.model_factory import ModelFactory
from model_training.evaluators.model_evaluator import ModelEvaluator
from model_training.visualization.visualizer import ModelVisualizer

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """Main pipeline for training protein-ligand binding prediction models."""
    
    def __init__(self, config: dict = None):
        """
        Initialize the model training pipeline.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = config or {}
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(
            protein_embedding_size=self.config.get('protein_embedding_size', PROTEIN_EMBEDDING_SIZE),
            fingerprint_size=self.config.get('fingerprint_size', FINGERPRINT_SIZE)
        )
        
        self.evaluator = ModelEvaluator()
        
        self.visualizer = ModelVisualizer(
            figsize=self.config.get('figure_size', FIGURE_SIZE),
            dpi=self.config.get('dpi', DPI),
            cmap=self.config.get('cmap', CMAP),
            save_dir=str(OUTPUT_DIR / "plots")
        )
        
        # Results storage
        self.models = {}
        self.model_results = {}
        self.feature_info = None
    
    def load_and_prepare_data(self, n_samples: int = None) -> tuple:
        """
        Load and prepare data for training.
        
        Args:
            n_samples: Number of samples to use (None for all)
            
        Returns:
            Tuple of (X, y_continuous, y_binary, feature_info)
        """
        logger.info("=" * 50)
        logger.info("STEP 1: DATA LOADING AND FEATURE ENGINEERING")
        logger.info("=" * 50)
        
        n_samples = n_samples or self.config.get('n_samples', N_SAMPLES)
        
        # Engineer features
        X, y_continuous, y_binary, feature_info = self.feature_engineer.engineer_features(
            combined_data_path=str(COMBINED_DATA_PATH),
            protein_embeddings_path=str(PROTEIN_EMBEDDINGS_PATH),
            ligand_data_path=str(LIGAND_DATA_PATH),
            n_samples=n_samples
        )
        
        self.feature_info = feature_info
        
        logger.info(f"Data preparation completed. Features: {X.shape}")
        return X, y_continuous, y_binary, feature_info
    
    def create_models(self, input_size: int, model_types: List[str] = None) -> Dict[str, Any]:
        """
        Create model instances.
        
        Args:
            input_size: Number of input features
            model_types: List of model types to create (None for all)
            
        Returns:
            Dictionary mapping model names to model instances
        """
        logger.info("=" * 50)
        logger.info("STEP 2: MODEL CREATION")
        logger.info("=" * 50)
        
        if model_types is None:
            model_types = ModelFactory.get_available_models()
        
        models = {}
        
        for model_type in model_types:
            try:
                # Get default parameters
                params = ModelFactory.get_default_params(model_type)
                
                # Override with config parameters
                if model_type == 'logistic_regression':
                    params.update(self.config.get('logistic_regression_params', LOGISTIC_REGRESSION_PARAMS))
                elif model_type == 'random_forest':
                    params.update(self.config.get('random_forest_params', RANDOM_FOREST_PARAMS))
                elif model_type == 'xgboost':
                    params.update(self.config.get('xgboost_params', XGBOOST_PARAMS))
                elif model_type == 'mlp':
                    params.update(self.config.get('mlp_params', MLP_PARAMS))
                    params['input_size'] = input_size
                
                # Create model
                models[model_type] = ModelFactory.create_model(model_type, **params)
                logger.info(f"Created {model_type} model")
                
            except Exception as e:
                logger.error(f"Failed to create {model_type} model: {e}")
        
        self.models = models
        logger.info(f"Created {len(models)} models: {list(models.keys())}")
        
        return models
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train all models.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary of trained models
        """
        logger.info("=" * 50)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("=" * 50)
        
        trained_models = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name}...")
                
                # Split data
                model.split_data(X, y)
                
                # Train model
                model.train()
                
                # Make predictions
                model.predict()
                
                trained_models[model_name] = model
                logger.info(f"{model_name} training completed")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
        
        self.models = trained_models
        logger.info(f"Training completed for {len(trained_models)} models")
        
        return trained_models
    
    def evaluate_models(self) -> Dict[str, Dict]:
        """
        Evaluate all trained models.
        
        Args:
            Dictionary mapping model names to evaluation results
        """
        logger.info("=" * 50)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("=" * 50)
        
        model_results = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Evaluating {model_name}...")
                
                # Evaluate model
                results = self.evaluator.evaluate_model(model)
                model_results[model_name] = results
                
                # Print summary
                metrics = results['metrics']
                logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                          f"F1: {metrics['f1']:.4f}, Training Time: {results['model_info']['training_time']:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
        
        self.model_results = model_results
        logger.info(f"Evaluation completed for {len(model_results)} models")
        
        return model_results
    
    def save_models(self) -> None:
        """Save all trained models."""
        logger.info("=" * 50)
        logger.info("STEP 5: MODEL SAVING")
        logger.info("=" * 50)
        
        for model_name, model in self.models.items():
            try:
                # Determine save path
                if model_name == 'mlp':
                    save_path = str(MLP_PATH)
                elif model_name == 'logistic_regression':
                    save_path = str(LOGISTIC_REGRESSION_PATH)
                elif model_name == 'random_forest':
                    save_path = str(RANDOM_FOREST_PATH)
                elif model_name == 'xgboost':
                    save_path = str(XGBOOST_PATH)
                else:
                    save_path = str(OUTPUT_DIR / f"{model_name}_model.pkl")
                
                # Save model
                model.save_model(save_path)
                
            except Exception as e:
                logger.error(f"Failed to save {model_name}: {e}")
    
    def create_visualizations(self, save: bool = True) -> None:
        """
        Create visualizations for model results.
        
        Args:
            save: Whether to save the plots
        """
        logger.info("=" * 50)
        logger.info("STEP 6: VISUALIZATION")
        logger.info("=" * 50)
        
        if not self.model_results:
            logger.warning("No model results available for visualization")
            return
        
        try:
            # Create summary plots
            self.visualizer.create_summary_plots(self.model_results, save=save)
            
            # Create individual confusion matrices
            for model_name, results in self.model_results.items():
                cm = results['metrics']['confusion_matrix']
                self.visualizer.plot_confusion_matrix(cm, model_name, save=save)
            
            # Create feature importance plots for applicable models
            for model_name, model in self.models.items():
                if hasattr(model, 'get_feature_importance'):
                    self.visualizer.plot_feature_importance(model, save=save)
            
            logger.info("Visualization completed")
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
    
    def get_best_model(self, metric: str = 'f1') -> tuple:
        """
        Get the best performing model.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (best_model_name, best_model_results)
        """
        if not self.model_results:
            raise ValueError("No model results available")
        
        return self.evaluator.get_best_model(self.model_results, metric)
    
    def run_complete_pipeline(self, n_samples: int = None, model_types: List[str] = None,
                            save_models: bool = True, create_plots: bool = True) -> Dict[str, Any]:
        """
        Run the complete model training pipeline.
        
        Args:
            n_samples: Number of samples to use
            model_types: List of model types to train
            save_models: Whether to save trained models
            create_plots: Whether to create visualizations
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("=" * 50)
        logger.info("STARTING COMPLETE MODEL TRAINING PIPELINE")
        logger.info("=" * 50)
        
        try:
            # Step 1: Load and prepare data
            X, y_continuous, y_binary, feature_info = self.load_and_prepare_data(n_samples)
            
            # Step 2: Create models
            models = self.create_models(X.shape[1], model_types)
            
            # Step 3: Train models
            trained_models = self.train_models(X, y_binary)
            
            # Step 4: Evaluate models
            model_results = self.evaluate_models()
            
            # Step 5: Save models
            if save_models:
                self.save_models()
            
            # Step 6: Create visualizations
            if create_plots:
                self.create_visualizations()
            
            # Get best model
            best_model_name, best_model_results = self.get_best_model()
            
            # Compile results
            pipeline_results = {
                'feature_info': feature_info,
                'models': trained_models,
                'model_results': model_results,
                'best_model': {
                    'name': best_model_name,
                    'results': best_model_results
                },
                'data_shape': X.shape,
                'target_distribution': y_binary.value_counts().to_dict()
            }
            
            logger.info("=" * 50)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 50)
            logger.info(f"Best model: {best_model_name}")
            logger.info(f"Best F1 score: {best_model_results['metrics']['f1']:.4f}")
            logger.info(f"Best accuracy: {best_model_results['metrics']['accuracy']:.4f}")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise


def main():
    """Main function to run the pipeline."""
    # Example configuration (can be customized)
    config = {
        'n_samples': 10000,  # Use 10K samples for testing
        'logistic_regression_params': LOGISTIC_REGRESSION_PARAMS,
        'random_forest_params': RANDOM_FOREST_PARAMS,
        'xgboost_params': XGBOOST_PARAMS,
        'mlp_params': MLP_PARAMS
    }
    
    # Initialize and run pipeline
    pipeline = ModelTrainingPipeline(config)
    
    try:
        results = pipeline.run_complete_pipeline(
            n_samples=10000,  # Use 10K samples for testing
            model_types=['logistic_regression', 'random_forest', 'xgboost'],  # Skip MLP for faster testing
            save_models=True,
            create_plots=True
        )
        
        print("\nPipeline completed successfully!")
        print(f"Best model: {results['best_model']['name']}")
        print(f"Best F1 score: {results['best_model']['results']['metrics']['f1']:.4f}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 