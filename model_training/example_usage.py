"""
Example usage of the model training pipeline.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from model_training.pipeline import ModelTrainingPipeline
from model_training.config.config import *


def example_basic_training():
    """Example of basic model training."""
    print("Running basic model training pipeline...")
    
    # Initialize pipeline with default configuration
    pipeline = ModelTrainingPipeline()
    
    # Run complete pipeline with smaller dataset for testing
    results = pipeline.run_complete_pipeline(
        n_samples=5000,  # Use 5K samples for testing
        model_types=['logistic_regression', 'random_forest'],  # Test with 2 models
        save_models=True,
        create_plots=True
    )
    
    print("Training completed!")
    print(f"Best model: {results['best_model']['name']}")
    print(f"Best F1 score: {results['best_model']['results']['metrics']['f1']:.4f}")


def example_custom_config():
    """Example with custom configuration."""
    print("Running pipeline with custom configuration...")
    
    # Custom configuration
    config = {
        'n_samples': 10000,
        'logistic_regression_params': {
            'penalty': 'l1',
            'C': 0.1,
            'test_size': 0.3,
            'random_state': 42
        },
        'random_forest_params': {
            'n_estimators': 200,
            'max_depth': 10,
            'test_size': 0.3,
            'random_state': 42
        }
    }
    
    # Initialize pipeline with custom config
    pipeline = ModelTrainingPipeline(config)
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        n_samples=10000,
        model_types=['logistic_regression', 'random_forest'],
        save_models=True,
        create_plots=True
    )
    
    print("Custom training completed!")
    print(f"Best model: {results['best_model']['name']}")
    print(f"Best F1 score: {results['best_model']['results']['metrics']['f1']:.4f}")


def example_step_by_step():
    """Example of running pipeline steps individually."""
    print("Running pipeline step by step...")
    
    # Initialize pipeline
    pipeline = ModelTrainingPipeline()
    
    # Step 1: Load and prepare data
    print("\nStep 1: Loading and preparing data...")
    X, y_continuous, y_binary, feature_info = pipeline.load_and_prepare_data(n_samples=3000)
    print(f"Data shape: {X.shape}")
    print(f"Target distribution: {y_binary.value_counts().to_dict()}")
    
    # Step 2: Create models
    print("\nStep 2: Creating models...")
    models = pipeline.create_models(X.shape[1], ['logistic_regression', 'random_forest'])
    print(f"Created {len(models)} models")
    
    # Step 3: Train models
    print("\nStep 3: Training models...")
    trained_models = pipeline.train_models(X, y_binary)
    print(f"Trained {len(trained_models)} models")
    
    # Step 4: Evaluate models
    print("\nStep 4: Evaluating models...")
    model_results = pipeline.evaluate_models()
    print(f"Evaluated {len(model_results)} models")
    
    # Step 5: Get best model
    print("\nStep 5: Getting best model...")
    best_model_name, best_model_results = pipeline.get_best_model()
    print(f"Best model: {best_model_name}")
    print(f"Best F1 score: {best_model_results['metrics']['f1']:.4f}")
    
    print("Step-by-step pipeline completed!")


def example_model_comparison():
    """Example of comparing different models."""
    print("Running model comparison...")
    
    # Initialize pipeline
    pipeline = ModelTrainingPipeline()
    
    # Load data
    X, y_continuous, y_binary, feature_info = pipeline.load_and_prepare_data(n_samples=5000)
    
    # Test different model configurations
    model_configs = {
        'logistic_regression_l1': {'penalty': 'l1', 'C': 0.1},
        'logistic_regression_l2': {'penalty': 'l2', 'C': 1.0},
        'random_forest_100': {'n_estimators': 100},
        'random_forest_200': {'n_estimators': 200},
        'xgboost_fast': {'n_estimators': 50, 'max_depth': 3},
        'xgboost_deep': {'n_estimators': 100, 'max_depth': 8}
    }
    
    results = {}
    
    for config_name, config_params in model_configs.items():
        print(f"\nTraining {config_name}...")
        
        # Create and train model
        model = pipeline.create_models(X.shape[1], ['logistic_regression'])[0]
        model.split_data(X, y_binary)
        model.train()
        model.predict()
        
        # Evaluate model
        model_result = pipeline.evaluator.evaluate_model(model)
        results[config_name] = model_result
        
        print(f"{config_name} - Accuracy: {model_result['metrics']['accuracy']:.4f}")
    
    # Compare results
    comparison_df = pipeline.evaluator.compare_models(results)
    print("\nModel Comparison:")
    print(comparison_df)
    
    print("Model comparison completed!")


if __name__ == "__main__":
    print("Model Training Pipeline Examples")
    print("=" * 40)
    
    # Choose which example to run
    example_choice = input("Choose example (1=basic, 2=custom, 3=step-by-step, 4=comparison): ").strip()
    
    if example_choice == "1":
        example_basic_training()
    elif example_choice == "2":
        example_custom_config()
    elif example_choice == "3":
        example_step_by_step()
    elif example_choice == "4":
        example_model_comparison()
    else:
        print("Invalid choice. Running basic example...")
        example_basic_training() 