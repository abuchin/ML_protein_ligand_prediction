# Protein-Ligand Binding Model Training Pipeline

This directory contains a modular and well-organized pipeline for training machine learning models to predict protein-ligand binding interactions.

## Overview

The pipeline consists of six main steps:
1. **Data Loading & Feature Engineering**: Load preprocessed data and create features
2. **Model Creation**: Initialize different types of models
3. **Model Training**: Train all models on the dataset
4. **Model Evaluation**: Evaluate model performance using various metrics
5. **Model Saving**: Save trained models to disk
6. **Visualization**: Create plots and visualizations of results

## Directory Structure

```
model_training/
├── config/
│   ├── __init__.py
│   └── config.py              # Configuration parameters
├── data_loader/
│   ├── __init__.py
│   └── feature_engineer.py    # Feature engineering
├── models/
│   ├── __init__.py
│   ├── base_model.py          # Abstract base model
│   ├── logistic_regression_model.py
│   ├── random_forest_model.py
│   ├── xgboost_model.py
│   ├── mlp_model.py
│   └── model_factory.py       # Model factory
├── evaluators/
│   ├── __init__.py
│   └── model_evaluator.py     # Model evaluation
├── visualization/
│   ├── __init__.py
│   └── visualizer.py          # Plotting and visualization
├── pipeline.py                # Main pipeline orchestrator
├── example_usage.py           # Usage examples
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r model_training/requirements.txt
```

2. Ensure you have the required data files from the data preparation pipeline:
   - `combined_data.csv` - Preprocessed dataset
   - `protein_embeddings.pkl` - Protein embeddings
   - `ligand_data.pkl` - Ligand molecular descriptors

## Usage

### Basic Usage

```python
from model_training.pipeline import ModelTrainingPipeline

# Initialize pipeline with default configuration
pipeline = ModelTrainingPipeline()

# Run complete pipeline
results = pipeline.run_complete_pipeline(
    n_samples=10000,  # Use 10K samples
    model_types=['logistic_regression', 'random_forest', 'xgboost'],
    save_models=True,
    create_plots=True
)

print(f"Best model: {results['best_model']['name']}")
print(f"Best F1 score: {results['best_model']['results']['metrics']['f1']:.4f}")
```

### Custom Configuration

```python
# Custom configuration
config = {
    'n_samples': 50000,
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
results = pipeline.run_complete_pipeline()
```

### Step-by-Step Processing

```python
pipeline = ModelTrainingPipeline()

# Step 1: Load and prepare data
X, y_continuous, y_binary, feature_info = pipeline.load_and_prepare_data(n_samples=5000)

# Step 2: Create models
models = pipeline.create_models(X.shape[1], ['logistic_regression', 'random_forest'])

# Step 3: Train models
trained_models = pipeline.train_models(X, y_binary)

# Step 4: Evaluate models
model_results = pipeline.evaluate_models()

# Step 5: Get best model
best_model_name, best_model_results = pipeline.get_best_model()
```

## Available Models

### 1. Logistic Regression
- **Type**: Linear classification
- **Pros**: Fast, interpretable, good baseline
- **Cons**: Limited to linear relationships
- **Parameters**: penalty, C, test_size, random_state

### 2. Random Forest
- **Type**: Ensemble of decision trees
- **Pros**: Handles non-linear relationships, feature importance
- **Cons**: Can be slow with large datasets
- **Parameters**: n_estimators, max_depth, test_size, random_state

### 3. XGBoost
- **Type**: Gradient boosting
- **Pros**: High performance, handles missing values
- **Cons**: More complex, requires tuning
- **Parameters**: n_estimators, max_depth, learning_rate, test_size, random_state

### 4. MLP (Multi-Layer Perceptron)
- **Type**: Neural network
- **Pros**: Can learn complex patterns
- **Cons**: Requires more data, longer training time
- **Parameters**: hidden_size, batch_size, learning_rate, epochs, test_size, random_state

## Configuration

The pipeline configuration is centralized in `config/config.py`. Key parameters include:

- **N_SAMPLES**: Number of samples to use for training (default: 500000)
- **TEST_SIZE**: Fraction of data for testing (default: 0.2)
- **RANDOM_STATE**: Random seed for reproducibility (default: 42)
- **Model-specific parameters**: See individual model classes for details

## Output Files

The pipeline generates the following output files:

1. **Trained Models**: Saved in the `Models/` directory
   - `logistic_regression_model.pkl`
   - `random_forest_model.pkl`
   - `xgboost_model.pkl`
   - `mlp_model.pth`

2. **Visualizations**: Saved in the `Models/plots/` directory
   - `accuracy_comparison.png`
   - `metrics_comparison.png`
   - `training_time_comparison.png`
   - Individual confusion matrices and feature importance plots

3. **Logs**: Training logs saved to `training.log`

## Key Features

### Feature Engineering
- Combines protein embeddings and ligand descriptors
- Handles missing data gracefully
- Provides feature information and statistics

### Model Training
- Unified interface for all models
- Automatic data splitting and preprocessing
- Training time tracking
- GPU support for MLP models

### Model Evaluation
- Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix generation
- Model comparison functionality
- Best model selection

### Visualization
- Accuracy comparison plots
- Metrics comparison across models
- Training time analysis
- Feature importance plots
- Confusion matrix visualization

## Performance Considerations

- **Data Size**: Large datasets may require significant memory
- **Model Selection**: Start with simpler models for quick testing
- **GPU Usage**: MLP models benefit from GPU acceleration
- **Parallel Processing**: Random Forest and XGBoost use parallel processing

## Example Script

Run the example script to see different usage patterns:

```bash
python model_training/example_usage.py
```

This will present options for:
1. Basic model training
2. Custom configuration usage
3. Step-by-step processing
4. Model comparison

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or number of samples
2. **CUDA Issues**: Ensure PyTorch is installed with CUDA support for MLP
3. **Missing Data**: Ensure all required data files are present
4. **Import Errors**: Install all requirements from `requirements.txt`

### Performance Tips

1. Start with smaller datasets for testing
2. Use simpler models first (Logistic Regression, Random Forest)
3. Adjust hyperparameters based on your data size
4. Monitor memory usage during training

## Contributing

When modifying the pipeline:
1. Update configuration parameters in `config/config.py`
2. Add new models by extending the `BaseModel` class
3. Update the `ModelFactory` for new model types
4. Add evaluation metrics in `ModelEvaluator`
5. Create visualizations in `ModelVisualizer`
6. Test with example data before running on full dataset

## Integration with Data Preparation Pipeline

This training pipeline is designed to work seamlessly with the data preparation pipeline:

1. **Input**: Uses output files from `data_prep/` pipeline
2. **Features**: Combines protein embeddings and ligand descriptors
3. **Output**: Trained models ready for prediction

The complete workflow is:
```
Data Preparation → Feature Engineering → Model Training → Evaluation → Deployment
``` 