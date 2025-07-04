"""
Configuration file for protein-ligand binding model training pipeline.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "Models"
MODEL_TRAINING_DIR = BASE_DIR / "model_training"

# Input data paths
COMBINED_DATA_PATH = DATA_DIR / "combined_data.csv"
PROTEIN_EMBEDDINGS_PATH = DATA_DIR / "protein_embeddings.pkl"
LIGAND_DATA_PATH = DATA_DIR / "ligand_data.pkl"

# Output paths
OUTPUT_DIR = MODELS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
LOGISTIC_REGRESSION_PATH = OUTPUT_DIR / "logistic_regression_model.pkl"
RANDOM_FOREST_PATH = OUTPUT_DIR / "random_forest_model.pkl"
XGBOOST_PATH = OUTPUT_DIR / "xgboost_model.pkl"
MLP_PATH = OUTPUT_DIR / "mlp_model.pth"

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SAMPLES = 500000  # Number of samples to use for training

# Model-specific parameters
LOGISTIC_REGRESSION_PARAMS = {
    'penalty': 'l2',
    'C': 1.0,
    'random_state': RANDOM_STATE
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

MLP_PARAMS = {
    'hidden_size': 256,
    'output_size': 2,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 10
}

# Feature engineering parameters
PROTEIN_EMBEDDING_SIZE = 1024  # Size of protein embeddings
FINGERPRINT_SIZE = 1024  # Size of molecular fingerprints
NUM_MOLECULAR_DESCRIPTORS = 8  # Number of molecular descriptors

# Evaluation parameters
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']
CROSS_VALIDATION_FOLDS = 5

# Visualization parameters
FIGURE_SIZE = (10, 6)
CMAP = 'Blues'
DPI = 300

# Device configuration
DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') is not None else 'cpu'

# Logging parameters
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = MODEL_TRAINING_DIR / 'training.log' 