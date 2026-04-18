"""
Configuration file for protein-ligand binding model training pipeline.
"""

import os
from pathlib import Path

# Prevent OMP library conflicts between PyTorch and XGBoost on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")

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

# Reproducibility
RANDOM_SEED = 42
RANDOM_STATE = RANDOM_SEED  # backwards-compat alias

# Training parameters
TEST_SIZE = 0.2
N_SAMPLES = 500000

# ── Test mode ──────────────────────────────────────────────────────────────────
# Set TEST_MODE=True for fast iterations (small data + small models).
# Flip to False for full production runs.
TEST_MODE = True

TEST_N_SAMPLES = 500
TEST_CV_FOLDS = 3

TEST_LOGISTIC_REGRESSION_PARAMS = {
    'penalty': 'l2',
    'C': 1.0,
    'class_weight': 'balanced',
    'max_iter': 200,
    'random_state': RANDOM_SEED,
}
TEST_LR_TUNE_GRID = {'C': [0.1, 1.0]}

TEST_RANDOM_FOREST_PARAMS = {
    'n_estimators': 5,
    'max_depth': 3,
    'class_weight': 'balanced',
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
}
TEST_RF_TUNE_GRID = {'n_estimators': [5, 10], 'max_depth': [3, 5]}

TEST_XGBOOST_PARAMS = {
    'n_estimators': 10,
    'max_depth': 3,
    'learning_rate': 0.1,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
}
TEST_XGB_TUNE_GRID = {'max_depth': [3], 'learning_rate': [0.1], 'n_estimators': [10]}

TEST_MLP_PARAMS = {
    'hidden_size': 64,
    'output_size': 2,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 10,
    'patience': 2,
}

# ── Production model parameters ───────────────────────────────────────────────
LOGISTIC_REGRESSION_PARAMS = {
    'penalty': 'l2',
    'C': 1.0,
    'class_weight': 'balanced',
    'max_iter': 1000,
    'random_state': RANDOM_SEED,
}
LR_TUNE_GRID = {'C': [0.01, 0.1, 1.0, 10.0], 'penalty': ['l2']}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'class_weight': 'balanced',
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
}
RF_TUNE_GRID = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}

XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
}
XGB_TUNE_GRID = {
    'max_depth': [3, 6],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [50, 100],
}

MLP_PARAMS = {
    'hidden_size': 256,
    'output_size': 2,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'patience': 5,
}

# Feature engineering parameters
PROTEIN_EMBEDDING_SIZE = 1024
FINGERPRINT_SIZE = 1024
NUM_MOLECULAR_DESCRIPTORS = 8

# Evaluation parameters
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'roc_auc', 'pr_auc', 'confusion_matrix']
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
