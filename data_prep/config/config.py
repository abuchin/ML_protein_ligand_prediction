"""
Configuration file for protein-ligand binding data preparation pipeline.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "Data"
DATA_PREP_DIR = BASE_DIR / "data_prep"

# Input data paths
INPUT_DATA_PATH = DATA_DIR / "Deloitte_DrugDiscovery_dataset.csv"
CID_SMILES_PATH = DATA_DIR / "CID-SMILES.txt"

# Protein embedding paths
PROT_BERT_TEST_EMBEDDINGS_PATH = DATA_DIR / "Pro_bert_test_embeddings.npy"
PROT_BERT_TRAIN_EMBEDDINGS_PATH = DATA_DIR / "Pro_bert_train_embeddings.npy"
PROT_BERT_TEST_IDS_PATH = DATA_DIR / "Prot_Bert_test_ids.npy"
PROT_BERT_TRAIN_IDS_PATH = DATA_DIR / "Prot_Bert_train_ids.npy"

# Output paths
OUTPUT_DIR = DATA_DIR
COMBINED_DATA_PATH = OUTPUT_DIR / "combined_data.csv"
PROTEIN_EMBEDDINGS_PATH = OUTPUT_DIR / "protein_embeddings.pkl"
PROTEIN_EMBEDDINGS_MISSED_PATH = OUTPUT_DIR / "protein_embeddings_missed.pkl"
CID_TO_SMILES_PATH = OUTPUT_DIR / "cid_to_smiles.pkl"
LIGAND_DATA_PATH = OUTPUT_DIR / "ligand_data.pkl"

# Data processing parameters
KIBA_THRESHOLD = 0.01
RANDOM_STATE = 37
BATCH_SIZE = 50
MAX_PROTEIN_SEQUENCE_LENGTH = 1024

# Model parameters
MODEL_NAME = "Rostlab/prot_bert_bfd"
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 1024

# Processing parameters
N_PROTEINS_TO_PROCESS = 50  # For testing, set to None for all proteins
LIGAND_BATCH_SIZE = 50

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 