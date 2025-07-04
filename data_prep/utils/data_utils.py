"""
Utility functions for data loading, saving, and basic operations.
"""

import pandas as pd
import numpy as np
import pickle
import requests
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: Path) -> pd.DataFrame:
    """Load data from CSV file."""
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Loaded data from {file_path} with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def save_data(data: pd.DataFrame, file_path: Path) -> None:
    """Save data to CSV file."""
    try:
        data.to_csv(file_path, index=False)
        logger.info(f"Saved data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise


def save_pickle(data, file_path: Path) -> None:
    """Save data to pickle file."""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved pickle data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving pickle data to {file_path}: {e}")
        raise


def load_pickle(file_path: Path):
    """Load data from pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded pickle data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading pickle data from {file_path}: {e}")
        raise


def fetch_protein_sequence(uniprot_id: str) -> str:
    """Fetch protein sequence from UniProt."""
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            fasta_data = response.text
            sequence = ''.join(fasta_data.split('\n')[1:])
            return sequence
        else:
            logger.warning(f"Error fetching sequence for {uniprot_id}: HTTP {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching sequence for {uniprot_id}: {e}")
        return None


def print_data_info(data: pd.DataFrame, title: str = "Dataset") -> None:
    """Print basic information about the dataset."""
    print(f"\n{title}")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Missing values:\n{data.isnull().sum()}")


def check_duplicates(data: pd.DataFrame, subset: list) -> pd.DataFrame:
    """Check for duplicates in specified columns."""
    duplicates = data[data.duplicated(subset=subset, keep=False)]
    if len(duplicates) > 0:
        logger.info(f"Found {len(duplicates)} duplicate rows based on {subset}")
    else:
        logger.info(f"No duplicates found based on {subset}")
    return duplicates 