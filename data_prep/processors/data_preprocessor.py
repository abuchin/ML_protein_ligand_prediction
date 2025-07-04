"""
Data preprocessing module for protein-ligand binding data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple
from ..utils.data_utils import print_data_info, check_duplicates

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class for preprocessing protein-ligand binding data."""
    
    def __init__(self, kiba_threshold: float = 0.01, random_state: int = 37):
        """
        Initialize the data preprocessor.
        
        Args:
            kiba_threshold: Threshold for KIBA score to determine binding
            random_state: Random state for reproducibility
        """
        self.kiba_threshold = kiba_threshold
        self.random_state = random_state
        
    def load_and_clean_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and perform initial cleaning of the dataset.
        
        Args:
            data_path: Path to the input CSV file
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Loading and cleaning data...")
        
        # Load data
        data = pd.read_csv(data_path)
        print_data_info(data, "Original Dataset")
        
        # Keep only reliably estimated values
        data = data[data['kiba_score_estimated'] == True]
        logger.info(f"After filtering estimated values: {data.shape}")
        
        # Remove NaN values
        data = data.dropna()
        logger.info(f"After removing NaN values: {data.shape}")
        
        return data
    
    def handle_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle duplicate protein-ligand pairs by averaging KIBA scores.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with duplicates handled
        """
        logger.info("Handling duplicates...")
        
        # Check for duplicates
        duplicates = check_duplicates(data, ['UniProt_ID', 'pubchem_cid'])
        
        if len(duplicates) > 0:
            # Calculate mean KIBA score for duplicate pairs
            data['kiba_score_mean'] = data.groupby(['UniProt_ID', 'pubchem_cid'])['kiba_score'].transform('mean')
            
            # Remove duplicates and keep mean scores
            data_unique = data.drop(columns='kiba_score').drop_duplicates(
                subset=['UniProt_ID', 'pubchem_cid']
            ).rename(columns={'kiba_score_mean': 'kiba_score'})
            
            logger.info(f"After handling duplicates: {data_unique.shape}")
            return data_unique
        else:
            logger.info("No duplicates found")
            return data
    
    def create_binding_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary binding labels based on KIBA threshold.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with binding labels
        """
        logger.info("Creating binding labels...")
        
        # Create binary labels: 0 - not bound, 1 - bound
        data['bound'] = (data['kiba_score'] > self.kiba_threshold).astype(int)
        
        # Print distribution
        bound_counts = data['bound'].value_counts()
        logger.info(f"Binding distribution: {dict(bound_counts)}")
        
        return data
    
    def create_negative_samples(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic negative samples by shuffling ligand IDs.
        
        Args:
            data: Input DataFrame with positive samples
            
        Returns:
            DataFrame with both positive and negative samples
        """
        logger.info("Creating negative samples...")
        
        # Create negative data by shuffling
        data_negative = data.copy()
        
        # Shuffle only pubchem_cid, preserve the order for uniport
        data_negative['pubchem_cid'] = np.random.permutation(data_negative['pubchem_cid'].values)
        
        # Set KIBA score to 0 for synthetic negative samples
        data_negative['kiba_score'] = 0
        data_negative['kiba_score_estimated'] = False
        data_negative['bound'] = 0
        
        # Check for randomly occurring pairs
        positive_pairs = set(tuple(x) for x in data[['UniProt_ID', 'pubchem_cid']].to_numpy())
        negative_pairs = set(tuple(x) for x in data_negative[['UniProt_ID', 'pubchem_cid']].to_numpy())
        common_pairs = positive_pairs.intersection(negative_pairs)
        
        if len(common_pairs) > 0:
            logger.warning(f"Found {len(common_pairs)} randomly occurring pairs, removing them...")
            # Remove common pairs from negative samples
            common_pairs_list = [list(pair) for pair in common_pairs]
            data_negative = data_negative[~data_negative[['UniProt_ID', 'pubchem_cid']].apply(tuple, axis=1).isin(common_pairs)]
        
        # Combine positive and negative samples
        combined_data = pd.concat([data, data_negative], ignore_index=True)
        
        # Shuffle the combined data
        shuffled_data = combined_data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        logger.info(f"Final dataset shape: {shuffled_data.shape}")
        logger.info(f"Final binding distribution: {dict(shuffled_data['bound'].value_counts())}")
        
        return shuffled_data
    
    def preprocess(self, data_path: str) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            data_path: Path to the input CSV file
            
        Returns:
            Preprocessed DataFrame ready for feature extraction
        """
        logger.info("Starting data preprocessing pipeline...")
        
        # Load and clean data
        data = self.load_and_clean_data(data_path)
        
        # Handle duplicates
        data = self.handle_duplicates(data)
        
        # Create binding labels
        data = self.create_binding_labels(data)
        
        # Create negative samples
        final_data = self.create_negative_samples(data)
        
        logger.info("Data preprocessing completed successfully!")
        
        return final_data 