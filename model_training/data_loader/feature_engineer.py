"""
Feature engineering module for protein-ligand binding prediction.
"""

import pandas as pd
import numpy as np
import pickle
import logging
from typing import Tuple, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Class for engineering features from protein embeddings and ligand data."""
    
    def __init__(self, protein_embedding_size: int = 1024, fingerprint_size: int = 1024):
        """
        Initialize the feature engineer.
        
        Args:
            protein_embedding_size: Size of protein embeddings
            fingerprint_size: Size of molecular fingerprints
        """
        self.protein_embedding_size = protein_embedding_size
        self.fingerprint_size = fingerprint_size
    
    def load_data(self, combined_data_path: str, protein_embeddings_path: str, 
                  ligand_data_path: str) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Load all required data files.
        
        Args:
            combined_data_path: Path to combined data CSV
            protein_embeddings_path: Path to protein embeddings pickle
            ligand_data_path: Path to ligand data pickle
            
        Returns:
            Tuple of (combined_data, protein_embeddings, ligand_data)
        """
        logger.info("Loading data files...")
        
        # Load combined data
        combined_data = pd.read_csv(combined_data_path)
        logger.info(f"Loaded combined data: {combined_data.shape}")
        
        # Load protein embeddings
        with open(protein_embeddings_path, 'rb') as f:
            protein_embeddings = pickle.load(f)
        logger.info(f"Loaded protein embeddings: {len(protein_embeddings)} proteins")
        
        # Load ligand data
        with open(ligand_data_path, 'rb') as f:
            ligand_data = pickle.load(f)
        logger.info(f"Loaded ligand data: {len(ligand_data)} ligands")
        
        return combined_data, protein_embeddings, ligand_data
    
    def create_features(self, combined_data: pd.DataFrame, protein_embeddings: Dict, 
                       ligand_data: Dict, n_samples: int = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Create features and targets from the data.
        
        Args:
            combined_data: Combined dataset
            protein_embeddings: Dictionary of protein embeddings
            ligand_data: Dictionary of ligand data
            n_samples: Number of samples to process (None for all)
            
        Returns:
            Tuple of (features_df, target_continuous, target_binary)
        """
        logger.info("Creating features and targets...")
        
        features = []
        target_continuous = []
        target_binary = []
        
        # Determine number of samples to process
        n_all = len(combined_data)
        n_range = n_samples if n_samples else n_all
        n_range = min(n_range, n_all)
        
        logger.info(f"Processing {n_range} samples out of {n_all} total samples")
        
        # Process samples
        for j in range(n_range):
            uniprot_id = combined_data['UniProt_ID'].iloc[j]
            pubchem_cid = int(combined_data['pubchem_cid'].iloc[j])
            
            # Check if both protein and ligand data are available
            if uniprot_id in protein_embeddings and pubchem_cid in ligand_data:
                # Get protein embedding
                protein_embedding = list(protein_embeddings[uniprot_id])
                
                # Get ligand features
                ligand_info = ligand_data[pubchem_cid]
                fingerprint = ligand_info['Fingerprint']
                molecular_weight = ligand_info['MolecularWeight']
                polar_surface_area = ligand_info['PolarSurfaceArea']
                num_rotatable_bonds = ligand_info['NumRotatableBonds']
                num_h_donors = ligand_info['NumHDonors']
                num_h_acceptors = ligand_info['NumHAcceptors']
                num_aromatic_rings = ligand_info['NumAromaticRings']
                fraction_csp3 = ligand_info['FractionCSP3']
                bertz_complexity = ligand_info['BertzComplexity']
                
                # Concatenate all features
                feature_vector = (protein_embedding + 
                                fingerprint + 
                                [molecular_weight, polar_surface_area, num_rotatable_bonds,
                                 num_h_donors, num_h_acceptors, num_aromatic_rings,
                                 fraction_csp3, bertz_complexity])
                
                features.append(feature_vector)
                target_continuous.append(combined_data['kiba_score'].iloc[j])
                target_binary.append(combined_data['bound'].iloc[j])
        
        # Create DataFrames and Series
        X = pd.DataFrame(features)
        y_continuous = pd.Series(target_continuous)
        y_binary = pd.Series(target_binary)
        
        logger.info(f"Created features: {X.shape}")
        logger.info(f"Target distribution (binary): {y_binary.value_counts().to_dict()}")
        
        return X, y_continuous, y_binary
    
    def get_feature_info(self, X: pd.DataFrame) -> Dict:
        """
        Get information about the features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary with feature information
        """
        feature_info = {
            'total_features': X.shape[1],
            'protein_embedding_features': self.protein_embedding_size,
            'fingerprint_features': self.fingerprint_size,
            'molecular_descriptor_features': 8,
            'feature_names': {
                'protein_embedding': f'protein_0 to protein_{self.protein_embedding_size-1}',
                'fingerprint': f'fingerprint_0 to fingerprint_{self.fingerprint_size-1}',
                'molecular_descriptors': [
                    'molecular_weight', 'polar_surface_area', 'num_rotatable_bonds',
                    'num_h_donors', 'num_h_acceptors', 'num_aromatic_rings',
                    'fraction_csp3', 'bertz_complexity'
                ]
            }
        }
        
        return feature_info
    
    def engineer_features(self, combined_data_path: str, protein_embeddings_path: str,
                         ligand_data_path: str, n_samples: int = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict]:
        """
        Complete feature engineering pipeline.
        
        Args:
            combined_data_path: Path to combined data CSV
            protein_embeddings_path: Path to protein embeddings pickle
            ligand_data_path: Path to ligand data pickle
            n_samples: Number of samples to process
            
        Returns:
            Tuple of (features_df, target_continuous, target_binary, feature_info)
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Load data
        combined_data, protein_embeddings, ligand_data = self.load_data(
            combined_data_path, protein_embeddings_path, ligand_data_path
        )
        
        # Create features
        X, y_continuous, y_binary = self.create_features(
            combined_data, protein_embeddings, ligand_data, n_samples
        )
        
        # Get feature information
        feature_info = self.get_feature_info(X)
        
        logger.info("Feature engineering completed successfully!")
        
        return X, y_continuous, y_binary, feature_info 