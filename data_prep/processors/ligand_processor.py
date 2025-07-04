"""
Ligand processing module for calculating molecular descriptors and fingerprints.
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
from ..utils.data_utils import save_pickle, load_pickle
from ..config.config import FINGERPRINT_RADIUS, FINGERPRINT_BITS, LIGAND_BATCH_SIZE

logger = logging.getLogger(__name__)


class LigandProcessor:
    """Class for processing ligands and calculating molecular descriptors."""
    
    def __init__(self, fingerprint_radius: int = FINGERPRINT_RADIUS, 
                 fingerprint_bits: int = FINGERPRINT_BITS):
        """
        Initialize the ligand processor.
        
        Args:
            fingerprint_radius: Radius for Morgan fingerprint
            fingerprint_bits: Number of bits for fingerprint
        """
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits
    
    def load_cid_smiles_mapping(self, cid_smiles_path: str) -> Dict[int, str]:
        """
        Load CID to SMILES mapping from file.
        
        Args:
            cid_smiles_path: Path to CID-SMILES mapping file
            
        Returns:
            Dictionary mapping CID to SMILES
        """
        logger.info("Loading CID to SMILES mapping...")
        
        try:
            # Load the mapping file
            cid_smiles_df = pd.read_csv(cid_smiles_path, sep=None, engine='python', header=None)
            cid_smiles_df.columns = ['pubchem_cid', 'smiles']
            
            # Create dictionary
            cid_to_smiles = dict(zip(cid_smiles_df['pubchem_cid'], cid_smiles_df['smiles']))
            
            logger.info(f"Loaded {len(cid_to_smiles)} CID-SMILES mappings")
            return cid_to_smiles
            
        except Exception as e:
            logger.error(f"Error loading CID-SMILES mapping: {e}")
            return {}
    
    def calculate_molecular_descriptors(self, smiles: str) -> Optional[Dict]:
        """
        Calculate molecular descriptors for a given SMILES string.
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            Dictionary containing molecular descriptors and fingerprint
        """
        try:
            # Convert SMILES to RDKit molecule object
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Calculate molecular fingerprint
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=self.fingerprint_radius, nBits=self.fingerprint_bits
            )
            fingerprint_bits = list(fingerprint)
            
            # Calculate molecular descriptors
            molecular_weight = Descriptors.MolWt(mol)
            logP = Descriptors.MolLogP(mol)
            polar_surface_area = Descriptors.TPSA(mol)
            num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            num_h_donors = Descriptors.NumHDonors(mol)
            num_h_acceptors = Descriptors.NumHAcceptors(mol)
            num_aromatic_rings = Descriptors.NumAromaticRings(mol)
            fraction_csp3 = Descriptors.FractionCSP3(mol)
            bertz_complexity = Descriptors.BertzCT(mol)
            
            return {
                'Fingerprint': fingerprint_bits,
                'SMILES': smiles,
                'MolecularWeight': molecular_weight,
                'LogP': logP,
                'PolarSurfaceArea': polar_surface_area,
                'NumRotatableBonds': num_rotatable_bonds,
                'NumHDonors': num_h_donors,
                'NumHAcceptors': num_h_acceptors,
                'NumAromaticRings': num_aromatic_rings,
                'FractionCSP3': fraction_csp3,
                'BertzComplexity': bertz_complexity
            }
            
        except Exception as e:
            logger.error(f"Error calculating descriptors for SMILES {smiles[:50]}...: {e}")
            return None
    
    def process_ligand_batch(self, pub_ids: List[int], cid_to_smiles: Dict[int, str]) -> Dict[int, Dict]:
        """
        Process a batch of ligands.
        
        Args:
            pub_ids: List of PubChem IDs to process
            cid_to_smiles: Dictionary mapping CID to SMILES
            
        Returns:
            Dictionary mapping PubChem ID to molecular descriptors
        """
        batch_results = {}
        
        for pub_id in pub_ids:
            smiles = cid_to_smiles.get(pub_id, None)
            if smiles is None:
                continue
            
            # Calculate descriptors
            descriptors = self.calculate_molecular_descriptors(smiles)
            if descriptors is not None:
                batch_results[pub_id] = descriptors
        
        return batch_results
    
    def process_ligands_parallel(self, pubchem_ids: List[int], cid_to_smiles: Dict[int, str],
                                batch_size: int = LIGAND_BATCH_SIZE) -> Dict[int, Dict]:
        """
        Process ligands in parallel using multiple processes.
        
        Args:
            pubchem_ids: List of PubChem IDs to process
            cid_to_smiles: Dictionary mapping CID to SMILES
            batch_size: Size of batches for parallel processing
            
        Returns:
            Dictionary mapping PubChem ID to molecular descriptors
        """
        logger.info(f"Processing {len(pubchem_ids)} ligands in parallel...")
        
        # Split pubchem_ids into batches
        batches = [pubchem_ids[i:i + batch_size] for i in range(0, len(pubchem_ids), batch_size)]
        
        ligand_data = {}
        start_time = time.time()
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self.process_ligand_batch, batch, cid_to_smiles): batch 
                for batch in batches
            }
            
            for future in as_completed(futures):
                result_batch = future.result()
                ligand_data.update(result_batch)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"Processed {len(ligand_data)} ligands in {processing_time:.2f} seconds")
        
        return ligand_data
    
    def process_ligands(self, pubchem_ids: List[int], cid_smiles_path: str,
                       output_path: str, batch_size: int = LIGAND_BATCH_SIZE) -> Dict[int, Dict]:
        """
        Complete ligand processing pipeline.
        
        Args:
            pubchem_ids: List of PubChem IDs to process
            cid_smiles_path: Path to CID-SMILES mapping file
            output_path: Path to save ligand data
            batch_size: Size of batches for parallel processing
            
        Returns:
            Dictionary mapping PubChem ID to molecular descriptors
        """
        logger.info("Starting ligand processing pipeline...")
        
        # Load CID to SMILES mapping
        cid_to_smiles = self.load_cid_smiles_mapping(cid_smiles_path)
        
        if not cid_to_smiles:
            logger.error("Failed to load CID-SMILES mapping")
            return {}
        
        # Process ligands in parallel
        ligand_data = self.process_ligands_parallel(pubchem_ids, cid_to_smiles, batch_size)
        
        # Save ligand data
        save_pickle(ligand_data, output_path)
        
        logger.info("Ligand processing completed successfully!")
        
        return ligand_data
    
    def combine_ligand_batches(self, batch_paths: List[str], output_path: str) -> Dict[int, Dict]:
        """
        Combine multiple ligand data batches into a single file.
        
        Args:
            batch_paths: List of paths to batch files
            output_path: Path to save combined data
            
        Returns:
            Combined dictionary of all ligand data
        """
        logger.info("Combining ligand data batches...")
        
        combined_data = {}
        
        for batch_path in batch_paths:
            try:
                batch_data = load_pickle(batch_path)
                combined_data.update(batch_data)
                logger.info(f"Loaded batch from {batch_path}: {len(batch_data)} ligands")
            except Exception as e:
                logger.error(f"Error loading batch from {batch_path}: {e}")
        
        # Save combined data
        save_pickle(combined_data, output_path)
        
        logger.info(f"Combined {len(combined_data)} total ligands")
        
        return combined_data 