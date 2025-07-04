"""
Main pipeline for protein-ligand binding data preparation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from data_prep.config.config import *
from data_prep.processors.data_preprocessor import DataPreprocessor
from data_prep.processors.protein_processor import ProteinProcessor
from data_prep.processors.ligand_processor import LigandProcessor
from data_prep.utils.data_utils import save_data, load_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_prep.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DataPreparationPipeline:
    """Main pipeline for preparing protein-ligand binding data."""
    
    def __init__(self, config: dict = None):
        """
        Initialize the data preparation pipeline.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = config or {}
        
        # Initialize processors
        self.data_preprocessor = DataPreprocessor(
            kiba_threshold=self.config.get('kiba_threshold', KIBA_THRESHOLD),
            random_state=self.config.get('random_state', RANDOM_STATE)
        )
        
        self.protein_processor = ProteinProcessor(
            model_name=self.config.get('model_name', MODEL_NAME),
            max_length=self.config.get('max_length', MAX_PROTEIN_SEQUENCE_LENGTH)
        )
        
        self.ligand_processor = LigandProcessor(
            fingerprint_radius=self.config.get('fingerprint_radius', FINGERPRINT_RADIUS),
            fingerprint_bits=self.config.get('fingerprint_bits', FINGERPRINT_BITS)
        )
    
    def run_data_preprocessing(self, input_path: str = None, output_path: str = None) -> str:
        """
        Run the data preprocessing step.
        
        Args:
            input_path: Path to input data file
            output_path: Path to save preprocessed data
            
        Returns:
            Path to the preprocessed data file
        """
        logger.info("=" * 50)
        logger.info("STEP 1: DATA PREPROCESSING")
        logger.info("=" * 50)
        
        input_path = input_path or str(INPUT_DATA_PATH)
        output_path = output_path or str(COMBINED_DATA_PATH)
        
        # Run preprocessing
        preprocessed_data = self.data_preprocessor.preprocess(input_path)
        
        # Save preprocessed data
        save_data(preprocessed_data, output_path)
        
        logger.info(f"Data preprocessing completed. Output saved to: {output_path}")
        return output_path
    
    def run_protein_processing(self, data_path: str = None, max_proteins: int = None) -> str:
        """
        Run the protein processing step.
        
        Args:
            data_path: Path to preprocessed data
            max_proteins: Maximum number of proteins to process (for testing)
            
        Returns:
            Path to the protein embeddings file
        """
        logger.info("=" * 50)
        logger.info("STEP 2: PROTEIN PROCESSING")
        logger.info("=" * 50)
        
        data_path = data_path or str(COMBINED_DATA_PATH)
        max_proteins = max_proteins or self.config.get('n_proteins_to_process', N_PROTEINS_TO_PROCESS)
        
        # Load preprocessed data
        data = load_data(data_path)
        
        # Get unique protein IDs
        protein_ids = data['UniProt_ID'].unique().tolist()
        
        # Process proteins
        protein_embeddings = self.protein_processor.process_proteins(
            protein_ids=protein_ids,
            test_embeddings_path=str(PROT_BERT_TEST_EMBEDDINGS_PATH),
            train_embeddings_path=str(PROT_BERT_TRAIN_EMBEDDINGS_PATH),
            test_ids_path=str(PROT_BERT_TEST_IDS_PATH),
            train_ids_path=str(PROT_BERT_TRAIN_IDS_PATH),
            output_path=str(PROTEIN_EMBEDDINGS_PATH),
            missed_output_path=str(PROTEIN_EMBEDDINGS_MISSED_PATH),
            max_proteins=max_proteins
        )
        
        logger.info(f"Protein processing completed. Embeddings saved to: {PROTEIN_EMBEDDINGS_PATH}")
        return str(PROTEIN_EMBEDDINGS_PATH)
    
    def run_ligand_processing(self, data_path: str = None, batch_size: int = None) -> str:
        """
        Run the ligand processing step.
        
        Args:
            data_path: Path to preprocessed data
            batch_size: Size of batches for parallel processing
            
        Returns:
            Path to the ligand data file
        """
        logger.info("=" * 50)
        logger.info("STEP 3: LIGAND PROCESSING")
        logger.info("=" * 50)
        
        data_path = data_path or str(COMBINED_DATA_PATH)
        batch_size = batch_size or self.config.get('ligand_batch_size', LIGAND_BATCH_SIZE)
        
        # Load preprocessed data
        data = load_data(data_path)
        
        # Get unique ligand IDs
        ligand_ids = data['pubchem_cid'].unique().astype(int).tolist()
        
        # Process ligands
        ligand_data = self.ligand_processor.process_ligands(
            pubchem_ids=ligand_ids,
            cid_smiles_path=str(CID_SMILES_PATH),
            output_path=str(LIGAND_DATA_PATH),
            batch_size=batch_size
        )
        
        logger.info(f"Ligand processing completed. Data saved to: {LIGAND_DATA_PATH}")
        return str(LIGAND_DATA_PATH)
    
    def run_complete_pipeline(self, input_path: str = None, 
                            max_proteins: int = None,
                            batch_size: int = None) -> dict:
        """
        Run the complete data preparation pipeline.
        
        Args:
            input_path: Path to input data file
            max_proteins: Maximum number of proteins to process
            batch_size: Size of batches for ligand processing
            
        Returns:
            Dictionary with paths to all output files
        """
        logger.info("=" * 50)
        logger.info("STARTING COMPLETE DATA PREPARATION PIPELINE")
        logger.info("=" * 50)
        
        try:
            # Step 1: Data preprocessing
            preprocessed_data_path = self.run_data_preprocessing(input_path)
            
            # Step 2: Protein processing
            protein_embeddings_path = self.run_protein_processing(
                preprocessed_data_path, max_proteins
            )
            
            # Step 3: Ligand processing
            ligand_data_path = self.run_ligand_processing(
                preprocessed_data_path, batch_size
            )
            
            # Prepare output summary
            output_files = {
                'preprocessed_data': preprocessed_data_path,
                'protein_embeddings': protein_embeddings_path,
                'protein_embeddings_missed': str(PROTEIN_EMBEDDINGS_MISSED_PATH),
                'ligand_data': ligand_data_path,
                'cid_to_smiles': str(CID_TO_SMILES_PATH)
            }
            
            logger.info("=" * 50)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 50)
            logger.info("Output files:")
            for key, path in output_files.items():
                logger.info(f"  {key}: {path}")
            
            return output_files
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise


def main():
    """Main function to run the pipeline."""
    # Example configuration (can be customized)
    config = {
        'kiba_threshold': KIBA_THRESHOLD,
        'random_state': RANDOM_STATE,
        'n_proteins_to_process': N_PROTEINS_TO_PROCESS,  # Set to None for all proteins
        'ligand_batch_size': LIGAND_BATCH_SIZE
    }
    
    # Initialize and run pipeline
    pipeline = DataPreparationPipeline(config)
    
    try:
        output_files = pipeline.run_complete_pipeline()
        print("\nPipeline completed successfully!")
        print("Output files:")
        for key, path in output_files.items():
            print(f"  {key}: {path}")
            
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 