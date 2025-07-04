"""
Protein processing module for generating protein embeddings using ProtBERT.
"""

import torch
import numpy as np
import logging
import time
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModel
from ..utils.data_utils import fetch_protein_sequence, save_pickle, load_pickle
from ..config.config import MODEL_NAME, MAX_PROTEIN_SEQUENCE_LENGTH

logger = logging.getLogger(__name__)


class ProteinProcessor:
    """Class for processing proteins and generating embeddings."""
    
    def __init__(self, model_name: str = MODEL_NAME, max_length: int = MAX_PROTEIN_SEQUENCE_LENGTH):
        """
        Initialize the protein processor.
        
        Args:
            model_name: Name of the ProtBERT model to use
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        logger.info(f"Initializing ProtBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on device: {self.device}")
    
    def get_protein_embedding(self, sequence: str) -> Optional[np.ndarray]:
        """
        Generate protein embedding from sequence.
        
        Args:
            sequence: Protein amino acid sequence
            
        Returns:
            Protein embedding as numpy array
        """
        try:
            # Add spaces to the sequence for proper tokenization
            sequence = " ".join(sequence)
            
            # Tokenize the sequence
            inputs = self.tokenizer(
                sequence, 
                return_tensors="pt", 
                is_split_into_words=False, 
                padding='max_length', 
                truncation=True, 
                add_special_tokens=True, 
                max_length=self.max_length
            )
            
            # Move inputs to device
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model(**inputs).last_hidden_state
                
            # Mean pooling to get fixed-size embedding
            embeddings = torch.mean(embeddings, dim=1)
            
            # Move to CPU and convert to numpy
            return embeddings.view(-1).cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error generating embedding for sequence: {e}")
            return None
    
    def load_existing_embeddings(self, test_embeddings_path: str, train_embeddings_path: str,
                                test_ids_path: str, train_ids_path: str) -> Dict[str, np.ndarray]:
        """
        Load existing protein embeddings from numpy files.
        
        Args:
            test_embeddings_path: Path to test embeddings
            train_embeddings_path: Path to train embeddings
            test_ids_path: Path to test IDs
            train_ids_path: Path to train IDs
            
        Returns:
            Dictionary mapping protein IDs to embeddings
        """
        logger.info("Loading existing protein embeddings...")
        
        try:
            # Load embeddings and IDs
            bert_data_test = np.load(test_embeddings_path)
            bert_data_train = np.load(train_embeddings_path)
            bert_data_test_ids = np.load(test_ids_path)
            bert_data_train_ids = np.load(train_ids_path)
            
            # Concatenate arrays
            bert_data_ids = np.concatenate((bert_data_test_ids, bert_data_train_ids), axis=0)
            bert_data = np.concatenate((bert_data_test, bert_data_train), axis=0)
            
            # Create dictionary
            protein_embeddings = dict(zip(bert_data_ids, bert_data))
            
            logger.info(f"Loaded {len(protein_embeddings)} existing protein embeddings")
            return protein_embeddings
            
        except Exception as e:
            logger.error(f"Error loading existing embeddings: {e}")
            return {}
    
    def generate_missing_embeddings(self, protein_ids: List[str], 
                                  existing_embeddings: Dict[str, np.ndarray],
                                  max_proteins: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for proteins that don't have existing embeddings.
        
        Args:
            protein_ids: List of protein IDs to process
            existing_embeddings: Dictionary of existing embeddings
            max_proteins: Maximum number of proteins to process (for testing)
            
        Returns:
            Dictionary of new protein embeddings
        """
        logger.info("Generating missing protein embeddings...")
        
        # Find proteins without existing embeddings
        missing_proteins = [pid for pid in protein_ids if pid not in existing_embeddings]
        
        if max_proteins:
            missing_proteins = missing_proteins[:max_proteins]
        
        logger.info(f"Found {len(missing_proteins)} proteins without embeddings")
        
        protein_embeddings_missed = {}
        start_time = time.time()
        
        for i, uniprot_id in enumerate(missing_proteins):
            if i % 10 == 0:
                logger.info(f"Processing protein {i+1}/{len(missing_proteins)}: {uniprot_id}")
            
            # Fetch protein sequence
            sequence = fetch_protein_sequence(uniprot_id)
            
            if sequence:
                # Generate embedding
                embedding = self.get_protein_embedding(sequence)
                if embedding is not None:
                    protein_embeddings_missed[uniprot_id] = embedding
            else:
                logger.warning(f"No sequence found for UniProt_ID: {uniprot_id}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"Generated {len(protein_embeddings_missed)} embeddings in {processing_time:.2f} seconds")
        
        return protein_embeddings_missed
    
    def combine_embeddings(self, existing_embeddings: Dict[str, np.ndarray],
                          new_embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Combine existing and new protein embeddings.
        
        Args:
            existing_embeddings: Dictionary of existing embeddings
            new_embeddings: Dictionary of new embeddings
            
        Returns:
            Combined dictionary of all embeddings
        """
        logger.info("Combining protein embeddings...")
        
        combined_embeddings = existing_embeddings.copy()
        combined_embeddings.update(new_embeddings)
        
        logger.info(f"Combined embeddings: {len(combined_embeddings)} total proteins")
        
        return combined_embeddings
    
    def process_proteins(self, protein_ids: List[str], 
                        test_embeddings_path: str, train_embeddings_path: str,
                        test_ids_path: str, train_ids_path: str,
                        output_path: str, missed_output_path: str,
                        max_proteins: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Complete protein processing pipeline.
        
        Args:
            protein_ids: List of protein IDs to process
            test_embeddings_path: Path to test embeddings
            train_embeddings_path: Path to train embeddings
            test_ids_path: Path to test IDs
            train_ids_path: Path to train IDs
            output_path: Path to save combined embeddings
            missed_output_path: Path to save missed embeddings
            max_proteins: Maximum number of proteins to process
            
        Returns:
            Dictionary of all protein embeddings
        """
        logger.info("Starting protein processing pipeline...")
        
        # Load existing embeddings
        existing_embeddings = self.load_existing_embeddings(
            test_embeddings_path, train_embeddings_path, test_ids_path, train_ids_path
        )
        
        # Generate missing embeddings
        new_embeddings = self.generate_missing_embeddings(
            protein_ids, existing_embeddings, max_proteins
        )
        
        # Combine embeddings
        combined_embeddings = self.combine_embeddings(existing_embeddings, new_embeddings)
        
        # Save embeddings
        save_pickle(combined_embeddings, output_path)
        save_pickle(new_embeddings, missed_output_path)
        
        logger.info("Protein processing completed successfully!")
        
        return combined_embeddings 