# Protein-Ligand Binding Data Preparation Pipeline

This directory contains a modular and well-organized pipeline for preparing protein-ligand binding data for machine learning models.

## Overview

The pipeline consists of three main steps:
1. **Data Preprocessing**: Clean and prepare the raw dataset
2. **Protein Processing**: Generate protein embeddings using ProtBERT
3. **Ligand Processing**: Calculate molecular descriptors and fingerprints using RDKit

## Directory Structure

```
data_prep/
├── config/
│   ├── __init__.py
│   └── config.py              # Configuration parameters
├── utils/
│   ├── __init__.py
│   └── data_utils.py          # Utility functions
├── processors/
│   ├── __init__.py
│   ├── data_preprocessor.py   # Data cleaning and preprocessing
│   ├── protein_processor.py   # Protein embedding generation
│   └── ligand_processor.py    # Molecular descriptor calculation
├── pipeline.py                # Main pipeline orchestrator
├── example_usage.py           # Usage examples
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r data_prep/requirements.txt
```

2. Ensure you have the required data files in the `Data/` directory:
   - `Deloitte_DrugDiscovery_dataset.csv` - Main dataset
   - `CID-SMILES.txt` - CID to SMILES mapping
   - `Pro_bert_test_embeddings.npy` - Pre-computed protein embeddings (test)
   - `Pro_bert_train_embeddings.npy` - Pre-computed protein embeddings (train)
   - `Prot_Bert_test_ids.npy` - Protein IDs for test embeddings
   - `Prot_Bert_train_ids.npy` - Protein IDs for train embeddings

## Usage

### Basic Usage

```python
from data_prep.pipeline import DataPreparationPipeline

# Initialize pipeline with default configuration
pipeline = DataPreparationPipeline()

# Run complete pipeline
output_files = pipeline.run_complete_pipeline()
```

### Custom Configuration

```python
# Custom configuration
config = {
    'kiba_threshold': 0.005,        # KIBA score threshold
    'random_state': 42,             # Random seed
    'n_proteins_to_process': 10,    # Number of proteins to process (for testing)
    'ligand_batch_size': 25         # Batch size for ligand processing
}

# Initialize pipeline with custom config
pipeline = DataPreparationPipeline(config)

# Run complete pipeline
output_files = pipeline.run_complete_pipeline()
```

### Step-by-Step Processing

```python
pipeline = DataPreparationPipeline()

# Step 1: Data preprocessing
preprocessed_path = pipeline.run_data_preprocessing()

# Step 2: Protein processing
protein_path = pipeline.run_protein_processing(
    data_path=preprocessed_path,
    max_proteins=5  # Process only 5 proteins for testing
)

# Step 3: Ligand processing
ligand_path = pipeline.run_ligand_processing(
    data_path=preprocessed_path,
    batch_size=10
)
```

## Configuration

The pipeline configuration is centralized in `config/config.py`. Key parameters include:

- **KIBA_THRESHOLD**: Threshold for determining protein-ligand binding (default: 0.01)
- **RANDOM_STATE**: Random seed for reproducibility (default: 37)
- **MODEL_NAME**: ProtBERT model name (default: "Rostlab/prot_bert_bfd")
- **FINGERPRINT_BITS**: Number of bits for molecular fingerprints (default: 1024)
- **BATCH_SIZE**: Batch size for parallel processing (default: 50)

## Output Files

The pipeline generates the following output files:

1. **combined_data.csv**: Preprocessed dataset with positive and negative samples
2. **protein_embeddings.pkl**: Dictionary mapping protein IDs to embeddings
3. **protein_embeddings_missed.pkl**: Newly generated protein embeddings
4. **ligand_data.pkl**: Dictionary mapping ligand IDs to molecular descriptors
5. **cid_to_smiles.pkl**: CID to SMILES mapping

## Key Features

### Data Preprocessing
- Filters reliable KIBA score estimates
- Handles duplicate protein-ligand pairs
- Creates synthetic negative samples
- Generates binary binding labels

### Protein Processing
- Loads existing ProtBERT embeddings
- Fetches missing protein sequences from UniProt
- Generates embeddings for new proteins
- Supports GPU acceleration

### Ligand Processing
- Calculates Morgan fingerprints
- Computes molecular descriptors (LogP, molecular weight, etc.)
- Parallel processing for efficiency
- Handles invalid SMILES strings gracefully

## Logging

The pipeline includes comprehensive logging that saves to both console and file (`data_prep.log`). Log levels include:
- INFO: General progress information
- WARNING: Non-critical issues
- ERROR: Critical errors that may affect processing

## Error Handling

The pipeline includes robust error handling:
- Graceful handling of missing protein sequences
- Invalid SMILES string filtering
- Network request timeouts
- Memory management for large datasets

## Performance Considerations

- **Protein Processing**: Use GPU if available for faster embedding generation
- **Ligand Processing**: Adjust batch size based on available CPU cores
- **Memory Usage**: Large datasets may require significant RAM
- **Network**: Protein sequence fetching requires internet connection

## Example Script

Run the example script to see different usage patterns:

```bash
python data_prep/example_usage.py
```

This will present options for:
1. Basic pipeline execution
2. Custom configuration usage
3. Step-by-step processing

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install all requirements from `requirements.txt`
2. **CUDA Issues**: Ensure PyTorch is installed with CUDA support if using GPU
3. **Memory Errors**: Reduce batch sizes or process fewer proteins at once
4. **Network Errors**: Check internet connection for protein sequence fetching

### Performance Tips

1. Use GPU for protein embedding generation
2. Adjust batch sizes based on available resources
3. Process proteins in smaller batches for testing
4. Monitor memory usage during large dataset processing

## Contributing

When modifying the pipeline:
1. Update configuration parameters in `config/config.py`
2. Add new utility functions to `utils/data_utils.py`
3. Extend processors in the `processors/` directory
4. Update this README with new features
5. Test with example data before running on full dataset 