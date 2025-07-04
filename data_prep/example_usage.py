"""
Example usage of the data preparation pipeline.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from data_prep.pipeline import DataPreparationPipeline
from data_prep.config.config import *


def example_basic_usage():
    """Example of basic pipeline usage."""
    print("Running basic data preparation pipeline...")
    
    # Initialize pipeline with default configuration
    pipeline = DataPreparationPipeline()
    
    # Run complete pipeline
    output_files = pipeline.run_complete_pipeline()
    
    print("Pipeline completed!")
    print("Output files:")
    for key, path in output_files.items():
        print(f"  {key}: {path}")


def example_custom_config():
    """Example with custom configuration."""
    print("Running pipeline with custom configuration...")
    
    # Custom configuration
    config = {
        'kiba_threshold': 0.005,  # More strict threshold
        'random_state': 42,
        'n_proteins_to_process': 10,  # Process only 10 proteins for testing
        'ligand_batch_size': 25
    }
    
    # Initialize pipeline with custom config
    pipeline = DataPreparationPipeline(config)
    
    # Run complete pipeline
    output_files = pipeline.run_complete_pipeline()
    
    print("Pipeline completed with custom config!")
    print("Output files:")
    for key, path in output_files.items():
        print(f"  {key}: {path}")


def example_step_by_step():
    """Example of running pipeline steps individually."""
    print("Running pipeline step by step...")
    
    # Initialize pipeline
    pipeline = DataPreparationPipeline()
    
    # Step 1: Data preprocessing
    print("\nStep 1: Data preprocessing")
    preprocessed_path = pipeline.run_data_preprocessing()
    
    # Step 2: Protein processing (only 5 proteins for testing)
    print("\nStep 2: Protein processing")
    protein_path = pipeline.run_protein_processing(
        data_path=preprocessed_path,
        max_proteins=5
    )
    
    # Step 3: Ligand processing
    print("\nStep 3: Ligand processing")
    ligand_path = pipeline.run_ligand_processing(
        data_path=preprocessed_path,
        batch_size=10
    )
    
    print("Step-by-step pipeline completed!")


if __name__ == "__main__":
    print("Data Preparation Pipeline Examples")
    print("=" * 40)
    
    # Choose which example to run
    example_choice = input("Choose example (1=basic, 2=custom, 3=step-by-step): ").strip()
    
    if example_choice == "1":
        example_basic_usage()
    elif example_choice == "2":
        example_custom_config()
    elif example_choice == "3":
        example_step_by_step()
    else:
        print("Invalid choice. Running basic example...")
        example_basic_usage() 