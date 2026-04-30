#!/usr/bin/env python3
"""Model training pipeline: load features → split → train → evaluate → save.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --split cold_protein   # recommended
    python scripts/run_training.py --split random         # baseline comparison
    python scripts/run_training.py --tune                 # enable HP tuning
    python scripts/run_training.py --n_samples 1000       # fast dev run
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plbind.config import CFG
from plbind.features.feature_builder import FeatureBuilder
from plbind.training.pipeline import TrainingPipeline
from plbind.utils.logging import setup_logging
from plbind.utils.seed import set_all_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protein-ligand model training")
    parser.add_argument(
        "--split",
        choices=["random", "cold_protein", "cold_ligand", "scaffold", "cold_both"],
        default="cold_protein",
        help="Train/test split strategy (default: cold_protein)",
    )
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--no_mlp", action="store_true", help="Skip InteractionMLP (faster smoke tests)")
    parser.add_argument("--device", default=None, help="Override device for MLP: auto|cpu|cuda|mps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override MLP batch size (default: CFG.batch_size)")
    parser.add_argument("--mlp_epochs", type=int, default=None, help="Override MLP max epochs (default: CFG.epochs)")
    parser.add_argument("--mlp_patience", type=int, default=None, help="Override MLP early-stopping patience (default: CFG.patience)")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--n_proteins", type=int, default=None,
                        help="Randomly select N proteins and keep all their rows (protein-level subsampling)")
    parser.add_argument("--output_dir", default=None, help="Override CFG.outputs_dir for this run")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def load_processed_data(processed_dir: Path) -> tuple:
    """Load all pre-computed feature files from disk."""
    combined = pd.read_csv(processed_dir / "combined_data.csv")

    with (processed_dir / "protein_embeddings.pkl").open("rb") as f:
        protein_embeddings: dict = pickle.load(f)

    with (processed_dir / "cid_to_row.pkl").open("rb") as f:
        cid_to_row: dict = pickle.load(f)

    fp_matrix = sp.load_npz(processed_dir / "ligand_fp.npz")
    desc_matrix = np.load(processed_dir / "ligand_desc.npy")

    aux_features = None
    aux_path = processed_dir / "aux_features.csv"
    if aux_path.exists():
        aux_features = pd.read_csv(aux_path, index_col="UniProt_ID")

    return combined, protein_embeddings, cid_to_row, fp_matrix, desc_matrix, aux_features


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else CFG.outputs_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "logs", level=args.log_level)
    set_all_seeds(CFG.random_seed)
    logger = logging.getLogger(__name__)

    logger.info("=== Training Pipeline START ===")
    logger.info("Split: %s | tune: %s | n_samples: %s", args.split, args.tune, args.n_samples)

    # ── Load data ─────────────────────────────────────────────────────────────
    combined, protein_embeddings, cid_to_row, fp_matrix, desc_matrix, aux_features = \
        load_processed_data(CFG.processed_dir)
    logger.info("Loaded %d rows, %d proteins, %d ligands.", len(combined),
                len(protein_embeddings), fp_matrix.shape[0])

    # Early subsample: filter combined BEFORE building the feature matrix so
    # the matrix never grows to full-dataset size in RAM.
    if args.n_proteins and args.n_proteins < combined["UniProt_ID"].nunique():
        rng = np.random.default_rng(CFG.random_seed)
        all_proteins = combined["UniProt_ID"].unique()
        chosen = rng.choice(all_proteins, size=args.n_proteins, replace=False)
        combined = combined[combined["UniProt_ID"].isin(chosen)].reset_index(drop=True)
        logger.info("Protein subsample: selected %d proteins → %d rows.", args.n_proteins, len(combined))
    elif args.n_samples and args.n_samples < len(combined):
        rng = np.random.default_rng(CFG.random_seed)
        idx = rng.choice(len(combined), size=args.n_samples, replace=False)
        combined = combined.iloc[idx].reset_index(drop=True)
        logger.info("Row subsample: %d rows.", len(combined))

    # ── Build feature matrix ──────────────────────────────────────────────────
    builder = FeatureBuilder(
        protein_embeddings=protein_embeddings,
        cid_to_row=cid_to_row,
        fp_matrix=fp_matrix,
        desc_matrix=desc_matrix,
        aux_features=aux_features,
    )

    X, y, block_map, df_filtered = builder.build(combined, log_attrition=True)
    protein_block, ligand_block, aux_block, _ = builder.build_blocks(df_filtered)
    feature_names = builder.feature_names

    logger.info("Feature matrix: %s  (protein=%d, ligand=%d, aux=%d)",
                X.shape, builder.protein_dim, builder.ligand_dim, builder.aux_dim)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    mlp_device = args.device or CFG.device
    mlp_batch_size = args.batch_size if args.batch_size else CFG.batch_size
    pipeline = TrainingPipeline(
        X=X,
        y=y,
        df=df_filtered,
        block_map=block_map,
        feature_names=feature_names,
        protein_block=None if args.no_mlp else protein_block,
        ligand_block=None if args.no_mlp else ligand_block,
        aux_block=None if args.no_mlp else aux_block,
        split_strategy=args.split,
        random_seed=CFG.random_seed,
        n_samples=None,  # already applied above
        tune=args.tune,
        output_dir=output_dir,
        mlp_device=mlp_device,
        mlp_batch_size=mlp_batch_size,
        mlp_epochs=args.mlp_epochs,
        mlp_patience=args.mlp_patience,
    )

    results = pipeline.run()
    logger.info("=== Training Pipeline DONE ===")

    # Print summary table
    scalar_results = {k: v for k, v in results.items()
                      if isinstance(v, dict) and "pr_auc" in v}
    if scalar_results:
        import pandas as pd
        from plbind.evaluation.evaluator import ModelEvaluator
        comparison = ModelEvaluator.compare_models(scalar_results)
        print("\n=== Model Comparison ===")
        print(comparison[["pr_auc", "roc_auc", "f1_binary", "accuracy"]].to_string())


if __name__ == "__main__":
    main()
