"""End-to-end model training pipeline.

Explicit imports only — no wildcard imports (they were hiding the missing
pandas import in the original pipeline.py and make refactoring risky).

Key changes from original pipeline:
  - Uses ProteinAwareSplitter (cold-protein by default) — no leakage.
  - Calls model.tune() when tune=True (the original wired tune grids but never called them).
  - Removes dead TEST_MODE flag; replaced with n_samples parameter.
  - Reports PR-AUC and EF@1% alongside ROC-AUC.
  - Saves per-prediction CSVs for error analysis.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from plbind.config import CFG
from plbind.evaluation.cold_start import ColdStartEvaluator
from plbind.evaluation.evaluator import ModelEvaluator
from plbind.models.factory import ModelFactory
from plbind.models.mlp import InteractionMLPModel
from plbind.training.splitter import Splitter
from plbind.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrate feature loading, splitting, training, and evaluation.

    Args:
        X:          Flat feature matrix (N, total_dim).
        y:          Binary labels (N,).
        df:         Raw DataFrame aligned with X (used for group labels and predictions).
        block_map:  Dict block_name → column slice (for MLP and SHAP aggregation).
        feature_names: List of feature names (for SHAP).
        protein_block: (N, protein_dim) protein-only features for InteractionMLP.
        ligand_block:  (N, ligand_dim)  ligand-only features for InteractionMLP.
        aux_block:     (N, aux_dim) optional auxiliary features.
        split_strategy: One of random | cold_protein | cold_ligand | scaffold | cold_both.
        random_seed:   RNG seed (set_all_seeds is called in run()).
        n_samples:     Subsample to this many rows (None = full dataset).
        tune:          Whether to run hyperparameter tuning for sklearn models.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame,
        block_map: dict,
        feature_names: Optional[List[str]] = None,
        protein_block: Optional[np.ndarray] = None,
        ligand_block: Optional[np.ndarray] = None,
        aux_block: Optional[np.ndarray] = None,
        split_strategy: str = "cold_protein",
        random_seed: int = 42,
        n_samples: Optional[int] = None,
        tune: bool = False,
        output_dir: Optional[Path] = None,
        mlp_device: str = "auto",
        mlp_batch_size: Optional[int] = None,
        mlp_epochs: Optional[int] = None,
        mlp_patience: Optional[int] = None,
    ) -> None:
        self.X = X
        self.y = y
        self.df = df
        self.block_map = block_map
        self.feature_names = feature_names
        self.protein_block = protein_block
        self.ligand_block = ligand_block
        self.aux_block = aux_block
        self.split_strategy = split_strategy
        self.random_seed = random_seed
        self.n_samples = n_samples
        self.tune = tune
        self.output_dir = Path(output_dir) if output_dir else CFG.outputs_dir
        self.mlp_device = mlp_device
        self.mlp_batch_size = mlp_batch_size if mlp_batch_size is not None else CFG.batch_size
        self.mlp_epochs = mlp_epochs if mlp_epochs is not None else CFG.epochs
        self.mlp_patience = mlp_patience if mlp_patience is not None else CFG.patience

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> Dict:
        set_all_seeds(self.random_seed)
        logger.info("=== Training Pipeline START ===")
        logger.info("Split strategy: %s | tune: %s | n_samples: %s",
                    self.split_strategy, self.tune, self.n_samples)

        X, y, df = self._subsample(self.X, self.y, self.df)

        # ── Split ──
        splitter = Splitter(
            test_size=CFG.test_size,
            val_size=CFG.val_size,
            random_state=self.random_seed,
        )
        train_df, val_df, test_df = splitter.split(df, self.split_strategy)

        train_idx = train_df.index
        val_idx = val_df.index
        test_idx = test_df.index

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        # ── Train sklearn models ──
        from sklearn.preprocessing import StandardScaler
        sklearn_models = {
            "logistic_regression": ModelFactory.create("logistic_regression", random_state=self.random_seed),
            "random_forest": ModelFactory.create("random_forest", random_state=self.random_seed),
            "xgboost": ModelFactory.create("xgboost", random_state=self.random_seed),
            "lightgbm": ModelFactory.create("lightgbm", random_state=self.random_seed),
        }

        results = {}
        for name, model in sklearn_models.items():
            logger.info("--- %s ---", name)
            # Fit scaler on train split only — never touch the full X
            model.scaler = StandardScaler()
            model.X_train = model.scaler.fit_transform(X_train).astype(np.float32)
            model.X_test = model.scaler.transform(X_test).astype(np.float32)
            model.y_train = y_train
            model.y_test = y_test
            logger.info("%s split: %d train / %d test  (pos_rate_train=%.3f)",
                        name, len(y_train), len(y_test), y_train.mean())

            if self.tune:
                model.tune()

            model.train()
            model.predict()

            y_proba = model.predict_proba() if hasattr(model, "predict_proba") else None
            metrics = ModelEvaluator.compute_metrics(model.y_test, model.y_pred, y_proba)
            results[name] = metrics
            logger.info(
                "%s → PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f",
                name,
                metrics.get("pr_auc", float("nan")),
                metrics.get("roc_auc", float("nan")),
                metrics.get("f1_binary", float("nan")),
            )
            model.save_model(self.output_dir / "models" / f"{name}.pkl")
            shap_result = self._run_shap(name, model)
            if shap_result is not None:
                results[name]["shap_group_importance"] = shap_result
            # Free scaled data matrices — no longer needed after save
            model.X_train = None
            model.X_test = None

        # ── CV with protein-aware groups (all sklearn models) ──
        # XGBoost is re-instantiated without early_stopping_rounds because
        # cross_validate does not pass eval_set and early stopping requires it.
        from xgboost import XGBClassifier
        xgb_cv = XGBClassifier(
            n_estimators=sklearn_models["xgboost"].n_estimators,
            max_depth=sklearn_models["xgboost"].max_depth,
            learning_rate=sklearn_models["xgboost"].learning_rate,
            scale_pos_weight=sklearn_models["xgboost"].scale_pos_weight,
            eval_metric="logloss",
            random_state=self.random_seed,
        )
        groups_all = df["UniProt_ID"].values
        cv_estimators = {
            "logistic_regression": sklearn_models["logistic_regression"].model,
            "random_forest":       sklearn_models["random_forest"].model,
            "xgboost":             xgb_cv,
            "lightgbm":            sklearn_models["lightgbm"].model,
        }
        for cv_name, estimator in cv_estimators.items():
            try:
                logger.info("CV: %s ...", cv_name)
                cv_result = ModelEvaluator.cross_validate(
                    estimator, X, y, groups=groups_all, cv=CFG.cv_folds,
                    random_state=self.random_seed,
                )
                results[f"cv_{cv_name}"] = cv_result
                logger.info(
                    "CV %s → ROC-AUC=%.4f ± %.4f  PR-AUC=%.4f ± %.4f",
                    cv_name,
                    cv_result.get("roc_auc", {}).get("mean", float("nan")),
                    cv_result.get("roc_auc", {}).get("std", float("nan")),
                    cv_result.get("average_precision", {}).get("mean", float("nan")),
                    cv_result.get("average_precision", {}).get("std", float("nan")),
                )
            except Exception as exc:
                logger.warning("CV failed for %s: %s", cv_name, exc)

        # Free training/validation flat matrices before MLP.
        # X_test is kept — the cold-start evaluator needs it for sklearn models.
        import gc
        del X_train, X_val
        self.X = None  # release self.X so gc can reclaim the full flat matrix
        self.y = None
        X = None  # local ref
        y = None
        gc.collect()

        # ── InteractionMLP ──
        if self.protein_block is not None and self.ligand_block is not None:
            logger.info("--- InteractionMLP ---")
            # sklearn CV leaves OpenMP threads in a broken barrier state; forcing
            # single-threaded mode prevents torch.randperm from deadlocking.
            import torch as _torch
            _torch.set_num_threads(1)
            try:
                aux_dim = self.aux_block.shape[1] if self.aux_block is not None else 0
                mlp = InteractionMLPModel(
                    protein_dim=self.protein_block.shape[1],
                    ligand_dim=self.ligand_block.shape[1],
                    aux_dim=aux_dim,
                    projection_dim=CFG.projection_dim,
                    aux_projection_dim=CFG.aux_projection_dim,
                    fusion_dims=CFG.fusion_dims,
                    dropout=CFG.dropout,
                    lr=CFG.lr,
                    batch_size=self.mlp_batch_size,
                    epochs=self.mlp_epochs,
                    patience=self.mlp_patience,
                    lr_scheduler_patience=CFG.lr_scheduler_patience,
                    device=self.mlp_device,
                    random_state=self.random_seed,
                )
                mlp.set_split_data(
                    self.protein_block[train_idx], self.ligand_block[train_idx], y_train,
                    self.protein_block[val_idx],   self.ligand_block[val_idx],   y_val,
                    self.protein_block[test_idx],  self.ligand_block[test_idx],  y_test,
                    self.aux_block[train_idx] if self.aux_block is not None else None,
                    self.aux_block[val_idx] if self.aux_block is not None else None,
                    self.aux_block[test_idx] if self.aux_block is not None else None,
                )
                mlp.train()
                mlp.predict()
                y_proba_mlp = mlp.predict_proba()
                mlp_metrics = ModelEvaluator.compute_metrics(mlp.y_test, mlp.y_pred, y_proba_mlp)
                results["interaction_mlp"] = mlp_metrics
                logger.info(
                    "InteractionMLP → PR-AUC=%.4f  ROC-AUC=%.4f",
                    mlp_metrics.get("pr_auc", float("nan")),
                    mlp_metrics.get("roc_auc", float("nan")),
                )
                mlp.save_model(self.output_dir / "models" / "interaction_mlp")
                sklearn_models["interaction_mlp"] = mlp
            except Exception as exc:
                logger.error("InteractionMLP failed: %s", exc, exc_info=True)

        # ── Cold-start evaluation ──
        cold_evaluator = ColdStartEvaluator(ef_cutoffs=CFG.ef_cutoffs)
        cold_results = cold_evaluator.evaluate_all_splits(
            models={n: m for n, m in sklearn_models.items()},
            split_data={self.split_strategy: (X_test, y_test)},
            df_splits={self.split_strategy: test_df},
            output_dir=self.output_dir / "predictions",
        )
        results["cold_start"] = cold_results.to_dict()

        # ── Comparison table ──
        scalar_results = {k: v for k, v in results.items() if isinstance(v, dict)
                          and "pr_auc" in v}
        if scalar_results:
            comparison = ModelEvaluator.compare_models(scalar_results)
            logger.info("\n%s", comparison.to_string())
            comparison.to_csv(self.output_dir / "model_comparison.csv")

        ModelEvaluator.save_results(results, self.output_dir / "results.json")
        logger.info("=== Training Pipeline DONE ===")
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _run_shap(self, name: str, model) -> Optional[dict]:
        """Compute SHAP values for one trained model; save plots and values to outputs/.

        Returns a dict of group importance fractions, or None if SHAP is unavailable.
        """
        try:
            import shap as _shap  # noqa: F401 — verify importability
            from plbind.evaluation.interpretability import SHAPAnalyzer
        except ImportError:
            return None
        try:
            analyzer = SHAPAnalyzer(
                feature_names=self.feature_names,
                block_map=self.block_map,
            )
            n_bg = min(CFG.shap_background_samples, len(model.X_train))
            shap_vals = analyzer.explain_tree(model, model.X_test, max_samples=n_bg)

            shap_dir = self.output_dir / "shap"
            shap_dir.mkdir(parents=True, exist_ok=True)
            raw = shap_vals.values if hasattr(shap_vals, "values") else shap_vals
            np.save(shap_dir / f"{name}_shap_values.npy", raw)

            group_imp = analyzer.feature_group_importance(shap_vals)
            fig_dir = self.output_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            analyzer.plot_summary(
                shap_vals, model.X_test,
                save_path=fig_dir / f"shap_{name}.png",
            )
            analyzer.plot_group_importance(
                shap_vals,
                save_path=fig_dir / f"shap_groups_{name}.png",
            )
            logger.info("SHAP done for %s — group importance: %s",
                        name, group_imp.set_index("block")["fraction"].to_dict())
            return group_imp.to_dict(orient="list")
        except Exception as exc:
            logger.warning("SHAP failed for %s: %s", name, exc)
            return None

    def _subsample(
        self, X: np.ndarray, y: np.ndarray, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        if self.n_samples is not None and self.n_samples < len(y):
            rng = np.random.RandomState(self.random_seed)
            idx = rng.choice(len(y), self.n_samples, replace=False)
            if self.protein_block is not None:
                self.protein_block = self.protein_block[idx]
            if self.ligand_block is not None:
                self.ligand_block = self.ligand_block[idx]
            if self.aux_block is not None:
                self.aux_block = self.aux_block[idx]
            return X[idx], y[idx], df.iloc[idx].reset_index(drop=True)
        return X, y, df.reset_index(drop=True)
