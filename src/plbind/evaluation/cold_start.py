"""Cold-start evaluation: measure generalization to unseen proteins and ligands.

Metrics reported:
  - ROC-AUC, PR-AUC (primary — threshold-free, good under imbalance)
  - F1 binary (positive class at threshold 0.5)
  - EF@1%, EF@5% — Enrichment Factor; the drug-discovery standard metric
  - Bedroc (α=20) — exponentially weighted version of EF; robust to early retrieval
  - Per-prediction CSV saved for error analysis

Reference:
    EF and Bedroc: Truchon & Bayly (2007) J. Chem. Inf. Model. 47, 488-508.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class ColdStartEvaluator:
    """Evaluate models under various cold-start conditions.

    Designed to work with any model that exposes predict_proba()
    and operates on pre-built feature matrices.
    """

    def __init__(self, ef_cutoffs: Tuple[float, ...] = (0.01, 0.05)) -> None:
        self.ef_cutoffs = ef_cutoffs

    # ── Main entry point ──────────────────────────────────────────────────────

    def evaluate_all_splits(
        self,
        models: Dict[str, Any],
        split_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        df_splits: Dict[str, pd.DataFrame],
        output_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Evaluate all models on all split types.

        Args:
            models:      dict model_name → fitted model with predict_proba(X).
            split_data:  dict split_type → (X_test, y_test).
            df_splits:   dict split_type → DataFrame (UniProt_ID, pubchem_cid, y_true, ...).
            output_dir:  If provided, save per-prediction CSVs and summary.

        Returns:
            DataFrame with columns [model, split_type, metric, value].
        """
        rows = []
        for split_type, (X_test, y_test) in split_data.items():
            df_meta = df_splits.get(split_type)
            for model_name, model in models.items():
                logger.info("Evaluating %s on %s split...", model_name, split_type)
                metrics = self.evaluate_one(model, X_test, y_test)

                if output_dir and df_meta is not None:
                    self._save_predictions(
                        model, X_test, y_test, df_meta,
                        output_dir / f"predictions_{model_name}_{split_type}.csv",
                    )

                for metric, value in metrics.items():
                    rows.append({
                        "model": model_name,
                        "split_type": split_type,
                        "metric": metric,
                        "value": value,
                    })

        result_df = pd.DataFrame(rows)
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_dir / "cold_start_summary.csv", index=False)
            logger.info("Cold-start summary saved → %s", output_dir / "cold_start_summary.csv")
        return result_df

    def evaluate_one(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Compute all metrics for one (model, test set) pair."""
        y_proba = self._get_proba(model, X_test)
        y_pred = (y_proba > 0.5).astype(int)

        metrics: Dict[str, float] = {}

        # Standard classification metrics
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float(average_precision_score(y_test, y_proba))
        metrics["f1_binary"] = float(f1_score(y_test, y_pred, average="binary", zero_division=0))

        # Drug-discovery metrics
        for cutoff in self.ef_cutoffs:
            ef_key = f"ef_at_{int(cutoff * 100)}pct"
            metrics[ef_key] = float(self._enrichment_factor(y_test, y_proba, cutoff))
        metrics["bedroc"] = float(self._bedroc(y_test, y_proba))

        return metrics

    # ── Drug-discovery metrics ────────────────────────────────────────────────

    @staticmethod
    def _enrichment_factor(y_true: np.ndarray, scores: np.ndarray, cutoff: float) -> float:
        """Enrichment Factor at top-k%.

        EF@k = (actives in top k%) / (total in top k%) / (actives_total / N_total)
        """
        n = len(y_true)
        n_top = max(1, int(np.ceil(cutoff * n)))
        top_idx = np.argsort(scores)[::-1][:n_top]
        n_actives_top = y_true[top_idx].sum()
        n_actives_total = y_true.sum()
        if n_actives_total == 0 or n_top == 0:
            return 0.0
        ef = (n_actives_top / n_top) / (n_actives_total / n)
        return float(ef)

    @staticmethod
    def _bedroc(y_true: np.ndarray, scores: np.ndarray, alpha: float = 20.0) -> float:
        """Boltzmann-Enhanced Discrimination of ROC (BEDROC, Truchon & Bayly 2007).

        BEDROC gives more weight to early retrieval (highest-scored compounds)
        than plain ROC-AUC. Ranges 0–1; higher is better.

        Falls back to PR-AUC if rdkit.ML.Scoring is not available.
        """
        try:
            from rdkit.ML.Scoring.Scoring import CalcBEDROC
            order = np.argsort(scores)[::-1]
            # CalcBEDROC expects a list of indexable rows [[label], ...] sorted
            # descending by score; col=0 points at the label column.
            scored = [[int(v)] for v in y_true[order]]
            n_actives = sum(row[0] for row in scored)
            if n_actives == 0:
                return 0.0
            return float(CalcBEDROC(scored, col=0, alpha=alpha))
        except ImportError:
            logger.debug("RDKit BEDROC unavailable; substituting PR-AUC.")
            return float(average_precision_score(y_true, scores))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_proba(model, X: np.ndarray) -> np.ndarray:
        """Extract positive-class probability from any model type.

        Handles three cases:
          1. InteractionMLPModel — uses block-specific scalers; predict_proba() reads
             its stored test tensors and takes no arguments.
          2. sklearn BaseModel wrappers (LR, RF, XGB, LGB) — apply base class scaler,
             then call the inner sklearn estimator's predict_proba.
          3. Raw sklearn estimators — call predict_proba directly.
        """
        # InteractionMLP stores block scalers separately; its predict_proba() is arg-free
        if hasattr(model, "_prot_scaler"):
            proba = model.predict_proba()          # returns (N, 2)
            return proba[:, 1] if proba.ndim == 2 else proba

        # sklearn BaseModel wrappers: scaler is on model.scaler, estimator on model.model
        if hasattr(model, "scaler") and model.scaler is not None and hasattr(model, "model"):
            X_scaled = model.scaler.transform(X).astype(np.float32)
            inner = model.model
            if hasattr(inner, "predict_proba"):
                proba = inner.predict_proba(X_scaled)
                return proba[:, 1] if proba.ndim == 2 else proba
            return inner.predict(X_scaled).astype(float)

        # Plain sklearn estimator
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return proba[:, 1] if proba.ndim == 2 else proba
        return model.predict(X).astype(float)

    @staticmethod
    def _save_predictions(
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        df_meta: pd.DataFrame,
        save_path: Path,
    ) -> None:
        """Save per-row predictions with UniProt_ID and pubchem_cid for error analysis."""
        proba = ColdStartEvaluator._get_proba(model, X_test)
        pred = (proba > 0.5).astype(int)

        out = df_meta[["UniProt_ID", "pubchem_cid"]].copy().reset_index(drop=True)
        out["y_true"] = y_test
        out["y_pred"] = pred
        out["y_proba"] = proba

        save_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(save_path, index=False)
        logger.info("Predictions saved → %s (%d rows)", save_path, len(out))
