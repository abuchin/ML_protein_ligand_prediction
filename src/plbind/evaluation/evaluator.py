"""ModelEvaluator — compute all metrics, cross-validate, and compare models.

Fixes vs. original:
  - PR-AUC (average_precision_score) was listed in METRICS config but never computed.
  - F1 now reported as binary (positive class) + macro; 'weighted' is less informative
    for imbalanced DTI data and can mask failures on the rare positive class.
  - cross_validate() accepts a groups= parameter for GroupKFold / StratifiedGroupKFold,
    enabling protein-aware CV without data leakage across folds.
  - Calibration methods added (CalibratedClassifierCV + reliability diagram).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Compute classification metrics, run CV, compare models, and calibrate probabilities."""

    # ── Metric computation ────────────────────────────────────────────────────

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Return a dict of all classification metrics.

        Metrics reported:
            accuracy, f1_binary (positive class), f1_macro, precision_binary,
            recall_binary, roc_auc, pr_auc, confusion_matrix.
        """
        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_binary": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_binary": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
            "recall_binary": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        if y_proba is not None:
            prob_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
            except ValueError:
                metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float(average_precision_score(y_true, prob_pos))

        return metrics

    # ── Cross-validation ──────────────────────────────────────────────────────

    @staticmethod
    def cross_validate(
        estimator,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        cv: int = 5,
        scoring: Optional[List[str]] = None,
        random_state: int = 42,
    ) -> Dict[str, Dict[str, float]]:
        """Run CV with optional protein-aware grouping.

        Args:
            estimator: sklearn-compatible estimator (with fit/predict_proba).
            X:         Feature matrix (already scaled).
            y:         Binary labels.
            groups:    Group labels for GroupKFold (e.g., UniProt_IDs).
                       If provided, uses StratifiedGroupKFold to prevent leakage.
            cv:        Number of folds.
            scoring:   Metric names; defaults to [f1_macro, roc_auc, average_precision].

        Returns:
            Dict of metric_name → {"mean": float, "std": float}.
        """
        if scoring is None:
            scoring = ["f1_macro", "roc_auc", "average_precision"]

        if groups is not None:
            splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=random_state)
            logger.info("CV: StratifiedGroupKFold (n_splits=%d) — protein-aware, no leakage.", cv)
        else:
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            logger.info("CV: StratifiedKFold (n_splits=%d).", cv)

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Wrap in pipeline so scaler is fitted inside each fold
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])

        cv_results = cross_validate(
            pipe, X, y,
            groups=groups,
            cv=splitter,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )

        summary: Dict[str, Dict[str, float]] = {}
        for metric in scoring:
            key = f"test_{metric}"
            if key in cv_results:
                vals = cv_results[key]
                summary[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                logger.info("CV %s: %.4f ± %.4f", metric, summary[metric]["mean"], summary[metric]["std"])
        return summary

    # ── Full model evaluation ─────────────────────────────────────────────────

    @classmethod
    def evaluate_model(cls, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Predict on X_test and compute all metrics."""
        y_pred = model.model.predict(X_test)
        y_proba = model.model.predict_proba(X_test) if hasattr(model.model, "predict_proba") else None
        return cls.compute_metrics(y_test, y_pred, y_proba)

    # ── Comparison ────────────────────────────────────────────────────────────

    @staticmethod
    def compare_models(model_results: Dict[str, Dict]) -> pd.DataFrame:
        """Return a DataFrame comparing models across all metrics."""
        rows = []
        for name, result in model_results.items():
            row = {"model": name}
            row.update({k: v for k, v in result.items() if not isinstance(v, (list, dict))})
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")

    @staticmethod
    def get_best_model(model_results: Dict, metric: str = "pr_auc") -> Tuple[str, Dict]:
        """Return (model_name, result_dict) for the model with the highest metric."""
        best_name = max(model_results, key=lambda n: model_results[n].get(metric, -1))
        return best_name, model_results[best_name]

    # ── Calibration ───────────────────────────────────────────────────────────

    @staticmethod
    def calibrate(model, X_val: np.ndarray, y_val: np.ndarray, method: str = "isotonic"):
        """Calibrate probability estimates using Platt scaling or isotonic regression.

        Returns a CalibratedClassifierCV fitted on val set.
        """
        calibrated = CalibratedClassifierCV(model.model, method=method, cv="prefit")
        calibrated.fit(X_val, y_val)
        logger.info("Calibrated %s with method='%s'.", model.model_name, method)
        return calibrated

    @staticmethod
    def plot_calibration_curves(
        calibrated_models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_bins: int = 10,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot reliability diagrams for all calibrated models."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

        for name, model in calibrated_models.items():
            prob_pos = model.predict_proba(X_test)[:, 1]
            frac_pos, mean_pred = calibration_curve(y_test, prob_pos, n_bins=n_bins)
            ax.plot(mean_pred, frac_pos, marker="o", label=name)

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration curves (reliability diagram)")
        ax.legend()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

    # ── Persistence ───────────────────────────────────────────────────────────

    @staticmethod
    def save_results(results: Dict, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Evaluation results saved → %s", path)
