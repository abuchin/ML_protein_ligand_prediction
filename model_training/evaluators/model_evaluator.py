"""
Model evaluation module for computing metrics and generating reports.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Computes metrics, cross-validates, and compares models."""

    def __init__(self):
        self.metrics = {}

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                        y_proba: np.ndarray = None) -> Dict[str, Any]:
        """Return a dict of classification metrics including macro averages and PR-AUC."""
        metrics = {}

        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Weighted averages
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Macro averages — less biased when classes are imbalanced
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)

        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        # Per-class breakdown
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)

        if y_proba is not None and y_proba.ndim > 1:
            pos_proba = y_proba[:, 1]
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, pos_proba)
            except Exception as e:
                logger.warning(f"ROC-AUC failed: {e}")
                metrics['roc_auc'] = None
            try:
                metrics['pr_auc'] = average_precision_score(y_true, pos_proba)
            except Exception as e:
                logger.warning(f"PR-AUC failed: {e}")
                metrics['pr_auc'] = None

        logger.info(
            f"Metrics — acc={metrics['accuracy']:.4f}  "
            f"f1_macro={metrics['f1_macro']:.4f}  "
            f"roc_auc={metrics.get('roc_auc', 'N/A')}  "
            f"pr_auc={metrics.get('pr_auc', 'N/A')}"
        )
        return metrics

    def cross_validate(self, sklearn_estimator, X: np.ndarray, y: np.ndarray,
                       cv: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Stratified k-fold CV for a raw sklearn estimator.

        Scales features inside each fold to prevent leakage.
        Returns mean and std for accuracy, f1_macro, and roc_auc.
        """
        pipeline = make_pipeline(StandardScaler(), sklearn_estimator)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        results = {}
        for scoring in ('accuracy', 'f1_macro', 'roc_auc'):
            try:
                scores = cross_val_score(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1)
                results[scoring] = {'mean': float(scores.mean()), 'std': float(scores.std())}
                logger.info(
                    f"CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f}"
                )
            except Exception as e:
                logger.warning(f"CV scoring '{scoring}' failed: {e}")
                results[scoring] = {'mean': None, 'std': None}

        return results

    def generate_classification_report(self, y_true, y_pred,
                                       target_names: List[str] = None) -> str:
        if target_names is None:
            target_names = ['Not Bound', 'Bound']
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

    def evaluate_model(self, model, X_test=None, y_test=None) -> Dict[str, Any]:
        """Evaluate a trained BaseModel instance."""
        logger.info(f"Evaluating {model.model_name}...")

        if hasattr(model, 'y_pred') and model.y_pred is not None:
            y_pred = model.y_pred
            y_true = model.y_test
        else:
            if X_test is None or y_test is None:
                raise ValueError("Provide X_test/y_test or train the model first")
            y_pred = model.predict()
            y_true = y_test

        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba()
            except Exception as e:
                logger.warning(f"predict_proba failed: {e}")

        metrics = self.compute_metrics(y_true, y_pred, y_proba)
        report = self.generate_classification_report(y_true, y_pred)
        model_info = model.get_model_info()

        return {
            'model_name': model.model_name,
            'metrics': metrics,
            'classification_report': report,
            'model_info': model_info,
            'predictions': y_pred,
            'true_labels': y_true,
            'probabilities': y_proba,
        }

    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        rows = []
        for name, res in model_results.items():
            m = res['metrics']
            rows.append({
                'Model': name,
                'Accuracy': m['accuracy'],
                'F1 (weighted)': m['f1'],
                'F1 (macro)': m['f1_macro'],
                'ROC-AUC': m.get('roc_auc'),
                'PR-AUC': m.get('pr_auc'),
                'Train time (s)': res['model_info'].get('training_time'),
            })
        df = pd.DataFrame(rows).sort_values('F1 (macro)', ascending=False)
        return df

    def get_best_model(self, model_results: Dict[str, Dict],
                       metric: str = 'f1_macro') -> Tuple[str, Dict]:
        best_name, best_score = None, -1.0
        for name, res in model_results.items():
            score = res['metrics'].get(metric, -1) or -1
            if score > best_score:
                best_score = score
                best_name = name
        if best_name is None:
            raise ValueError(f"No models found with metric '{metric}'")
        return best_name, model_results[best_name]

    def save_evaluation_results(self, results: Dict[str, Any], filepath: str) -> None:
        import json

        def _serialize(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, dict):
                return {k: _serialize(vv) for k, vv in v.items()}
            if isinstance(v, (np.integer, np.floating)):
                return v.item()
            return v

        with open(filepath, 'w') as f:
            json.dump({k: _serialize(v) for k, v in results.items()}, f, indent=2)
        logger.info(f"Results saved to {filepath}")
