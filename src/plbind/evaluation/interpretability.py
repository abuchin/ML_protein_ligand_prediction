"""SHAP-based feature importance and feature-group attribution.

Two explainer types:
  - TreeExplainer: exact SHAP for RF, XGBoost, LightGBM. Fast and exact.
  - DeepExplainer:  gradient-based SHAP for PyTorch models. Approximate.

feature_group_importance() aggregates SHAP values by feature block
(protein / ligand / aux), answering the question: "which type of
information drives predictions?" — key for the submission analysis.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """Compute and visualize SHAP values for trained models.

    Args:
        feature_names: Full list of feature names (protein + ligand + aux).
        block_map:     Dict mapping block name → column slice in the flat matrix.
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        block_map: Optional[Dict[str, slice]] = None,
    ) -> None:
        self.feature_names = feature_names
        self.block_map = block_map

    # ── Explainers ────────────────────────────────────────────────────────────

    def explain_tree(self, model, X: np.ndarray, max_samples: int = 1000):
        """Exact SHAP for RF / XGBoost / LightGBM (TreeExplainer)."""
        try:
            import shap
        except ImportError:
            raise ImportError("Install shap: pip install shap")

        sk_model = model.model if hasattr(model, "model") else model
        explainer = shap.TreeExplainer(sk_model)

        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X = X[idx]

        logger.info("Computing SHAP values (TreeExplainer) for %d samples...", len(X))
        shap_values = explainer(X)
        return shap_values

    def explain_mlp(self, model, X: np.ndarray, background_samples: int = 200):
        """Gradient-based SHAP for PyTorch InteractionMLP (DeepExplainer)."""
        try:
            import shap
            import torch
        except ImportError:
            raise ImportError("Install shap: pip install shap")

        net = model._net if hasattr(model, "_net") else model
        net.eval()

        bg_idx = np.random.choice(len(X), min(background_samples, len(X)), replace=False)
        background = torch.tensor(X[bg_idx], dtype=torch.float32)
        test_data = torch.tensor(X, dtype=torch.float32)

        explainer = shap.DeepExplainer(net, background)
        logger.info("Computing SHAP values (DeepExplainer) for %d samples...", len(X))
        shap_values = explainer.shap_values(test_data)
        return shap_values

    # ── Aggregation ───────────────────────────────────────────────────────────

    def feature_group_importance(self, shap_values) -> pd.DataFrame:
        """Aggregate mean |SHAP| by feature block (protein / ligand / aux).

        Args:
            shap_values: SHAP Explanation object (from TreeExplainer) or ndarray.

        Returns:
            DataFrame with columns [block, mean_abs_shap, fraction].
        """
        if self.block_map is None:
            raise ValueError("block_map must be provided to aggregate by group.")

        # Handle both Explanation objects and raw ndarrays
        if hasattr(shap_values, "values"):
            vals = np.abs(shap_values.values)  # (N, F) or (N, F, C)
        else:
            vals = np.abs(shap_values)

        # For multi-class: take positive class (index 1)
        if vals.ndim == 3:
            vals = vals[:, :, 1]

        rows = []
        total = vals.sum()
        for block_name, slc in self.block_map.items():
            block_importance = vals[:, slc].sum()
            rows.append({
                "block": block_name,
                "mean_abs_shap": float(block_importance / len(vals)),
                "fraction": float(block_importance / total) if total > 0 else 0.0,
            })

        return pd.DataFrame(rows).sort_values("fraction", ascending=False)

    def top_features(self, shap_values, n: int = 20) -> pd.DataFrame:
        """Return top-n features by mean |SHAP| value."""
        if hasattr(shap_values, "values"):
            vals = np.abs(shap_values.values)
        else:
            vals = np.abs(shap_values)

        if vals.ndim == 3:
            vals = vals[:, :, 1]

        mean_importance = vals.mean(axis=0)
        names = self.feature_names or [f"feat_{i}" for i in range(len(mean_importance))]

        return (
            pd.DataFrame({"feature": names, "mean_abs_shap": mean_importance})
            .sort_values("mean_abs_shap", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    # ── Visualization ─────────────────────────────────────────────────────────

    def plot_summary(
        self,
        shap_values,
        X: np.ndarray,
        max_display: int = 20,
        save_path: Optional[Path] = None,
    ) -> None:
        """SHAP summary (beeswarm) plot."""
        import shap
        import matplotlib.pyplot as plt

        feature_names = self.feature_names or [f"feat_{i}" for i in range(X.shape[1])]
        if hasattr(shap_values, "values"):
            shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
        else:
            shap.summary_plot(shap_values, X, feature_names=feature_names,
                              max_display=max_display, show=False)
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            logger.info("SHAP summary saved → %s", save_path)
        plt.show()

    def plot_group_importance(
        self,
        shap_values,
        save_path: Optional[Path] = None,
    ) -> None:
        """Horizontal bar chart showing fraction of importance per feature block."""
        import matplotlib.pyplot as plt

        df = self.feature_group_importance(shap_values)
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.barh(df["block"], df["fraction"] * 100)
        ax.set_xlabel("% of total SHAP importance")
        ax.set_title("Feature group importance")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()
