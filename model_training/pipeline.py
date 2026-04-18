"""
Main pipeline for protein-ligand binding model training.
"""

import logging
import random
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.append(str(Path(__file__).parent.parent))

from model_training.config.config import (
    COMBINED_DATA_PATH, PROTEIN_EMBEDDINGS_PATH, LIGAND_DATA_PATH,
    OUTPUT_DIR, LOG_LEVEL, LOG_FORMAT, LOG_FILE,
    LOGISTIC_REGRESSION_PATH, RANDOM_FOREST_PATH, XGBOOST_PATH, MLP_PATH,
    PROTEIN_EMBEDDING_SIZE, FINGERPRINT_SIZE,
    FIGURE_SIZE, DPI, CMAP,
    RANDOM_SEED, TEST_MODE,
    # Production params
    N_SAMPLES, CROSS_VALIDATION_FOLDS,
    LOGISTIC_REGRESSION_PARAMS, LR_TUNE_GRID,
    RANDOM_FOREST_PARAMS, RF_TUNE_GRID,
    XGBOOST_PARAMS, XGB_TUNE_GRID,
    MLP_PARAMS,
    # Test-mode params
    TEST_N_SAMPLES, TEST_CV_FOLDS,
    TEST_LOGISTIC_REGRESSION_PARAMS, TEST_LR_TUNE_GRID,
    TEST_RANDOM_FOREST_PARAMS, TEST_RF_TUNE_GRID,
    TEST_XGBOOST_PARAMS, TEST_XGB_TUNE_GRID,
    TEST_MLP_PARAMS,
)
from model_training.data_loader.feature_engineer import FeatureEngineer
from model_training.models.model_factory import ModelFactory
from model_training.evaluators.model_evaluator import ModelEvaluator
from model_training.visualization.visualizer import ModelVisualizer

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


class ModelTrainingPipeline:
    """Orchestrates feature engineering, training, CV, tuning, evaluation, and saving."""

    def __init__(self, test_mode: bool = TEST_MODE, config: dict = None):
        self.test_mode = test_mode
        self.config = config or {}

        _set_global_seeds(RANDOM_SEED)

        # Pick param sets based on mode
        if test_mode:
            logger.info("Running in TEST MODE (small data + small models)")
            self._n_samples = TEST_N_SAMPLES
            self._cv_folds = TEST_CV_FOLDS
            self._lr_params = TEST_LOGISTIC_REGRESSION_PARAMS
            self._lr_grid = TEST_LR_TUNE_GRID
            self._rf_params = TEST_RANDOM_FOREST_PARAMS
            self._rf_grid = TEST_RF_TUNE_GRID
            self._xgb_params = TEST_XGBOOST_PARAMS
            self._xgb_grid = TEST_XGB_TUNE_GRID
            self._mlp_params = TEST_MLP_PARAMS
        else:
            self._n_samples = N_SAMPLES
            self._cv_folds = CROSS_VALIDATION_FOLDS
            self._lr_params = LOGISTIC_REGRESSION_PARAMS
            self._lr_grid = LR_TUNE_GRID
            self._rf_params = RANDOM_FOREST_PARAMS
            self._rf_grid = RF_TUNE_GRID
            self._xgb_params = XGBOOST_PARAMS
            self._xgb_grid = XGB_TUNE_GRID
            self._mlp_params = MLP_PARAMS

        self.feature_engineer = FeatureEngineer(
            protein_embedding_size=PROTEIN_EMBEDDING_SIZE,
            fingerprint_size=FINGERPRINT_SIZE,
        )
        self.evaluator = ModelEvaluator()
        self.visualizer = ModelVisualizer(
            figsize=FIGURE_SIZE, dpi=DPI, cmap=CMAP,
            save_dir=str(OUTPUT_DIR / "plots"),
        )

        self.models: Dict[str, Any] = {}
        self.model_results: Dict[str, Any] = {}
        self.cv_results: Dict[str, Any] = {}
        self.feature_info = None

    # ── Step 1: data ──────────────────────────────────────────────────────────
    def load_and_prepare_data(self) -> tuple:
        logger.info("STEP 1: DATA LOADING AND FEATURE ENGINEERING")
        X, y_continuous, y_binary, feature_info = self.feature_engineer.engineer_features(
            combined_data_path=str(COMBINED_DATA_PATH),
            protein_embeddings_path=str(PROTEIN_EMBEDDINGS_PATH),
            ligand_data_path=str(LIGAND_DATA_PATH),
            n_samples=self._n_samples,
        )
        self.feature_info = feature_info
        logger.info(f"Features shape: {X.shape}  class dist: {y_binary.value_counts().to_dict()}")
        return X, y_continuous, y_binary, feature_info

    # ── Step 2: create models ─────────────────────────────────────────────────
    def create_models(self, input_size: int,
                      model_types: List[str] = None) -> Dict[str, Any]:
        logger.info("STEP 2: MODEL CREATION")
        if model_types is None:
            model_types = ModelFactory.get_available_models()

        models: Dict[str, Any] = {}
        for mtype in model_types:
            try:
                if mtype == 'logistic_regression':
                    params = {**self._lr_params, 'test_size': 0.2, 'random_state': RANDOM_SEED}
                elif mtype == 'random_forest':
                    params = {**self._rf_params, 'test_size': 0.2, 'random_state': RANDOM_SEED}
                elif mtype == 'xgboost':
                    params = {**self._xgb_params, 'test_size': 0.2, 'random_state': RANDOM_SEED}
                elif mtype == 'mlp':
                    params = {**self._mlp_params, 'input_size': input_size,
                              'test_size': 0.2, 'random_state': RANDOM_SEED}
                else:
                    params = ModelFactory.get_default_params(mtype)

                models[mtype] = ModelFactory.create_model(mtype, **params)
                logger.info(f"Created {mtype}")
            except Exception as e:
                logger.error(f"Failed to create {mtype}: {e}")

        self.models = models
        return models

    # ── Step 3: split + optional tune + train ─────────────────────────────────
    def train_models(self, X, y, tune: bool = False,
                     y_full=None) -> Dict[str, Any]:
        logger.info(f"STEP 3: MODEL TRAINING  (tune={tune})")

        trained: Dict[str, Any] = {}
        for name, model in self.models.items():
            try:
                model.split_data(X, y)

                if tune and hasattr(model, 'tune'):
                    grid = {
                        'logistic_regression': self._lr_grid,
                        'random_forest': self._rf_grid,
                        'xgboost': self._xgb_grid,
                    }.get(name)
                    if grid:
                        model.tune(param_grid=grid, cv=self._cv_folds)

                model.train()
                model.predict()
                trained[name] = model
                logger.info(f"{name} done")
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")

        self.models = trained
        return trained

    # ── Step 4: cross-validation (sklearn models only) ────────────────────────
    def run_cross_validation(self, X, y) -> Dict[str, Any]:
        logger.info("STEP 4: CROSS-VALIDATION")
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        y_arr = y.values if hasattr(y, 'values') else np.array(y)

        cv_results: Dict[str, Any] = {}
        sklearn_types = ('logistic_regression', 'random_forest', 'xgboost')
        for name, model in self.models.items():
            if name not in sklearn_types:
                continue
            try:
                logger.info(f"CV for {name} ({self._cv_folds} folds)...")
                cv_results[name] = self.evaluator.cross_validate(
                    model.model, X_arr, y_arr, cv=self._cv_folds
                )
            except Exception as e:
                logger.error(f"CV failed for {name}: {e}")

        self.cv_results = cv_results
        return cv_results

    # ── Step 5: evaluate ──────────────────────────────────────────────────────
    def evaluate_models(self) -> Dict[str, Any]:
        logger.info("STEP 5: EVALUATION")
        results: Dict[str, Any] = {}
        for name, model in self.models.items():
            try:
                res = self.evaluator.evaluate_model(model)
                results[name] = res
                m = res['metrics']
                logger.info(
                    f"{name} — acc={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}  "
                    f"roc_auc={m.get('roc_auc', 'N/A')}  pr_auc={m.get('pr_auc', 'N/A')}"
                )
            except Exception as e:
                logger.error(f"Evaluation failed for {name}: {e}")
        self.model_results = results
        return results

    # ── Step 6: save ──────────────────────────────────────────────────────────
    def save_models(self) -> None:
        logger.info("STEP 6: SAVING MODELS")
        path_map = {
            'logistic_regression': str(LOGISTIC_REGRESSION_PATH),
            'random_forest': str(RANDOM_FOREST_PATH),
            'xgboost': str(XGBOOST_PATH),
            'mlp': str(MLP_PATH),
        }
        for name, model in self.models.items():
            try:
                model.save_model(path_map.get(name, str(OUTPUT_DIR / f"{name}_model.pkl")))
            except Exception as e:
                logger.error(f"Save failed for {name}: {e}")

    # ── Step 7: visualise ─────────────────────────────────────────────────────
    def create_visualizations(self, save: bool = True) -> None:
        logger.info("STEP 7: VISUALIZATION")
        if not self.model_results:
            logger.warning("No results to visualise")
            return
        try:
            self.visualizer.create_summary_plots(self.model_results, save=save)
            for name, res in self.model_results.items():
                cm = res['metrics']['confusion_matrix']
                self.visualizer.plot_confusion_matrix(cm, name, save=save)
            for name, model in self.models.items():
                if hasattr(model, 'get_feature_importance'):
                    self.visualizer.plot_feature_importance(model, save=save)
        except Exception as e:
            logger.error(f"Visualisation failed: {e}")

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def run_complete_pipeline(self, model_types: List[str] = None,
                              tune: bool = True,
                              save_models: bool = True,
                              create_plots: bool = True) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info(f"STARTING PIPELINE  test_mode={self.test_mode}  tune={tune}")
        logger.info("=" * 60)

        X, y_continuous, y_binary, feature_info = self.load_and_prepare_data()
        self.create_models(X.shape[1], model_types)
        self.train_models(X, y_binary, tune=tune)
        cv_results = self.run_cross_validation(X, y_binary)
        model_results = self.evaluate_models()

        if save_models:
            self.save_models()
        if create_plots:
            self.create_visualizations()

        best_name, best_res = self.evaluator.get_best_model(model_results)

        logger.info("=" * 60)
        logger.info(f"DONE — best model: {best_name}  "
                    f"f1_macro={best_res['metrics']['f1_macro']:.4f}")
        logger.info("=" * 60)

        return {
            'feature_info': feature_info,
            'models': self.models,
            'model_results': model_results,
            'cv_results': cv_results,
            'best_model': {'name': best_name, 'results': best_res},
            'data_shape': X.shape,
            'target_distribution': y_binary.value_counts().to_dict(),
        }


def main():
    pipeline = ModelTrainingPipeline(test_mode=TEST_MODE)

    results = pipeline.run_complete_pipeline(
        model_types=['logistic_regression', 'random_forest', 'xgboost'],
        tune=True,
        save_models=True,
        create_plots=True,
    )

    print("\n=== Pipeline complete ===")
    print(f"Best model : {results['best_model']['name']}")
    print(f"F1 (macro) : {results['best_model']['results']['metrics']['f1_macro']:.4f}")
    print(f"ROC-AUC    : {results['best_model']['results']['metrics'].get('roc_auc', 'N/A')}")
    print(f"PR-AUC     : {results['best_model']['results']['metrics'].get('pr_auc', 'N/A')}")

    if results['cv_results']:
        print("\n=== Cross-validation (mean ± std) ===")
        for model_name, cv in results['cv_results'].items():
            parts = [f"{k}: {v['mean']:.4f}±{v['std']:.4f}"
                     for k, v in cv.items() if v['mean'] is not None]
            print(f"  {model_name}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
