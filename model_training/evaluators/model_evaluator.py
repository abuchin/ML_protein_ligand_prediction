"""
Model evaluation module for computing metrics and generating reports.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class for evaluating model performance."""
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.metrics = {}
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Compute various evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary of computed metrics
        """
        logger.info("Computing evaluation metrics...")
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # ROC AUC (if probabilities are provided)
        if y_proba is not None and len(y_proba.shape) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1], average='weighted')
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
                metrics['roc_auc'] = None
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None)
        
        logger.info(f"Computed metrics: {list(metrics.keys())}")
        
        return metrics
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     target_names: List[str] = None) -> str:
        """
        Generate a detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names for target classes
            
        Returns:
            Classification report string
        """
        if target_names is None:
            target_names = ['Not Bound', 'Bound']
        
        report = classification_report(y_true, y_pred, target_names=target_names)
        return report
    
    def evaluate_model(self, model, X_test: pd.DataFrame = None, y_test: np.ndarray = None) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model instance
            X_test: Test features (if not already split in model)
            y_test: Test labels (if not already split in model)
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {model.model_name}...")
        
        # Get predictions
        if hasattr(model, 'y_pred') and model.y_pred is not None:
            y_pred = model.y_pred
            y_true = model.y_test
        else:
            if X_test is None or y_test is None:
                raise ValueError("Either model must have predictions or X_test/y_test must be provided")
            y_pred = model.predict()
            y_true = y_test
        
        # Get probabilities if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba()
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")
        
        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred, y_proba)
        
        # Generate classification report
        report = self.generate_classification_report(y_true, y_pred)
        
        # Get model information
        model_info = model.get_model_info()
        
        # Compile results
        results = {
            'model_name': model.model_name,
            'metrics': metrics,
            'classification_report': report,
            'model_info': model_info,
            'predictions': y_pred,
            'true_labels': y_true,
            'probabilities': y_proba
        }
        
        logger.info(f"Evaluation completed for {model.model_name}")
        
        return results
    
    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models based on their metrics.
        
        Args:
            model_results: Dictionary mapping model names to evaluation results
            
        Returns:
            DataFrame with comparison metrics
        """
        logger.info("Comparing models...")
        
        comparison_data = []
        
        for model_name, results in model_results.items():
            metrics = results['metrics']
            
            row = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'ROC-AUC': metrics.get('roc_auc', None),
                'Training_Time': results['model_info'].get('training_time', None),
                'Prediction_Time': results['model_info'].get('prediction_time', None)
            }
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by F1-score (or accuracy if F1 is not available)
        if 'F1-Score' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        else:
            comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        logger.info("Model comparison completed")
        
        return comparison_df
    
    def get_best_model(self, model_results: Dict[str, Dict], metric: str = 'f1') -> Tuple[str, Dict]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            model_results: Dictionary mapping model names to evaluation results
            metric: Metric to use for comparison ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')
            
        Returns:
            Tuple of (best_model_name, best_model_results)
        """
        best_model = None
        best_score = -1
        
        for model_name, results in model_results.items():
            score = results['metrics'].get(metric, -1)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        if best_model is None:
            raise ValueError(f"No models found with metric '{metric}'")
        
        return best_model, model_results[best_model]
    
    def save_evaluation_results(self, results: Dict[str, Any], filepath: str) -> None:
        """
        Save evaluation results to a file.
        
        Args:
            results: Evaluation results dictionary
            filepath: Path to save the results
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_results[key][k] = v.tolist()
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filepath}") 