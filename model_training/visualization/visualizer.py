"""
Visualization module for model training results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """Class for creating visualizations of model training results."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6), dpi: int = 300, 
                 cmap: str = 'Blues', save_dir: str = None):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
            dpi: DPI for saved figures
            cmap: Default colormap
            save_dir: Directory to save figures
        """
        self.figsize = figsize
        self.dpi = dpi
        self.cmap = cmap
        self.save_dir = Path(save_dir) if save_dir else None
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str = "Model",
                            labels: List[str] = None, save: bool = False) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
            labels: Class labels
            save: Whether to save the plot
        """
        if labels is None:
            labels = ['Not Bound', 'Bound']
        
        plt.figure(figsize=self.figsize)
        sns.heatmap(cm, annot=True, fmt="d", cmap=self.cmap,
                    xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        
        if save and self.save_dir:
            filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {self.save_dir / filename}")
        
        plt.show()
    
    def plot_metrics_comparison(self, model_results: Dict[str, Dict], 
                              metrics: List[str] = None, save: bool = False) -> None:
        """
        Plot comparison of metrics across models.
        
        Args:
            model_results: Dictionary mapping model names to evaluation results
            metrics: List of metrics to plot
            save: Whether to save the plot
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Extract data
        model_names = list(model_results.keys())
        metric_data = {metric: [] for metric in metrics}
        
        for model_name in model_names:
            for metric in metrics:
                value = model_results[model_name]['metrics'].get(metric, 0)
                metric_data[metric].append(value)
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                axes[i].bar(model_names, metric_data[metric])
                axes[i].set_title(f'{metric.title()} Comparison')
                axes[i].set_ylabel(metric.title())
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].set_ylim(0, 1)
                
                # Add baseline
                axes[i].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Baseline (0.5)')
                axes[i].legend()
        
        plt.tight_layout()
        
        if save and self.save_dir:
            filename = "metrics_comparison.png"
            plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Metrics comparison saved to {self.save_dir / filename}")
        
        plt.show()
    
    def plot_accuracy_comparison(self, model_results: Dict[str, Dict], save: bool = False) -> None:
        """
        Plot accuracy comparison across models.
        
        Args:
            model_results: Dictionary mapping model names to evaluation results
            save: Whether to save the plot
        """
        model_names = list(model_results.keys())
        accuracies = [model_results[name]['metrics']['accuracy'] for name in model_names]
        
        plt.figure(figsize=self.figsize)
        bars = plt.bar(model_names, accuracies, color=sns.color_palette("husl", len(model_names)))
        
        # Add baseline
        plt.axhline(y=0.5, color='r', linestyle='--', label='Baseline (0.5)')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.ylim(0, 1)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save and self.save_dir:
            filename = "accuracy_comparison.png"
            plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Accuracy comparison saved to {self.save_dir / filename}")
        
        plt.show()
    
    def plot_feature_importance(self, model, top_n: int = 20, save: bool = False) -> None:
        """
        Plot feature importance for models that support it.
        
        Args:
            model: Trained model with feature importance
            top_n: Number of top features to show
            save: Whether to save the plot
        """
        if not hasattr(model, 'get_feature_importance'):
            logger.warning(f"{model.model_name} does not support feature importance")
            return
        
        try:
            importance = model.get_feature_importance()
            
            # Get top features
            if len(importance) > top_n:
                top_indices = np.argsort(importance)[-top_n:]
                top_importance = importance[top_indices]
                feature_names = [f'Feature {i}' for i in top_indices]
            else:
                top_importance = importance
                feature_names = [f'Feature {i}' for i in range(len(importance))]
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(top_importance)), top_importance)
            plt.yticks(range(len(feature_names)), feature_names)
            plt.xlabel('Feature Importance')
            plt.title(f'Top {len(top_importance)} Feature Importances - {model.model_name}')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, top_importance)):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{val:.4f}', va='center')
            
            plt.tight_layout()
            
            if save and self.save_dir:
                filename = f"feature_importance_{model.model_name.lower().replace(' ', '_')}.png"
                plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Feature importance saved to {self.save_dir / filename}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance for {model.model_name}: {e}")
    
    def plot_training_time_comparison(self, model_results: Dict[str, Dict], save: bool = False) -> None:
        """
        Plot training time comparison across models.
        
        Args:
            model_results: Dictionary mapping model names to evaluation results
            save: Whether to save the plot
        """
        model_names = []
        training_times = []
        
        for model_name, results in model_results.items():
            training_time = results['model_info'].get('training_time')
            if training_time is not None:
                model_names.append(model_name)
                training_times.append(training_time)
        
        if not training_times:
            logger.warning("No training time data available")
            return
        
        plt.figure(figsize=self.figsize)
        bars = plt.bar(model_names, training_times, color=sns.color_palette("husl", len(model_names)))
        
        # Add value labels
        for bar, time_val in zip(bars, training_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time_val:.1f}s', ha='center', va='bottom')
        
        plt.xlabel('Model')
        plt.ylabel('Training Time (seconds)')
        plt.title('Model Training Time Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save and self.save_dir:
            filename = "training_time_comparison.png"
            plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Training time comparison saved to {self.save_dir / filename}")
        
        plt.show()
    
    def plot_dataset_size_comparison(self, results_by_size: Dict[str, Dict], 
                                   metric: str = 'accuracy', save: bool = False) -> None:
        """
        Plot model performance across different dataset sizes.
        
        Args:
            results_by_size: Dictionary mapping dataset sizes to model results
            metric: Metric to plot
            save: Whether to save the plot
        """
        dataset_sizes = list(results_by_size.keys())
        model_names = list(results_by_size[dataset_sizes[0]].keys())
        
        # Extract data
        data = []
        for size in dataset_sizes:
            for model_name in model_names:
                if model_name in results_by_size[size]:
                    value = results_by_size[size][model_name]['metrics'].get(metric, 0)
                    data.append([size, model_name, value])
        
        df = pd.DataFrame(data, columns=['Dataset Size', 'Model', metric.title()])
        
        plt.figure(figsize=self.figsize)
        sns.lineplot(data=df, x='Dataset Size', y=metric.title(), hue='Model', marker='o')
        plt.xlabel('Dataset Size')
        plt.ylabel(metric.title())
        plt.title(f'Model {metric.title()} vs. Training Data Size')
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save and self.save_dir:
            filename = f"dataset_size_comparison_{metric}.png"
            plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Dataset size comparison saved to {self.save_dir / filename}")
        
        plt.show()
    
    def create_summary_plots(self, model_results: Dict[str, Dict], save: bool = True) -> None:
        """
        Create a comprehensive set of summary plots.
        
        Args:
            model_results: Dictionary mapping model names to evaluation results
            save: Whether to save the plots
        """
        logger.info("Creating summary plots...")
        
        # Accuracy comparison
        self.plot_accuracy_comparison(model_results, save=save)
        
        # Metrics comparison
        self.plot_metrics_comparison(model_results, save=save)
        
        # Training time comparison
        self.plot_training_time_comparison(model_results, save=save)
        
        logger.info("Summary plots completed") 