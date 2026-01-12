"""
Comprehensive evaluation metrics for financial sentiment analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SentimentEvaluator:
    """Comprehensive evaluator for sentiment analysis models."""
    
    def __init__(self, class_names: List[str] = None):
        """Initialize evaluator."""
        self.class_names = class_names or ['negative', 'neutral', 'positive']
        self.label_map = {name: i for i, name in enumerate(self.class_names)}
        
    def evaluate_classification(
        self, 
        y_true: List[str], 
        y_pred: List[str], 
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Evaluate classification performance."""
        logger.info("Computing classification metrics")
        
        # Convert to numeric labels
        y_true_num = [self.label_map[label] for label in y_true]
        y_pred_num = [self.label_map[label] for label in y_pred]
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true_num, y_pred_num)
        metrics['precision_macro'] = precision_score(y_true_num, y_pred_num, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true_num, y_pred_num, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true_num, y_pred_num, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true_num, y_pred_num, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true_num, y_pred_num, average=None, zero_division=0)
        recall_per_class = recall_score(y_true_num, y_pred_num, average=None, zero_division=0)
        f1_per_class = f1_score(y_true_num, y_pred_num, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # Confusion matrix
        cm = confusion_matrix(y_true_num, y_pred_num)
        metrics['confusion_matrix'] = cm
        
        # ROC AUC (if probabilities available)
        if y_proba is not None:
            try:
                # One-vs-rest ROC AUC
                metrics['roc_auc_ovr'] = roc_auc_score(y_true_num, y_proba, multi_class='ovr', average='macro')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true_num, y_proba, multi_class='ovo', average='macro')
                
                # Per-class ROC AUC
                for i, class_name in enumerate(self.class_names):
                    y_binary = (np.array(y_true_num) == i).astype(int)
                    if len(np.unique(y_binary)) > 1:  # Check if class exists
                        metrics[f'roc_auc_{class_name}'] = roc_auc_score(y_binary, y_proba[:, i])
                
                # Average Precision (PR AUC)
                metrics['pr_auc'] = average_precision_score(y_true_num, y_proba, average='macro')
                
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
        
        return metrics
    
    def evaluate_calibration(
        self, 
        y_true: List[str], 
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """Evaluate calibration of predicted probabilities."""
        logger.info("Computing calibration metrics")
        
        y_true_num = [self.label_map[label] for label in y_true]
        
        calibration_metrics = {}
        
        # Expected Calibration Error (ECE)
        ece = self._compute_ece(y_true_num, y_proba, n_bins)
        calibration_metrics['ece'] = ece
        
        # Maximum Calibration Error (MCE)
        mce = self._compute_mce(y_true_num, y_proba, n_bins)
        calibration_metrics['mce'] = mce
        
        # Reliability diagrams for each class
        reliability_diagrams = {}
        for i, class_name in enumerate(self.class_names):
            y_binary = (np.array(y_true_num) == i).astype(int)
            if len(np.unique(y_binary)) > 1:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_binary, y_proba[:, i], n_bins=n_bins
                )
                reliability_diagrams[class_name] = {
                    'fraction_of_positives': fraction_of_positives,
                    'mean_predicted_value': mean_predicted_value
                }
        
        calibration_metrics['reliability_diagrams'] = reliability_diagrams
        
        return calibration_metrics
    
    def _compute_ece(self, y_true: List[int], y_proba: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba.max(axis=1) > bin_lower) & (y_proba.max(axis=1) <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (y_true == np.argmax(y_proba, axis=1))[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].max(axis=1).mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _compute_mce(self, y_true: List[int], y_proba: np.ndarray, n_bins: int = 10) -> float:
        """Compute Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba.max(axis=1) > bin_lower) & (y_proba.max(axis=1) <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (y_true == np.argmax(y_proba, axis=1))[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].max(axis=1).mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def create_evaluation_report(
        self, 
        y_true: List[str], 
        y_pred: List[str], 
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """Create comprehensive evaluation report."""
        logger.info(f"Creating evaluation report for {model_name}")
        
        report = {
            'model_name': model_name,
            'classification_metrics': self.evaluate_classification(y_true, y_pred, y_proba)
        }
        
        if y_proba is not None:
            report['calibration_metrics'] = self.evaluate_calibration(y_true, y_proba)
        
        # Classification report
        report['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        
        return report
    
    def plot_confusion_matrix(
        self, 
        cm: np.ndarray, 
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_curves(
        self, 
        reliability_diagrams: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """Plot calibration curves."""
        fig, axes = plt.subplots(1, len(reliability_diagrams), figsize=(5*len(reliability_diagrams), 5))
        if len(reliability_diagrams) == 1:
            axes = [axes]
        
        for i, (class_name, data) in enumerate(reliability_diagrams.items()):
            axes[i].plot(
                data['mean_predicted_value'], 
                data['fraction_of_positives'], 
                'o-', 
                label=f'{class_name}'
            )
            axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[i].set_xlabel('Mean Predicted Probability')
            axes[i].set_ylabel('Fraction of Positives')
            axes[i].set_title(f'Calibration Curve - {class_name.title()}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(
        self, 
        y_true: List[str], 
        y_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot ROC curves for each class."""
        plt.figure(figsize=(10, 8))
        
        y_true_num = [self.label_map[label] for label in y_true]
        
        for i, class_name in enumerate(self.class_names):
            y_binary = (np.array(y_true_num) == i).astype(int)
            if len(np.unique(y_binary)) > 1:
                fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
                auc = roc_auc_score(y_binary, y_proba[:, i])
                plt.plot(fpr, tpr, label=f'{class_name.title()} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ModelComparison:
    """Compare multiple sentiment analysis models."""
    
    def __init__(self, evaluator: SentimentEvaluator):
        """Initialize model comparison."""
        self.evaluator = evaluator
        self.results = {}
    
    def add_model_results(
        self, 
        model_name: str, 
        y_true: List[str], 
        y_pred: List[str], 
        y_proba: Optional[np.ndarray] = None
    ) -> None:
        """Add model results for comparison."""
        report = self.evaluator.create_evaluation_report(y_true, y_pred, y_proba, model_name)
        self.results[model_name] = report
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create comparison table of all models."""
        comparison_data = []
        
        for model_name, results in self.results.items():
            metrics = results['classification_metrics']
            row = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'F1 Macro': metrics['f1_macro'],
                'F1 Weighted': metrics['f1_weighted'],
                'Precision Macro': metrics['precision_macro'],
                'Recall Macro': metrics['recall_macro']
            }
            
            if 'roc_auc_ovr' in metrics:
                row['ROC AUC (OVR)'] = metrics['roc_auc_ovr']
                row['PR AUC'] = metrics['pr_auc']
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data).round(4)
    
    def plot_model_comparison(self, metric: str = 'f1_macro', save_path: Optional[str] = None) -> None:
        """Plot comparison of models for a specific metric."""
        models = list(self.results.keys())
        values = [self.results[model]['classification_metrics'][metric] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, values, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
        plt.xlabel('Models')
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
