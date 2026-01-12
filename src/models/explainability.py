"""
Model explainability and interpretability for financial sentiment analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import shap
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SentimentExplainer:
    """Comprehensive explainability for sentiment analysis models."""
    
    def __init__(self, model, tokenizer=None, device=None):
        """Initialize explainer."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize SHAP explainer
        self.shap_explainer = None
        self.lime_explainer = LimeTextExplainer(class_names=['negative', 'neutral', 'positive'])
        
    def explain_with_shap(
        self, 
        texts: List[str], 
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """Generate SHAP explanations for sentiment predictions."""
        logger.info("Generating SHAP explanations")
        
        if self.shap_explainer is None:
            self._initialize_shap_explainer(texts[:max_samples])
        
        # Generate explanations
        shap_values = self.shap_explainer(texts[:max_samples])
        
        return {
            'shap_values': shap_values,
            'base_values': shap_values.base_values,
            'data': shap_values.data,
            'values': shap_values.values
        }
    
    def _initialize_shap_explainer(self, sample_texts: List[str]):
        """Initialize SHAP explainer with sample texts."""
        try:
            # Create a wrapper function for the model
            def model_predict(texts):
                if isinstance(texts, str):
                    texts = [texts]
                
                predictions = []
                for text in texts:
                    if hasattr(self.model, 'predict_proba'):
                        probs = self.model.predict_proba([text])[0]
                    else:
                        _, probs = self.model.predict([text])
                        probs = probs[0] if isinstance(probs, list) else probs
                    
                    predictions.append(probs)
                
                return np.array(predictions)
            
            # Initialize SHAP explainer
            self.shap_explainer = shap.Explainer(model_predict, sample_texts)
            
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.shap_explainer = None
    
    def explain_with_lime(
        self, 
        text: str, 
        num_features: int = 10
    ) -> Dict[str, Any]:
        """Generate LIME explanation for a single text."""
        logger.info(f"Generating LIME explanation for text: {text[:50]}...")
        
        def predict_proba(texts):
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(texts)
            else:
                _, probs = self.model.predict(texts)
                return np.array(probs)
        
        # Generate explanation
        explanation = self.lime_explainer.explain_instance(
            text, 
            predict_proba, 
            num_features=num_features
        )
        
        return {
            'explanation': explanation,
            'feature_importance': explanation.as_list(),
            'prediction': explanation.predict_proba
        }
    
    def visualize_shap_explanations(
        self, 
        shap_results: Dict[str, Any],
        texts: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """Visualize SHAP explanations."""
        if self.shap_explainer is None:
            logger.warning("SHAP explainer not initialized")
            return
        
        # Create SHAP plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Waterfall plot for first text
        if len(texts) > 0:
            shap.waterfall_plot(shap_results['shap_values'][0], show=False)
            axes[0, 0].set_title('SHAP Waterfall Plot')
        
        # Summary plot
        shap.summary_plot(shap_results['shap_values'], texts, show=False)
        axes[0, 1].set_title('SHAP Summary Plot')
        
        # Bar plot
        shap.plots.bar(shap_results['shap_values'], show=False)
        axes[1, 0].set_title('SHAP Feature Importance')
        
        # Heatmap
        shap.plots.heatmap(shap_results['shap_values'], show=False)
        axes[1, 1].set_title('SHAP Heatmap')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_lime_explanation(
        self, 
        lime_results: Dict[str, Any],
        text: str,
        save_path: Optional[str] = None
    ) -> None:
        """Visualize LIME explanation."""
        explanation = lime_results['explanation']
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature importance bar plot
        features, scores = zip(*lime_results['feature_importance'])
        colors = ['red' if score < 0 else 'green' for score in scores]
        
        axes[0].barh(features, scores, color=colors, alpha=0.7)
        axes[0].set_title('LIME Feature Importance')
        axes[0].set_xlabel('Importance Score')
        axes[0].grid(True, alpha=0.3)
        
        # Prediction probabilities
        classes = ['negative', 'neutral', 'positive']
        probs = lime_results['prediction'][0]
        
        axes[1].bar(classes, probs, color=['red', 'gray', 'green'], alpha=0.7)
        axes[1].set_title('Prediction Probabilities')
        axes[1].set_ylabel('Probability')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_attention_patterns(
        self, 
        texts: List[str],
        model_name: str = "ProsusAI/finbert"
    ) -> Dict[str, Any]:
        """Analyze attention patterns for transformer models."""
        logger.info("Analyzing attention patterns")
        
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        attention_data = []
        
        for text in texts[:5]:  # Limit to 5 texts for visualization
            # Tokenize text
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512
            )
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Extract attention weights (if available)
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    attention = outputs.attentions[-1]  # Last layer attention
                    attention = attention.squeeze(0)  # Remove batch dimension
                    
                    # Average across heads
                    attention_avg = attention.mean(dim=0)
                    
                    # Get tokens
                    tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                    
                    attention_data.append({
                        'text': text,
                        'tokens': tokens,
                        'attention': attention_avg.cpu().numpy()
                    })
        
        return {'attention_data': attention_data}
    
    def visualize_attention(
        self, 
        attention_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """Visualize attention patterns."""
        attention_data = attention_results['attention_data']
        
        if not attention_data:
            logger.warning("No attention data to visualize")
            return
        
        fig, axes = plt.subplots(len(attention_data), 1, figsize=(15, 4*len(attention_data)))
        if len(attention_data) == 1:
            axes = [axes]
        
        for i, data in enumerate(attention_data):
            tokens = data['tokens']
            attention = data['attention']
            
            # Create heatmap
            im = axes[i].imshow(attention, cmap='Blues', aspect='auto')
            
            # Set labels
            axes[i].set_xticks(range(len(tokens)))
            axes[i].set_xticklabels(tokens, rotation=45, ha='right')
            axes[i].set_yticks(range(len(tokens)))
            axes[i].set_yticklabels(tokens)
            
            axes[i].set_title(f'Attention Pattern - Text {i+1}')
            axes[i].set_xlabel('Key Tokens')
            axes[i].set_ylabel('Query Tokens')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_explanation_report(
        self, 
        text: str,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """Create comprehensive explanation report for a single text."""
        logger.info(f"Creating explanation report for: {text[:50]}...")
        
        report = {
            'text': text,
            'model_name': model_name,
            'timestamp': pd.Timestamp.now()
        }
        
        # Get prediction
        if hasattr(self.model, 'predict'):
            pred, probs = self.model.predict([text])
            report['prediction'] = pred[0]
            report['probabilities'] = probs[0] if isinstance(probs, list) else probs[0]
        
        # LIME explanation
        try:
            lime_results = self.explain_with_lime(text)
            report['lime_explanation'] = lime_results['feature_importance']
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
            report['lime_explanation'] = None
        
        # SHAP explanation (if available)
        try:
            shap_results = self.explain_with_shap([text])
            if shap_results['shap_values'] is not None:
                report['shap_explanation'] = shap_results['values'][0].tolist()
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            report['shap_explanation'] = None
        
        return report
    
    def analyze_model_bias(
        self, 
        test_data: pd.DataFrame,
        sensitive_attributes: List[str] = None
    ) -> Dict[str, Any]:
        """Analyze model bias across different groups."""
        logger.info("Analyzing model bias")
        
        if sensitive_attributes is None:
            sensitive_attributes = ['company', 'source']
        
        bias_results = {}
        
        for attr in sensitive_attributes:
            if attr not in test_data.columns:
                continue
            
            # Group by sensitive attribute
            groups = test_data.groupby(attr)
            
            group_metrics = {}
            for group_name, group_data in groups:
                if len(group_data) < 10:  # Skip small groups
                    continue
                
                # Get predictions for this group
                texts = group_data['headline'].tolist()
                true_labels = group_data['sentiment'].tolist()
                
                if hasattr(self.model, 'predict'):
                    pred_labels, _ = self.model.predict(texts)
                else:
                    pred_labels = [self.model.predict([text])[0] for text in texts]
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, f1_score
                
                group_metrics[group_name] = {
                    'accuracy': accuracy_score(true_labels, pred_labels),
                    'f1_score': f1_score(true_labels, pred_labels, average='weighted'),
                    'sample_size': len(group_data)
                }
            
            bias_results[attr] = group_metrics
        
        return bias_results
    
    def visualize_bias_analysis(
        self, 
        bias_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """Visualize bias analysis results."""
        n_attrs = len(bias_results)
        fig, axes = plt.subplots(1, n_attrs, figsize=(6*n_attrs, 6))
        if n_attrs == 1:
            axes = [axes]
        
        for i, (attr, group_metrics) in enumerate(bias_results.items()):
            groups = list(group_metrics.keys())
            accuracies = [group_metrics[g]['accuracy'] for g in groups]
            f1_scores = [group_metrics[g]['f1_score'] for g in groups]
            
            x = np.arange(len(groups))
            width = 0.35
            
            axes[i].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.7)
            axes[i].bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.7)
            
            axes[i].set_xlabel(attr.title())
            axes[i].set_ylabel('Score')
            axes[i].set_title(f'Model Performance by {attr.title()}')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(groups, rotation=45)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
