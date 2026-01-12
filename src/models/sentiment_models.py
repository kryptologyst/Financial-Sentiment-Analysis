"""
Advanced NLP models for financial sentiment analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VADERSentimentAnalyzer:
    """VADER sentiment analyzer for baseline comparison."""
    
    def __init__(self, thresholds: Dict[str, float] = None):
        """Initialize VADER analyzer."""
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        
        self.sia = SentimentIntensityAnalyzer()
        self.thresholds = thresholds or {'positive': 0.1, 'negative': -0.1}
        
    def predict(self, texts: List[str]) -> Tuple[List[float], List[str]]:
        """Predict sentiment scores and classes."""
        scores = []
        classes = []
        
        for text in texts:
            score = self.sia.polarity_scores(text)['compound']
            scores.append(score)
            
            if score > self.thresholds['positive']:
                classes.append('positive')
            elif score < self.thresholds['negative']:
                classes.append('negative')
            else:
                classes.append('neutral')
        
        return scores, classes
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict sentiment probabilities."""
        probabilities = []
        
        for text in texts:
            scores = self.sia.polarity_scores(text)
            # Convert to probabilities (normalize)
            prob_neg = max(0, -scores['compound'])
            prob_pos = max(0, scores['compound'])
            prob_neu = 1 - prob_neg - prob_pos
            
            probabilities.append([prob_neg, prob_neu, prob_pos])
        
        return np.array(probabilities)


class FinBERTModel:
    """FinBERT model for financial sentiment analysis."""
    
    def __init__(
        self, 
        model_name: str = "ProsusAI/finbert",
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """Initialize FinBERT model."""
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=3  # negative, neutral, positive
        ).to(self.device)
        
        logger.info(f"Loaded FinBERT model: {model_name}")
    
    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize texts for FinBERT."""
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def predict(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        """Predict sentiment labels and probabilities."""
        self.model.eval()
        
        # Tokenize texts
        inputs = self.tokenize_texts(texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions.cpu().numpy().tolist(), probabilities.cpu().numpy()
    
    def train(
        self, 
        train_texts: List[str], 
        train_labels: List[int],
        val_texts: List[str], 
        val_labels: List[int],
        epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        output_dir: str = "assets/models/finbert"
    ) -> None:
        """Fine-tune FinBERT on financial sentiment data."""
        logger.info("Starting FinBERT fine-tuning")
        
        # Create datasets
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            learning_rate=learning_rate,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"FinBERT training completed. Model saved to {output_dir}")
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        precision, recall, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'f1_macro': f1_macro
        }


class SentimentDataset(torch.utils.data.Dataset):
    """Dataset class for sentiment analysis."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        """Initialize dataset."""
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TransformerSentimentModel:
    """Generic transformer-based sentiment analysis model."""
    
    def __init__(
        self, 
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """Initialize transformer model."""
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        
        logger.info(f"Loaded transformer model: {model_name}")
    
    def predict(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """Predict sentiment labels and probabilities."""
        self.model.eval()
        
        predictions = []
        probabilities = []
        
        for text in texts:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                pred = torch.argmax(logits, dim=-1)
            
            # Map predictions to sentiment labels
            label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            predictions.append(label_map[pred.item()])
            probabilities.append(probs.cpu().numpy().flatten())
        
        return predictions, probabilities


class EnsembleSentimentModel:
    """Ensemble model combining multiple sentiment analyzers."""
    
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        """Initialize ensemble model."""
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        logger.info(f"Initialized ensemble with {len(models)} models")
    
    def predict(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """Predict using ensemble of models."""
        all_predictions = []
        all_probabilities = []
        
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(texts)
                preds = np.argmax(probs, axis=1)
            else:
                preds, probs = model.predict(texts)
                if isinstance(probs, list):
                    probs = np.array(probs)
            
            all_predictions.append(preds)
            all_probabilities.append(probs)
        
        # Weighted average of probabilities
        ensemble_probs = np.zeros_like(all_probabilities[0])
        for probs, weight in zip(all_probabilities, self.weights):
            ensemble_probs += weight * probs
        
        # Final predictions
        final_predictions = np.argmax(ensemble_probs, axis=1)
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        final_labels = [label_map[pred] for pred in final_predictions]
        
        return final_labels, ensemble_probs.tolist()
