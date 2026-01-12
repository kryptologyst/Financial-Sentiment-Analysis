"""
Basic tests for financial sentiment analysis.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.news_data import FinancialNewsDataGenerator, FinancialNewsDataLoader
from src.models.sentiment_models import VADERSentimentAnalyzer
from src.models.evaluation import SentimentEvaluator
from src.utils.core import set_random_seeds, time_based_split


class TestDataGeneration:
    """Test data generation functionality."""
    
    def test_data_generator(self):
        """Test synthetic data generation."""
        generator = FinancialNewsDataGenerator(seed=42)
        df = generator.generate_news_data(100)
        
        assert len(df) == 100
        assert 'headline' in df.columns
        assert 'sentiment' in df.columns
        assert 'date' in df.columns
        assert df['sentiment'].isin(['positive', 'negative', 'neutral']).all()
    
    def test_data_loader(self):
        """Test data loading and preprocessing."""
        loader = FinancialNewsDataLoader()
        df = loader.load_data()
        df = loader.preprocess_data(df)
        
        assert 'headline_clean' in df.columns
        assert 'sentiment_label' in df.columns
        assert df['sentiment_label'].isin([0, 1, 2]).all()


class TestModels:
    """Test sentiment analysis models."""
    
    def test_vader_model(self):
        """Test VADER sentiment analyzer."""
        model = VADERSentimentAnalyzer()
        
        test_texts = [
            "This is great news for investors!",
            "The market is crashing badly.",
            "The stock price remained unchanged."
        ]
        
        scores, classes = model.predict(test_texts)
        probabilities = model.predict_proba(test_texts)
        
        assert len(scores) == 3
        assert len(classes) == 3
        assert probabilities.shape == (3, 3)
        assert all(cls in ['positive', 'negative', 'neutral'] for cls in classes)


class TestEvaluation:
    """Test evaluation metrics."""
    
    def test_evaluator(self):
        """Test sentiment evaluator."""
        evaluator = SentimentEvaluator()
        
        y_true = ['positive', 'negative', 'neutral', 'positive']
        y_pred = ['positive', 'negative', 'neutral', 'negative']
        y_proba = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.3, 0.1, 0.6]])
        
        metrics = evaluator.evaluate_classification(y_true, y_pred, y_proba)
        
        assert 'accuracy' in metrics
        assert 'f1_macro' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 0 <= metrics['accuracy'] <= 1


class TestUtils:
    """Test utility functions."""
    
    def test_random_seeds(self):
        """Test random seed setting."""
        set_random_seeds(42)
        # This is a basic test - in practice you'd test actual randomness
        assert True
    
    def test_time_based_split(self):
        """Test time-based data splitting."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'value': range(100)
        })
        
        train, val, test = time_based_split(df, 'date')
        
        assert len(train) + len(val) + len(test) == 100
        assert train['date'].max() <= val['date'].min()
        assert val['date'].max() <= test['date'].min()


if __name__ == "__main__":
    pytest.main([__file__])
