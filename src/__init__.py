"""
Financial Sentiment Analysis Package
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@example.com"
__description__ = "Financial sentiment analysis using advanced NLP models"

# Import main modules
from .data.news_data import FinancialNewsDataGenerator, FinancialNewsDataLoader
from .models.sentiment_models import VADERSentimentAnalyzer, FinBERTModel, TransformerSentimentModel
from .models.evaluation import SentimentEvaluator, ModelComparison
from .models.explainability import SentimentExplainer
from .backtest.trading_strategy import SentimentTradingStrategy, MarketDataDownloader
from .utils.core import set_random_seeds, load_config, get_device

__all__ = [
    "FinancialNewsDataGenerator",
    "FinancialNewsDataLoader", 
    "VADERSentimentAnalyzer",
    "FinBERTModel",
    "TransformerSentimentModel",
    "SentimentEvaluator",
    "ModelComparison",
    "SentimentExplainer",
    "SentimentTradingStrategy",
    "MarketDataDownloader",
    "set_random_seeds",
    "load_config",
    "get_device"
]
