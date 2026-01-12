"""
Data pipeline for financial sentiment analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class FinancialNewsDataGenerator:
    """Generate synthetic financial news data for research purposes."""
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Financial news templates
        self.positive_templates = [
            "{} reports strong quarterly earnings, beating analyst expectations",
            "{} stock surges on positive market outlook and growth prospects", 
            "{} announces breakthrough innovation in {} technology",
            "{} receives regulatory approval for new product launch",
            "{} partnership with {} expected to drive significant revenue growth",
            "{} stock hits new all-time high amid bullish market sentiment",
            "{} reports record revenue growth in latest quarterly results",
            "{} expansion into {} market shows promising early results"
        ]
        
        self.negative_templates = [
            "{} faces regulatory challenges and potential fines",
            "{} stock drops sharply following disappointing earnings report",
            "{} announces major restructuring and workforce reduction",
            "{} faces supply chain disruptions affecting production",
            "{} stock volatility increases amid market uncertainty",
            "{} reports lower than expected revenue in latest quarter",
            "{} faces increased competition in {} market segment",
            "{} stock declines on concerns over {} regulatory changes"
        ]
        
        self.neutral_templates = [
            "{} maintains stable performance in current market conditions",
            "{} announces routine quarterly earnings call scheduled",
            "{} stock shows minimal movement in today's trading session",
            "{} provides update on ongoing {} development project",
            "{} reports standard quarterly financial results",
            "{} maintains current dividend policy for shareholders",
            "{} announces standard corporate governance updates",
            "{} provides routine market update and outlook"
        ]
        
        # Company names and sectors
        self.companies = [
            "Apple", "Microsoft", "Google", "Amazon", "Tesla", "Meta", "Netflix",
            "NVIDIA", "AMD", "Intel", "IBM", "Oracle", "Salesforce", "Adobe",
            "PayPal", "Square", "Zoom", "Slack", "Uber", "Lyft", "Airbnb"
        ]
        
        self.sectors = [
            "technology", "healthcare", "finance", "energy", "consumer goods",
            "automotive", "telecommunications", "retail", "manufacturing"
        ]
        
        self.technologies = [
            "AI", "blockchain", "cloud computing", "machine learning", "IoT",
            "quantum computing", "5G", "cybersecurity", "data analytics"
        ]

    def generate_news_data(self, n_samples: int = 1000, start_date: str = "2023-01-01") -> pd.DataFrame:
        """Generate synthetic financial news data."""
        logger.info(f"Generating {n_samples} synthetic news samples")
        
        headlines = []
        sentiments = []
        dates = []
        sources = []
        companies = []
        
        # Generate date range
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        date_range = [start_dt + timedelta(days=i) for i in range(n_samples)]
        
        # News sources
        news_sources = [
            "Reuters", "Bloomberg", "CNBC", "Wall Street Journal", "Financial Times",
            "MarketWatch", "Yahoo Finance", "Seeking Alpha", "Investor's Business Daily"
        ]
        
        # Sentiment distribution (realistic for financial news)
        sentiment_probs = [0.3, 0.4, 0.3]  # negative, neutral, positive
        
        for i in range(n_samples):
            sentiment_idx = np.random.choice(3, p=sentiment_probs)
            sentiment_labels = ['negative', 'neutral', 'positive']
            sentiment = sentiment_labels[sentiment_idx]
            
            # Generate headline based on sentiment
            if sentiment == 'positive':
                template = random.choice(self.positive_templates)
            elif sentiment == 'negative':
                template = random.choice(self.negative_templates)
            else:
                template = random.choice(self.neutral_templates)
            
            # Fill template with random values
            company = random.choice(self.companies)
            sector = random.choice(self.sectors)
            tech = random.choice(self.technologies)
            
            headline = template.format(company, sector, tech)
            
            headlines.append(headline)
            sentiments.append(sentiment)
            dates.append(date_range[i])
            sources.append(random.choice(news_sources))
            companies.append(company)
        
        # Create DataFrame
        df = pd.DataFrame({
            'headline': headlines,
            'sentiment': sentiments,
            'date': dates,
            'source': sources,
            'company': companies,
            'text_length': [len(h) for h in headlines],
            'word_count': [len(h.split()) for h in headlines]
        })
        
        # Add some noise to make it more realistic
        df = self._add_realistic_noise(df)
        
        logger.info(f"Generated data shape: {df.shape}")
        logger.info(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
        
        return df
    
    def _add_realistic_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic noise to the synthetic data."""
        # Add some misclassified samples (realistic noise)
        n_noise = int(len(df) * 0.05)  # 5% noise
        noise_indices = np.random.choice(len(df), n_noise, replace=False)
        
        for idx in noise_indices:
            current_sentiment = df.loc[idx, 'sentiment']
            # Flip sentiment with some probability
            if np.random.random() < 0.3:
                if current_sentiment == 'positive':
                    df.loc[idx, 'sentiment'] = 'negative'
                elif current_sentiment == 'negative':
                    df.loc[idx, 'sentiment'] = 'positive'
        
        return df


class FinancialNewsDataLoader:
    """Load and preprocess financial news data."""
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize the data loader."""
        self.data_path = Path(data_path) if data_path else Path("data/raw")
        self.processed_path = Path("data/processed")
        
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load financial news data from file or generate synthetic data."""
        if file_path and Path(file_path).exists():
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
        else:
            logger.info("No data file found, generating synthetic data")
            generator = FinancialNewsDataGenerator()
            df = generator.generate_news_data()
            
            # Save synthetic data
            self.processed_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.processed_path / "synthetic_news_data.csv", index=False)
            logger.info(f"Saved synthetic data to {self.processed_path / 'synthetic_news_data.csv'}")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the financial news data."""
        logger.info("Preprocessing financial news data")
        
        # Clean text
        df['headline_clean'] = df['headline'].str.lower()
        df['headline_clean'] = df['headline_clean'].str.replace(r'[^\w\s]', '', regex=True)
        df['headline_clean'] = df['headline_clean'].str.replace(r'\s+', ' ', regex=True)
        df['headline_clean'] = df['headline_clean'].str.strip()
        
        # Add text features
        df['char_count'] = df['headline'].str.len()
        df['word_count'] = df['headline'].str.split().str.len()
        df['sentence_count'] = df['headline'].str.count(r'[.!?]+') + 1
        
        # Convert sentiment to numeric labels
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        df['sentiment_label'] = df['sentiment'].map(sentiment_map)
        
        # Add date features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])
        
        logger.info(f"Preprocessed data shape: {df.shape}")
        return df
    
    def get_train_val_test_split(
        self, 
        df: pd.DataFrame, 
        date_col: str = 'date',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets based on time."""
        logger.info("Creating time-based train/validation/test split")
        
        # Sort by date to ensure proper time-based split
        df_sorted = df.sort_values(date_col)
        n_samples = len(df_sorted)
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
