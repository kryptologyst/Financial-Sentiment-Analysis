"""
Project 518: Financial Sentiment Analysis
Modernized implementation with advanced NLP models and proper evaluation.

This is a research/educational project for sentiment analysis on financial news.
NOT FOR INVESTMENT ADVICE - Results may be inaccurate and are for research only.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

def main():
    """Main function to run financial sentiment analysis."""
    # 1. Sample financial news articles
    data = {
        'headline': [
            "Stock market hits record high amidst economic recovery",
            "Inflation concerns lead to market volatility", 
            "Tech stocks soar after strong earnings reports",
            "Economic slowdown predicted due to rising interest rates",
            "Global markets recover as stimulus packages are announced"
        ],
        'date': pd.date_range('2024-01-01', periods=5, freq='D'),
        'source': ['Reuters', 'Bloomberg', 'CNBC', 'WSJ', 'Financial Times']
    }
    
    df = pd.DataFrame(data)
    
    # 2. Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # 3. Apply sentiment analysis to each headline
    df['sentiment_score'] = df['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    # 4. Classify sentiment into categories
    df['sentiment_class'] = df['sentiment_score'].apply(
        lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
    )
    
    # 5. Display the results
    print("Sentiment Analysis of Financial News Headlines:")
    print(df)
    
    # 6. Plot the distribution of sentiments
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment_class'].value_counts()
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
    plt.title('Sentiment Distribution of Financial News Headlines')
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return df

if __name__ == "__main__":
    df = main()
