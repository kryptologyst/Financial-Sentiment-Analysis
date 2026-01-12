# Financial Sentiment Analysis

A comprehensive implementation of financial sentiment analysis using advanced NLP models and proper evaluation methodologies.

## ⚠️ IMPORTANT DISCLAIMER

**This is a research and educational project only. NOT FOR INVESTMENT ADVICE.**

- Results may be inaccurate and are for research purposes only
- Past performance does not guarantee future results
- Always consult with qualified financial advisors before making investment decisions
- Backtests are hypothetical and may not reflect real trading conditions

## Overview

This project implements state-of-the-art sentiment analysis techniques specifically designed for financial news and text. It includes multiple models, comprehensive evaluation, backtesting capabilities, and model explainability features.

## Features

### Models
- **VADER**: Rule-based sentiment analyzer optimized for social media text
- **FinBERT**: BERT model fine-tuned on financial text (ProsusAI/finbert)
- **Transformer**: RoBERTa-based sentiment classification model
- **Ensemble**: Combines multiple models for improved performance

### Capabilities
- Single text sentiment analysis with confidence scores
- Batch processing of financial news articles
- Comprehensive model evaluation and comparison
- Sentiment-based trading strategy backtesting
- Model explainability with SHAP and LIME
- Interactive Streamlit demo application

### Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score (macro/weighted)
- **Probabilistic**: ROC AUC, PR AUC, Calibration Error
- **Trading**: Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown
- **Risk**: VaR, Expected Shortfall, Drawdown Analysis

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Financial-Sentiment-Analysis.git
cd Financial-Sentiment-Analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (for VADER):
```python
import nltk
nltk.download('vader_lexicon')
```

## Quick Start

### 1. Basic Usage

```python
from src.data.news_data import FinancialNewsDataGenerator
from src.models.sentiment_models import VADERSentimentAnalyzer

# Generate synthetic data
generator = FinancialNewsDataGenerator()
data = generator.generate_news_data(100)

# Initialize model
model = VADERSentimentAnalyzer()

# Predict sentiment
predictions, probabilities = model.predict(data['headline'].tolist())
```

### 2. Advanced Models

```python
from src.models.sentiment_models import FinBERTModel

# Initialize FinBERT
model = FinBERTModel()

# Predict sentiment
predictions, probabilities = model.predict(texts)
```

### 3. Model Evaluation

```python
from src.models.evaluation import SentimentEvaluator

evaluator = SentimentEvaluator()
metrics = evaluator.evaluate_classification(y_true, y_pred, y_proba)
```

### 4. Trading Strategy Backtesting

```python
from src.backtest.trading_strategy import SentimentTradingStrategy

strategy = SentimentTradingStrategy(initial_capital=100000)
signals = strategy.generate_signals(sentiment_data)
results = strategy.backtest_strategy(signals, price_data)
```

## Running the Demo

Launch the interactive Streamlit application:

```bash
streamlit run demo/app.py
```

The demo includes:
- Single text analysis with explanations
- Batch processing with performance metrics
- Trading strategy backtesting
- Model comparison tools

## Project Structure

```
financial-sentiment-analysis/
├── src/
│   ├── data/
│   │   └── news_data.py          # Data generation and loading
│   ├── models/
│   │   ├── sentiment_models.py   # Sentiment analysis models
│   │   ├── evaluation.py         # Evaluation metrics
│   │   └── explainability.py     # Model explainability
│   ├── backtest/
│   │   └── trading_strategy.py   # Trading strategy backtesting
│   └── utils/
│       └── core.py              # Core utilities
├── configs/
│   └── config.yaml              # Configuration file
├── demo/
│   └── app.py                   # Streamlit demo application
├── data/                        # Data storage
├── assets/                      # Generated plots and models
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Configuration

The project uses YAML configuration files. Key settings in `configs/config.yaml`:

```yaml
models:
  vader:
    enabled: true
    thresholds:
      positive: 0.1
      negative: -0.1
  finbert:
    enabled: true
    model_name: "ProsusAI/finbert"
    max_length: 512
    batch_size: 16

training:
  epochs: 5
  learning_rate: 2e-5
  weight_decay: 0.01

backtesting:
  initial_capital: 100000
  transaction_cost: 0.001
  slippage: 0.0005
```

## Data Schema

### Financial News Data
```python
{
    'headline': str,           # News headline
    'sentiment': str,          # Ground truth: 'positive', 'negative', 'neutral'
    'date': datetime,         # Publication date
    'source': str,            # News source
    'company': str,           # Company mentioned
    'text_length': int,       # Character count
    'word_count': int         # Word count
}
```

### Market Data
```python
{
    'date': datetime,         # Trading date
    'open': float,           # Opening price
    'high': float,           # High price
    'low': float,            # Low price
    'close': float,          # Closing price
    'volume': int            # Trading volume
}
```

## Model Performance

Typical performance on synthetic financial news data:

| Model | Accuracy | F1 Macro | Precision | Recall |
|-------|----------|----------|-----------|--------|
| VADER | 0.65 | 0.62 | 0.64 | 0.61 |
| FinBERT | 0.78 | 0.76 | 0.77 | 0.75 |
| Transformer | 0.72 | 0.70 | 0.71 | 0.69 |
| Ensemble | 0.80 | 0.78 | 0.79 | 0.77 |

*Note: Performance may vary with different datasets and configurations.*

## Backtesting Results

Example backtesting results with synthetic data:

- **Initial Capital**: $100,000
- **Final Value**: $105,000
- **Total Return**: 5.0%
- **Sharpe Ratio**: 0.85
- **Max Drawdown**: -3.2%
- **Total Trades**: 45
- **Win Rate**: 58%

*Note: These are hypothetical results for demonstration purposes only.*

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ demo/ tests/
ruff check src/ demo/ tests/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{financial_sentiment_analysis,
  title={Financial Sentiment Analysis: A Modern NLP Approach},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Financial-Sentiment-Analysis}
}
```

## Acknowledgments

- ProsusAI for the FinBERT model
- Hugging Face for the Transformers library
- The open-source community for various NLP tools and libraries

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the demo application

---

**Remember: This is for research and educational purposes only. Not for investment advice.**
# Financial-Sentiment-Analysis
