# Financial Sentiment Analysis - Quick Start Guide

## Quick Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download NLTK data:**
```python
import nltk
nltk.download('vader_lexicon')
```

3. **Run the demo:**
```bash
streamlit run demo/app.py
```

## Basic Usage

```python
from src.data.news_data import FinancialNewsDataGenerator
from src.models.sentiment_models import VADERSentimentAnalyzer

# Generate sample data
generator = FinancialNewsDataGenerator()
data = generator.generate_news_data(100)

# Analyze sentiment
model = VADERSentimentAnalyzer()
predictions, probabilities = model.predict(data['headline'].tolist())
```

## Training Models

```bash
python scripts/train.py --model ensemble --epochs 5 --backtest --explain
```

## Running Tests

```bash
pytest tests/
```

## Important Notes

- This is for **research and educational purposes only**
- **NOT for investment advice**
- Results may be inaccurate
- Always consult qualified financial advisors

## Project Structure

- `src/` - Core source code
- `demo/` - Streamlit application
- `configs/` - Configuration files
- `data/` - Data storage
- `assets/` - Generated outputs
- `tests/` - Unit tests

## Support

Check the README.md for detailed documentation and examples.
