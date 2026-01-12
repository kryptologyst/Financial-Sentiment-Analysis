"""
Main training and evaluation script for financial sentiment analysis.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.news_data import FinancialNewsDataLoader
from src.models.sentiment_models import VADERSentimentAnalyzer, FinBERTModel, EnsembleSentimentModel
from src.models.evaluation import SentimentEvaluator, ModelComparison
from src.models.explainability import SentimentExplainer
from src.backtest.trading_strategy import SentimentTradingStrategy, MarketDataDownloader
from src.utils.core import set_random_seeds, load_config, create_directories, setup_logging

def main():
    """Main training and evaluation function."""
    parser = argparse.ArgumentParser(description="Financial Sentiment Analysis Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--data-path", type=str, help="Path to data file")
    parser.add_argument("--output-dir", type=str, default="assets/results", help="Output directory")
    parser.add_argument("--model", type=str, choices=["vader", "finbert", "transformer", "ensemble"], 
                       default="ensemble", help="Model to train")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting")
    parser.add_argument("--explain", action="store_true", help="Generate explanations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup
    set_random_seeds(args.seed)
    create_directories(".")
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("Starting financial sentiment analysis training")
    
    # Load configuration
    config = load_config(args.config)
    
    # Load data
    data_loader = FinancialNewsDataLoader(args.data_path)
    df = data_loader.load_data()
    df = data_loader.preprocess_data(df)
    
    # Split data
    train_df, val_df, test_df = data_loader.get_train_val_test_split(df)
    
    logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize models
    models = {}
    
    if args.model in ["vader", "ensemble"]:
        models['vader'] = VADERSentimentAnalyzer(
            thresholds=config['models']['vader']['thresholds']
        )
    
    if args.model in ["finbert", "ensemble"]:
        models['finbert'] = FinBERTModel(
            model_name=config['models']['finbert']['model_name'],
            max_length=config['models']['finbert']['max_length']
        )
    
    if args.model in ["transformer", "ensemble"]:
        models['transformer'] = TransformerSentimentModel(
            model_name=config['models']['transformer']['model_name'],
            max_length=config['models']['transformer']['max_length']
        )
    
    # Train models
    for model_name, model in models.items():
        if hasattr(model, 'train') and model_name != 'vader':
            logger.info(f"Training {model_name}")
            model.train(
                train_df['headline'].tolist(),
                train_df['sentiment_label'].tolist(),
                val_df['headline'].tolist(),
                val_df['sentiment_label'].tolist(),
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size
            )
    
    # Create ensemble if requested
    if args.model == "ensemble":
        ensemble_model = EnsembleSentimentModel(
            list(models.values()),
            weights=[1.0/len(models)] * len(models)
        )
        models['ensemble'] = ensemble_model
    
    # Evaluate models
    evaluator = SentimentEvaluator()
    comparison = ModelComparison(evaluator)
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}")
        
        # Get predictions
        if hasattr(model, 'predict'):
            predictions, probabilities = model.predict(test_df['headline'].tolist())
        else:
            predictions = [model.predict([text])[0] for text in test_df['headline']]
            probabilities = None
        
        # Add to comparison
        comparison.add_model_results(
            model_name,
            test_df['sentiment'].tolist(),
            predictions,
            probabilities
        )
    
    # Create comparison report
    comparison_df = comparison.create_comparison_table()
    logger.info("Model comparison results:")
    logger.info(f"\n{comparison_df}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    
    # Backtesting
    if args.backtest:
        logger.info("Running backtesting")
        
        # Generate market data
        downloader = MarketDataDownloader()
        price_data = downloader.generate_synthetic_price_data(
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        # Run backtest for each model
        for model_name, model in models.items():
            logger.info(f"Backtesting {model_name}")
            
            strategy = SentimentTradingStrategy(
                initial_capital=config['backtesting']['initial_capital'],
                transaction_cost=config['backtesting']['transaction_cost'],
                slippage=config['backtesting']['slippage']
            )
            
            signals = strategy.generate_signals(test_df)
            results = strategy.backtest_strategy(signals, price_data)
            
            if results:
                metrics = strategy.calculate_performance_metrics(results)
                logger.info(f"{model_name} backtest results: {metrics}")
    
    # Explainability
    if args.explain:
        logger.info("Generating explanations")
        
        sample_texts = test_df['headline'].head(5).tolist()
        
        for model_name, model in models.items():
            logger.info(f"Generating explanations for {model_name}")
            
            explainer = SentimentExplainer(model)
            
            for text in sample_texts:
                try:
                    report = explainer.create_explanation_report(text, model_name)
                    logger.info(f"Explanation generated for: {text[:50]}...")
                except Exception as e:
                    logger.warning(f"Could not generate explanation: {e}")
    
    logger.info("Training and evaluation completed")

if __name__ == "__main__":
    main()
