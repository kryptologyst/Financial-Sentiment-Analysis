"""
Backtesting module for sentiment-based trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import yfinance as yf
from pathlib import Path

logger = logging.getLogger(__name__)


class SentimentTradingStrategy:
    """Sentiment-based trading strategy."""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        rebalance_frequency: str = "1D"
    ):
        """Initialize trading strategy."""
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.rebalance_frequency = rebalance_frequency
        
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        
    def generate_signals(
        self, 
        sentiment_data: pd.DataFrame,
        sentiment_threshold: float = 0.1
    ) -> pd.DataFrame:
        """Generate trading signals from sentiment data."""
        logger.info("Generating trading signals from sentiment data")
        
        signals = sentiment_data.copy()
        
        # Convert sentiment to numeric scores
        sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
        signals['sentiment_score'] = signals['sentiment'].map(sentiment_map)
        
        # Generate signals based on sentiment
        signals['signal'] = 0
        signals.loc[signals['sentiment_score'] > sentiment_threshold, 'signal'] = 1  # Buy
        signals.loc[signals['sentiment_score'] < -sentiment_threshold, 'signal'] = -1  # Sell
        
        # Add signal strength
        signals['signal_strength'] = np.abs(signals['sentiment_score'])
        
        return signals
    
    def backtest_strategy(
        self, 
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        symbol: str = "SPY"
    ) -> Dict[str, Any]:
        """Backtest the sentiment-based trading strategy."""
        logger.info(f"Backtesting strategy for {symbol}")
        
        # Merge signals with price data
        merged_data = pd.merge(
            signals, 
            price_data, 
            left_on='date', 
            right_on='Date', 
            how='inner'
        ).sort_values('date')
        
        if merged_data.empty:
            logger.error("No overlapping data between signals and prices")
            return {}
        
        # Initialize portfolio tracking
        portfolio_values = []
        trades = []
        current_position = 0
        cash = self.initial_capital
        
        for idx, row in merged_data.iterrows():
            date = row['date']
            price = row['Close']
            signal = row['signal']
            signal_strength = row['signal_strength']
            
            # Calculate position size based on signal strength
            if signal != 0:
                position_size = signal_strength * 0.1  # 10% of portfolio per signal
                target_position = position_size * self.portfolio_value / price
                
                # Calculate trade
                position_change = target_position - current_position
                
                if abs(position_change) > 0:
                    # Calculate costs
                    trade_value = abs(position_change) * price
                    total_cost = trade_value * (self.transaction_cost + self.slippage)
                    
                    # Execute trade
                    if position_change > 0:  # Buy
                        if cash >= trade_value + total_cost:
                            cash -= (trade_value + total_cost)
                            current_position += position_change
                            trade_type = "BUY"
                        else:
                            continue  # Insufficient cash
                    else:  # Sell
                        if current_position >= abs(position_change):
                            cash += (trade_value - total_cost)
                            current_position += position_change
                            trade_type = "SELL"
                        else:
                            continue  # Insufficient position
                    
                    # Record trade
                    trade = {
                        'date': date,
                        'type': trade_type,
                        'price': price,
                        'quantity': abs(position_change),
                        'value': trade_value,
                        'cost': total_cost,
                        'signal': signal,
                        'signal_strength': signal_strength
                    }
                    trades.append(trade)
            
            # Calculate portfolio value
            portfolio_value = cash + current_position * price
            portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'position': current_position,
                'price': price
            })
        
        # Create results
        results = {
            'portfolio_history': pd.DataFrame(portfolio_values),
            'trades': pd.DataFrame(trades),
            'final_value': portfolio_values[-1]['portfolio_value'] if portfolio_values else self.initial_capital,
            'total_return': (portfolio_values[-1]['portfolio_value'] / self.initial_capital - 1) if portfolio_values else 0
        }
        
        logger.info(f"Backtest completed. Final value: ${results['final_value']:.2f}")
        return results
    
    def calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not results or 'portfolio_history' not in results:
            return {}
        
        portfolio_df = results['portfolio_history']
        returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = results['total_return']
        metrics['annualized_return'] = (1 + results['total_return']) ** (252 / len(returns)) - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
        
        # Risk metrics
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        metrics['sortino_ratio'] = metrics['annualized_return'] / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # Trade metrics
        if 'trades' in results and not results['trades'].empty:
            trades_df = results['trades']
            metrics['total_trades'] = len(trades_df)
            metrics['win_rate'] = len(trades_df[trades_df['value'] > 0]) / len(trades_df) if len(trades_df) > 0 else 0
            
            # Calculate trade P&L (simplified)
            trade_pnl = []
            for _, trade in trades_df.iterrows():
                if trade['type'] == 'BUY':
                    # Find corresponding sell
                    future_trades = trades_df[trades_df['date'] > trade['date']]
                    if not future_trades.empty:
                        sell_trade = future_trades.iloc[0]
                        if sell_trade['type'] == 'SELL':
                            pnl = (sell_trade['price'] - trade['price']) * trade['quantity'] - trade['cost'] - sell_trade['cost']
                            trade_pnl.append(pnl)
            
            if trade_pnl:
                metrics['avg_trade_pnl'] = np.mean(trade_pnl)
                metrics['total_trade_pnl'] = np.sum(trade_pnl)
        
        return metrics


class MarketDataDownloader:
    """Download market data for backtesting."""
    
    def __init__(self):
        """Initialize market data downloader."""
        pass
    
    def download_price_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Download price data for a symbol."""
        logger.info(f"Downloading price data for {symbol}")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Reset index to get Date as column
            data = data.reset_index()
            data.columns = [col.lower() for col in data.columns]
            
            logger.info(f"Downloaded {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def generate_synthetic_price_data(
        self, 
        start_date: str, 
        end_date: str,
        initial_price: float = 100,
        volatility: float = 0.02,
        drift: float = 0.0001
    ) -> pd.DataFrame:
        """Generate synthetic price data for testing."""
        logger.info("Generating synthetic price data")
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate price using geometric Brownian motion
        n_days = len(dates)
        dt = 1/252  # Daily timestep
        
        # Random shocks
        shocks = np.random.normal(0, 1, n_days)
        
        # Calculate returns
        returns = drift * dt + volatility * np.sqrt(dt) * shocks
        
        # Calculate prices
        prices = [initial_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = prices[1:]  # Remove initial price
        
        # Create DataFrame
        data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n_days)
        })
        
        # Ensure high >= low and high/low >= open/close
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])
        data['high'] = np.maximum(data['high'], data['open'])
        data['low'] = np.minimum(data['low'], data['open'])
        
        logger.info(f"Generated synthetic data with {len(data)} records")
        return data


class BacktestAnalyzer:
    """Analyze and visualize backtest results."""
    
    def __init__(self):
        """Initialize backtest analyzer."""
        pass
    
    def plot_portfolio_performance(
        self, 
        portfolio_history: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot portfolio performance over time."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(portfolio_history['date'], portfolio_history['portfolio_value'], 
                label='Portfolio', linewidth=2)
        
        if benchmark_data is not None:
            plt.plot(benchmark_data['date'], benchmark_data['close'], 
                    label='Benchmark', linewidth=2, alpha=0.7)
        
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        returns = portfolio_history['portfolio_value'].pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        
        plt.fill_between(portfolio_history['date'][1:], drawdown, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_trade_analysis(
        self, 
        trades: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """Plot trade analysis."""
        if trades.empty:
            logger.warning("No trades to analyze")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Trade frequency over time
        trades['date'] = pd.to_datetime(trades['date'])
        trade_counts = trades.groupby(trades['date'].dt.date).size()
        
        axes[0, 0].plot(trade_counts.index, trade_counts.values, marker='o')
        axes[0, 0].set_title('Trade Frequency Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Number of Trades')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Signal distribution
        signal_counts = trades['signal'].value_counts()
        axes[0, 1].bar(signal_counts.index, signal_counts.values)
        axes[0, 1].set_title('Signal Distribution')
        axes[0, 1].set_xlabel('Signal')
        axes[0, 1].set_ylabel('Count')
        
        # Signal strength distribution
        axes[1, 0].hist(trades['signal_strength'], bins=20, alpha=0.7)
        axes[1, 0].set_title('Signal Strength Distribution')
        axes[1, 0].set_xlabel('Signal Strength')
        axes[1, 0].set_ylabel('Frequency')
        
        # Trade value distribution
        axes[1, 1].hist(trades['value'], bins=20, alpha=0.7)
        axes[1, 1].set_title('Trade Value Distribution')
        axes[1, 1].set_xlabel('Trade Value ($)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_report(
        self, 
        results: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> pd.DataFrame:
        """Create comprehensive performance report."""
        report_data = []
        
        for metric, value in metrics.items():
            report_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Value': f"{value:.4f}" if isinstance(value, float) else str(value)
            })
        
        return pd.DataFrame(report_data)
