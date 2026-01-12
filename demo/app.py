"""
Streamlit demo app for financial sentiment analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.news_data import FinancialNewsDataGenerator, FinancialNewsDataLoader
from src.models.sentiment_models import VADERSentimentAnalyzer, FinBERTModel, TransformerSentimentModel
from src.models.evaluation import SentimentEvaluator
from src.models.explainability import SentimentExplainer
from src.backtest.trading_strategy import SentimentTradingStrategy, MarketDataDownloader, BacktestAnalyzer
from src.utils.core import set_random_seeds, load_config

# Page configuration
st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <p>This application is NOT for investment advice. Results may be inaccurate and are for research purposes only. 
    Past performance does not guarantee future results. Always consult with qualified financial advisors before making investment decisions.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">📈 Financial Sentiment Analysis</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")
st.sidebar.markdown("### Model Settings")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["VADER", "FinBERT", "Transformer", "Ensemble"],
    help="Choose the sentiment analysis model to use"
)

# Load configuration
try:
    config = load_config("configs/config.yaml")
except:
    config = {
        'models': {
            'vader': {'enabled': True, 'thresholds': {'positive': 0.1, 'negative': -0.1}},
            'finbert': {'enabled': True, 'model_name': 'ProsusAI/finbert'},
            'transformer': {'enabled': True, 'model_name': 'cardiffnlp/twitter-roberta-base-sentiment-latest'}
        }
    }

# Initialize models
@st.cache_resource
def load_models():
    """Load and cache models."""
    models = {}
    
    try:
        models['vader'] = VADERSentimentAnalyzer(
            thresholds=config['models']['vader']['thresholds']
        )
    except Exception as e:
        st.error(f"Error loading VADER: {e}")
    
    try:
        models['finbert'] = FinBERTModel(
            model_name=config['models']['finbert']['model_name']
        )
    except Exception as e:
        st.error(f"Error loading FinBERT: {e}")
    
    try:
        models['transformer'] = TransformerSentimentModel(
            model_name=config['models']['transformer']['model_name']
        )
    except Exception as e:
        st.error(f"Error loading Transformer: {e}")
    
    return models

# Load models
with st.spinner("Loading models..."):
    models = load_models()

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Single Text Analysis", 
    "📊 Batch Analysis", 
    "📈 Backtesting", 
    "🧠 Model Comparison", 
    "📋 About"
])

# Tab 1: Single Text Analysis
with tab1:
    st.header("Single Text Sentiment Analysis")
    
    # Text input
    text_input = st.text_area(
        "Enter financial news text:",
        value="Apple reports strong quarterly earnings, beating analyst expectations",
        height=100,
        help="Enter any financial news text to analyze sentiment"
    )
    
    if st.button("Analyze Sentiment", type="primary"):
        if text_input.strip():
            with st.spinner("Analyzing sentiment..."):
                # Get selected model
                if model_type == "VADER":
                    model = models.get('vader')
                    if model:
                        scores, classes = model.predict([text_input])
                        probabilities = model.predict_proba([text_input])
                        prediction = classes[0]
                        confidence = probabilities[0]
                elif model_type == "FinBERT":
                    model = models.get('finbert')
                    if model:
                        pred_labels, probabilities = model.predict([text_input])
                        prediction = ['negative', 'neutral', 'positive'][pred_labels[0]]
                        confidence = probabilities[0]
                elif model_type == "Transformer":
                    model = models.get('transformer')
                    if model:
                        predictions, probabilities = model.predict([text_input])
                        prediction = predictions[0]
                        confidence = probabilities[0]
                else:  # Ensemble
                    # Use VADER as fallback
                    model = models.get('vader')
                    if model:
                        scores, classes = model.predict([text_input])
                        probabilities = model.predict_proba([text_input])
                        prediction = classes[0]
                        confidence = probabilities[0]
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prediction", prediction.title())
                
                with col2:
                    max_prob = np.max(confidence)
                    st.metric("Confidence", f"{max_prob:.2%}")
                
                with col3:
                    # Sentiment color
                    color_map = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}
                    st.metric("Sentiment", color_map.get(prediction, '❓'))
                
                # Probability distribution
                st.subheader("Probability Distribution")
                
                prob_df = pd.DataFrame({
                    'Sentiment': ['Negative', 'Neutral', 'Positive'],
                    'Probability': confidence
                })
                
                fig = px.bar(
                    prob_df, 
                    x='Sentiment', 
                    y='Probability',
                    color='Sentiment',
                    color_discrete_map={
                        'Negative': '#ff4444',
                        'Neutral': '#ffaa00', 
                        'Positive': '#44ff44'
                    }
                )
                fig.update_layout(
                    title="Sentiment Probability Distribution",
                    xaxis_title="Sentiment",
                    yaxis_title="Probability",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Model explanation (if available)
                if st.checkbox("Show Model Explanation"):
                    try:
                        explainer = SentimentExplainer(model)
                        lime_results = explainer.explain_with_lime(text_input)
                        
                        st.subheader("Feature Importance (LIME)")
                        
                        # Create explanation dataframe
                        features, scores = zip(*lime_results['feature_importance'])
                        exp_df = pd.DataFrame({
                            'Feature': features,
                            'Importance': scores
                        })
                        
                        # Sort by absolute importance
                        exp_df['abs_importance'] = exp_df['Importance'].abs()
                        exp_df = exp_df.sort_values('abs_importance', ascending=True)
                        
                        # Plot
                        fig = px.bar(
                            exp_df, 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            color='Importance',
                            color_continuous_scale='RdBu_r'
                        )
                        fig.update_layout(
                            title="Feature Importance",
                            xaxis_title="Importance Score",
                            yaxis_title="Features"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.warning(f"Could not generate explanation: {e}")

# Tab 2: Batch Analysis
with tab2:
    st.header("Batch Sentiment Analysis")
    
    # Data source selection
    data_source = st.radio(
        "Choose data source:",
        ["Generate Synthetic Data", "Upload CSV File"],
        help="Select whether to generate synthetic data or upload your own"
    )
    
    if data_source == "Generate Synthetic Data":
        n_samples = st.slider("Number of samples:", 10, 1000, 100)
        
        if st.button("Generate and Analyze Data"):
            with st.spinner("Generating synthetic data..."):
                generator = FinancialNewsDataGenerator()
                df = generator.generate_news_data(n_samples)
                
                # Analyze with selected model
                if model_type == "VADER":
                    model = models.get('vader')
                    if model:
                        scores, classes = model.predict(df['headline'].tolist())
                        probabilities = model.predict_proba(df['headline'].tolist())
                        df['predicted_sentiment'] = classes
                        df['confidence'] = np.max(probabilities, axis=1)
                
                # Display results
                st.subheader("Analysis Results")
                st.dataframe(df.head(10))
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sentiment Distribution")
                    sentiment_counts = df['sentiment'].value_counts()
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Actual Sentiment Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Predicted vs Actual")
                    if 'predicted_sentiment' in df.columns:
                        pred_counts = df['predicted_sentiment'].value_counts()
                        fig = px.pie(
                            values=pred_counts.values,
                            names=pred_counts.index,
                            title="Predicted Sentiment Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics
                if 'predicted_sentiment' in df.columns:
                    st.subheader("Performance Metrics")
                    
                    evaluator = SentimentEvaluator()
                    metrics = evaluator.evaluate_classification(
                        df['sentiment'].tolist(),
                        df['predicted_sentiment'].tolist(),
                        probabilities if 'probabilities' in locals() else None
                    )
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                    with col2:
                        st.metric("F1 Macro", f"{metrics['f1_macro']:.3f}")
                    with col3:
                        st.metric("Precision", f"{metrics['precision_macro']:.3f}")
                    with col4:
                        st.metric("Recall", f"{metrics['recall_macro']:.3f}")
    
    else:  # Upload CSV
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file with 'headline' and 'sentiment' columns"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head())
            
            if 'headline' in df.columns:
                if st.button("Analyze Uploaded Data"):
                    with st.spinner("Analyzing uploaded data..."):
                        # Analyze with selected model
                        if model_type == "VADER":
                            model = models.get('vader')
                            if model:
                                scores, classes = model.predict(df['headline'].tolist())
                                probabilities = model.predict_proba(df['headline'].tolist())
                                df['predicted_sentiment'] = classes
                                df['confidence'] = np.max(probabilities, axis=1)
                        
                        # Display results
                        st.subheader("Analysis Results")
                        st.dataframe(df.head(10))
                        
                        # Performance metrics (if sentiment column exists)
                        if 'sentiment' in df.columns:
                            evaluator = SentimentEvaluator()
                            metrics = evaluator.evaluate_classification(
                                df['sentiment'].tolist(),
                                df['predicted_sentiment'].tolist()
                            )
                            
                            st.subheader("Performance Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                            with col2:
                                st.metric("F1 Macro", f"{metrics['f1_macro']:.3f}")
                            with col3:
                                st.metric("Precision", f"{metrics['precision_macro']:.3f}")
                            with col4:
                                st.metric("Recall", f"{metrics['recall_macro']:.3f}")

# Tab 3: Backtesting
with tab3:
    st.header("Sentiment-Based Trading Strategy Backtesting")
    
    st.markdown("""
    <div class="disclaimer">
        <p><strong>Backtesting Disclaimer:</strong> This is a hypothetical backtest for research purposes only. 
        Past performance does not guarantee future results. Real trading involves additional costs and risks not captured here.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Strategy parameters
    col1, col2 = st.columns(2)
    
    with col1:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=10000,
            max_value=1000000,
            value=100000,
            step=10000
        )
        
        transaction_cost = st.slider(
            "Transaction Cost (%)",
            min_value=0.0,
            max_value=0.01,
            value=0.001,
            step=0.0001,
            format="%.4f"
        )
    
    with col2:
        sentiment_threshold = st.slider(
            "Sentiment Threshold",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.01
        )
        
        rebalance_freq = st.selectbox(
            "Rebalance Frequency",
            ["1D", "1W", "1M"],
            help="How often to rebalance the portfolio"
        )
    
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            # Generate synthetic data
            generator = FinancialNewsDataGenerator()
            sentiment_data = generator.generate_news_data(100)
            
            # Generate synthetic price data
            downloader = MarketDataDownloader()
            price_data = downloader.generate_synthetic_price_data(
                start_date="2023-01-01",
                end_date="2023-12-31"
            )
            
            # Initialize strategy
            strategy = SentimentTradingStrategy(
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                rebalance_frequency=rebalance_freq
            )
            
            # Generate signals
            signals = strategy.generate_signals(sentiment_data, sentiment_threshold)
            
            # Run backtest
            results = strategy.backtest_strategy(signals, price_data)
            
            if results:
                # Calculate performance metrics
                metrics = strategy.calculate_performance_metrics(results)
                
                # Display results
                st.subheader("Backtest Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Final Value", f"${results['final_value']:,.2f}")
                with col2:
                    st.metric("Total Return", f"{results['total_return']:.2%}")
                with col3:
                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
                with col4:
                    st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
                
                # Portfolio performance chart
                st.subheader("Portfolio Performance")
                
                portfolio_df = results['portfolio_history']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=portfolio_df['date'],
                    y=portfolio_df['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title="Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade analysis
                if not results['trades'].empty:
                    st.subheader("Trade Analysis")
                    
                    trades_df = results['trades']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Trades", len(trades_df))
                        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2%}")
                    
                    with col2:
                        st.metric("Avg Trade P&L", f"${metrics.get('avg_trade_pnl', 0):,.2f}")
                        st.metric("Total Trade P&L", f"${metrics.get('total_trade_pnl', 0):,.2f}")
                    
                    # Trade distribution
                    trade_counts = trades_df['signal'].value_counts()
                    fig = px.bar(
                        x=trade_counts.index,
                        y=trade_counts.values,
                        title="Trade Signal Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Model Comparison
with tab4:
    st.header("Model Comparison")
    
    if st.button("Compare All Models", type="primary"):
        with st.spinner("Comparing models..."):
            # Generate test data
            generator = FinancialNewsDataGenerator()
            test_data = generator.generate_news_data(200)
            
            # Test all available models
            comparison_results = {}
            
            for model_name, model in models.items():
                if model:
                    try:
                        if hasattr(model, 'predict'):
                            predictions, probabilities = model.predict(test_data['headline'].tolist())
                            if isinstance(predictions, list) and len(predictions) > 0:
                                if isinstance(predictions[0], str):
                                    pred_labels = predictions
                                else:
                                    pred_labels = [['negative', 'neutral', 'positive'][p] for p in predictions]
                            else:
                                pred_labels = predictions
                        else:
                            pred_labels = [model.predict([text])[0] for text in test_data['headline']]
                        
                        # Evaluate
                        evaluator = SentimentEvaluator()
                        metrics = evaluator.evaluate_classification(
                            test_data['sentiment'].tolist(),
                            pred_labels,
                            probabilities if 'probabilities' in locals() else None
                        )
                        
                        comparison_results[model_name.upper()] = metrics
                        
                    except Exception as e:
                        st.error(f"Error evaluating {model_name}: {e}")
            
            if comparison_results:
                # Create comparison table
                st.subheader("Model Performance Comparison")
                
                comparison_data = []
                for model_name, metrics in comparison_results.items():
                    comparison_data.append({
                        'Model': model_name,
                        'Accuracy': f"{metrics['accuracy']:.3f}",
                        'F1 Macro': f"{metrics['f1_macro']:.3f}",
                        'F1 Weighted': f"{metrics['f1_weighted']:.3f}",
                        'Precision': f"{metrics['precision_macro']:.3f}",
                        'Recall': f"{metrics['recall_macro']:.3f}"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualization
                st.subheader("Performance Visualization")
                
                metrics_to_plot = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
                
                fig = go.Figure()
                
                for metric in metrics_to_plot:
                    fig.add_trace(go.Bar(
                        name=metric.replace('_', ' ').title(),
                        x=list(comparison_results.keys()),
                        y=[comparison_results[model][metric] for model in comparison_results.keys()]
                    ))
                
                fig.update_layout(
                    title="Model Performance Comparison",
                    xaxis_title="Models",
                    yaxis_title="Score",
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Tab 5: About
with tab5:
    st.header("About This Application")
    
    st.markdown("""
    ## Financial Sentiment Analysis Demo
    
    This application demonstrates advanced NLP techniques for analyzing sentiment in financial news and text.
    
    ### Features
    
    - **Multiple Models**: VADER, FinBERT, and Transformer-based sentiment analysis
    - **Single Text Analysis**: Analyze individual financial news articles
    - **Batch Processing**: Process multiple texts with performance metrics
    - **Trading Strategy Backtesting**: Test sentiment-based trading strategies
    - **Model Comparison**: Compare different models side-by-side
    - **Explainability**: Understand model predictions with LIME explanations
    
    ### Models Used
    
    1. **VADER**: Rule-based sentiment analyzer optimized for social media text
    2. **FinBERT**: BERT model fine-tuned on financial text
    3. **Transformer**: RoBERTa-based sentiment classification model
    
    ### Technical Stack
    
    - **Frontend**: Streamlit
    - **ML Libraries**: Transformers, scikit-learn, PyTorch
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Backtesting**: Custom implementation with realistic costs
    
    ### Important Notes
    
    - This is a research and educational demonstration
    - Results may not be accurate for real-world applications
    - Not intended for investment advice
    - Always consult qualified financial advisors
    
    ### Contact
    
    For questions or issues, please refer to the project documentation.
    """)
    
    # Technical details
    with st.expander("Technical Details"):
        st.code("""
        # Key dependencies
        streamlit>=1.25.0
        transformers>=4.30.0
        torch>=2.0.0
        pandas>=2.0.0
        numpy>=1.24.0
        plotly>=5.15.0
        scikit-learn>=1.3.0
        """, language="python")
        
        st.markdown("""
        ### Model Architecture
        
        - **FinBERT**: 12-layer transformer with 768 hidden dimensions
        - **VADER**: Lexicon and rule-based approach
        - **RoBERTa**: 12-layer transformer optimized for sentiment
        
        ### Evaluation Metrics
        
        - Accuracy, Precision, Recall, F1-Score
        - ROC AUC and PR AUC
        - Calibration Error
        - Confusion Matrix Analysis
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Financial Sentiment Analysis Demo | Research & Educational Use Only</p>
    <p>⚠️ Not for Investment Advice | Results May Be Inaccurate</p>
</div>
""", unsafe_allow_html=True)
