"""
Configuration Settings for Institutional Trading System v3
COMPLETE VERSION - ₹10K Capital | Top 3 Positions | Enhanced Hybrid Models
Last Updated: 2025-11-21
"""

import os
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROJECT ROOT AND DIRECTORIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Define all project directories
PROJECT_DIRECTORIES = {
    'MODELS': str(PROJECT_ROOT / 'models'),
    'FEATURE_STORE': str(PROJECT_ROOT / 'data_cache' / 'feature_store'),
    'PRICE_CACHE': str(PROJECT_ROOT / 'data_cache' / 'pickle_data'),
    'METADATA_DB': str(PROJECT_ROOT / 'data_cache' / 'metadata.db'),
    'RESULTS': str(PROJECT_ROOT / 'results'),
    'SIGNALS': str(PROJECT_ROOT / 'signals'),
    'LOGS': str(PROJECT_ROOT / 'logs'),
    'DATA_CACHE': str(PROJECT_ROOT / 'data_cache'),
}

# Create directories if they don't exist
for dir_name, dir_path in PROJECT_DIRECTORIES.items():
    if dir_name != 'METADATA_DB':  # Skip DB file
        os.makedirs(dir_path, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRADING UNIVERSE - NSE Nifty 100 Stocks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRADE_UNIVERSE = [
    # Large Cap - Blue Chips
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS',
    'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS',
    
    # IT & Technology
    'WIPRO.NS', 'TECHM.NS', 'LTTS.NS', 'MPHASIS.NS', 'COFORGE.NS',
    
    # Banking & Finance
    'INDUSINDBK.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'ICICIPRULI.NS', 'ICICIGI.NS',
    'BANDHANBNK.NS', 'BANKBARODA.NS', 'PNB.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS',
    
    # Energy & Resources
    'ULTRACEMCO.NS', 'BPCL.NS', 'IOC.NS', 'TATAPOWER.NS', 'GAIL.NS',
    'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'COALINDIA.NS', 'VEDL.NS',
    'ADANIGREEN.NS', 'ADANIENT.NS', 'ADANITRANS.NS', 'ADANIPORTS.NS',
    
    # Auto & Manufacturing
    'TATAMOTORS.NS', 'M&M.NS', 'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS',
    'BOSCHLTD.NS', 'MOTHERSON.NS',
    
    # Pharma & Healthcare
    'SUNPHARMA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'CIPLA.NS', 'LUPIN.NS',
    'BIOCON.NS', 'TORNTPHARM.NS', 'AUROPHARMA.NS', 'APOLLOHOSP.NS',
    
    # Metals & Mining
    'JSWSTEEL.NS', 'TATASTEEL.NS', 'HINDALCO.NS', 'GRASIM.NS',
    
    # Chemicals & Materials
    'UPL.NS', 'PIDILITIND.NS', 'SRF.NS', 'AMBUJACEM.NS', 'ACC.NS', 'SHREECEM.NS',
    
    # FMCG & Consumer
    'BRITANNIA.NS', 'TITAN.NS', 'NESTLEIND.NS', 'TATACONSUM.NS',
    'GODREJCP.NS', 'DABUR.NS', 'MARICO.NS', 'COLPAL.NS', 'MCDOWELL-N.NS',
    
    # Real Estate & Infrastructure
    'DLF.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS',
    
    # Paints & Building
    'BERGEPAINT.NS', 'HAVELLS.NS',
    
    # Conglomerates
    'SIEMENS.NS', 'ABB.NS', 'PAGEIND.NS', 'PGHH.NS', 'BAJAJHLDNG.NS',
    
    # Retail & E-commerce
    'TRENT.NS', 'NYKAA.NS', 'ZOMATO.NS', 'PAYTM.NS',
    
    # Travel & Hospitality
    'INDIGO.NS', 'IRCTC.NS',
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CAPITAL & POSITION CONFIGURATION - ₹10K CAPITAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CAPITAL_CONFIG = {
    # Total trading capital
    'TOTAL_CAPITAL': 10000,  # ₹10,000
    
    # Position limits - TOP 3 ONLY
    'MAX_POSITIONS': 3,  # Maximum 3 concurrent positions
    'MIN_POSITION_VALUE': 2000,  # Minimum ₹2k per position
    
    # Reserve cash
    'RESERVE_CASH_PCT': 0.05,  # Keep 5% as reserve (₹500)
    
    # Position sizing - AGGRESSIVE for concentrated portfolio
    'DEFAULT_POSITION_SIZE': 0.30,  # 30% of capital per position
    'TOP_POSITION_SIZE': 0.35,      # 35% for highest confidence (₹3,500)
    'SECOND_POSITION_SIZE': 0.32,   # 32% for second best (₹3,200)
    'THIRD_POSITION_SIZE': 0.28,    # 28% for third (₹2,800)
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RISK MANAGEMENT PARAMETERS - OPTIMIZED FOR ₹10K
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RISK_PARAMS = {
    # Portfolio-level risk
    'MAX_PORTFOLIO_RISK': 0.03,  # 3% max total portfolio risk (₹300)
    'MAX_DAILY_LOSS': 0.05,      # 5% max daily loss (₹500)
    'MAX_DRAWDOWN': 0.20,        # 20% max drawdown (₹2,000)
    
    # Position-level risk - HIGHER for concentrated positions
    'MAX_SINGLE_POSITION': 0.35,  # 35% max capital per position (₹3,500)
    'MAX_POSITION_RISK': 0.01,    # 1% max risk per position (₹100)
    
    # Stop loss settings - TIGHTER for small capital
    'DEFAULT_STOP_LOSS_PCT': 0.03,  # 3% stop loss
    'TRAILING_STOP_PCT': 0.02,      # 2% trailing stop
    
    # Kelly Criterion - MORE aggressive for growth
    'KELLY_FRACTION': 0.50,  # Use 50% of full Kelly (more aggressive)
    'MIN_KELLY_SIZE': 0.20,  # Minimum 20% position (₹2,000)
    'MAX_KELLY_SIZE': 0.35,  # Maximum 35% position (₹3,500)
    
    # Diversification - LESS restrictive with 3 positions
    'MAX_SECTOR_EXPOSURE': 0.50,  # 50% max in single sector
    'MAX_CORRELATION': 0.80,      # Allow higher correlation
    
    # Risk metrics
    'VAR_CONFIDENCE': 0.95,  # 95% Value at Risk
    'CVAR_ALPHA': 0.05,      # 5% CVaR (Expected Shortfall)
    
    # Reward/Risk requirements - HIGHER for selectivity
    'MIN_RISK_REWARD': 2.0,  # Minimum 2:1 reward/risk (more selective)
    'TARGET_RISK_REWARD': 3.0,  # Target 3:1 reward/risk
    
    # Position duration - SHORTER for small capital
    'MAX_HOLDING_DAYS': 3,  # Max 3 trading days
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIGNAL GENERATION PARAMETERS - HIGHLY SELECTIVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SIGNAL_PARAMS = {
    # Confidence thresholds - HIGHER for quality
    'MIN_CONFIDENCE': 0.70,  # Minimum 70% confidence (more selective)
    'HIGH_CONFIDENCE': 0.80,  # High confidence threshold
    
    # ML model thresholds
    'MIN_ML_PROBABILITY': 0.65,  # Minimum 65% ML probability
    'MIN_ENSEMBLE_AGREEMENT': 0.75,  # 75% ensemble agreement
    
    # Technical requirements
    'MIN_VOLUME_RATIO': 1.5,  # Volume 1.5x average (more liquid)
    'MIN_ATR_MULTIPLE': 0.8,  # Minimum 0.8 ATR move expected
    
    # Signal limits - TOP 3 ONLY
    'MAX_SIGNALS_PER_DAY': 3,  # Max 3 signals (one per position)
    'MAX_SIGNALS_PER_STOCK': 1,  # Max 1 signal per stock per day
    
    # Pattern requirements
    'MIN_PATTERN_STRENGTH': 0.70,  # 70% pattern strength
    'REQUIRE_PATTERN_CONFIRMATION': True,  # Require pattern + ML agreement
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATTERN DETECTION CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN_CONFIG = {
    # Lookback periods
    'LOOKBACK_DAYS': 60,  # Days to look back for patterns
    'SR_LOOKBACK': 50,    # Support/Resistance lookback
    'PIVOT_LOOKBACK': 20,  # Pivot points lookback
    
    # Pattern strength
    'MIN_PATTERN_STRENGTH': 0.70,  # 70% minimum
    'MIN_PATTERN_OCCURRENCES': 2,
    
    # Support/Resistance
    'SUPPORT_RESISTANCE_TOLERANCE': 0.02,  # 2% tolerance
    'MIN_SR_TOUCHES': 3,  # Minimum 3 touches for valid S/R
    'SR_PROXIMITY_PCT': 0.03,  # Within 3% of S/R to be "at level"
    
    # Trendlines
    'MIN_TRENDLINE_POINTS': 3,
    'TRENDLINE_TOLERANCE': 0.015,  # 1.5% tolerance
    
    # Fibonacci
    'FIB_LEVELS': [0.236, 0.382, 0.5, 0.618, 0.786],
    'FIB_TOLERANCE': 0.01,  # 1% tolerance
    
    # Candlestick patterns
    'ENABLE_CANDLESTICK_PATTERNS': True,
    'MIN_CANDLE_SIZE': 0.005,  # Minimum 0.5% body size
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TECHNICAL INDICATOR PARAMETERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INDICATOR_PARAMS = {
    # Moving Averages
    'SMA_PERIODS': [10, 20, 50, 200],
    'EMA_PERIODS': [12, 26, 50],
    
    # Momentum Indicators
    'RSI_PERIOD': 14,
    'RSI_OVERBOUGHT': 70,
    'RSI_OVERSOLD': 30,
    
    # MACD
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    
    # Bollinger Bands
    'BB_PERIOD': 20,
    'BB_STD': 2,
    
    # ADX
    'ADX_PERIOD': 14,
    'ADX_STRONG_TREND': 25,
    
    # ATR
    'ATR_PERIOD': 14,
    
    # Stochastic
    'STOCH_K': 14,
    'STOCH_D': 3,
    'STOCH_OVERBOUGHT': 80,
    'STOCH_OVERSOLD': 20,
    
    # CCI
    'CCI_PERIOD': 20,
    'CCI_OVERBOUGHT': 100,
    'CCI_OVERSOLD': -100,
    
    # MFI
    'MFI_PERIOD': 14,
    'MFI_OVERBOUGHT': 80,
    'MFI_OVERSOLD': 20,
    
    # Williams %R
    'WILLIAMS_PERIOD': 14,
    'WILLIAMS_OVERBOUGHT': -20,
    'WILLIAMS_OVERSOLD': -80,
    
    # ROC (Rate of Change)
    'ROC_PERIOD': 12,
    
    # TSI (True Strength Index)
    'TSI_LONG': 25,
    'TSI_SHORT': 13,
    
    # Volume indicators
    'VOLUME_SMA': 20,
    'OBV_ENABLED': True,
    'CMF_PERIOD': 20,
    'VWAP_ENABLED': True,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MACHINE LEARNING CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ML_CONFIG = {
    # Data split
    'TRAIN_TEST_SPLIT': 0.2,  # 80/20 train/test split
    'VALIDATION_SPLIT': 0.2,   # 20% of train for validation
    'RANDOM_STATE': 42,
    
    # Cross-validation
    'CV_FOLDS': 5,  # 5-fold cross-validation
    'CV_STRATEGY': 'StratifiedKFold',
    
    # Ensemble configuration
    'ENSEMBLE_METHOD': 'dynamic',  # 'dynamic', 'voting', 'stacking'
    'DYNAMIC_LOOKBACK': 20,  # Days to track performance
    'MIN_MODEL_ACCURACY': 0.55,  # Minimum accuracy to keep model active
    
    # Feature selection
    'FEATURE_SELECTION': False,
    'MAX_FEATURES': 57,  # Total features
    
    # Class imbalance handling
    'HANDLE_IMBALANCE': True,
    'IMBALANCE_METHOD': 'class_weight',  # 'class_weight' or 'smote'
    
    # Model optimization
    'ENABLE_OPTIMIZATION': False,  # Hyperparameter tuning (slow)
    'OPTIMIZATION_TRIALS': 20,
}

# Deep Learning Configuration
DL_SEQUENCE_LENGTH = 60  # 60-day sequences

DL_CONFIG = {
    'SEQUENCE_LENGTH': 60,
    'BATCH_SIZE': 32,
    'EPOCHS': 100,
    'LEARNING_RATE': 0.0005,
    'EARLY_STOPPING_PATIENCE': 15,
    'REDUCE_LR_PATIENCE': 5,
    'REDUCE_LR_FACTOR': 0.5,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE DEFINITIONS (57 FEATURES TOTAL)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Base Technical Indicators (30 features)
ML_FEATURES_BASE = [
    # Moving Averages
    'sma_10', 'sma_20', 'sma_50', 'sma_200',
    'ema_12', 'ema_26', 'ema_50',
    
    # Momentum Indicators
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'stoch_k', 'stoch_d', 'williams_r', 'roc', 'tsi',
    
    # Volatility Indicators
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'atr',
    
    # Trend Indicators
    'adx', 'cci',
    
    # Volume Indicators
    'mfi', 'obv', 'cmf', 'volume_ratio', 'vwap', 'pvt', 'force_index',
]

# Pattern Features (20 features)
ML_FEATURES_PATTERNS = [
    # Support/Resistance
    'distance_to_support_pct', 'distance_to_resistance_pct',
    'in_support_zone', 'in_resistance_zone',
    'support_strength_score', 'resistance_strength_score',
    
    # Pivot Points
    'above_pivot', 'pivot_position_score',
    
    # Fibonacci
    'nearest_fib_distance_pct', 'fib_retracement_level',
    
    # Chart Patterns
    'bullish_pattern_detected', 'bearish_pattern_detected',
    'pattern_confidence_score',
    
    # Trendlines
    'above_support_trendline', 'below_resistance_trendline',
    'trend_strength', 'higher_highs_lows', 'lower_highs_lows',
    
    # Candlestick Patterns
    'bullish_engulfing', 'bearish_engulfing',
]

# Dynamic Features (7 features - from ML/DL models)
ML_FEATURES_DYNAMIC = [
    # Time Series Forecasts
    'arima_forecast_1d', 'arima_confidence',
    'egarch_volatility', 'direction_prob',
    
    # Deep Learning Predictions
    'cnn_lstm_prob', 'transformer_prob', 'hybrid_lstm_trans_prob',
]

# Complete Feature Set (57 features total)
ML_FEATURES = ML_FEATURES_BASE + ML_FEATURES_PATTERNS + ML_FEATURES_DYNAMIC

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIME SERIES MODEL CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIME_SERIES_CONFIG = {
    # ARIMA
    'ARIMA_ORDER': (5, 1, 2),  # (p, d, q)
    'ARIMA_SEASONAL': False,
    
    # GARCH/EGARCH
    'GARCH_P': 1,  # Lag order
    'GARCH_Q': 1,  # MA order
    'USE_EGARCH': True,  # Use EGARCH (better for stocks)
    
    # LSTM for time series
    'TS_LSTM_UNITS': 64,
    'TS_SEQUENCE_LENGTH': 60,
    
    # Hybrid weighting
    'ARIMA_WEIGHT': 0.33,
    'LSTM_WEIGHT': 0.33,
    'EGARCH_WEIGHT': 0.34,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SENTIMENT ANALYSIS (Optional - Disabled by default)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SENTIMENT_CONFIG = {
    'ENABLED': False,  # Set to True to enable
    
    # News API
    'NEWS_API_KEY': None,  # Add your API key
    'NEWS_SOURCES': ['moneycontrol', 'economictimes', 'business-standard'],
    'MAX_NEWS_ITEMS': 10,
    'NEWS_LOOKBACK_DAYS': 3,
    
    # Sentiment weighting
    'SENTIMENT_WEIGHT': 0.05,  # 5% weight in signals
    'MIN_SENTIMENT_SCORE': -0.3,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TELEGRAM ALERTS (Optional - Disabled by default)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TELEGRAM_CONFIG = {
    'ENABLED': False,  # Set to True to enable
    
    # Bot credentials (get from @BotFather)
    'BOT_TOKEN': None,  # Your bot token
    'CHAT_ID': None,    # Your chat ID
    
    # Alert settings
    'ALERT_ON_SIGNAL': True,
    'ALERT_ON_TRADE': True,
    'ALERT_ON_ERROR': True,
    'DAILY_SUMMARY': True,
    'SUMMARY_TIME': '18:00',
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BACKTEST CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BACKTEST_CONFIG = {
    # Test period
    'START_DATE': '2022-01-01',
    'END_DATE': '2024-12-31',
    
    # Capital - UPDATED TO ₹10K
    'INITIAL_CAPITAL': 10000,  # ₹10,000
    
    # Transaction costs
    'COMMISSION_PCT': 0.001,  # 0.1% commission
    'SLIPPAGE_PCT': 0.002,    # 0.2% slippage (higher for small capital)
    'IMPACT_COST_PCT': 0.001, # 0.1% impact cost
    
    # Execution
    'FILL_PROBABILITY': 0.90,  # 90% order fill rate
    'PARTIAL_FILL': False,
    
    # Performance tracking
    'BENCHMARK': '^NSEI',  # Nifty 50
    'RISK_FREE_RATE': 0.07,  # 7% risk-free rate
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA MANAGEMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATA_CONFIG = {
    # Data source
    'DATA_SOURCE': 'yfinance',
    'FALLBACK_SOURCE': 'nsepy',
    
    # Fetch settings
    'MAX_RETRIES': 5,
    'RETRY_DELAY': 5,  # seconds
    'TIMEOUT': 30,
    
    # Cache settings
    'ENABLE_CACHE': True,
    'CACHE_EXPIRY_DAYS': 1,
    
    # Validation
    'MIN_DATA_POINTS': 200,
    'CHECK_DATA_QUALITY': True,
    'REMOVE_OUTLIERS': True,
    'OUTLIER_THRESHOLD': 4,  # Z-score
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOGGING CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LOGGING_CONFIG = {
    'LEVEL': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'DATE_FORMAT': '%Y-%m-%d %H:%M:%S',
    
    # File logging
    'LOG_TO_FILE': True,
    'LOG_FILE': str(PROJECT_ROOT / 'logs' / 'trading_system.log'),
    'MAX_LOG_SIZE': 10 * 1024 * 1024,  # 10 MB
    'BACKUP_COUNT': 5,
    
    # Console logging
    'LOG_TO_CONSOLE': True,
    'CONSOLE_LEVEL': 'INFO',
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PERFORMANCE OPTIMIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTIMIZATION_CONFIG = {
    # Parallel processing
    'ENABLE_MULTIPROCESSING': False,
    'NUM_WORKERS': 4,
    
    # Memory management
    'BATCH_PROCESSING': True,
    'BATCH_SIZE': 10,
    
    # GPU acceleration
    'USE_GPU': False,
    'GPU_MEMORY_FRACTION': 0.7,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SYSTEM VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Validate critical settings
assert CAPITAL_CONFIG['TOTAL_CAPITAL'] > 0, "Total capital must be positive"
assert CAPITAL_CONFIG['MAX_POSITIONS'] > 0, "Max positions must be positive"
assert 0 < RISK_PARAMS['MAX_SINGLE_POSITION'] <= 1, "Max position size must be 0-1"
assert len(TRADE_UNIVERSE) > 0, "Trade universe cannot be empty"
assert len(ML_FEATURES) == len(ML_FEATURES_BASE) + len(ML_FEATURES_PATTERNS) + len(ML_FEATURES_DYNAMIC), "Feature count mismatch"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION SUMMARY (for verification)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    print("=" * 80)
    print("INSTITUTIONAL TRADING SYSTEM - CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"\n📁 Project Root: {PROJECT_ROOT}")
    print(f"\n💰 CAPITAL CONFIGURATION:")
    print(f"   Total Capital: ₹{CAPITAL_CONFIG['TOTAL_CAPITAL']:,}")
    print(f"   Max Positions: {CAPITAL_CONFIG['MAX_POSITIONS']}")
    print(f"   Position Sizes: {CAPITAL_CONFIG['TOP_POSITION_SIZE']:.0%}, {CAPITAL_CONFIG['SECOND_POSITION_SIZE']:.0%}, {CAPITAL_CONFIG['THIRD_POSITION_SIZE']:.0%}")
    print(f"\n⚠️  RISK PARAMETERS:")
    print(f"   Max Single Position: {RISK_PARAMS['MAX_SINGLE_POSITION']:.0%}")
    print(f"   Max Portfolio Risk: {RISK_PARAMS['MAX_PORTFOLIO_RISK']:.1%}")
    print(f"   Default Stop Loss: {RISK_PARAMS['DEFAULT_STOP_LOSS_PCT']:.1%}")
    print(f"   Min Risk/Reward: {RISK_PARAMS['MIN_RISK_REWARD']:.1f}:1")
    print(f"\n📊 TRADING SETUP:")
    print(f"   Stocks: {len(TRADE_UNIVERSE)}")
    print(f"   Total Features: {len(ML_FEATURES)}")
    print(f"   DL Sequence Length: {DL_SEQUENCE_LENGTH} days")
    print(f"   Ensemble Method: {ML_CONFIG['ENSEMBLE_METHOD']}")
    print(f"\n📈 BACKTEST:")
    print(f"   Period: {BACKTEST_CONFIG['START_DATE']} to {BACKTEST_CONFIG['END_DATE']}")
    print(f"   Initial Capital: ₹{BACKTEST_CONFIG['INITIAL_CAPITAL']:,}")
    print(f"\n📁 DIRECTORIES:")
    for name, path in PROJECT_DIRECTORIES.items():
        if name != 'METADATA_DB':
            print(f"   {name}: {path}")
    print("\n" + "=" * 80)
    print("✅ Configuration loaded successfully!")
    print("=" * 80 + "\n")
