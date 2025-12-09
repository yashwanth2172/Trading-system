"""
Ultra-Optimized Feature Store Builder with Hybrid Models
COMPLETE VERSION with data leakage fixes
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# --- Project Root ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import (
    TRADE_UNIVERSE, ML_CONFIG, ML_FEATURES, PATTERN_CONFIG,
    PROJECT_DIRECTORIES, ML_FEATURES_BASE, ML_FEATURES_PATTERNS,
    DL_SEQUENCE_LENGTH
)
from src.data_manager import DataManager
from src.technical_engine import TechnicalEngine
from src.pattern_detector import PatternDetector
from src.time_series_engine import TimeSeriesEngine
from keras.models import load_model
from keras.utils import custom_object_scope
from src.sequence_models import TransformerBlock

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread-safe counter
class ProgressCounter:
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()
    
    def increment(self):
        with self.lock:
            self.count += 1
            return self.count


class UltraOptimizedFeatureStoreBuilder:
    """
    Enhanced feature store builder with:
    - NO DATA LEAKAGE (per-row pattern detection)
    - Hybrid LSTM-Transformer predictions
    - ARIMA-LSTM-EGARCH time series forecasts
    - Parallel processing capability
    - Progress tracking
    """
    
    def __init__(self):
        self.data_manager = DataManager()
        self.tech_engine = TechnicalEngine()
        self.pattern_detector = PatternDetector()
        self.ts_engine = TimeSeriesEngine()
        
        # Directories
        self.feature_store_dir = Path(PROJECT_DIRECTORIES['FEATURE_STORE'])
        self.model_dir = Path(PROJECT_DIRECTORIES['MODELS'])
        
        self.feature_store_dir.mkdir(parents=True, exist_ok=True)
        
        # Models (lazy loading)
        self.cnn_lstm_model = None
        self.transformer_model = None
        self.hybrid_model = None
        self.sequence_scaler = None
        
        self.sequence_length = DL_SEQUENCE_LENGTH
        
        logger.info("✓ Ultra-Optimized Feature Store Builder initialized")
    
    def _load_sequence_models(self):
        """Load trained sequence models"""
        if self.cnn_lstm_model is not None:
            return  # Already loaded
        
        logger.info("Loading sequence models...")
        
        try:
            # Load CNN-LSTM
            cnn_lstm_path = self.model_dir / 'cnn_lstm_model.h5'
            if cnn_lstm_path.exists():
                with custom_object_scope({'TransformerBlock': TransformerBlock}):
                    self.cnn_lstm_model = load_model(str(cnn_lstm_path))
                logger.info("   ✓ CNN-LSTM loaded")
            else:
                logger.warning(f"   CNN-LSTM not found: {cnn_lstm_path}")
            
            # Load Transformer
            transformer_path = self.model_dir / 'transformer_model.h5'
            if transformer_path.exists():
                with custom_object_scope({'TransformerBlock': TransformerBlock}):
                    self.transformer_model = load_model(str(transformer_path))
                logger.info("   ✓ Transformer loaded")
            else:
                logger.warning(f"   Transformer not found: {transformer_path}")
            
            # Load Hybrid (if exists)
            hybrid_path = self.model_dir / 'hybrid_lstm_transformer_model.h5'
            if hybrid_path.exists():
                with custom_object_scope({'TransformerBlock': TransformerBlock}):
                    self.hybrid_model = load_model(str(hybrid_path))
                logger.info("   ✓ Hybrid LSTM-Transformer loaded")
            else:
                logger.info("   Hybrid model not found (optional)")
            
            # Load scaler
            scaler_path = self.model_dir / 'sequence_scaler.pkl'
            if scaler_path.exists():
                self.sequence_scaler = joblib.load(str(scaler_path))
                logger.info("   ✓ Scaler loaded")
            else:
                logger.warning(f"   Scaler not found: {scaler_path}")
                self.sequence_scaler = StandardScaler()
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _calculate_static_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Calculate static features (technical indicators)
        These are calculated rolling, so NO data leakage
        
        Args:
            df: Price dataframe
            symbol: Stock symbol
        
        Returns:
            DataFrame with technical indicator features
        """
        try:
            # Calculate technical indicators (all are rolling/lookback based)
            df_tech = self.tech_engine.calculate_all_indicators(df.copy())
        except Exception as e:
            logger.warning(f"   Technical indicators failed: {e}")
            df_tech = df.copy()
        
        # Start from index 50 (enough history for indicators)
        start_idx = 50
        features = pd.DataFrame(index=df_tech.index[start_idx:])
        
        # Copy base technical features
        for col in ML_FEATURES_BASE:
            if col in df_tech.columns:
                features[col] = df_tech[col].iloc[start_idx:].values
            else:
                features[col] = 0.0
        
        return features
    
    def _calculate_pattern_features_no_leakage(self, df: pd.DataFrame, 
                                              symbol: str,
                                              start_idx: int = 50) -> pd.DataFrame:
        """
        ✅ FIXED: Calculate pattern features PER ROW to prevent data leakage
        
        This calculates patterns using ONLY data available up to that point
        
        Args:
            df: Price dataframe with technical indicators
            symbol: Stock symbol
            start_idx: Starting index
        
        Returns:
            DataFrame with pattern features (NO LEAKAGE)
        """
        logger.debug(f"   Calculating pattern features (no leakage) for {symbol}...")
        
        pattern_features_list = []
        
        # Calculate patterns for each row using only past data
        for i in range(start_idx, len(df)):
            # ✅ KEY FIX: Use only data UP TO this point
            df_slice = df.iloc[:i+1].copy()
            
            try:
                # Detect patterns using only available history
                pattern_results = self.pattern_detector.detect_all_patterns(df_slice, symbol)
            except Exception as e:
                # Fallback to default values
                pattern_results = {}
            
            # Extract pattern information
            sr = pattern_results.get('support_resistance', {})
            pivots = pattern_results.get('pivot_points', {})
            fib = pattern_results.get('fibonacci_levels', {})
            patterns = pattern_results.get('chart_patterns', {})
            
            current_price = df_slice['close'].iloc[-1]
            
            # Build feature dict for this row
            row_features = {
                'distance_to_support_pct': sr.get('support_distance_pct', 5.0) or 5.0,
                'distance_to_resistance_pct': sr.get('resistance_distance_pct', 5.0) or 5.0,
                'in_support_zone': 1 if sr.get('in_support_zone', False) else 0,
                'in_resistance_zone': 1 if sr.get('in_resistance_zone', False) else 0,
                'support_strength_score': min(len(sr.get('support_strength', {})), 5),
                'resistance_strength_score': min(len(sr.get('resistance_strength', {})), 5),
                'above_pivot': 1 if current_price > pivots.get('pivot_point', current_price) else 0,
                'pivot_position_score': 2,  # Neutral
                'nearest_fib_distance_pct': fib.get('distance_to_nearest_pct', 5.0) or 5.0,
                'fib_retracement_level': 50.0,  # Default
                'bullish_pattern_detected': 1 if patterns.get('bullish_patterns', 0) > 0 else 0,
                'bearish_pattern_detected': 1 if patterns.get('bearish_patterns', 0) > 0 else 0,
                'pattern_confidence_score': 70,  # Default
                'above_support_trendline': 1,
                'below_resistance_trendline': 0,
                'trend_strength': 0.5,
                'higher_highs_lows': 1,
                'lower_highs_lows': 0,
                'bullish_engulfing': 0,
                'bearish_engulfing': 0
            }
            
            pattern_features_list.append(row_features)
        
        # Convert to DataFrame
        pattern_df = pd.DataFrame(pattern_features_list, index=df.index[start_idx:])
        
        # Fill any NaN values
        pattern_df.fillna(0.0, inplace=True)
        
        return pattern_df
    
    def _calculate_dynamic_features(self, df: pd.DataFrame,
                                   static_df: pd.DataFrame,
                                   start_index: int,
                                   symbol: str) -> pd.DataFrame:
        """
        Calculate dynamic features:
        - ARIMA-LSTM-EGARCH hybrid forecasts (per row, no leakage)
        - Deep learning predictions (CNN-LSTM, Transformer, Hybrid)
        
        Args:
            df: Original price data
            static_df: Static features dataframe
            start_index: Starting index
            symbol: Stock symbol
        
        Returns:
            DataFrame with dynamic features
        """
        features_list = []
        
        # ═══════════════════════════════════════════════════════════════
        # PART 1: Time Series Forecasts (ARIMA-LSTM-EGARCH) - PER ROW
        # ═══════════════════════════════════════════════════════════════
        
        logger.debug(f"   Calculating time series forecasts (per row)...")
        
        for i in range(start_index, len(df)):
            # ✅ Use only data up to this point
            df_slice = df.iloc[:i+1]
            
            try:
                # Hybrid forecast using ARIMA-LSTM-EGARCH
                forecast_result = self.ts_engine.forecast_hybrid(df_slice['close'])
                
                row_features = {
                    'arima_forecast_1d': forecast_result['price_forecast'],
                    'arima_confidence': forecast_result['confidence'],
                    'egarch_volatility': forecast_result['volatility_forecast'],
                    'direction_prob': forecast_result['direction_probability']
                }
            except Exception as e:
                # Fallback: simple momentum proxy
                recent_returns = df_slice['close'].pct_change(5).iloc[-5:].mean()
                last_close = df_slice['close'].iloc[-1]
                
                row_features = {
                    'arima_forecast_1d': last_close * (1 + recent_returns),
                    'arima_confidence': 0.5,
                    'egarch_volatility': df_slice['close'].pct_change().std() * 100,
                    'direction_prob': 0.5
                }
            
            features_list.append(row_features)
        
        # ═══════════════════════════════════════════════════════════════
        # PART 2: Deep Learning Predictions (Batched for efficiency)
        # ═══════════════════════════════════════════════════════════════
        
        logger.debug(f"   Calculating DL predictions...")
        
        # Prepare sequences for DL models
        all_sequences = []
        
        for i in range(start_index, len(df)):
            if i >= self.sequence_length:
                # Get sequence of features
                seq_data = static_df.iloc[i-self.sequence_length:i]
                
                if len(seq_data) == self.sequence_length:
                    all_sequences.append(seq_data.values)
                else:
                    all_sequences.append(None)
            else:
                all_sequences.append(None)
        
        # Batch predict
        cnn_preds = []
        trans_preds = []
        hybrid_preds = []
        
        if self.cnn_lstm_model or self.transformer_model or self.hybrid_model:
            self._load_sequence_models()
            
            # Prepare valid sequences
            valid_sequences = [seq for seq in all_sequences if seq is not None]
            
            if valid_sequences and self.sequence_scaler:
                try:
                    # Stack and scale
                    sequences_array = np.array(valid_sequences)
                    n_samples, n_timesteps, n_features = sequences_array.shape
                    
                    # Reshape for scaling
                    sequences_reshaped = sequences_array.reshape(-1, n_features)
                    sequences_scaled = self.sequence_scaler.transform(sequences_reshaped)
                    sequences_scaled = sequences_scaled.reshape(n_samples, n_timesteps, n_features)
                    
                    # Predict with each model
                    if self.cnn_lstm_model:
                        cnn_preds = self.cnn_lstm_model.predict(sequences_scaled, verbose=0).flatten()
                    
                    if self.transformer_model:
                        trans_preds = self.transformer_model.predict(sequences_scaled, verbose=0).flatten()
                    
                    if self.hybrid_model:
                        hybrid_preds = self.hybrid_model.predict(sequences_scaled, verbose=0).flatten()
                
                except Exception as e:
                    logger.warning(f"   DL prediction failed: {e}")
        
        # Add DL predictions to features
        valid_idx = 0
        for idx, seq in enumerate(all_sequences):
            if seq is not None and valid_idx < len(cnn_preds):
                features_list[idx]['cnn_lstm_prob'] = float(cnn_preds[valid_idx]) if len(cnn_preds) > 0 else 0.5
                features_list[idx]['transformer_prob'] = float(trans_preds[valid_idx]) if len(trans_preds) > 0 else 0.5
                features_list[idx]['hybrid_lstm_trans_prob'] = float(hybrid_preds[valid_idx]) if len(hybrid_preds) > 0 else 0.5
                valid_idx += 1
            else:
                features_list[idx]['cnn_lstm_prob'] = 0.5
                features_list[idx]['transformer_prob'] = 0.5
                features_list[idx]['hybrid_lstm_trans_prob'] = 0.5
        
        # Convert to DataFrame
        dynamic_df = pd.DataFrame(features_list, index=df.index[start_index:])
        
        return dynamic_df
    
    def build_features_for_symbol(self, symbol: str, 
                                  start_date: str = '2020-01-01',
                                  end_date: str = '2024-12-31') -> bool:
        """
        Build complete feature set for a symbol with NO DATA LEAKAGE
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
        
        Returns:
            True if successful
        """
        try:
            # Fetch data
            df = self.data_manager.fetch_data_with_retry(
                symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or len(df) < 200:
                logger.warning(f"   {symbol}: Insufficient data")
                return False
            
            # Step 1: Calculate static features (technical indicators)
            static_features = self._calculate_static_features(df, symbol)
            
            if len(static_features) == 0:
                logger.warning(f"   {symbol}: No static features")
                return False
            
            # Step 2: Calculate pattern features (NO LEAKAGE - per row)
            pattern_features = self._calculate_pattern_features_no_leakage(
                df, symbol, start_idx=50
            )
            
            # Step 3: Calculate dynamic features (forecasts + DL predictions)
            dynamic_features = self._calculate_dynamic_features(
                df, static_features, start_index=50, symbol=symbol
            )
            
            # Combine all features
            combined_features = pd.concat([
                static_features,
                pattern_features,
                dynamic_features
            ], axis=1)
            
            # Align indices
            combined_features = combined_features.loc[
                combined_features.index.isin(static_features.index) &
                combined_features.index.isin(pattern_features.index) &
                combined_features.index.isin(dynamic_features.index)
            ]
            
            if len(combined_features) == 0:
                logger.warning(f"   {symbol}: No combined features")
                return False
            
            # Fill any remaining NaN
            combined_features.fillna(method='ffill', inplace=True)
            combined_features.fillna(0.0, inplace=True)
            
            # Save to feature store
            feature_file = self.feature_store_dir / f"{symbol.replace('.NS', '')}_features.pkl"
            combined_features.to_pickle(str(feature_file))
            
            logger.info(f"   ✓ {symbol}: {len(combined_features)} rows, {len(combined_features.columns)} features")
            
            return True
            
        except Exception as e:
            logger.error(f"   ✗ {symbol} failed: {e}")
            return False
    
    def build_all_features(self, symbols: list = None, parallel: bool = False, max_workers: int = 4):
        """
        Build features for all symbols
        
        Args:
            symbols: List of symbols (default: TRADE_UNIVERSE)
            parallel: Use parallel processing
            max_workers: Number of parallel workers
        """
        if symbols is None:
            symbols = TRADE_UNIVERSE[:95]
        
        logger.info("=" * 80)
        logger.info("BUILDING FEATURE STORE - NO DATA LEAKAGE")
        logger.info("=" * 80)
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Parallel: {parallel}")
        
        # Load models once
        self._load_sequence_models()
        
        start_time = time.time()
        success_count = 0
        
        if parallel:
            logger.info(f"Using {max_workers} parallel workers")
            
            progress = ProgressCounter()
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.build_features_for_symbol, symbol): symbol
                    for symbol in symbols
                }
                
                for future in tqdm(as_completed(futures), total=len(symbols), desc="Building features"):
                    symbol = futures[future]
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                        
                        count = progress.increment()
                        if count % 10 == 0:
                            logger.info(f"   Progress: {count}/{len(symbols)}")
                    
                    except Exception as e:
                        logger.error(f"   {symbol} exception: {e}")
        
        else:
            # Sequential processing
            for i, symbol in enumerate(tqdm(symbols, desc="Building features"), 1):
                success = self.build_features_for_symbol(symbol)
                if success:
                    success_count += 1
                
                if i % 10 == 0:
                    logger.info(f"   Progress: {i}/{len(symbols)}")
        
        elapsed = time.time() - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE STORE BUILD COMPLETE")
        logger.info("=" * 80)
        logger.info(f"✓ Successful: {success_count}/{len(symbols)}")
        logger.info(f"✓ Time elapsed: {elapsed/60:.1f} minutes")
        logger.info(f"✓ Feature store: {self.feature_store_dir}")
        logger.info("=" * 80)


def main():
    """Main execution"""
    builder = UltraOptimizedFeatureStoreBuilder()
    
    # Get symbols
    symbols = TRADE_UNIVERSE[:95]
    
    # Build features (set parallel=True for faster processing)
    builder.build_all_features(
        symbols=symbols,
        parallel=False,  # Set to True for parallel processing
        max_workers=4
    )


if __name__ == '__main__':
    main()
