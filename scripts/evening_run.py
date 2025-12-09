"""
Evening Run - Generate Trading Signals
COMPLETE VERSION with position sizing fix and hybrid models
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

# --- Add project root to path ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# ---------------------------------

from config.settings import (
    TRADE_UNIVERSE, CAPITAL_CONFIG, SIGNAL_PARAMS, PATTERN_CONFIG,
    ML_CONFIG, SENTIMENT_CONFIG, PROJECT_DIRECTORIES,
    ML_FEATURES, ML_FEATURES_BASE, ML_FEATURES_PATTERNS,
    RISK_PARAMS, DL_SEQUENCE_LENGTH
)

from src.data_manager import DataManager
from src.technical_engine import TechnicalEngine
from src.ml_ensemble import Enhanced13ModelEnsemble
from src.pattern_detector import PatternDetector
from src.sentiment_analyzer import SentimentAnalyzer
from src.risk_manager import RiskManager
from src.alerts import AlertManager
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


class EveningSignalGenerator:
    """
    Complete evening signal generator with:
    - Hybrid LSTM-Transformer predictions
    - ARIMA-LSTM-EGARCH forecasts
    - Dynamic weighted ensemble
    - FIXED position sizing (remaining capital tracking)
    - Pattern detection
    - Sentiment analysis
    - Risk management
    """
    
    def __init__(self):
        # Initialize components
        self.data_manager = DataManager()
        self.tech_engine = TechnicalEngine()
        self.pattern_detector = PatternDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.ts_engine = TimeSeriesEngine()
        
        # Risk manager with initial capital
        total_capital = CAPITAL_CONFIG['TOTAL_CAPITAL']
        self.risk_manager = RiskManager(total_capital=total_capital)
        
        # Alert manager (optional)
        try:
            self.alert_manager = AlertManager()
            self.alerts_enabled = True
        except:
            self.alert_manager = None
            self.alerts_enabled = False
            logger.warning("Alert manager disabled")
        
        # Directories
        self.model_dir = Path(PROJECT_DIRECTORIES['MODELS'])
        self.feature_store_dir = Path(PROJECT_DIRECTORIES['FEATURE_STORE'])
        self.signals_dir = Path(PROJECT_DIRECTORIES['SIGNALS'])
        self.signals_dir.mkdir(exist_ok=True)
        
        # Models (lazy loading)
        self.ml_ensemble = None
        self.cnn_lstm_model = None
        self.transformer_model = None
        self.hybrid_model = None
        self.sequence_scaler = None
        
        self.sequence_length = DL_SEQUENCE_LENGTH
        
        logger.info("✓ Evening Signal Generator initialized")
        logger.info(f"   Capital: ₹{total_capital:,.0f}")
        logger.info(f"   Max positions: {CAPITAL_CONFIG['MAX_POSITIONS']}")
    
    def _load_models(self):
        """Load all ML/DL models"""
        if self.ml_ensemble is not None:
            return  # Already loaded
        
        logger.info("Loading models...")
        
        # Load ML Ensemble
        ensemble_path = self.model_dir / 'ENSEMBLE_ensemble.pkl'
        if ensemble_path.exists():
            try:
                self.ml_ensemble = Enhanced13ModelEnsemble.load(str(ensemble_path))
                logger.info("   ✓ ML Ensemble loaded")
            except Exception as e:
                logger.error(f"   ML Ensemble load failed: {e}")
        
        # Load sequence models
        try:
            cnn_lstm_path = self.model_dir / 'cnn_lstm_model.h5'
            if cnn_lstm_path.exists():
                with custom_object_scope({'TransformerBlock': TransformerBlock}):
                    self.cnn_lstm_model = load_model(str(cnn_lstm_path))
                logger.info("   ✓ CNN-LSTM loaded")
            
            transformer_path = self.model_dir / 'transformer_model.h5'
            if transformer_path.exists():
                with custom_object_scope({'TransformerBlock': TransformerBlock}):
                    self.transformer_model = load_model(str(transformer_path))
                logger.info("   ✓ Transformer loaded")
            
            hybrid_path = self.model_dir / 'hybrid_lstm_transformer_model.h5'
            if hybrid_path.exists():
                with custom_object_scope({'TransformerBlock': TransformerBlock}):
                    self.hybrid_model = load_model(str(hybrid_path))
                logger.info("   ✓ Hybrid LSTM-Transformer loaded")
            
            scaler_path = self.model_dir / 'sequence_scaler.pkl'
            if scaler_path.exists():
                self.sequence_scaler = joblib.load(str(scaler_path))
                logger.info("   ✓ Scaler loaded")
        
        except Exception as e:
            logger.error(f"   Sequence models load failed: {e}")
    
    def load_features(self, symbol: str) -> pd.DataFrame:
        """Load pre-built features from feature store"""
        feature_file = self.feature_store_dir / f"{symbol.replace('.NS', '')}_features.pkl"
        
        if not feature_file.exists():
            logger.warning(f"   No features for {symbol}")
            return None
        
        try:
            features = pd.read_pickle(str(feature_file))
            return features
        except Exception as e:
            logger.error(f"   Error loading features for {symbol}: {e}")
            return None
    
    def calculate_stop_loss_target(self, current_price: float, 
                                   volatility_forecast: float,
                                   confidence: float) -> dict:
        """
        Calculate stop loss and target based on volatility
        
        Args:
            current_price: Current stock price
            volatility_forecast: Forecasted volatility (%)
            confidence: Model confidence (0-1)
        
        Returns:
            Dict with stop_loss, target, risk_reward_ratio
        """
        # Risk amount based on volatility
        vol_multiplier = 2.0 if confidence > 0.7 else 2.5
        risk_amount = current_price * (volatility_forecast / 100) * vol_multiplier
        
        # Stop loss
        stop_loss = current_price - risk_amount
        
        # Target (reward/risk ratio based on confidence)
        if confidence > 0.75:
            reward_mult = 2.5  # High confidence = higher target
        elif confidence > 0.65:
            reward_mult = 2.0
        else:
            reward_mult = 1.5
        
        reward_amount = risk_amount * reward_mult
        target = current_price + reward_amount
        
        return {
            'stop_loss': stop_loss,
            'target': target,
            'risk_amount': risk_amount,
            'reward_amount': reward_amount,
            'reward_risk_ratio': reward_mult
        }
    
    def generate_signal(self, symbol: str, date: str = None) -> dict:
        """
        Generate complete trading signal for a symbol
        
        Returns:
            Signal dict or None
        """
        try:
            # Load features
            features_df = self.load_features(symbol)
            
            if features_df is None or len(features_df) == 0:
                return None
            
            # Get latest features (most recent day)
            if date:
                try:
                    latest_features = features_df.loc[date]
                except:
                    latest_features = features_df.iloc[-1]
            else:
                latest_features = features_df.iloc[-1]
            
            # Get current price
            df_price = self.data_manager.fetch_data_with_retry(symbol, period='5d')
            if df_price is None or len(df_price) == 0:
                return None
            
            current_price = float(df_price['close'].iloc[-1])
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 1: ML Ensemble Prediction
            # ═══════════════════════════════════════════════════════════════
            
            ml_prediction = None
            if self.ml_ensemble:
                try:
                    # Prepare features for ML
                    feature_vector = latest_features[ML_FEATURES].values.reshape(1, -1)
                    ml_prediction = self.ml_ensemble.predict(feature_vector)
                except Exception as e:
                    logger.warning(f"   ML prediction failed for {symbol}: {e}")
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 2: Time Series Forecast (ARIMA-LSTM-EGARCH)
            # ═══════════════════════════════════════════════════════════════
            
            ts_forecast = None
            try:
                prices = df_price['close']
                ts_forecast = self.ts_engine.forecast_hybrid(prices)
            except Exception as e:
                logger.warning(f"   Time series forecast failed for {symbol}: {e}")
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 3: Deep Learning Predictions
            # ═══════════════════════════════════════════════════════════════
            
            dl_predictions = {
                'cnn_lstm': latest_features.get('cnn_lstm_prob', 0.5),
                'transformer': latest_features.get('transformer_prob', 0.5),
                'hybrid': latest_features.get('hybrid_lstm_trans_prob', 0.5)
            }
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 4: Pattern Analysis
            # ═══════════════════════════════════════════════════════════════
            
            pattern_score = 0.5
            try:
                # Calculate pattern score from features
                bullish_patterns = latest_features.get('bullish_pattern_detected', 0)
                bearish_patterns = latest_features.get('bearish_pattern_detected', 0)
                support_strength = latest_features.get('support_strength_score', 0)
                
                if bullish_patterns > 0 and support_strength > 2:
                    pattern_score = 0.7
                elif bearish_patterns > 0:
                    pattern_score = 0.3
            except:
                pass
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 5: Sentiment Analysis (Optional)
            # ═══════════════════════════════════════════════════════════════
            
            sentiment_score = 0.5
            try:
                if SENTIMENT_CONFIG.get('ENABLED', False):
                    sentiment = self.sentiment_analyzer.analyze_stock_sentiment(symbol)
                    sentiment_score = sentiment.get('compound_score', 0.5)
            except:
                pass
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 6: Combine All Signals (Weighted Ensemble)
            # ═══════════════════════════════════════════════════════════════
            
            weights = {
                'ml': 0.30,
                'ts': 0.20,
                'dl_hybrid': 0.25,
                'dl_cnn': 0.10,
                'dl_trans': 0.05,
                'pattern': 0.05,
                'sentiment': 0.05
            }
            
            # Calculate weighted score
            scores = {
                'ml': ml_prediction['probability'] if ml_prediction else 0.5,
                'ts': ts_forecast['direction_probability'] if ts_forecast else 0.5,
                'dl_hybrid': dl_predictions['hybrid'],
                'dl_cnn': dl_predictions['cnn_lstm'],
                'dl_trans': dl_predictions['transformer'],
                'pattern': pattern_score,
                'sentiment': sentiment_score
            }
            
            final_score = sum(scores[k] * weights[k] for k in scores.keys())
            
            # Calculate confidence (inverse of prediction variance)
            score_variance = np.var(list(scores.values()))
            confidence = max(0.3, 1.0 - score_variance)
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 7: Generate Signal
            # ═══════════════════════════════════════════════════════════════
            
            # Signal threshold
            buy_threshold = SIGNAL_PARAMS.get('MIN_CONFIDENCE', 0.65)
            
            if final_score < buy_threshold:
                return None  # No signal
            
            # Calculate stop loss and target
            volatility = ts_forecast['volatility_forecast'] if ts_forecast else 2.0
            sl_target = self.calculate_stop_loss_target(current_price, volatility, confidence)
            
            # Build signal
            signal = {
                'symbol': symbol,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'signal_type': 'BUY',
                'confidence': confidence,
                'final_score': final_score,
                'stop_loss': sl_target['stop_loss'],
                'target': sl_target['target'],
                'reward_risk_ratio': sl_target['reward_risk_ratio'],
                'volatility_forecast': volatility,
                
                # Component scores
                'ml_probability': scores['ml'],
                'ts_probability': scores['ts'],
                'dl_hybrid_prob': scores['dl_hybrid'],
                'dl_cnn_prob': scores['dl_cnn'],
                'dl_trans_prob': scores['dl_trans'],
                'pattern_score': scores['pattern'],
                'sentiment_score': scores['sentiment'],
                
                # Forecasts
                'price_forecast': ts_forecast['price_forecast'] if ts_forecast else current_price,
                'expected_return_pct': ts_forecast['expected_return_pct'] if ts_forecast else 0.0,
                
                # Technical indicators (for reference)
                'rsi': latest_features.get('rsi', 50),
                'macd': latest_features.get('macd', 0),
                'volume_ratio': latest_features.get('volume_ratio', 1.0),
            }
            
            return signal
        
        except Exception as e:
            logger.error(f"   Signal generation failed for {symbol}: {e}")
            return None
    
    def apply_position_sizing(self, signals: list) -> list:
        """
        ✅ FIXED: Apply position sizing with REMAINING CAPITAL tracking
        
        Args:
            signals: List of raw signals
        
        Returns:
            List of signals with position sizing (approved only)
        """
        max_positions = CAPITAL_CONFIG['MAX_POSITIONS']
        total_capital = CAPITAL_CONFIG['TOTAL_CAPITAL']
        remaining_capital = total_capital  # ✅ Track remaining capital
        
        # Sort signals by confidence (best first)
        sorted_signals = sorted(signals, key=lambda x: x['confidence'], reverse=True)
        
        # Take top N signals
        top_signals = sorted_signals[:max_positions]
        
        approved_signals = []
        
        for signal in top_signals:
            # ✅ UPDATE: Set available cash to remaining capital
            self.risk_manager.available_cash = remaining_capital
            
            # Calculate position size
            ml_prob = signal.get('final_score', 0.5)
            
            position_result = self.risk_manager.calculate_position_size(
                signal=signal,
                current_price=signal['current_price'],
                method='kelly',
                ml_probability=ml_prob
            )
            
            # Add position info to signal
            signal.update(position_result)
            
            # Check if approved
            if position_result['validation']['approved']:
                # ✅ DEDUCT from remaining capital
                remaining_capital -= position_result['position_value']
                approved_signals.append(signal)
                
                logger.info(
                    f"   ✓ {signal['symbol']}: Allocated ₹{position_result['position_value']:,.0f}, "
                    f"Remaining: ₹{remaining_capital:,.0f}"
                )
            else:
                logger.warning(
                    f"   ✗ {signal['symbol']}: REJECTED - {position_result['validation']['reason']}"
                )
        
        logger.info(f"\n✓ Approved {len(approved_signals)}/{len(top_signals)} signals")
        logger.info(f"   Capital allocated: ₹{total_capital - remaining_capital:,.0f}")
        logger.info(f"   Capital remaining: ₹{remaining_capital:,.0f}")
        
        return approved_signals
    
    def generate_all_signals(self, symbols: list = None) -> pd.DataFrame:
        """
        Generate signals for all symbols
        
        Returns:
            DataFrame with all signals
        """
        if symbols is None:
            symbols = TRADE_UNIVERSE[:95]
        
        logger.info("=" * 80)
        logger.info("EVENING RUN - GENERATING SIGNALS")
        logger.info("=" * 80)
        logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Symbols: {len(symbols)}")
        
        # Load models
        self._load_models()
        
        # Generate signals
        all_signals = []
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
            
            signal = self.generate_signal(symbol)
            
            if signal:
                all_signals.append(signal)
                logger.info(f"   ✓ Signal generated: Score={signal['final_score']:.3f}, Conf={signal['confidence']:.3f}")
            else:
                logger.info(f"   ✗ No signal")
        
        logger.info(f"\n✓ Generated {len(all_signals)} raw signals")
        
        if len(all_signals) == 0:
            logger.warning("No signals generated!")
            return pd.DataFrame()
        
        # Apply position sizing (with remaining capital tracking)
        approved_signals = self.apply_position_sizing(all_signals)
        
        if len(approved_signals) == 0:
            logger.warning("No signals approved after risk management!")
            return pd.DataFrame()
        
        # Convert to DataFrame
        signals_df = pd.DataFrame(approved_signals)
        
        # Save signals
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        signals_file = self.signals_dir / f'evening_signals_{timestamp}.csv'
        signals_df.to_csv(str(signals_file), index=False)
        
        logger.info(f"\n✓ Signals saved: {signals_file}")
        
        # Send alerts
        if self.alerts_enabled and self.alert_manager:
            try:
                self._send_alerts(approved_signals)
            except Exception as e:
                logger.error(f"Alert sending failed: {e}")
        
        # Print summary
        self._print_summary(signals_df)
        
        return signals_df
    
    def _send_alerts(self, signals: list):
        """Send Telegram alerts for signals"""
        if not signals:
            return
        
        message = f"🎯 *EVENING SIGNALS - {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        message += f"📊 Generated {len(signals)} signals\n\n"
        
        for sig in signals[:5]:  # Top 5
            message += f"*{sig['symbol']}*\n"
            message += f"Price: ₹{sig['current_price']:.2f}\n"
            message += f"Target: ₹{sig['target']:.2f} (+{((sig['target']/sig['current_price'])-1)*100:.1f}%)\n"
            message += f"Stop: ₹{sig['stop_loss']:.2f}\n"
            message += f"Shares: {sig['shares']}\n"
            message += f"Confidence: {sig['confidence']:.1%}\n\n"
        
        self.alert_manager.send_message(message)
    
    def _print_summary(self, signals_df: pd.DataFrame):
        """Print signal summary"""
        logger.info("\n" + "=" * 80)
        logger.info("SIGNAL SUMMARY")
        logger.info("=" * 80)
        
        if len(signals_df) == 0:
            logger.info("No signals to display")
            return
        
        # Sort by confidence
        signals_df_sorted = signals_df.sort_values('confidence', ascending=False)
        
        logger.info(f"\n{'Symbol':<15} {'Price':>10} {'Target':>10} {'Stop':>10} {'Shares':>8} {'Conf':>8}")
        logger.info("-" * 80)
        
        for _, sig in signals_df_sorted.iterrows():
            logger.info(
                f"{sig['symbol']:<15} "
                f"₹{sig['current_price']:>9.2f} "
                f"₹{sig['target']:>9.2f} "
                f"₹{sig['stop_loss']:>9.2f} "
                f"{sig['shares']:>8} "
                f"{sig['confidence']:>7.1%}"
            )
        
        logger.info("-" * 80)
        
        # Statistics
        total_allocation = signals_df['position_value'].sum()
        avg_confidence = signals_df['confidence'].mean()
        avg_rr = signals_df['reward_risk_ratio'].mean()
        
        logger.info(f"\nTotal Allocation: ₹{total_allocation:,.0f}")
        logger.info(f"Average Confidence: {avg_confidence:.1%}")
        logger.info(f"Average R/R Ratio: {avg_rr:.2f}")
        logger.info("=" * 80)


def main():
    """Main execution"""
    generator = EveningSignalGenerator()
    
    # Get symbols
    symbols = TRADE_UNIVERSE[:95]
    
    # Generate signals
    signals_df = generator.generate_all_signals(symbols)
    
    return signals_df


if __name__ == '__main__':
    main()
