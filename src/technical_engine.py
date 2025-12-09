
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from config.settings import INDICATOR_PARAMS

logger = logging.getLogger(__name__)


class TechnicalEngine:
    
    def __init__(self):
        self.params = INDICATOR_PARAMS
        logger.info("✓ Technical Engine initialized")
    
    # ========== PUBLIC METHODS ==========
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict:
        if df is None or df.empty or len(df) < 20:
            return self._no_signal(symbol, "Insufficient data")
        
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            # Analyze components
            trend = self._analyze_trend(df)
            momentum = self._analyze_momentum(df)
            volatility = self._analyze_volatility(df)
            volume = self._analyze_volume(df)
            
            # Determine signal direction
            bullish_score = 0
            bearish_score = 0
            reasons = []
            
            # Trend indicators
            if trend['direction'] == 'BULLISH':
                bullish_score += trend['strength']
                reasons.append(f"Bullish trend (strength: {trend['strength']:.1f})")
            elif trend['direction'] == 'BEARISH':
                bearish_score += trend['strength']
                reasons.append(f"Bearish trend (strength: {trend['strength']:.1f})")
            
            # Momentum indicators
            if momentum['status'] == 'BULLISH':
                bullish_score += 1
                reasons.append(f"Bullish momentum (RSI: {momentum['rsi']:.0f})")
            elif momentum['status'] == 'BEARISH':
                bearish_score += 1
                reasons.append(f"Bearish momentum (RSI: {momentum['rsi']:.0f})")
            
            # MACD signal
            if momentum['macd_signal'] == 'BULLISH':
                bullish_score += 1
                reasons.append("MACD bullish crossover")
            elif momentum['macd_signal'] == 'BEARISH':
                bearish_score += 1
                reasons.append("MACD bearish crossover")
            
            # Volume confirmation
            if volume['status'] == 'HIGH':
                if bullish_score > bearish_score:
                    bullish_score += 0.5
                    reasons.append("High volume confirmation")
                elif bearish_score > bullish_score:
                    bearish_score += 0.5
                    reasons.append("High volume confirmation")
            
            # Volatility assessment
            if volatility['level'] == 'LOW':
                # Low volatility = potential breakout
                if bullish_score > bearish_score:
                    reasons.append("Low volatility (potential upside)")
                elif bearish_score > bullish_score:
                    reasons.append("Low volatility (potential downside)")
            
            # Determine final signal
            score_diff = abs(bullish_score - bearish_score)
            min_score_diff = 1.5  # Minimum difference to generate signal
            
            if bullish_score > bearish_score and score_diff >= min_score_diff:
                signal_type = 'BUY'
                confidence = min(0.95, 0.50 + (score_diff * 0.15))
            elif bearish_score > bullish_score and score_diff >= min_score_diff:
                signal_type = 'SELL'
                confidence = min(0.95, 0.50 + (score_diff * 0.15))
            else:
                return self._no_signal(symbol, "Conflicting signals")
            
            # Calculate entry, target, and stop loss
            current_price = latest['close']
            atr = latest.get('atr', current_price * 0.02)
            
            if signal_type == 'BUY':
                entry_price = current_price
                stop_loss = current_price - (2 * atr)
                target = current_price + (3 * atr)  # 1.5:1 reward/risk
            else:  # SELL
                entry_price = current_price
                stop_loss = current_price + (2 * atr)
                target = current_price - (3 * atr)
            
            return {
                'symbol': symbol,
                'signal': signal_type,
                'confidence': confidence,
                'entry_price': float(entry_price),
                'target': float(target),
                'stop_loss': float(stop_loss),
                'current_price': float(current_price),
                'reasons': reasons,
                'indicators': {
                    'trend': trend,
                    'momentum': momentum,
                    'volatility': volatility,
                    'volume': volume,
                    'rsi': float(latest.get('rsi', 50)),
                    'macd': float(latest.get('macd', 0)),
                    'atr': float(atr),
                },
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return self._no_signal(symbol, f"Error: {str(e)}")
    
    def batch_generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        signals = []
        
        for symbol, df in market_data.items():
            try:
                signal = self.generate_signal(df, symbol)
                if signal['signal'] != 'HOLD':
                    signals.append(signal)
            except Exception as e:
                logger.warning(f"Failed to generate signal for {symbol}: {e}")
                continue
        
        logger.info(f"Generated {len(signals)} actionable signals from {len(market_data)} symbols")
        return signals
    
    # ========== ANALYSIS METHODS ==========
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Analyze trend direction and strength"""
        try:
            latest = df.iloc[-1]
            
            # Get moving averages
            sma_20 = latest.get('sma_20', latest['close'])
            sma_50 = latest.get('sma_50', latest['close'])
            sma_200 = latest.get('sma_200', latest['close'])
            price = latest['close']
            
            # Determine trend
            if price > sma_20 > sma_50 > sma_200:
                direction = 'BULLISH'
                strength = 3.0
            elif price > sma_20 > sma_50:
                direction = 'BULLISH'
                strength = 2.0
            elif price > sma_20:
                direction = 'BULLISH'
                strength = 1.0
            elif price < sma_20 < sma_50 < sma_200:
                direction = 'BEARISH'
                strength = 3.0
            elif price < sma_20 < sma_50:
                direction = 'BEARISH'
                strength = 2.0
            elif price < sma_20:
                direction = 'BEARISH'
                strength = 1.0
            else:
                direction = 'NEUTRAL'
                strength = 0.0
            
            return {
                'direction': direction,
                'strength': strength,
                'sma_20': float(sma_20),
                'sma_50': float(sma_50),
                'sma_200': float(sma_200)
            }
        
        except Exception as e:
            logger.debug(f"Trend analysis error: {e}")
            return {'direction': 'NEUTRAL', 'strength': 0.0}
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze momentum indicators"""
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            rsi = latest.get('rsi', 50)
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            
            # RSI status
            if rsi > 70:
                rsi_status = 'OVERBOUGHT'
            elif rsi < 30:
                rsi_status = 'OVERSOLD'
            elif rsi > 50:
                rsi_status = 'BULLISH'
            elif rsi < 50:
                rsi_status = 'BEARISH'
            else:
                rsi_status = 'NEUTRAL'
            
            # MACD signal
            prev_macd = prev.get('macd', 0)
            prev_signal = prev.get('macd_signal', 0)
            
            if macd > macd_signal and prev_macd <= prev_signal:
                macd_crossover = 'BULLISH'
            elif macd < macd_signal and prev_macd >= prev_signal:
                macd_crossover = 'BEARISH'
            else:
                macd_crossover = 'NEUTRAL'
            
            return {
                'status': rsi_status,
                'rsi': float(rsi),
                'macd': float(macd),
                'macd_signal_line': float(macd_signal),
                'macd_signal': macd_crossover
            }
        
        except Exception as e:
            logger.debug(f"Momentum analysis error: {e}")
            return {'status': 'NEUTRAL', 'rsi': 50.0, 'macd_signal': 'NEUTRAL'}
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """Analyze volatility levels"""
        try:
            latest = df.iloc[-1]
            
            atr = latest.get('atr', latest['close'] * 0.02)
            bb_upper = latest.get('bb_upper', latest['close'] * 1.02)
            bb_lower = latest.get('bb_lower', latest['close'] * 0.98)
            price = latest['close']
            
            # ATR-based volatility
            atr_pct = (atr / price) * 100
            
            if atr_pct > 3:
                vol_level = 'HIGH'
            elif atr_pct < 1:
                vol_level = 'LOW'
            else:
                vol_level = 'NORMAL'
            
            # Bollinger Band position
            bb_range = bb_upper - bb_lower
            if bb_range > 0:
                bb_position = (price - bb_lower) / bb_range
            else:
                bb_position = 0.5
            
            return {
                'level': vol_level,
                'atr': float(atr),
                'atr_pct': float(atr_pct),
                'bb_position': float(bb_position)
            }
        
        except Exception as e:
            logger.debug(f"Volatility analysis error: {e}")
            return {'level': 'NORMAL', 'atr_pct': 2.0}
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Analyze volume patterns"""
        try:
            latest = df.iloc[-1]
            
            volume = latest.get('volume', 0)
            volume_ma = latest.get('volume_ma', volume)
            
            if volume_ma > 0:
                volume_ratio = volume / volume_ma
            else:
                volume_ratio = 1.0
            
            if volume_ratio > 1.5:
                status = 'HIGH'
            elif volume_ratio < 0.7:
                status = 'LOW'
            else:
                status = 'NORMAL'
            
            return {
                'status': status,
                'volume': float(volume),
                'volume_ma': float(volume_ma),
                'volume_ratio': float(volume_ratio)
            }
        
        except Exception as e:
            logger.debug(f"Volume analysis error: {e}")
            return {'status': 'NORMAL', 'volume_ratio': 1.0}
    
    def _no_signal(self, symbol: str, reason: str) -> Dict:
        """Return a HOLD signal with reason"""
        return {
            'symbol': symbol,
            'signal': 'HOLD',
            'confidence': 0.0,
            'entry_price': 0.0,
            'target': 0.0,
            'stop_loss': 0.0,
            'current_price': 0.0,
            'reasons': [reason],
            'indicators': {},
            'timestamp': datetime.now().isoformat()
        }


logger.info("✓ TechnicalEngine module loaded successfully")
