
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

class PatternDetector:
    """
    Advanced Technical Pattern Detection System
    """
    
    def __init__(self):
        self.patterns_detected = {}
        logger.info("✓ Pattern Detector initialized")
    
    def detect_all_patterns(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict:
        """
        Detect all chart patterns and support/resistance levels
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
        
        Returns:
            Dictionary with all detected patterns and levels
        """
        if df is None or df.empty or len(df) < 30:
            return self._empty_result()
        
        try:
            results = {
                'symbol': symbol,
                'support_resistance': self.find_support_resistance(df),
                'pivot_points': self.calculate_pivot_points(df),
                'fibonacci_levels': self.calculate_fibonacci_retracement(df),
                'chart_patterns': self.detect_chart_patterns(df),
                'trend_lines': self.detect_trend_lines(df),
                'price_action': self.analyze_price_action(df)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"{symbol}: Pattern detection error: {e}")
            return self._empty_result()
    
    def find_support_resistance(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        """
        Find Support and Resistance levels using pivot points clustering
        
        Method:
        1. Find local maxima (resistance candidates)
        2. Find local minima (support candidates)
        3. Cluster nearby levels using DBSCAN
        4. Calculate strength based on touches
        """
        try:
            # Get recent data
            recent_df = df.tail(100).copy()
            
            # Find local extrema
            local_max_idx = argrelextrema(recent_df['high'].values, np.greater, order=5)[0]
            local_min_idx = argrelextrema(recent_df['low'].values, np.less, order=5)[0]
            
            # Extract price levels
            resistance_levels = recent_df.iloc[local_max_idx]['high'].values
            support_levels = recent_df.iloc[local_min_idx]['low'].values
            
            # Cluster resistance levels
            clustered_resistance = self._cluster_levels(resistance_levels)
            clustered_support = self._cluster_levels(support_levels)
            
            # Calculate current price
            current_price = df['close'].iloc[-1]
            
            # Find nearest levels
            nearest_support = self._find_nearest_level(current_price, clustered_support, direction='below')
            nearest_resistance = self._find_nearest_level(current_price, clustered_resistance, direction='above')
            
            # Calculate strength (number of touches)
            support_strength = self._calculate_level_strength(df, clustered_support)
            resistance_strength = self._calculate_level_strength(df, clustered_resistance)
            
            return {
                'support_levels': clustered_support.tolist() if len(clustered_support) > 0 else [],
                'resistance_levels': clustered_resistance.tolist() if len(clustered_resistance) > 0 else [],
                'nearest_support': float(nearest_support) if nearest_support else None,
                'nearest_resistance': float(nearest_resistance) if nearest_resistance else None,
                'support_distance_pct': ((current_price - nearest_support) / current_price * 100) if nearest_support else None,
                'resistance_distance_pct': ((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else None,
                'support_strength': support_strength,
                'resistance_strength': resistance_strength,
                'in_support_zone': self._is_near_level(current_price, clustered_support, tolerance=0.015),
                'in_resistance_zone': self._is_near_level(current_price, clustered_resistance, tolerance=0.015)
            }
            
        except Exception as e:
            logger.debug(f"Support/Resistance detection error: {e}")
            return {}
    
    def _cluster_levels(self, levels: np.ndarray, eps_pct: float = 0.02) -> np.ndarray:
        """Cluster price levels using DBSCAN"""
        if len(levels) < 2:
            return levels
        
        # Calculate epsilon as percentage of price
        mean_price = np.mean(levels)
        eps = mean_price * eps_pct
        
        # Reshape for DBSCAN
        levels_reshaped = levels.reshape(-1, 1)
        
        # Cluster
        clustering = DBSCAN(eps=eps, min_samples=2).fit(levels_reshaped)
        
        # Get cluster centers
        unique_labels = set(clustering.labels_)
        cluster_centers = []
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            cluster_points = levels[clustering.labels_ == label]
            cluster_centers.append(np.mean(cluster_points))
        
        # Add noise points as individual levels if significant
        noise_points = levels[clustering.labels_ == -1]
        cluster_centers.extend(noise_points)
        
        return np.array(sorted(cluster_centers))
    
    def _find_nearest_level(self, price: float, levels: np.ndarray, direction: str = 'both') -> Optional[float]:
        """Find nearest support or resistance level"""
        if len(levels) == 0:
            return None
        
        if direction == 'below':
            below_levels = levels[levels < price]
            return below_levels[-1] if len(below_levels) > 0 else None
        elif direction == 'above':
            above_levels = levels[levels > price]
            return above_levels[0] if len(above_levels) > 0 else None
        else:  # both
            distances = np.abs(levels - price)
            return levels[np.argmin(distances)]
    
    def _calculate_level_strength(self, df: pd.DataFrame, levels: np.ndarray, tolerance: float = 0.015) -> Dict:
        """Calculate strength of support/resistance levels based on touches"""
        strength = {}
        
        for level in levels:
            touches = 0
            bounces = 0
            
            for i in range(1, len(df)):
                price_range = (df['low'].iloc[i], df['high'].iloc[i])
                prev_close = df['close'].iloc[i-1]
                curr_close = df['close'].iloc[i]
                
                # Check if price touched the level
                if (price_range[0] <= level * (1 + tolerance) and 
                    price_range[1] >= level * (1 - tolerance)):
                    touches += 1
                    
                    # Check if it bounced
                    if prev_close < level < curr_close or prev_close > level > curr_close:
                        bounces += 1
            
            strength[level] = {
                'touches': touches,
                'bounces': bounces,
                'strength_score': touches + (bounces * 2)  # Bounces count double
            }
        
        return strength
    
    def _is_near_level(self, price: float, levels: np.ndarray, tolerance: float = 0.015) -> bool:
        """Check if price is near any level"""
        if len(levels) == 0:
            return False
        
        for level in levels:
            if abs(price - level) / level <= tolerance:
                return True
        
        return False
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Classic Pivot Points (used by floor traders)
        
        Formulas:
        PP = (High + Low + Close) / 3
        R1 = 2*PP - Low
        S1 = 2*PP - High
        R2 = PP + (High - Low)
        S2 = PP - (High - Low)
        R3 = High + 2*(PP - Low)
        S3 = Low - 2*(High - PP)
        """
        try:
            # Use previous day's data
            prev_high = df['high'].iloc[-2]
            prev_low = df['low'].iloc[-2]
            prev_close = df['close'].iloc[-2]
            
            # Pivot Point
            pp = (prev_high + prev_low + prev_close) / 3
            
            # Support and Resistance
            r1 = 2 * pp - prev_low
            s1 = 2 * pp - prev_high
            
            r2 = pp + (prev_high - prev_low)
            s2 = pp - (prev_high - prev_low)
            
            r3 = prev_high + 2 * (pp - prev_low)
            s3 = prev_low - 2 * (prev_high - pp)
            
            current_price = df['close'].iloc[-1]
            
            return {
                'pivot_point': float(pp),
                'resistance_1': float(r1),
                'resistance_2': float(r2),
                'resistance_3': float(r3),
                'support_1': float(s1),
                'support_2': float(s2),
                'support_3': float(s3),
                'current_position': self._get_pivot_position(current_price, pp, r1, s1)
            }
            
        except Exception as e:
            logger.debug(f"Pivot points calculation error: {e}")
            return {}
    
    def _get_pivot_position(self, price: float, pp: float, r1: float, s1: float) -> str:
        """Determine price position relative to pivots"""
        if price > r1:
            return "Above R1 (Strong Bullish)"
        elif price > pp:
            return "Above Pivot (Bullish)"
        elif price > s1:
            return "Below Pivot (Bearish)"
        else:
            return "Below S1 (Strong Bearish)"
    
    def calculate_fibonacci_retracement(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
        """
        Calculate Fibonacci Retracement levels
        
        Levels: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%
        """
        try:
            recent_df = df.tail(lookback)
            
            # Find swing high and low
            swing_high = recent_df['high'].max()
            swing_low = recent_df['low'].min()
            
            # Determine trend direction
            if df['close'].iloc[-1] > df['close'].iloc[-lookback]:
                trend = "UPTREND"
                # In uptrend, measure from low to high
                diff = swing_high - swing_low
                base = swing_low
            else:
                trend = "DOWNTREND"
                # In downtrend, measure from high to low
                diff = swing_high - swing_low
                base = swing_high
            
            # Fibonacci levels
            fib_levels = {
                '0.0': swing_high if trend == "DOWNTREND" else swing_low,
                '23.6': base + (diff * 0.236) if trend == "UPTREND" else base - (diff * 0.236),
                '38.2': base + (diff * 0.382) if trend == "UPTREND" else base - (diff * 0.382),
                '50.0': base + (diff * 0.50) if trend == "UPTREND" else base - (diff * 0.50),
                '61.8': base + (diff * 0.618) if trend == "UPTREND" else base - (diff * 0.618),
                '78.6': base + (diff * 0.786) if trend == "UPTREND" else base - (diff * 0.786),
                '100.0': swing_low if trend == "DOWNTREND" else swing_high
            }
            
            current_price = df['close'].iloc[-1]
            
            # Find nearest Fib level
            nearest_fib = min(fib_levels.items(), key=lambda x: abs(x[1] - current_price))
            
            return {
                'trend': trend,
                'swing_high': float(swing_high),
                'swing_low': float(swing_low),
                'fib_0': float(fib_levels['0.0']),
                'fib_236': float(fib_levels['23.6']),
                'fib_382': float(fib_levels['38.2']),
                'fib_50': float(fib_levels['50.0']),
                'fib_618': float(fib_levels['61.8']),
                'fib_786': float(fib_levels['78.6']),
                'fib_100': float(fib_levels['100.0']),
                'nearest_fib_level': nearest_fib[0],
                'nearest_fib_price': float(nearest_fib[1]),
                'distance_to_nearest_pct': abs(current_price - nearest_fib[1]) / current_price * 100
            }
            
        except Exception as e:
            logger.debug(f"Fibonacci calculation error: {e}")
            return {}
    
    def detect_chart_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Detect classic chart patterns:
        - Head and Shoulders / Inverse H&S
        - Double Top / Double Bottom
        - Triangle (Ascending, Descending, Symmetric)
        - Flag and Pennant
        - Wedge (Rising, Falling)
        """
        patterns = {
            'head_and_shoulders': self._detect_head_and_shoulders(df),
            'double_top_bottom': self._detect_double_top_bottom(df),
            'triangle': self._detect_triangle(df),
            'flag_pennant': self._detect_flag_pennant(df),
            'wedge': self._detect_wedge(df)
        }
        
        # Filter detected patterns
        detected = {k: v for k, v in patterns.items() if v.get('detected', False)}
        
        return {
            'patterns_found': list(detected.keys()),
            'pattern_count': len(detected),
            'details': detected,
            'bullish_patterns': sum(1 for p in detected.values() if p.get('signal') == 'BULLISH'),
            'bearish_patterns': sum(1 for p in detected.values() if p.get('signal') == 'BEARISH')
        }
    
    def _detect_head_and_shoulders(self, df: pd.DataFrame, lookback: int = 30) -> Dict:
        """Detect Head and Shoulders pattern"""
        try:
            recent_df = df.tail(lookback)
            
            # Find peaks
            peaks_idx = argrelextrema(recent_df['high'].values, np.greater, order=3)[0]
            
            if len(peaks_idx) < 3:
                return {'detected': False}
            
            # Take last 3 peaks
            peaks = peaks_idx[-3:]
            peak_prices = recent_df.iloc[peaks]['high'].values
            
            # Head and Shoulders: peak2 > peak1 and peak2 > peak3, peak1 ≈ peak3
            if (peak_prices[1] > peak_prices[0] and 
                peak_prices[1] > peak_prices[2] and
                abs(peak_prices[0] - peak_prices[2]) / peak_prices[0] < 0.05):
                
                return {
                    'detected': True,
                    'pattern': 'Head and Shoulders',
                    'signal': 'BEARISH',
                    'left_shoulder': float(peak_prices[0]),
                    'head': float(peak_prices[1]),
                    'right_shoulder': float(peak_prices[2]),
                    'neckline': float(recent_df.iloc[peaks]['low'].min()),
                    'target': None  # Calculate based on neckline break
                }
            
            # Inverse Head and Shoulders (bullish)
            troughs_idx = argrelextrema(recent_df['low'].values, np.less, order=3)[0]
            
            if len(troughs_idx) >= 3:
                troughs = troughs_idx[-3:]
                trough_prices = recent_df.iloc[troughs]['low'].values
                
                if (trough_prices[1] < trough_prices[0] and 
                    trough_prices[1] < trough_prices[2] and
                    abs(trough_prices[0] - trough_prices[2]) / trough_prices[0] < 0.05):
                    
                    return {
                        'detected': True,
                        'pattern': 'Inverse Head and Shoulders',
                        'signal': 'BULLISH',
                        'left_shoulder': float(trough_prices[0]),
                        'head': float(trough_prices[1]),
                        'right_shoulder': float(trough_prices[2]),
                        'neckline': float(recent_df.iloc[troughs]['high'].max()),
                        'target': None
                    }
            
            return {'detected': False}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def _detect_double_top_bottom(self, df: pd.DataFrame, lookback: int = 25) -> Dict:
        """Detect Double Top/Bottom patterns"""
        try:
            recent_df = df.tail(lookback)
            
            # Double Top
            peaks_idx = argrelextrema(recent_df['high'].values, np.greater, order=3)[0]
            
            if len(peaks_idx) >= 2:
                last_two_peaks = peaks_idx[-2:]
                peak_prices = recent_df.iloc[last_two_peaks]['high'].values
                
                # Check if peaks are similar (within 2%)
                if abs(peak_prices[0] - peak_prices[1]) / peak_prices[0] < 0.02:
                    trough_between = recent_df.iloc[last_two_peaks[0]:last_two_peaks[1]]['low'].min()
                    
                    return {
                        'detected': True,
                        'pattern': 'Double Top',
                        'signal': 'BEARISH',
                        'first_top': float(peak_prices[0]),
                        'second_top': float(peak_prices[1]),
                        'valley': float(trough_between),
                        'confirmation_level': float(trough_between),
                        'target': float(trough_between - (peak_prices[0] - trough_between))
                    }
            
            # Double Bottom
            troughs_idx = argrelextrema(recent_df['low'].values, np.less, order=3)[0]
            
            if len(troughs_idx) >= 2:
                last_two_troughs = troughs_idx[-2:]
                trough_prices = recent_df.iloc[last_two_troughs]['low'].values
                
                if abs(trough_prices[0] - trough_prices[1]) / trough_prices[0] < 0.02:
                    peak_between = recent_df.iloc[last_two_troughs[0]:last_two_troughs[1]]['high'].max()
                    
                    return {
                        'detected': True,
                        'pattern': 'Double Bottom',
                        'signal': 'BULLISH',
                        'first_bottom': float(trough_prices[0]),
                        'second_bottom': float(trough_prices[1]),
                        'peak': float(peak_between),
                        'confirmation_level': float(peak_between),
                        'target': float(peak_between + (peak_between - trough_prices[0]))
                    }
            
            return {'detected': False}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def _detect_triangle(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        """Detect Triangle patterns (Ascending, Descending, Symmetric)"""
        try:
            recent_df = df.tail(lookback)
            
            highs = recent_df['high'].values
            lows = recent_df['low'].values
            
            # Fit trend lines
            x = np.arange(len(recent_df))
            
            # Upper trend line (resistance)
            upper_slope = np.polyfit(x, highs, 1)[0]
            
            # Lower trend line (support)
            lower_slope = np.polyfit(x, lows, 1)[0]
            
            # Determine triangle type
            if abs(upper_slope) < 0.01 and lower_slope > 0.05:
                pattern_type = "Ascending Triangle"
                signal = "BULLISH"
            elif upper_slope < -0.05 and abs(lower_slope) < 0.01:
                pattern_type = "Descending Triangle"
                signal = "BEARISH"
            elif abs(upper_slope + lower_slope) < 0.05:  # Converging
                pattern_type = "Symmetric Triangle"
                signal = "NEUTRAL"
            else:
                return {'detected': False}
            
            return {
                'detected': True,
                'pattern': pattern_type,
                'signal': signal,
                'upper_slope': float(upper_slope),
                'lower_slope': float(lower_slope),
                'apex_distance': int(lookback / 2)  # Rough estimate
            }
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def _detect_flag_pennant(self, df: pd.DataFrame, lookback: int = 15) -> Dict:
        """Detect Flag and Pennant patterns"""
        try:
            # Need strong prior move
            if len(df) < 30:
                return {'detected': False}
            
            recent_df = df.tail(lookback)
            prior_df = df.tail(30).head(15)
            
            # Check for strong prior move (>5% in 15 days)
            prior_move = (prior_df['close'].iloc[-1] - prior_df['close'].iloc[0]) / prior_df['close'].iloc[0]
            
            if abs(prior_move) < 0.05:
                return {'detected': False}
            
            # Check for consolidation (range < 3%)
            consolidation_range = (recent_df['high'].max() - recent_df['low'].min()) / recent_df['close'].mean()
            
            if consolidation_range < 0.03:
                if prior_move > 0:
                    return {
                        'detected': True,
                        'pattern': 'Bull Flag',
                        'signal': 'BULLISH',
                        'flagpole_move_pct': float(prior_move * 100),
                        'consolidation_range_pct': float(consolidation_range * 100),
                        'expected_move': float(prior_move)  # Often repeats flagpole move
                    }
                else:
                    return {
                        'detected': True,
                        'pattern': 'Bear Flag',
                        'signal': 'BEARISH',
                        'flagpole_move_pct': float(prior_move * 100),
                        'consolidation_range_pct': float(consolidation_range * 100),
                        'expected_move': float(prior_move)
                    }
            
            return {'detected': False}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def _detect_wedge(self, df: pd.DataFrame, lookback: int = 25) -> Dict:
        """Detect Wedge patterns (Rising, Falling)"""
        try:
            recent_df = df.tail(lookback)
            
            highs = recent_df['high'].values
            lows = recent_df['low'].values
            x = np.arange(len(recent_df))
            
            # Fit trend lines
            upper_slope = np.polyfit(x, highs, 1)[0]
            lower_slope = np.polyfit(x, lows, 1)[0]
            
            # Both lines should be moving in same direction and converging
            if upper_slope > 0 and lower_slope > 0 and upper_slope < lower_slope:
                return {
                    'detected': True,
                    'pattern': 'Rising Wedge',
                    'signal': 'BEARISH',  # Rising wedge is bearish
                    'upper_slope': float(upper_slope),
                    'lower_slope': float(lower_slope)
                }
            
            elif upper_slope < 0 and lower_slope < 0 and upper_slope > lower_slope:
                return {
                    'detected': True,
                    'pattern': 'Falling Wedge',
                    'signal': 'BULLISH',  # Falling wedge is bullish
                    'upper_slope': float(upper_slope),
                    'lower_slope': float(lower_slope)
                }
            
            return {'detected': False}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def detect_trend_lines(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
        """Detect major trend lines"""
        try:
            recent_df = df.tail(lookback)
            
            # Find peaks and troughs
            peaks_idx = argrelextrema(recent_df['high'].values, np.greater, order=5)[0]
            troughs_idx = argrelextrema(recent_df['low'].values, np.less, order=5)[0]
            
            # Fit resistance line (connect peaks)
            if len(peaks_idx) >= 2:
                peak_prices = recent_df.iloc[peaks_idx]['high'].values
                peak_x = peaks_idx
                resistance_slope, resistance_intercept = np.polyfit(peak_x, peak_prices, 1)
            else:
                resistance_slope, resistance_intercept = 0, recent_df['high'].max()
            
            # Fit support line (connect troughs)
            if len(troughs_idx) >= 2:
                trough_prices = recent_df.iloc[troughs_idx]['low'].values
                trough_x = troughs_idx
                support_slope, support_intercept = np.polyfit(trough_x, trough_prices, 1)
            else:
                support_slope, support_intercept = 0, recent_df['low'].min()
            
            # Current levels
            current_x = len(recent_df) - 1
            current_resistance = resistance_slope * current_x + resistance_intercept
            current_support = support_slope * current_x + support_intercept
            
            return {
                'resistance_trend_line': float(current_resistance),
                'support_trend_line': float(current_support),
                'resistance_slope': float(resistance_slope),
                'support_slope': float(support_slope),
                'trend': 'UPTREND' if support_slope > 0.1 else 'DOWNTREND' if support_slope < -0.1 else 'SIDEWAYS'
            }
            
        except Exception as e:
            logger.debug(f"Trend line detection error: {e}")
            return {}
    
    def analyze_price_action(self, df: pd.DataFrame) -> Dict:
        """
        Analyze recent price action patterns:
        - Higher highs, higher lows
        - Lower highs, lower lows
        - Inside bars, outside bars
        - Engulfing patterns
        """
        try:
            if len(df) < 5:
                return {}
            
            recent = df.tail(5)
            
            # Check for higher highs and higher lows
            highs = recent['high'].values
            lows = recent['low'].values
            
            higher_highs = all(highs[i] > highs[i-1] for i in range(1, len(highs)))
            higher_lows = all(lows[i] > lows[i-1] for i in range(1, len(lows)))
            lower_highs = all(highs[i] < highs[i-1] for i in range(1, len(highs)))
            lower_lows = all(lows[i] < lows[i-1] for i in range(1, len(lows)))
            
            # Candlestick patterns (last 2 candles)
            prev = recent.iloc[-2]
            curr = recent.iloc[-1]
            
            # Bullish Engulfing
            bullish_engulfing = (
                prev['close'] < prev['open'] and  # Prev was bearish
                curr['close'] > curr['open'] and  # Curr is bullish
                curr['open'] < prev['close'] and  # Opens below prev close
                curr['close'] > prev['open']       # Closes above prev open
            )
            
            # Bearish Engulfing
            bearish_engulfing = (
                prev['close'] > prev['open'] and  # Prev was bullish
                curr['close'] < curr['open'] and  # Curr is bearish
                curr['open'] > prev['close'] and  # Opens above prev close
                curr['close'] < prev['open']       # Closes below prev open
            )
            
            # Inside Bar
            inside_bar = (
                curr['high'] < prev['high'] and
                curr['low'] > prev['low']
            )
            
            # Outside Bar
            outside_bar = (
                curr['high'] > prev['high'] and
                curr['low'] < prev['low']
            )
            
            return {
                'higher_highs_lows': higher_highs and higher_lows,
                'lower_highs_lows': lower_highs and lower_lows,
                'bullish_engulfing': bullish_engulfing,
                'bearish_engulfing': bearish_engulfing,
                'inside_bar': inside_bar,
                'outside_bar': outside_bar,
                'price_action_signal': (
                    'BULLISH' if (higher_highs and higher_lows) or bullish_engulfing else
                    'BEARISH' if (lower_highs and lower_lows) or bearish_engulfing else
                    'NEUTRAL'
                )
            }
            
        except Exception as e:
            logger.debug(f"Price action analysis error: {e}")
            return {}
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'support_resistance': {},
            'pivot_points': {},
            'fibonacci_levels': {},
            'chart_patterns': {'patterns_found': [], 'pattern_count': 0},
            'trend_lines': {},
            'price_action': {}
        }


if __name__ == '__main__':
    # Test pattern detector
    logging.basicConfig(level=logging.INFO)
    
    # Create mock data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        'Date': dates,
        'open': prices + np.random.randn(100) * 0.5,
        'high': prices + abs(np.random.randn(100) * 1.5),
        'low': prices - abs(np.random.randn(100) * 1.5),
        'close': prices + np.random.randn(100) * 0.5,
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    detector = PatternDetector()
    results = detector.detect_all_patterns(df, "TEST")
    
    print("\n" + "="*60)
    print("PATTERN DETECTION RESULTS")
    print("="*60)
    
    print("\n📊 Support & Resistance:")
    sr = results['support_resistance']
    print(f"  Support Levels: {sr.get('support_levels', [])}")
    print(f"  Resistance Levels: {sr.get('resistance_levels', [])}")
    print(f"  Nearest Support: {sr.get('nearest_support')}")
    print(f"  Nearest Resistance: {sr.get('nearest_resistance')}")
    
    print("\n📍 Pivot Points:")
    pivots = results['pivot_points']
    print(f"  Pivot: {pivots.get('pivot_point')}")
    print(f"  R1/S1: {pivots.get('resistance_1')}/{pivots.get('support_1')}")
    
    print("\n📈 Chart Patterns:")
    patterns = results['chart_patterns']
    print(f"  Patterns Found: {patterns.get('patterns_found', [])}")
    print(f"  Count: {patterns.get('pattern_count', 0)}")
    
    print("\n✅ Test Complete")
