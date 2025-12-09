
import logging
import os
from typing import Dict, List, Optional
from datetime import datetime
import requests
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

class AlertManager:
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        """
        Initialize Alert Manager
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID
        """
        self.bot_token = bot_token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if self.enabled:
            logger.info("✓ Alert Manager initialized (Telegram enabled)")
        else:
            logger.warning("⚠️  Alert Manager initialized (Telegram disabled - no credentials)")
    
    def send_message(self, message: str) -> bool:
        """
        Send message to Telegram
        
        Args:
            message: Message text
        
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.debug("Telegram disabled, message not sent")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.debug("✓ Telegram message sent")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def send_trade_signal_detailed(self, signal: Dict) -> bool:
        """
        Send detailed trade signal with all 13 models + pattern analysis
        
        Args:
            signal: Signal dictionary with all levels
        
        Returns:
            True if sent successfully
        """
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            direction = signal.get('direction', 'BUY')
            final_score = signal.get('final_score', 0)
            price = signal.get('current_price', 0)
            
            # Direction emoji
            direction_emoji = "🟢" if direction == "BUY" else "🔴"
            
            # Build message
            message = f"🎯 <b>TRADE SIGNAL: {symbol}</b>\n"
            message += "=" * 40 + "\n"
            message += f"{direction_emoji} <b>Direction:</b> {direction}\n"
            message += f"💯 <b>Final Score:</b> {final_score:.1f}/100\n\n"
            
            # Multi-Level Analysis
            message += "📊 <b>MULTI-LEVEL ANALYSIS</b>\n"
            message += "-" * 40 + "\n"
            
            # Level 1 - Technical
            level1_score = signal.get('level1_score', 0)
            level1_emoji = "✅" if level1_score >= 60 else "⚠️"
            message += f"Level 1 - Technical: {level1_score:.1f}/100 {level1_emoji}\n"
            
            # Level 2 - ML Ensemble (13 models)
            ml_result = signal.get('ml_prediction', {})
            ml_confidence = ml_result.get('probability', 0) * 100
            ml_emoji = "✅" if ml_confidence >= 65 else "⚠️"
            message += f"Level 2 - ML Ensemble: {ml_confidence:.1f}% {ml_emoji}\n"
            
            # Level 3 - Pattern Analysis
            level3_score = signal.get('level3_score', 0)
            level3_emoji = "✅" if level3_score >= 65 else "⚠️"
            message += f"Level 3 - Pattern Analysis: {level3_score:.1f}/100 {level3_emoji}\n\n"
            
            # 13 ML Models Breakdown
            model_scores = ml_result.get('model_scores', {})
            if model_scores:
                message += "🤖 <b>13-MODEL BREAKDOWN</b>\n"
                message += "-" * 40 + "\n"
                
                # Group models
                tree_models = {k: v for k, v in model_scores.items() if k in ['rf', 'et', 'xgb', 'lgbm', 'catboost']}
                boosting_models = {k: v for k, v in model_scores.items() if k in ['adaboost', 'gb']}
                linear_models = {k: v for k, v in model_scores.items() if k in ['lr', 'ridge']}
                advanced_models = {k: v for k, v in model_scores.items() if k in ['svm', 'mlp', 'gnb', 'knn']}
                ensemble_models = {k: v for k, v in model_scores.items() if k in ['stacking', 'voting']}
                
                # Tree Models
                if tree_models:
                    message += "<b>Tree Models:</b>\n"
                    for model, score in tree_models.items():
                        model_name = self._get_model_display_name(model)
                        message += f"  {model_name}: {score*100:.1f}%\n"
                
                # Boosting Models
                if boosting_models:
                    message += "\n<b>Boosting Models:</b>\n"
                    for model, score in boosting_models.items():
                        model_name = self._get_model_display_name(model)
                        message += f"  {model_name}: {score*100:.1f}%\n"
                
                # Linear Models
                if linear_models:
                    message += "\n<b>Linear Models:</b>\n"
                    for model, score in linear_models.items():
                        model_name = self._get_model_display_name(model)
                        message += f"  {model_name}: {score*100:.1f}%\n"
                
                # Advanced Models
                if advanced_models:
                    message += "\n<b>Advanced Models:</b>\n"
                    for model, score in advanced_models.items():
                        model_name = self._get_model_display_name(model)
                        message += f"  {model_name}: {score*100:.1f}%\n"
                
                # Meta Ensembles
                if ensemble_models:
                    message += "\n<b>Meta Ensembles:</b>\n"
                    for model, score in ensemble_models.items():
                        model_name = self._get_model_display_name(model)
                        message += f"  {model_name}: {score*100:.1f}%\n"
                
                # Model Agreement
                agreement = ml_result.get('model_agreement', 0)
                buy_votes = ml_result.get('buy_votes', 0)
                total_votes = ml_result.get('total_votes', 0)
                message += f"\n<b>Agreement:</b> {agreement*100:.1f}% ({buy_votes}/{total_votes} models)\n\n"
            
            # Pattern Analysis Details
            pattern_details = signal.get('pattern_details', {})
            if pattern_details:
                message += "🔍 <b>PATTERN DETAILS</b>\n"
                message += "-" * 40 + "\n"
                
                # Support & Resistance
                sr = pattern_details.get('support_resistance', {})
                if sr:
                    nearest_support = sr.get('nearest_support')
                    nearest_resistance = sr.get('nearest_resistance')
                    support_dist = sr.get('support_distance_pct', 0)
                    resistance_dist = sr.get('resistance_distance_pct', 0)
                    in_support_zone = sr.get('in_support_zone', False)
                    
                    message += "📍 <b>Support & Resistance:</b>\n"
                    if nearest_support:
                        support_emoji = "🟢" if in_support_zone else ""
                        message += f"  Support: ₹{nearest_support:.2f} ({support_dist:.1f}% below) {support_emoji}\n"
                    if nearest_resistance:
                        message += f"  Resistance: ₹{nearest_resistance:.2f} ({resistance_dist:.1f}% above)\n"
                    
                    if in_support_zone:
                        message += f"  <b>Status:</b> Near support zone ✅\n"
                    message += "\n"
                
                # Chart Patterns
                chart_patterns = pattern_details.get('chart_patterns', {})
                patterns_found = chart_patterns.get('patterns_found', [])
                if patterns_found:
                    message += "📐 <b>Chart Patterns:</b>\n"
                    pattern_details_dict = chart_patterns.get('details', {})
                    
                    for pattern_name in patterns_found:
                        pattern_info = pattern_details_dict.get(pattern_name, {})
                        pattern_type = pattern_info.get('pattern', 'Unknown')
                        signal_type = pattern_info.get('signal', 'NEUTRAL')
                        
                        signal_emoji = "✅" if signal_type == "BULLISH" else "⚠️" if signal_type == "BEARISH" else "➖"
                        message += f"  Pattern: {pattern_type} ({signal_type}) {signal_emoji}\n"
                        
                        # Add target if available
                        target = pattern_info.get('target')
                        if target:
                            target_pct = ((target - price) / price * 100) if price > 0 else 0
                            message += f"  Target: ₹{target:.2f} ({target_pct:+.1f}%)\n"
                    
                    message += "\n"
                
                # Fibonacci
                fib = pattern_details.get('fibonacci_levels', {})
                if fib:
                    nearest_fib = fib.get('nearest_fib_level')
                    nearest_fib_price = fib.get('nearest_fib_price')
                    fib_dist = fib.get('distance_to_nearest_pct', 0)
                    
                    if nearest_fib:
                        message += "📊 <b>Fibonacci:</b>\n"
                        message += f"  Level: {nearest_fib} retracement\n"
                        if nearest_fib_price:
                            message += f"  Price: ₹{nearest_fib_price:.2f}\n"
                        if fib_dist < 1.0:
                            message += f"  Distance: {fib_dist:.1f}% (KEY LEVEL) ✅\n"
                        message += "\n"
                
                # Pivot Points
                pivots = pattern_details.get('pivot_points', {})
                if pivots:
                    pp = pivots.get('pivot_point')
                    r1 = pivots.get('resistance_1')
                    s1 = pivots.get('support_1')
                    position = pivots.get('current_position', '')
                    
                    if pp:
                        message += "⚖️ <b>Pivot Points:</b>\n"
                        message += f"  Position: {position}\n"
                        message += f"  PP: ₹{pp:.2f}"
                        if r1:
                            message += f" | R1: ₹{r1:.2f}"
                        if s1:
                            message += f" | S1: ₹{s1:.2f}"
                        message += "\n\n"
                
                # Trend Lines
                trend = pattern_details.get('trend_lines', {})
                if trend:
                    trend_type = trend.get('trend', 'SIDEWAYS')
                    support_line = trend.get('support_trend_line')
                    
                    if trend_type != 'SIDEWAYS':
                        message += "📈 <b>Trend Lines:</b>\n"
                        if support_line:
                            message += f"  Support Line: ₹{support_line:.2f}\n"
                        message += f"  Trend: {trend_type}"
                        
                        if trend_type == 'UPTREND':
                            message += " ✅\n"
                        elif trend_type == 'DOWNTREND':
                            message += " ⚠️\n"
                        message += "\n"
                
                # Price Action
                price_action = pattern_details.get('price_action', {})
                if price_action:
                    pa_signal = price_action.get('price_action_signal', 'NEUTRAL')
                    bullish_engulfing = price_action.get('bullish_engulfing', False)
                    bearish_engulfing = price_action.get('bearish_engulfing', False)
                    higher_highs_lows = price_action.get('higher_highs_lows', False)
                    
                    if pa_signal != 'NEUTRAL':
                        message += "🕯️ <b>Price Action:</b>\n"
                        
                        if bullish_engulfing:
                            message += "  Pattern: Bullish Engulfing ✅\n"
                        elif bearish_engulfing:
                            message += "  Pattern: Bearish Engulfing ⚠️\n"
                        
                        if higher_highs_lows:
                            message += "  Higher Lows: Yes ✅\n"
                        
                        message += f"  Signal: {pa_signal}\n\n"
            
            # Position Details
            message += "💰 <b>POSITION DETAILS</b>\n"
            message += "-" * 40 + "\n"
            message += f"Price: ₹{price:.2f}\n"
            
            position_size = signal.get('position_size', 0)
            capital_pct = signal.get('capital_allocation_pct', 0)
            message += f"Capital: ₹{position_size:.0f} ({capital_pct:.0f}%)\n"
            
            stop_loss = signal.get('stop_loss', 0)
            stop_loss_pct = ((stop_loss - price) / price * 100) if price > 0 else 0
            message += f"Stop Loss: ₹{stop_loss:.2f} ({stop_loss_pct:.1f}%)\n"
            
            # Check if stop loss is at support
            nearest_support = pattern_details.get('support_resistance', {}).get('nearest_support')
            if nearest_support and abs(stop_loss - nearest_support) / nearest_support < 0.01:
                message += "  [AT SUPPORT] ✅\n"
            
            target = signal.get('target_price', 0)
            target_pct = ((target - price) / price * 100) if price > 0 else 0
            message += f"Target: ₹{target:.2f} ({target_pct:+.1f}%)\n"
            
            # Check if target is pattern-based
            if pattern_details.get('chart_patterns', {}).get('patterns_found'):
                message += "  [PATTERN TARGET] 📐\n"
            
            max_holding = signal.get('max_holding_days', 5)
            message += f"Max Holding: {max_holding} days\n"
            
            # News sentiment (if available)
            sentiment_score = signal.get('sentiment_score', 0)
            if sentiment_score != 0:
                message += f"\n📰 News Sentiment: {sentiment_score:+.2f}\n"
            
            # Send message
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error creating detailed signal message: {e}")
            return False
    
    def _get_model_display_name(self, model_key: str) -> str:
        """Get display name for model"""
        names = {
            'rf': 'Random Forest',
            'et': 'Extra Trees',
            'xgb': 'XGBoost',
            'lgbm': 'LightGBM',
            'catboost': 'CatBoost',
            'adaboost': 'AdaBoost',
            'gb': 'Gradient Boosting',
            'lr': 'Logistic Regression',
            'ridge': 'Ridge Classifier',
            'svm': 'SVM (RBF)',
            'mlp': 'Neural Network',
            'gnb': 'Naive Bayes',
            'knn': 'K-Neighbors',
            'stacking': 'Stacking Meta',
            'voting': 'Soft Voting'
        }
        return names.get(model_key, model_key.upper())
    
    def send_daily_summary(self, signals: List[Dict]) -> bool:
        """
        Send daily summary of all signals
        
        Args:
            signals: List of signal dictionaries
        
        Returns:
            True if sent successfully
        """
        try:
            message = "📊 <b>DAILY TRADING SUMMARY</b>\n"
            message += "=" * 40 + "\n"
            message += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
            message += f"Total Signals: {len(signals)}\n\n"
            
            if not signals:
                message += "No signals generated today.\n"
                return self.send_message(message)
            
            # Sort by final score
            sorted_signals = sorted(signals, key=lambda x: x.get('final_score', 0), reverse=True)
            
            for i, signal in enumerate(sorted_signals, 1):
                symbol = signal.get('symbol', 'UNKNOWN')
                direction = signal.get('direction', 'BUY')
                score = signal.get('final_score', 0)
                price = signal.get('current_price', 0)
                
                direction_emoji = "🟢" if direction == "BUY" else "🔴"
                
                message += f"{i}. {direction_emoji} <b>{symbol}</b>\n"
                message += f"   Score: {score:.1f}/100 | Price: ₹{price:.2f}\n"
                
                # ML confidence
                ml_result = signal.get('ml_prediction', {})
                ml_conf = ml_result.get('probability', 0) * 100
                message += f"   ML: {ml_conf:.1f}% | Agreement: {ml_result.get('model_agreement', 0)*100:.0f}%\n\n"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error creating daily summary: {e}")
            return False
    
    def send_error_alert(self, error_msg: str, context: str = "") -> bool:
        """
        Send error alert to Telegram
        
        Args:
            error_msg: Error message
            context: Additional context
        
        Returns:
            True if sent successfully
        """
        try:
            message = "⚠️ <b>SYSTEM ERROR</b>\n"
            message += "=" * 40 + "\n"
            message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            message += f"<b>Error:</b> {error_msg}\n"
            
            if context:
                message += f"\n<b>Context:</b> {context}\n"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending error alert: {e}")
            return False


if __name__ == '__main__':
    # Test alerts
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("TESTING ALERT SYSTEM")
    print("="*60)
    
    # Mock signal with 13 models + patterns
    test_signal = {
        'symbol': 'RELIANCE.NS',
        'direction': 'BUY',
        'final_score': 89.4,
        'current_price': 2500.0,
        'level1_score': 75.0,
        'level3_score': 85.0,
        'ml_prediction': {
            'probability': 0.78,
            'model_agreement': 0.92,
            'buy_votes': 12,
            'total_votes': 13,
            'model_scores': {
                'rf': 0.79,
                'et': 0.77,
                'xgb': 0.85,
                'lgbm': 0.84,
                'catboost': 0.86,
                'adaboost': 0.75,
                'gb': 0.80,
                'lr': 0.72,
                'ridge': 0.70,
                'svm': 0.74,
                'mlp': 0.76,
                'gnb': 0.68,
                'knn': 0.71,
                'stacking': 0.83,
                'voting': 0.81
            }
        },
        'pattern_details': {
            'support_resistance': {
                'nearest_support': 2450.0,
                'nearest_resistance': 2600.0,
                'support_distance_pct': 2.0,
                'resistance_distance_pct': 4.0,
                'in_support_zone': True
            },
            'chart_patterns': {
                'patterns_found': ['double_top_bottom'],
                'details': {
                    'double_top_bottom': {
                        'pattern': 'Double Bottom',
                        'signal': 'BULLISH',
                        'target': 2650.0
                    }
                }
            },
            'fibonacci_levels': {
                'nearest_fib_level': '61.8',
                'nearest_fib_price': 2505.0,
                'distance_to_nearest_pct': 0.2
            },
            'pivot_points': {
                'pivot_point': 2475.0,
                'resistance_1': 2525.0,
                'support_1': 2425.0,
                'current_position': 'Above Pivot (Bullish)'
            },
            'trend_lines': {
                'trend': 'UPTREND',
                'support_trend_line': 2430.0
            },
            'price_action': {
                'price_action_signal': 'BULLISH',
                'bullish_engulfing': True,
                'higher_highs_lows': True
            }
        },
        'position_size': 3500,
        'capital_allocation_pct': 35,
        'stop_loss': 2425,
        'target_price': 2650,
        'max_holding_days': 5,
        'sentiment_score': 0.15
    }
    
    # Initialize alert manager (will use env variables if available)
    alerts = AlertManager()
    
    if alerts.enabled:
        print("\n✓ Telegram enabled - sending test signal...")
        alerts.send_trade_signal_detailed(test_signal)
    else:
        print("\n⚠️  Telegram disabled - displaying message format:")
        print("\n" + "-"*60)
        # Would print formatted message here
    
    print("\n✅ Test Complete!")
