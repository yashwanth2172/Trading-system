import logging
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import glob
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config.settings import TRADE_UNIVERSE
from src.data_manager import DataManager
from src.sentiment_analyzer import SentimentAnalyzer
from src.alerts import AlertManager

# Logging setup
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'morning_run.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def morning_validation():
    logger.info("=" * 80)
    logger.info("☀️ MORNING PRE-MARKET VALIDATION")
    logger.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    today = datetime.now()
    if today.weekday() >= 5:  
        logger.info("⚠️ Weekend detected. No market today.")
        return
    
    current_hour = today.hour
    if current_hour >= 9 and today.minute >= 15:
        logger.warning("⚠️ Market already opened. Run this before 9:15 AM IST.")
    
    logger.info("\n📦 Initializing components...")
    try:
        data_manager = DataManager(max_workers=1)
        sentiment_analyzer = SentimentAnalyzer()
        alert_manager = AlertManager()
        logger.info("✓ Components initialized")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("📂 STEP 1: LOADING YESTERDAY'S SIGNALS")
    logger.info("=" * 80)
    
    signals_dir = os.path.join(PROJECT_ROOT, 'signals')
    os.makedirs(signals_dir, exist_ok=True)
    
    try:
        signal_files = sorted(glob.glob(os.path.join(signals_dir, 'evening_*.csv')), reverse=True)
        
        if not signal_files:
            logger.warning("⚠️ No evening signals found. Run evening_run.py first.")
            alert_manager.send_alert("No evening signals found for morning validation", priority="WARNING")
            return
        
        latest_signal_file = signal_files[0]
        signals_df = pd.read_csv(latest_signal_file)
        
        logger.info(f"✓ Loaded {len(signals_df)} signals from {os.path.basename(latest_signal_file)}")
        logger.info(f"  BUY signals: {(signals_df['signal'] == 'BUY').sum()}")
        logger.info(f"  SELL signals: {(signals_df['signal'] == 'SELL').sum()}")
    
    except Exception as e:
        logger.error(f"❌ Failed to load signals: {e}")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("📰 STEP 2: CHECKING OVERNIGHT NEWS & EVENTS")
    logger.info("=" * 80)
    
    try:
        symbols_to_check = signals_df['symbol'].unique().tolist()[:10]  
        overnight_sentiment = {}
        sentiment_changes = []
        
        for symbol in symbols_to_check:
            try:
                sentiment = sentiment_analyzer.analyze_sentiment(
                    symbol,
                    days_back=1,  
                    max_articles=10
                )
                
                overnight_sentiment[symbol] = sentiment
                sentiment_col = None
                if 'sentiment_score' in signals_df.columns:
                    sentiment_col = 'sentiment_score'
                elif 'sentiment' in signals_df.columns:
                    sentiment_col = 'sentiment'
                else:
                    logger.debug(f"No sentiment column found in signals file for {symbol}")
                    continue
                
                if symbol in signals_df['symbol'].values:
                    old_sentiment = signals_df[signals_df['symbol'] == symbol][sentiment_col].iloc[0]
                else:
                    old_sentiment = 0
                
                new_sentiment = sentiment['sentiment_score']
                sentiment_diff = new_sentiment - old_sentiment
                
                if abs(sentiment_diff) > 0.3:  
                    sentiment_changes.append({
                        'symbol': symbol,
                        'old': old_sentiment,
                        'new': new_sentiment,
                        'change': sentiment_diff,
                        'classification': sentiment['classification']
                    })
                
                logger.debug(f"{symbol}: {sentiment['classification']} ({sentiment['sentiment_score']:.2f})")
            
            except Exception as e:
                logger.debug(f"Sentiment check failed for {symbol}: {e}")
        
        logger.info(f"✓ Checked overnight sentiment for {len(overnight_sentiment)} symbols")
        
        if sentiment_changes:
            logger.warning(f"⚠️ Significant sentiment changes detected for {len(sentiment_changes)} symbols:")
            for change in sentiment_changes:
                logger.warning(
                    f"  {change['symbol']}: {change['old']:.2f} → {change['new']:.2f} "
                    f"({change['change']:+.2f})"
                )
    
    except Exception as e:
        logger.warning(f"⚠️ Overnight sentiment check failed: {e}")
        overnight_sentiment = {}
        sentiment_changes = []
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 STEP 3: CHECKING PRE-MARKET DATA")
    logger.info("=" * 80)
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)  
        symbols_to_fetch = signals_df['symbol'].unique().tolist()
        
        logger.info(f"Fetching latest prices for {len(symbols_to_fetch)} symbols...")
        
        market_data = data_manager.fetch_data(
            symbols=symbols_to_fetch,
            start_date=start_date,
            end_date=end_date,
            use_cache=False 
        )
        
        price_updates = {}
        for symbol, df in market_data.items():
            if df is not None and not df.empty:
                latest_price = df.iloc[-1]['close']
                price_updates[symbol] = latest_price
        
        logger.info(f"✓ Updated prices for {len(price_updates)} symbols")
    
    except Exception as e:
        logger.warning(f"⚠️ Price update failed: {e}")
        price_updates = {}
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ STEP 4: VALIDATING SIGNALS")
    logger.info("=" * 80)
    
    try:
        validated_signals = []
        rejected_signals = []
        
        for idx, row in signals_df.iterrows():
            symbol = row['symbol']
            signal_type = row['signal']
            original_price = row['entry_price']
            validation_reasons = []
            is_valid = True
            
            if symbol in overnight_sentiment:
                new_sentiment = overnight_sentiment[symbol]['sentiment_score']
                if signal_type == 'BUY' and new_sentiment < -0.3:
                    is_valid = False
                    validation_reasons.append(f"Bearish overnight sentiment ({new_sentiment:.2f})")
                elif signal_type == 'SELL' and new_sentiment > 0.3:
                    is_valid = False
                    validation_reasons.append(f"Bullish overnight sentiment ({new_sentiment:.2f})")
            
            if symbol in price_updates:
                latest_price = price_updates[symbol]
                price_change = (latest_price - original_price) / original_price
                if signal_type == 'BUY' and price_change < -0.03: 
                    is_valid = False
                    validation_reasons.append(f"Price dropped {price_change:.1%}")
                elif signal_type == 'SELL' and price_change > 0.03:  
                    is_valid = False
                    validation_reasons.append(f"Price rose {price_change:.1%}")
            
            signal_dict = row.to_dict()
            signal_dict['validation_reasons'] = validation_reasons
            
            if is_valid:
                validated_signals.append(signal_dict)
            else:
                rejected_signals.append(signal_dict)
        
        logger.info(f"✓ Validation complete:")
        logger.info(f"  Validated: {len(validated_signals)}")
        logger.info(f"  Rejected: {len(rejected_signals)}")
        
        if rejected_signals:
            logger.warning("⚠️ Rejected signals:")
            for sig in rejected_signals:
                logger.warning(f"  {sig['symbol']}: {', '.join(sig['validation_reasons'])}")
    
    except Exception as e:
        logger.error(f"❌ Signal validation failed: {e}")
        validated_signals = signals_df.to_dict('records')
        rejected_signals = []
    
    logger.info("\n" + "=" * 80)
    logger.info("📧 STEP 5: SENDING MORNING BRIEF")
    logger.info("=" * 80)
    
    try:
        brief_msg = f"""
☀️ MORNING MARKET BRIEF
📅 {datetime.now().strftime('%A, %B %d, %Y')}

Signal Validation:
• Total Signals: {len(signals_df)}
• ✅ Validated: {len(validated_signals)}
• ❌ Rejected: {len(rejected_signals)}

Validated Trades (Ready for Execution):
"""
        
        for i, sig in enumerate(validated_signals[:5], 1):
            brief_msg += f"\n{i}. {sig['symbol']} - {sig['signal']}"
            brief_msg += f"\n   Entry: ₹{sig['entry_price']:.2f}"
            brief_msg += f"\n   Target: ₹{sig['target']:.2f}"
            brief_msg += f"\n   Shares: {int(sig['shares'])}"
            
            if sig['symbol'] in overnight_sentiment:
                sent_class = overnight_sentiment[sig['symbol']]['classification']
                brief_msg += f"\n   Sentiment: {sent_class}"
        
        if rejected_signals:
            brief_msg += f"\n\n⚠️ Rejected Signals:"
            for sig in rejected_signals[:3]:
                brief_msg += f"\n• {sig['symbol']}: {sig['validation_reasons'][0]}"
        
        if sentiment_changes:
            brief_msg += f"\n\n📰 Overnight Sentiment Changes:"
            for change in sentiment_changes[:3]:
                brief_msg += f"\n• {change['symbol']}: {change['classification']}"
        
        brief_msg += f"\n\nMarket Opens: 9:15 AM IST"
        brief_msg += f"\nRecommendations: {len(validated_signals)} trades ready"
        alert_manager.send_alert(brief_msg, priority="INFO")
        logger.info("✓ Morning brief sent")
        for sig in validated_signals[:3]:
            alert_manager.send_trade_signal(
                symbol=sig['symbol'],
                signal=sig['signal'],
                confidence=sig['confidence'],
                price=sig['entry_price'],
                target=sig['target'],
                stop_loss=sig.get('stop_loss', sig['entry_price'] * 0.98),
                additional_info={
                    'Shares': int(sig['shares']),
                    'Investment': f"₹{sig['investment']:,.0f}",
                    'Status': '✅ VALIDATED'
                }
            )
    
    except Exception as e:
        logger.warning(f"⚠️ Morning brief failed: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("💾 SAVING VALIDATED SIGNALS")
    logger.info("=" * 80)
    
    try:
        if validated_signals:
            validated_df = pd.DataFrame(validated_signals)
            filename = os.path.join(
                signals_dir,
                f"morning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            validated_df.to_csv(filename, index=False)
            logger.info(f"✓ Validated signals saved to {filename}")
        
        if rejected_signals:
            rejected_df = pd.DataFrame(rejected_signals)
            filename = os.path.join(
                signals_dir,
                f"rejected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            rejected_df.to_csv(filename, index=False)
            logger.info(f"✓ Rejected signals saved to {filename}")
    
    except Exception as e:
        logger.warning(f"⚠️ Failed to save signals: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ MORNING VALIDATION COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Ready for market open at 9:15 AM IST")
    logger.info(f"Validated trades: {len(validated_signals)}")
    
    if validated_signals:
        total_investment = sum(s.get('investment', 0) for s in validated_signals)
        logger.info(f"Total investment: ₹{total_investment:,.0f}")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'signals'), exist_ok=True)
    
    try:
        morning_validation()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Validation interrupted by user")
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")
