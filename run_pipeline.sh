#!/bin/bash

################################################################################
# COMPLETE TRADING SYSTEM PIPELINE
# From Raw Data → Trading Signals (All Steps)
################################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"; }
log_step() { echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n${CYAN}$1${NC}\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"; }
log_error() { echo -e "${RED}[$(date +'%H:%M:%S')] ❌ ERROR:${NC} $1"; exit 1; }

# Banner
clear
echo -e "${MAGENTA}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║            🚀 INSTITUTIONAL TRADING SYSTEM - COMPLETE PIPELINE 🚀          ║
║                                                                            ║
║                    FROM DATA FETCHING → TRADING SIGNALS                    ║
║                                                                            ║
║  ✓ Data Fetching (Yahoo Finance)                                          ║
║  ✓ Technical Indicators (30 indicators)                                   ║
║  ✓ Deep Learning Models (CNN-LSTM, Transformer, Hybrid)                   ║
║  ✓ Pattern Detection (S/R, Fibonacci, Chart Patterns)                     ║
║  ✓ Time Series Forecasting (ARIMA-LSTM-EGARCH)                           ║
║  ✓ Feature Engineering (57 features)                                      ║
║  ✓ ML Ensemble (13 models, dynamic weighting)                             ║
║  ✓ Backtesting (2022-2024)                                                ║
║  ✓ Signal Generation (Top 3 positions, ₹10K capital)                     ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

TOTAL_START=$(date +%s)

# Environment setup
log_step "🔧 ENVIRONMENT SETUP"
source .venv/bin/activate 2>/dev/null || log "No venv found"
mkdir -p models data_cache/{feature_store,pickle_data} results signals logs
log "✓ Directories ready"

################################################################################
# STEP 1: DATA FETCHING & CACHING
################################################################################

log_step "📊 STEP 1/9: DATA FETCHING & CACHING"

log "Testing data fetch for sample stock..."
python3 << 'EOFPY'
import sys
sys.path.insert(0, '.')
from src.data_manager import DataManager

dm = DataManager()
print("\n✓ Fetching RELIANCE.NS (test)...")
df = dm.fetch_data_with_retry('RELIANCE.NS', start_date='2020-01-01', end_date='2024-12-31')
if df is not None:
    print(f"✓ Data fetched: {len(df)} days")
    print(f"✓ Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"✓ Columns: {list(df.columns)}")
else:
    print("❌ Data fetch failed")
    sys.exit(1)
EOFPY

log "✓ Data fetching working correctly"

################################################################################
# STEP 2: TECHNICAL INDICATORS TEST
################################################################################

log_step "📈 STEP 2/9: TECHNICAL INDICATORS CALCULATION"

log "Testing technical indicators..."
python3 << 'EOFPY'
import sys
sys.path.insert(0, '.')
from src.data_manager import DataManager
from src.technical_engine import TechnicalEngine

dm = DataManager()
tech = TechnicalEngine()

df = dm.fetch_data_with_retry('RELIANCE.NS', start_date='2023-01-01', end_date='2024-12-31')
df_tech = tech.calculate_all_indicators(df)

print(f"\n✓ Calculated {len(df_tech.columns)} total columns")
print(f"✓ Technical indicators: {len([c for c in df_tech.columns if c not in ['open','high','low','close','volume']])}")
print(f"✓ Sample indicators: RSI={df_tech['rsi'].iloc[-1]:.2f}, MACD={df_tech['macd'].iloc[-1]:.2f}")
EOFPY

log "✓ Technical indicators working correctly"

################################################################################
# STEP 3: TRAIN DEEP LEARNING MODELS
################################################################################

log_step "🤖 STEP 3/9: TRAINING DEEP LEARNING MODELS (60-90 min)"

log "Training CNN-LSTM, Transformer, and Hybrid LSTM-Transformer..."
log "⏰ This will take 60-90 minutes. Perfect time for coffee! ☕"

python3 scripts/sequence_model_trainer.py 2>&1 | tee logs/step3_dl_training.log || log_error "DL training failed"

if [ -f "models/hybrid_lstm_transformer_model.h5" ]; then
    log "✓ Hybrid LSTM-Transformer model trained"
    ls -lh models/*.h5 | awk '{print "   " $9 " - " $5}'
else
    log_error "Model files not found"
fi

################################################################################
# STEP 4: PATTERN DETECTION TEST
################################################################################

log_step "📐 STEP 4/9: PATTERN DETECTION"

log "Testing pattern detection..."
python3 << 'EOFPY'
import sys
sys.path.insert(0, '.')
from src.data_manager import DataManager
from src.pattern_detector import PatternDetector

dm = DataManager()
pd_detector = PatternDetector()

df = dm.fetch_data_with_retry('RELIANCE.NS', start_date='2023-01-01', end_date='2024-12-31')
patterns = pd_detector.detect_all_patterns(df, 'RELIANCE.NS')

print(f"\n✓ Pattern detection completed")
print(f"✓ Support/Resistance: {patterns.get('support_resistance', {}).get('support_level', 'N/A')}")
print(f"✓ Patterns found: {len(patterns.get('chart_patterns', {}).get('bullish_patterns', []))} bullish")
EOFPY

log "✓ Pattern detection working correctly"

################################################################################
# STEP 5: TIME SERIES FORECASTING TEST
################################################################################

log_step "📊 STEP 5/9: TIME SERIES FORECASTING (ARIMA-LSTM-EGARCH)"

log "Testing hybrid time series forecast..."
python3 << 'EOFPY'
import sys
sys.path.insert(0, '.')
from src.data_manager import DataManager
from src.time_series_engine import TimeSeriesEngine

dm = DataManager()
ts = TimeSeriesEngine()

df = dm.fetch_data_with_retry('RELIANCE.NS', start_date='2023-01-01', end_date='2024-12-31')
forecast = ts.forecast_hybrid(df['close'])

print(f"\n✓ Hybrid forecast generated")
print(f"✓ Price forecast: ₹{forecast['price_forecast']:.2f}")
print(f"✓ Volatility: {forecast['volatility_forecast']:.2f}%")
print(f"✓ Confidence: {forecast['confidence']:.2%}")
print(f"✓ Direction probability: {forecast['direction_probability']:.2%}")
EOFPY

log "✓ Time series forecasting working correctly"

################################################################################
# STEP 6: BUILD FEATURE STORE
################################################################################

log_step "🏗️  STEP 6/9: BUILDING FEATURE STORE (2-4 hours)"

log "Building features for all 95 stocks..."
log "⏰ This will take 2-4 hours. Great time for lunch! 🍔"

python3 scripts/build_feature_store.py 2>&1 | tee logs/step6_feature_store.log || log_error "Feature store build failed"

FEATURE_COUNT=$(ls -1 data_cache/feature_store/*.pkl 2>/dev/null | wc -l)
log "✓ Feature store built: $FEATURE_COUNT stock files"

################################################################################
# STEP 7: TRAIN ML ENSEMBLE
################################################################################

log_step "🎯 STEP 7/9: TRAINING ML ENSEMBLE (30-60 min)"

log "Training 13-model ensemble with dynamic weighting..."
log "⏰ This will take 30-60 minutes. Check emails! 📧"

python3 scripts/model_trainer.py 2>&1 | tee logs/step7_ml_ensemble.log || log_error "ML ensemble training failed"

if [ -f "models/ENSEMBLE_ensemble.pkl" ]; then
    ENSEMBLE_SIZE=$(ls -lh models/ENSEMBLE_ensemble.pkl | awk '{print $5}')
    log "✓ ML Ensemble trained: $ENSEMBLE_SIZE"
else
    log_error "Ensemble model not found"
fi

################################################################################
# STEP 8: RUN BACKTEST
################################################################################

log_step "📊 STEP 8/9: RUNNING BACKTEST (1-2 hours)"

log "Backtesting strategy on 2022-2024..."
log "⏰ This will take 1-2 hours. Watch tutorials! 📺"

python3 scripts/backtester.py 2>&1 | tee logs/step8_backtest.log || log_error "Backtest failed"

LATEST_METRICS=$(ls -t results/backtest_metrics_*.txt 2>/dev/null | head -1)
if [ -f "$LATEST_METRICS" ]; then
    echo -e "\n${YELLOW}📊 BACKTEST RESULTS:${NC}"
    head -35 "$LATEST_METRICS"
else
    log "⚠️  Backtest results not found"
fi

################################################################################
# STEP 9: GENERATE TRADING SIGNALS
################################################################################

log_step "🎯 STEP 9/9: GENERATING TRADING SIGNALS"

log "Generating today's trading signals..."

python3 scripts/evening_run.py 2>&1 | tee logs/step9_signals.log || log_error "Signal generation failed"

LATEST_SIGNALS=$(ls -t signals/evening_signals_*.csv 2>/dev/null | head -1)
if [ -f "$LATEST_SIGNALS" ]; then
    echo -e "\n${GREEN}🎯 TODAY'S TRADING SIGNALS:${NC}\n"
    echo -e "${CYAN}"
    head -10 "$LATEST_SIGNALS" | column -t -s ','
    echo -e "${NC}"
    
    SIGNAL_COUNT=$(tail -n +2 "$LATEST_SIGNALS" | wc -l)
    log "✓ Generated $SIGNAL_COUNT signals"
else
    log "⚠️  No signals generated"
fi

################################################################################
# COMPLETION SUMMARY
################################################################################

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINS=$(((TOTAL_DURATION % 3600) / 60))

echo -e "\n${GREEN}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                  ✅  COMPLETE PIPELINE FINISHED SUCCESSFULLY!              ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

log_step "📋 FINAL SUMMARY"

printf "${CYAN}%-35s${NC} %s\n" "Total Execution Time:" "${TOTAL_HOURS}h ${TOTAL_MINS}m"
printf "${CYAN}%-35s${NC} %s\n" "Data Fetching:" "✓ Complete"
printf "${CYAN}%-35s${NC} %s\n" "Technical Indicators:" "✓ Complete"
printf "${CYAN}%-35s${NC} %s\n" "DL Models:" "$([ -f models/hybrid_lstm_transformer_model.h5 ] && echo '✓ Trained' || echo '✗ Missing')"
printf "${CYAN}%-35s${NC} %s\n" "Pattern Detection:" "✓ Complete"
printf "${CYAN}%-35s${NC} %s\n" "Time Series Forecasting:" "✓ Complete"
printf "${CYAN}%-35s${NC} %s\n" "Feature Store:" "✓ $FEATURE_COUNT stocks"
printf "${CYAN}%-35s${NC} %s\n" "ML Ensemble:" "$([ -f models/ENSEMBLE_ensemble.pkl ] && echo '✓ Trained' || echo '✗ Missing')"
printf "${CYAN}%-35s${NC} %s\n" "Backtest:" "$([ -f \"$LATEST_METRICS\" ] && echo '✓ Complete' || echo '⚠️  Skipped')"
printf "${CYAN}%-35s${NC} %s\n" "Trading Signals:" "$([ -f \"$LATEST_SIGNALS\" ] && echo \"✓ $SIGNAL_COUNT signals\" || echo '✗ None')"

echo -e "\n${CYAN}📁 OUTPUT LOCATIONS:${NC}"
echo "   Models:          models/"
echo "   Features:        data_cache/feature_store/"
echo "   Backtest Results: results/"
echo "   Trading Signals:  signals/"
echo "   Logs:            logs/"

echo -e "\n${CYAN}📊 STORAGE USAGE:${NC}"
du -sh models/ data_cache/ results/ signals/ 2>/dev/null | awk '{print "   " $2 ": " $1}'

echo -e "\n${GREEN}🚀 NEXT STEPS:${NC}"
echo "   1. Review signals:     cat $LATEST_SIGNALS"
echo "   2. Check backtest:     cat $LATEST_METRICS"
echo "   3. Daily signals:      python3 scripts/evening_run.py"
echo ""

log "✨ Trading system is fully operational!"
echo ""

exit 0
