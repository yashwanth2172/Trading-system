"""
Enhanced Backtester with Capital Management Fix
COMPLETE VERSION - No negative capital bug
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import (
    TRADE_UNIVERSE, CAPITAL_CONFIG, PROJECT_DIRECTORIES,
    ML_FEATURES, RISK_PARAMS
)
from src.data_manager import DataManager
from src.technical_engine import TechnicalEngine
from src.ml_ensemble import Enhanced13ModelEnsemble
from src.risk_manager import RiskManager
from src.time_series_engine import TimeSeriesEngine
from src.pattern_detector import PatternDetector
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedBacktester:
    """
    Complete backtester with:
    - FIXED capital management (no negative capital!)
    - Hybrid ML predictions
    - Time series forecasts
    - Realistic position sizing
    - Stop loss / Take profit execution
    - Performance metrics
    """
    
    def __init__(self, initial_capital: float = 1000000):
        """
        Args:
            initial_capital: Starting capital for backtest
        """
        self.initial_capital = initial_capital
        
        # Initialize components
        self.data_manager = DataManager()
        self.tech_engine = TechnicalEngine()
        self.pattern_detector = PatternDetector()
        self.ts_engine = TimeSeriesEngine()
        
        # ✅ Risk manager with PROPER capital tracking
        self.risk_manager = RiskManager(total_capital=initial_capital)
        
        # Directories
        self.model_dir = Path(PROJECT_DIRECTORIES['MODELS'])
        self.feature_store_dir = Path(PROJECT_DIRECTORIES['FEATURE_STORE'])
        self.results_dir = Path(PROJECT_DIRECTORIES['RESULTS'])
        self.results_dir.mkdir(exist_ok=True)
        
        # Models (lazy loading)
        self.ml_ensemble = None
        
        # Tracking
        self.portfolio_history = []
        self.trades_history = []
        self.daily_positions = []
        
        logger.info("=" * 80)
        logger.info("ENHANCED BACKTESTER INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Initial Capital: ₹{initial_capital:,.0f}")
        logger.info(f"Max Positions: {CAPITAL_CONFIG['MAX_POSITIONS']}")
        logger.info("=" * 80)
    
    def _load_models(self):
        """Load ML ensemble"""
        if self.ml_ensemble is not None:
            return
        
        logger.info("Loading ML ensemble...")
        
        ensemble_path = self.model_dir / 'ENSEMBLE_ensemble.pkl'
        if ensemble_path.exists():
            try:
                self.ml_ensemble = Enhanced13ModelEnsemble.load(str(ensemble_path))
                logger.info("✓ ML Ensemble loaded")
            except Exception as e:
                logger.error(f"ML Ensemble load failed: {e}")
    
    def load_features(self, symbol: str) -> pd.DataFrame:
        """Load features from feature store"""
        feature_file = self.feature_store_dir / f"{symbol.replace('.NS', '')}_features.pkl"
        
        if not feature_file.exists():
            return None
        
        try:
            features = pd.read_pickle(str(feature_file))
            return features
        except Exception as e:
            logger.error(f"Error loading features for {symbol}: {e}")
            return None
    
    def generate_signal_for_date(self, symbol: str, date: pd.Timestamp, 
                                 features_df: pd.DataFrame) -> dict:
        """
        Generate trading signal for specific date
        
        Returns:
            Signal dict or None
        """
        try:
            # Get features for this date
            if date not in features_df.index:
                return None
            
            features = features_df.loc[date]
            
            # Get current price
            price_data = self.data_manager.get_cached_data(symbol)
            if price_data is None or date not in price_data.index:
                return None
            
            current_price = float(price_data.loc[date]['close'])
            
            # ML Prediction
            ml_prediction = None
            if self.ml_ensemble:
                try:
                    feature_vector = features[ML_FEATURES].values.reshape(1, -1)
                    ml_prediction = self.ml_ensemble.predict(feature_vector)
                except:
                    ml_prediction = None
            
            if ml_prediction is None:
                return None
            
            # Get forecasts from features
            ts_prob = features.get('direction_prob', 0.5)
            dl_hybrid_prob = features.get('hybrid_lstm_trans_prob', 0.5)
            volatility = features.get('egarch_volatility', 2.0)
            
            # Combine signals
            final_score = (
                ml_prediction['probability'] * 0.5 +
                ts_prob * 0.3 +
                dl_hybrid_prob * 0.2
            )
            
            # Calculate confidence
            scores = [ml_prediction['probability'], ts_prob, dl_hybrid_prob]
            confidence = max(0.3, 1.0 - np.var(scores))
            
            # Signal threshold
            if final_score < 0.65 or confidence < 0.6:
                return None
            
            # Calculate stop loss and target
            risk_amount = current_price * (volatility / 100) * 2.0
            stop_loss = current_price - risk_amount
            
            reward_mult = 2.0 if confidence > 0.7 else 1.5
            target = current_price + (risk_amount * reward_mult)
            
            signal = {
                'symbol': symbol,
                'date': date,
                'current_price': current_price,
                'confidence': confidence,
                'final_score': final_score,
                'stop_loss': stop_loss,
                'target': target,
                'reward_risk_ratio': reward_mult,
                'volatility_forecast': volatility,
                'ml_probability': ml_prediction['probability']
            }
            
            return signal
        
        except Exception as e:
            logger.debug(f"Signal generation failed for {symbol} on {date}: {e}")
            return None
    
    def check_exit_conditions(self, symbol: str, entry_info: dict, 
                             current_price: float, current_date: pd.Timestamp) -> tuple:
        """
        Check if position should be exited
        
        Returns:
            (should_exit: bool, exit_reason: str)
        """
        # Check stop loss
        if current_price <= entry_info['stop_loss']:
            return True, 'stop_loss'
        
        # Check target
        if current_price >= entry_info['target']:
            return True, 'target'
        
        # Check max holding period (e.g., 5 days)
        days_held = (current_date - entry_info['entry_date']).days
        if days_held >= 5:
            return True, 'max_holding'
        
        return False, None
    
    def run_backtest(self, symbols: list = None, 
                    start_date: str = '2022-01-01',
                    end_date: str = '2024-12-31') -> dict:
        """
        Run complete backtest simulation
        
        Args:
            symbols: List of symbols to trade
            start_date: Backtest start date
            end_date: Backtest end date
        
        Returns:
            Dict with results
        """
        if symbols is None:
            symbols = TRADE_UNIVERSE[:95]
        
        logger.info("=" * 80)
        logger.info("RUNNING BACKTEST SIMULATION (FAST)")
        logger.info("=" * 80)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Initial Capital: ₹{self.initial_capital:,.0f}")
        logger.info("=" * 80)
        
        # Load models
        self._load_models()
        
        # Load all features
        logger.info("\nLoading features...")
        stock_features = {}
        stock_data = {}
        
        for symbol in symbols:
            features = self.load_features(symbol)
            if features is not None:
                stock_features[symbol] = features
                
                # Cache price data
                df = self.data_manager.fetch_data_with_retry(
                    symbol, start_date=start_date, end_date=end_date
                )
                if df is not None:
                    stock_data[symbol] = df
        
        logger.info(f"✓ Loaded features for {len(stock_features)} symbols")
        
        # Get trading days
        sample_df = list(stock_data.values())[0]
        trading_days = pd.date_range(start=start_date, end=end_date, freq='B')
        trading_days = [d for d in trading_days if d in sample_df.index]
        
        logger.info(f"\nSimulating {len(trading_days)} trading days...")
        
        # Simulate day by day
        for day_idx, current_date in enumerate(tqdm(trading_days, desc="Simulating Days")):
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 1: Check exits for existing positions
            # ═══════════════════════════════════════════════════════════════
            
            positions_to_exit = []
            
            for symbol in list(self.risk_manager.current_positions.keys()):
                if symbol not in stock_data:
                    continue
                
                # Get current price
                if current_date not in stock_data[symbol].index:
                    continue
                
                current_price = float(stock_data[symbol].loc[current_date]['close'])
                
                # Update position value
                self.risk_manager.update_position(symbol, current_price)
                
                # Check exit conditions
                position_info = self.risk_manager.current_positions[symbol]
                should_exit, exit_reason = self.check_exit_conditions(
                    symbol, position_info, current_price, current_date
                )
                
                if should_exit:
                    # Exit position
                    exit_info = self.risk_manager.remove_position(symbol, current_price, exit_reason)
                    
                    if exit_info:
                        exit_info['exit_date'] = current_date
                        self.trades_history.append(exit_info)
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 2: Generate new signals
            # ═══════════════════════════════════════════════════════════════
            
            daily_signals = []
            
            for symbol in symbols:
                # Skip if already in position
                if symbol in self.risk_manager.current_positions:
                    continue
                
                # Skip if no data
                if symbol not in stock_features or symbol not in stock_data:
                    continue
                
                if current_date not in stock_data[symbol].index:
                    continue
                
                # Generate signal
                signal = self.generate_signal_for_date(
                    symbol, current_date, stock_features[symbol]
                )
                
                if signal:
                    daily_signals.append(signal)
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 3: Apply position sizing and enter positions
            # ═══════════════════════════════════════════════════════════════
            
            if daily_signals:
                # Sort by confidence
                daily_signals.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Limit to max positions
                max_new_positions = CAPITAL_CONFIG['MAX_POSITIONS'] - len(self.risk_manager.current_positions)
                daily_signals = daily_signals[:max_new_positions]
                
                for signal in daily_signals:
                    # Calculate position size
                    position_result = self.risk_manager.calculate_position_size(
                        signal=signal,
                        current_price=signal['current_price'],
                        method='kelly',
                        ml_probability=signal['ml_probability']
                    )
                    
                    # Check if approved
                    if position_result['validation']['approved']:
                        # Add entry info
                        position_result['entry_date'] = current_date
                        position_result['entry_reason'] = 'signal'
                        
                        # Enter position
                        success = self.risk_manager.add_position(position_result)
                        
                        if not success:
                            logger.warning(f"Failed to enter {signal['symbol']}")
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 4: Record daily portfolio state
            # ═══════════════════════════════════════════════════════════════
            
            portfolio_summary = self.risk_manager.get_portfolio_summary()
            
            self.portfolio_history.append({
                'date': current_date,
                'total_value': portfolio_summary['total_value'],
                'available_cash': portfolio_summary['available_cash'],
                'positions_value': portfolio_summary['positions_value'],
                'num_positions': portfolio_summary['total_positions'],
                'total_pnl': portfolio_summary['total_pnl'],
                'total_pnl_pct': portfolio_summary['total_pnl_pct']
            })
            
            # Log progress
            if (day_idx + 1) % 100 == 0:
                logger.info(
                    f"  Day {day_idx+1}/{len(trading_days)} ({current_date.date()}): "
                    f"₹{portfolio_summary['total_value']:,.0f} ({portfolio_summary['total_positions']} positions)"
                )
        
        logger.info(f"\n✓ Simulation complete: {len(self.trades_history)} trades executed")
        
        # ═══════════════════════════════════════════════════════════════════
        # CALCULATE PERFORMANCE METRICS
        # ═══════════════════════════════════════════════════════════════════
        
        logger.info("\n" + "=" * 80)
        logger.info("CALCULATING PERFORMANCE METRICS")
        logger.info("=" * 80)
        
        results = self.calculate_metrics()
        
        # ═══════════════════════════════════════════════════════════════════
        # SAVE RESULTS
        # ═══════════════════════════════════════════════════════════════════
        
        logger.info("\n" + "=" * 80)
        logger.info("SAVING RESULTS")
        logger.info("=" * 80)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save trades
        if self.trades_history:
            trades_df = pd.DataFrame(self.trades_history)
            trades_file = self.results_dir / f'backtest_trades_{timestamp}.csv'
            trades_df.to_csv(str(trades_file), index=False)
            logger.info(f"\n✓ Trades saved: {trades_file}")
        
        # Save portfolio history
        if self.portfolio_history:
            portfolio_df = pd.DataFrame(self.portfolio_history)
            portfolio_file = self.results_dir / f'backtest_portfolio_{timestamp}.csv'
            portfolio_df.to_csv(str(portfolio_file), index=False)
            logger.info(f"✓ Portfolio values saved: {portfolio_file}")
        
        # Save metrics
        metrics_file = self.results_dir / f'backtest_metrics_{timestamp}.txt'
        with open(metrics_file, 'w') as f:
            f.write("BACKTEST RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Initial Capital: ₹{results['initial_capital']:,.0f}\n")
            f.write(f"Final Value: ₹{results['final_value']:,.0f}\n")
            f.write(f"Total Return: {results['total_return_pct']:.2f}%\n")
            f.write(f"Total P&L: ₹{results['total_pnl']:,.0f}\n\n")
            
            f.write("TRADE STATISTICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Trades: {results['total_trades']}\n")
            f.write(f"Winning Trades: {results['winning_trades']}\n")
            f.write(f"Losing Trades: {results['losing_trades']}\n")
            f.write(f"Win Rate: {results['win_rate']:.2f}%\n\n")
            
            f.write("P&L ANALYSIS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Average P&L: ₹{results['avg_pnl']:,.0f} ({results['avg_pnl_pct']:.2f}%)\n")
            f.write(f"Average Win: ₹{results['avg_win']:,.0f}\n")
            f.write(f"Average Loss: ₹{results['avg_loss']:,.0f}\n")
            f.write(f"Profit Factor: {results['profit_factor']:.2f}\n\n")
            
            f.write("RISK METRICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Max Drawdown: {results['max_drawdown_pct']:.2f}% (₹{results['max_drawdown']:,.0f})\n")
            f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
        
        logger.info(f"✓ Metrics saved: {metrics_file}")
        
        logger.info("\n✅ Backtest complete")
        
        return results
    
    def calculate_metrics(self) -> dict:
        """Calculate comprehensive performance metrics"""
        
        # Portfolio values
        portfolio_df = pd.DataFrame(self.portfolio_history)
        
        initial_capital = self.initial_capital
        final_value = portfolio_df['total_value'].iloc[-1] if len(portfolio_df) > 0 else initial_capital
        total_pnl = final_value - initial_capital
        total_return_pct = (total_pnl / initial_capital) * 100
        
        # Trade statistics
        trades_df = pd.DataFrame(self.trades_history) if self.trades_history else pd.DataFrame()
        
        total_trades = len(trades_df)
        
        if total_trades > 0:
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            win_rate = (winning_trades / total_trades) * 100
            
            avg_pnl = trades_df['pnl'].mean()
            avg_pnl_pct = trades_df['pnl_pct'].mean()
            
            wins = trades_df[trades_df['pnl'] > 0]
            losses = trades_df[trades_df['pnl'] < 0]
            
            avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
            avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
            
            total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
            total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
            
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        else:
            winning_trades = losing_trades = 0
            win_rate = avg_pnl = avg_pnl_pct = avg_win = avg_loss = profit_factor = 0
        
        # Risk metrics
        if len(portfolio_df) > 1:
            # Calculate returns
            portfolio_df['returns'] = portfolio_df['total_value'].pct_change()
            
            # Sharpe ratio (annualized)
            returns_std = portfolio_df['returns'].std()
            if returns_std > 0:
                sharpe_ratio = (portfolio_df['returns'].mean() / returns_std) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Max drawdown
            cummax = portfolio_df['total_value'].cummax()
            drawdown = portfolio_df['total_value'] - cummax
            max_drawdown = drawdown.min()
            max_drawdown_pct = (max_drawdown / cummax[drawdown.idxmin()]) * 100
        else:
            sharpe_ratio = 0
            max_drawdown = 0
            max_drawdown_pct = 0
        
        # Print results
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 80)
        
        logger.info("\n📊 PERFORMANCE SUMMARY:")
        logger.info("-" * 80)
        logger.info(f"Initial Capital: ₹{initial_capital:,.0f}")
        logger.info(f"Final Value: ₹{final_value:,.0f}")
        logger.info(f"Total Return: {total_return_pct:.2f}%")
        logger.info(f"Total P&L: ₹{total_pnl:,.0f}")
        
        logger.info("\n📈 TRADE STATISTICS:")
        logger.info("-" * 80)
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Winning Trades: {winning_trades}")
        logger.info(f"Losing Trades: {losing_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        
        logger.info("\n💰 P&L ANALYSIS:")
        logger.info("-" * 80)
        logger.info(f"Average P&L: ₹{avg_pnl:,.0f} ({avg_pnl_pct:.2f}%)")
        logger.info(f"Average Win: ₹{avg_win:,.0f}")
        logger.info(f"Average Loss: ₹{avg_loss:,.0f}")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        
        logger.info("\n⚠️ RISK METRICS:")
        logger.info("-" * 80)
        logger.info(f"Max Drawdown: {max_drawdown_pct:.2f}% (₹{max_drawdown:,.0f})")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        logger.info("=" * 80)
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'avg_pnl_pct': avg_pnl_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio
        }


def main():
    """Main execution"""
    
    # Create backtester with initial capital
    initial_capital = CAPITAL_CONFIG.get('TOTAL_CAPITAL', 1000000)
    backtester = EnhancedBacktester(initial_capital=initial_capital)
    
    # Get symbols
    symbols = TRADE_UNIVERSE[:95]
    
    # Run backtest
    results = backtester.run_backtest(
        symbols=symbols,
        start_date='2022-01-01',
        end_date='2024-12-31'
    )
    
    return results


if __name__ == '__main__':
    main()
