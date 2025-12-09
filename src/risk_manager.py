"""
Enhanced Risk Manager with Proper Capital Tracking
FIXES: Capital going negative bug
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from scipy.optimize import minimize
from scipy import stats

from config.settings import RISK_PARAMS

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Enhanced Risk Manager with STRICT capital enforcement
    
    Key Features:
    - Proper cash tracking (no negative capital!)
    - Kelly Criterion position sizing
    - Portfolio-level risk limits
    - Dynamic position sizing based on volatility
    - Real-time capital allocation
    """
    
    def __init__(self, total_capital: float = 1000000, params: Optional[Dict] = None):
        """
        Args:
            total_capital: Initial capital
            params: Risk parameters (optional)
        """
        self.total_capital = total_capital
        self.available_cash = total_capital  # ✅ NEW: Track actual cash
        self.params = params or RISK_PARAMS
        
        # Risk parameters
        self.max_portfolio_risk = self.params.get('MAX_PORTFOLIO_RISK', 0.02)
        self.max_single_position = self.params.get('MAX_SINGLE_POSITION', 0.15)
        self.kelly_fraction = self.params.get('KELLY_FRACTION', 0.25)
        self.max_sector_exposure = self.params.get('MAX_SECTOR_EXPOSURE', 0.30)
        self.max_correlation = self.params.get('MAX_CORRELATION', 0.7)
        self.var_confidence = self.params.get('VAR_CONFIDENCE', 0.95)
        self.default_win_rate = 0.55
        
        # Position tracking
        self.current_positions = {}
        self.position_history = []
        
        logger.info(f"✓ Risk Manager initialized with ₹{total_capital:,.0f} capital")
        logger.info(f"   Available cash: ₹{self.available_cash:,.0f}")
        logger.info(f"   Max position: {self.max_single_position:.0%}, Max risk: {self.max_portfolio_risk:.0%}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # POSITION SIZING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def calculate_position_size(
        self,
        signal: Dict,
        current_price: float,
        method: str = 'kelly',
        ml_probability: Optional[float] = None
    ) -> Dict:
        """
        Calculate position size with STRICT capital enforcement
        
        Args:
            signal: Trading signal dict
            current_price: Current stock price
            method: 'kelly', 'volatility', or 'fixed'
            ml_probability: ML model probability (0-1)
        
        Returns:
            Dict with position details or validation rejection
        """
        symbol = signal['symbol']
        confidence = signal.get('confidence', 0.5)
        target = signal.get('target', current_price * 1.05)
        stop_loss = signal.get('stop_loss', current_price * 0.98)
        
        # Calculate risk metrics
        risk_per_share = abs(current_price - stop_loss)
        if risk_per_share == 0:
            risk_per_share = current_price * 0.02
        
        reward_per_share = abs(target - current_price)
        reward_risk_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 1.0
        
        # Win probability
        if ml_probability is not None and 0 <= ml_probability <= 1:
            win_probability = ml_probability
        elif confidence is not None and 0 <= confidence <= 1:
            win_probability = confidence
        else:
            win_probability = self.default_win_rate
        
        # Calculate position size percentage
        if method == 'kelly':
            position_size_pct = self._kelly_criterion(win_probability, reward_risk_ratio)
        elif method == 'volatility':
            position_size_pct = self._volatility_based_sizing(signal, current_price)
        else:
            position_size_pct = self.max_single_position
        
        # Cap at max position size
        position_size_pct = min(position_size_pct, self.max_single_position)
        
        # ✅ CRITICAL: Calculate based on available cash (not total capital)
        position_value = self.available_cash * position_size_pct
        
        # Calculate shares
        shares = int(position_value / current_price)
        actual_position_value = shares * current_price
        
        # ✅ FINAL CHECK: Ensure we have enough cash
        if actual_position_value > self.available_cash:
            logger.warning(
                f"✗ {symbol}: Insufficient cash. Need ₹{actual_position_value:,.0f}, "
                f"Available: ₹{self.available_cash:,.0f}"
            )
            return {
                'validation': {
                    'approved': False,
                    'reason': f'Insufficient cash (need ₹{actual_position_value:,.0f})'
                },
                'symbol': symbol
            }
        
        # Calculate metrics
        total_portfolio_value = self.get_total_portfolio_value()
        actual_position_pct = actual_position_value / total_portfolio_value
        risk_amount = shares * risk_per_share
        risk_pct = risk_amount / total_portfolio_value
        
        result = {
            'symbol': symbol,
            'shares': shares,
            'position_value': actual_position_value,
            'position_pct': actual_position_pct,
            'risk_amount': risk_amount,
            'risk_pct': risk_pct,
            'entry_price': current_price,
            'target': target,
            'stop_loss': stop_loss,
            'reward_risk_ratio': reward_risk_ratio,
            'method_used': method,
            'confidence': confidence,
            'ml_probability': ml_probability,
            'win_probability_used': win_probability,
            'timestamp': datetime.now().isoformat()
        }
        
        # Validate position
        validation = self._validate_position(result)
        result['validation'] = validation
        
        if validation['approved']:
            logger.info(
                f"✓ {symbol}: {shares} shares @ ₹{current_price:.2f} "
                f"(₹{actual_position_value:,.0f}, {actual_position_pct:.1%}) "
                f"Risk: {risk_pct:.2%}, Win Prob: {win_probability:.1%}"
            )
        else:
            logger.warning(f"✗ {symbol} position REJECTED: {validation['reason']}")
        
        return result
    
    def _kelly_criterion(self, win_prob: float, reward_risk_ratio: float) -> float:
        """Kelly Criterion with fractional Kelly for safety"""
        if reward_risk_ratio <= 0 or win_prob <= 0:
            return 0.01
        
        q = 1 - win_prob
        kelly_pct = (win_prob * reward_risk_ratio - q) / reward_risk_ratio
        
        # Apply fractional Kelly for safety
        kelly_pct = kelly_pct * self.kelly_fraction
        
        # Ensure positive and within bounds
        kelly_pct = max(0.01, min(kelly_pct, self.max_single_position))
        
        return kelly_pct
    
    def _volatility_based_sizing(self, signal: Dict, current_price: float) -> float:
        """Position sizing based on volatility (inverse relationship)"""
        indicators = signal.get('indicators', {})
        atr = indicators.get('atr', current_price * 0.02)
        volatility_pct = atr / current_price
        
        # Lower position size for higher volatility
        if volatility_pct > 0.05:
            position_pct = 0.05
        elif volatility_pct > 0.03:
            position_pct = 0.08
        elif volatility_pct > 0.02:
            position_pct = 0.12
        else:
            position_pct = 0.15
        
        return min(position_pct, self.max_single_position)
    
    def _validate_position(self, position: Dict) -> Dict:
        """Validate position against all risk limits"""
        symbol = position['symbol']
        
        # Check 1: Minimum shares
        if position['shares'] < 1:
            return {'approved': False, 'reason': 'Position too small (< 1 share)'}
        
        # Check 2: Max position size
        if position['position_pct'] > self.max_single_position:
            return {
                'approved': False,
                'reason': f'Exceeds max position size ({self.max_single_position:.0%})'
            }
        
        # Check 3: Portfolio risk
        total_risk = self._calculate_total_portfolio_risk()
        new_total_risk = total_risk + position['risk_pct']
        
        if new_total_risk > self.max_portfolio_risk:
            return {
                'approved': False,
                'reason': f'Exceeds portfolio risk limit ({self.max_portfolio_risk:.0%})'
            }
        
        # Check 4: Available cash (CRITICAL)
        if position['position_value'] > self.available_cash:
            return {
                'approved': False,
                'reason': f'Insufficient cash (available: ₹{self.available_cash:,.0f})'
            }
        
        # Check 5: Minimum reward/risk ratio
        min_rr = self.params.get('MIN_RISK_REWARD', 1.5)
        if position['reward_risk_ratio'] < min_rr:
            return {
                'approved': False,
                'reason': f'R/R {position["reward_risk_ratio"]:.2f} < min {min_rr:.2f}'
            }
        
        return {'approved': True, 'reason': 'All risk checks passed'}
    
    def _calculate_total_portfolio_risk(self) -> float:
        """Calculate total portfolio risk from all positions"""
        return sum(pos.get('risk_pct', 0) for pos in self.current_positions.values())
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # POSITION MANAGEMENT (FIXED)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def add_position(self, position: Dict) -> bool:
        """
        ✅ FIXED: Add position and deduct cash
        
        Args:
            position: Position dict from calculate_position_size
        
        Returns:
            True if added successfully
        """
        symbol = position['symbol']
        cost = position['position_value']
        
        # Double check we have cash
        if cost > self.available_cash:
            logger.error(
                f"❌ Cannot add {symbol}: Insufficient cash. "
                f"Need ₹{cost:,.0f}, Available: ₹{self.available_cash:,.0f}"
            )
            return False
        
        # ✅ DEDUCT CASH
        self.available_cash -= cost
        
        # Store position
        self.current_positions[symbol] = position
        self.position_history.append({
            'action': 'OPEN',
            'timestamp': datetime.now().isoformat(),
            **position
        })
        
        logger.info(
            f"✓ Added position: {symbol} | "
            f"Cost: ₹{cost:,.0f} | Remaining cash: ₹{self.available_cash:,.0f}"
        )
        return True
    
    def remove_position(self, symbol: str, exit_price: float, 
                       exit_reason: str = 'signal') -> Optional[Dict]:
        """
        ✅ FIXED: Remove position and return cash
        
        Args:
            symbol: Stock symbol
            exit_price: Exit price
            exit_reason: Reason for exit
        
        Returns:
            Exit info dict
        """
        if symbol not in self.current_positions:
            return None
        
        pos = self.current_positions.pop(symbol)
        shares = pos['shares']
        entry_cost = pos['position_value']
        entry_price = pos['entry_price']
        
        # ✅ CALCULATE EXIT VALUE
        exit_value = shares * exit_price
        pnl = exit_value - entry_cost
        pnl_pct = (pnl / entry_cost) * 100 if entry_cost > 0 else 0
        
        # ✅ RETURN CASH
        self.available_cash += exit_value
        
        exit_info = {
            'action': 'CLOSE',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'shares': shares,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_cost': entry_cost,
            'exit_value': exit_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason
        }
        
        self.position_history.append(exit_info)
        
        logger.info(
            f"✓ Closed position: {symbol} | "
            f"P&L: ₹{pnl:,.0f} ({pnl_pct:+.2f}%) | "
            f"Cash now: ₹{self.available_cash:,.0f}"
        )
        
        return exit_info
    
    def update_position(self, symbol: str, current_price: float):
        """Update position with current market price"""
        if symbol in self.current_positions:
            pos = self.current_positions[symbol]
            pos['current_price'] = current_price
            pos['current_value'] = pos['shares'] * current_price
            pos['unrealized_pnl'] = (current_price - pos['entry_price']) * pos['shares']
            pos['unrealized_pnl_pct'] = (current_price / pos['entry_price'] - 1) * 100
            pos['last_updated'] = datetime.now().isoformat()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PORTFOLIO SUMMARY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def get_total_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)"""
        positions_value = sum(
            pos.get('current_value', pos['position_value'])
            for pos in self.current_positions.values()
        )
        return self.available_cash + positions_value
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio state"""
        if not self.current_positions:
            return {
                'total_positions': 0,
                'total_value': self.available_cash,
                'positions_value': 0,
                'available_cash': self.available_cash,
                'total_pnl': self.available_cash - self.total_capital,
                'total_pnl_pct': ((self.available_cash / self.total_capital) - 1) * 100,
                'utilization': 0,
                'total_risk': 0,
                'positions': []
            }
        
        positions_value = sum(
            pos.get('current_value', pos['position_value'])
            for pos in self.current_positions.values()
        )
        
        total_value = self.available_cash + positions_value
        total_pnl = total_value - self.total_capital
        
        return {
            'total_positions': len(self.current_positions),
            'total_value': total_value,
            'positions_value': positions_value,
            'available_cash': self.available_cash,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / self.total_capital) * 100,
            'utilization': positions_value / self.total_capital,
            'total_risk': self._calculate_total_portfolio_risk(),
            'positions': list(self.current_positions.keys())
        }
    
    def get_position_details(self, symbol: str) -> Optional[Dict]:
        """Get details of specific position"""
        return self.current_positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all current positions"""
        return self.current_positions.copy()
    
    def clear_positions(self):
        """Clear all positions (emergency liquidation)"""
        logger.warning("⚠️ Clearing all positions from risk manager")
        self.current_positions.clear()
    
    def get_position_history(self) -> List[Dict]:
        """Get complete position history"""
        return self.position_history.copy()


logger.info("✓ RiskManager module loaded (with capital tracking fix)")
