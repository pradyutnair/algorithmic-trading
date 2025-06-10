"""
Base strategy class for all trading strategies.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.001):
        """
        Initialize the strategy.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate (e.g., 0.001 = 0.1%)
            slippage: Slippage rate
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.positions = {}
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.trades = []
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Args:
            data: OHLCV data
            
        Returns:
            DataFrame with signals
        """
        pass
    
    def calculate_position_size(self, 
                              signal: float, 
                              price: float, 
                              volatility: float = None) -> float:
        """
        Calculate position size based on signal strength and risk management.
        
        Args:
            signal: Signal strength (-1 to 1)
            price: Current price
            volatility: Price volatility
            
        Returns:
            Position size
        """
        # Simple position sizing: use fixed percentage of portfolio
        max_position_value = self.portfolio_value * 0.1  # 10% max per position
        shares = int(max_position_value / price)
        return shares * signal
    
    def execute_trade(self, 
                     symbol: str, 
                     signal: float, 
                     price: float, 
                     timestamp: pd.Timestamp):
        """
        Execute a trade based on the signal.
        
        Args:
            symbol: Asset symbol
            signal: Signal strength (-1 to 1)
            price: Execution price
            timestamp: Trade timestamp
        """
        if abs(signal) < 0.1:  # No trade for weak signals
            return
            
        position_size = self.calculate_position_size(signal, price)
        current_position = self.positions.get(symbol, 0)
        
        # Calculate trade size
        trade_size = position_size - current_position
        
        if abs(trade_size) < 1:  # Minimum trade size
            return
            
        # Calculate costs
        trade_value = abs(trade_size) * price
        commission_cost = trade_value * self.commission
        slippage_cost = trade_value * self.slippage
        total_cost = commission_cost + slippage_cost
        
        # Check if we have enough cash for buy orders
        if trade_size > 0 and (trade_value + total_cost) > self.cash:
            # Adjust trade size based on available cash
            available_for_trade = self.cash * 0.95  # Keep 5% buffer
            trade_size = int(available_for_trade / price)
            trade_value = trade_size * price
            total_cost = trade_value * (self.commission + self.slippage)
        
        if trade_size == 0:
            return
            
        # Execute the trade
        self.cash -= trade_size * price + total_cost
        self.positions[symbol] = self.positions.get(symbol, 0) + trade_size
        
        # Record the trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'trade_size': trade_size,
            'price': price,
            'commission': commission_cost,
            'slippage': slippage_cost,
            'cash_after': self.cash
        }
        self.trades.append(trade_record)
    
    def update_portfolio_value(self, prices: Dict[str, float]):
        """
        Update portfolio value based on current prices.
        
        Args:
            prices: Dictionary of current prices
        """
        positions_value = sum(
            self.positions.get(symbol, 0) * price 
            for symbol, price in prices.items()
        )
        self.portfolio_value = self.cash + positions_value
    
    def get_portfolio_stats(self) -> Dict:
        """
        Calculate portfolio statistics.
        
        Returns:
            Dictionary of portfolio statistics
        """
        if not self.trades:
            return {}
            
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate returns
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        # Calculate trade statistics
        winning_trades = trades_df[trades_df['trade_size'] > 0]
        losing_trades = trades_df[trades_df['trade_size'] < 0]
        
        stats = {
            'total_return': total_return,
            'final_portfolio_value': self.portfolio_value,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,
            'total_commission': trades_df['commission'].sum(),
            'total_slippage': trades_df['slippage'].sum()
        }
        
        return stats 