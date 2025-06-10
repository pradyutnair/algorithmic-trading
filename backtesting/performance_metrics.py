"""
Performance metrics calculation for backtesting results.
"""

import pandas as pd
import numpy as np
from typing import Dict


class PerformanceMetrics:
    """
    Calculate various performance metrics for trading strategies.
    """
    
    def __init__(self, portfolio_values: pd.DataFrame, trades: pd.DataFrame):
        """
        Initialize performance metrics calculator.
        
        Args:
            portfolio_values: DataFrame with portfolio values over time
            trades: DataFrame with individual trades
        """
        self.portfolio_values = portfolio_values
        self.trades = trades
        self.returns = self._calculate_returns()
        
    def _calculate_returns(self) -> pd.Series:
        """Calculate portfolio returns."""
        if 'portfolio_value' in self.portfolio_values.columns:
            return self.portfolio_values['portfolio_value'].pct_change().dropna()
        else:
            return pd.Series()
    
    def total_return(self) -> float:
        """Calculate total return."""
        if len(self.portfolio_values) == 0:
            return 0.0
        start_value = self.portfolio_values['portfolio_value'].iloc[0]
        end_value = self.portfolio_values['portfolio_value'].iloc[-1]
        return (end_value - start_value) / start_value
    
    def annualized_return(self) -> float:
        """Calculate annualized return."""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        total_ret = self.total_return()
        days = (self.portfolio_values.index[-1] - self.portfolio_values.index[0]).days
        years = days / 365.25
        
        if years <= 0:
            return 0.0
        
        return (1 + total_ret) ** (1 / years) - 1
    
    def annualized_volatility(self) -> float:
        """Calculate annualized volatility."""
        if len(self.returns) == 0:
            return 0.0
        return self.returns.std() * np.sqrt(252)
    
    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        ann_return = self.annualized_return()
        ann_vol = self.annualized_volatility()
        
        if ann_vol == 0:
            return 0.0
        
        return (ann_return - risk_free_rate) / ann_vol
    
    def sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        ann_return = self.annualized_return()
        negative_returns = self.returns[self.returns < 0]
        
        if len(negative_returns) == 0:
            return np.inf
        
        downside_vol = negative_returns.std() * np.sqrt(252)
        
        if downside_vol == 0:
            return np.inf
        
        return (ann_return - risk_free_rate) / downside_vol
    
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.portfolio_values) == 0:
            return 0.0
        
        cumulative = self.portfolio_values['portfolio_value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calmar_ratio(self) -> float:
        """Calculate Calmar ratio."""
        ann_return = self.annualized_return()
        max_dd = abs(self.max_drawdown())
        
        if max_dd == 0:
            return np.inf
        
        return ann_return / max_dd
    
    def value_at_risk(self, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk."""
        if len(self.returns) == 0:
            return 0.0
        return np.percentile(self.returns, confidence_level * 100)
    
    def expected_shortfall(self, confidence_level: float = 0.05) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if len(self.returns) == 0:
            return 0.0
        
        var = self.value_at_risk(confidence_level)
        return self.returns[self.returns <= var].mean()
    
    def win_rate(self) -> float:
        """Calculate win rate from trades."""
        if len(self.trades) == 0:
            return 0.0
        
        if 'trade_size' not in self.trades.columns:
            return 0.0
        
        # Simplified win rate calculation
        profitable_trades = len(self.trades[self.trades['trade_size'] > 0])
        total_trades = len(self.trades)
        
        return profitable_trades / total_trades if total_trades > 0 else 0.0
    
    def average_trade_return(self) -> float:
        """Calculate average trade return."""
        if len(self.trades) == 0:
            return 0.0
        
        # This is a simplified calculation
        # In practice, you'd need to track individual trade P&L
        return self.returns.mean() if len(self.returns) > 0 else 0.0
    
    def profit_factor(self) -> float:
        """Calculate profit factor."""
        if len(self.returns) == 0:
            return 0.0
        
        positive_returns = self.returns[self.returns > 0].sum()
        negative_returns = abs(self.returns[self.returns < 0].sum())
        
        if negative_returns == 0:
            return np.inf
        
        return positive_returns / negative_returns
    
    def information_ratio(self, benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio against a benchmark."""
        if len(self.returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align returns
        aligned_returns = pd.concat([self.returns, benchmark_returns], axis=1).dropna()
        if len(aligned_returns) == 0:
            return 0.0
        
        excess_returns = aligned_returns.iloc[:, 0] - aligned_returns.iloc[:, 1]
        tracking_error = excess_returns.std()
        
        if tracking_error == 0:
            return 0.0
        
        return excess_returns.mean() / tracking_error
    
    def calculate_all_metrics(self) -> Dict[str, float]:
        """
        Calculate all performance metrics.
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'total_return': self.total_return(),
            'annualized_return': self.annualized_return(),
            'annualized_volatility': self.annualized_volatility(),
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'max_drawdown': self.max_drawdown(),
            'calmar_ratio': self.calmar_ratio(),
            'var_95': self.value_at_risk(0.05),
            'expected_shortfall': self.expected_shortfall(0.05),
            'win_rate': self.win_rate(),
            'avg_trade_return': self.average_trade_return(),
            'profit_factor': self.profit_factor(),
            'total_trades': len(self.trades)
        }
        
        # Add best and worst trade if trades exist
        if len(self.trades) > 0 and 'trade_size' in self.trades.columns:
            # Simplified - in practice you'd calculate actual P&L per trade
            metrics['best_trade'] = self.returns.max() if len(self.returns) > 0 else 0.0
            metrics['worst_trade'] = self.returns.min() if len(self.returns) > 0 else 0.0
        else:
            metrics['best_trade'] = 0.0
            metrics['worst_trade'] = 0.0
        
        return metrics 