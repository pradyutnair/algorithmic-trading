"""
Risk management and portfolio utility functions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR)
    
    Args:
        returns: Return series
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
    
    Returns:
        VaR value
    """
    return np.percentile(returns.dropna(), confidence_level * 100)


def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (CVaR)
    
    Args:
        returns: Return series
        confidence_level: Confidence level
    
    Returns:
        CVaR value
    """
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()


def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate maximum drawdown
    
    Args:
        prices: Price series
    
    Returns:
        Maximum drawdown percentage
    """
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino ratio
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    
    if downside_std == 0:
        return np.inf
    
    return np.sqrt(252) * excess_returns.mean() / downside_std


def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate beta coefficient
    
    Args:
        stock_returns: Stock return series
        market_returns: Market return series
    
    Returns:
        Beta value
    """
    covariance = np.cov(stock_returns.dropna(), market_returns.dropna())[0][1]
    market_variance = np.var(market_returns.dropna())
    return covariance / market_variance


def portfolio_optimization_equal_weight(assets: List[str]) -> Dict[str, float]:
    """
    Equal weight portfolio optimization
    
    Args:
        assets: List of asset names
    
    Returns:
        Dictionary of asset weights
    """
    weight = 1.0 / len(assets)
    return {asset: weight for asset in assets}


def portfolio_optimization_risk_parity(returns: pd.DataFrame) -> Dict[str, float]:
    """
    Risk parity portfolio optimization
    
    Args:
        returns: DataFrame of asset returns
    
    Returns:
        Dictionary of asset weights
    """
    # Simple risk parity: inverse volatility weighting
    volatilities = returns.std()
    inv_vol = 1 / volatilities
    weights = inv_vol / inv_vol.sum()
    return weights.to_dict()


def calculate_position_size(account_balance: float, 
                          risk_per_trade: float, 
                          entry_price: float, 
                          stop_loss_price: float) -> int:
    """
    Calculate position size based on risk management
    
    Args:
        account_balance: Total account balance
        risk_per_trade: Risk per trade as percentage (e.g., 0.02 for 2%)
        entry_price: Entry price per share
        stop_loss_price: Stop loss price per share
    
    Returns:
        Number of shares to trade
    """
    risk_amount = account_balance * risk_per_trade
    price_diff = abs(entry_price - stop_loss_price)
    
    if price_diff == 0:
        return 0
    
    shares = int(risk_amount / price_diff)
    return shares


def calculate_kelly_criterion(win_rate: float, 
                            avg_win: float, 
                            avg_loss: float) -> float:
    """
    Calculate Kelly Criterion for optimal position sizing
    
    Args:
        win_rate: Probability of winning (0-1)
        avg_win: Average winning amount
        avg_loss: Average losing amount
    
    Returns:
        Kelly percentage (0-1)
    """
    if avg_loss == 0:
        return 0
    
    b = avg_win / avg_loss  # Ratio of win to loss
    kelly = (win_rate * (1 + b) - 1) / b
    
    # Cap at 25% for safety
    return max(0, min(kelly, 0.25)) 