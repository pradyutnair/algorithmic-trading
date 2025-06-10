"""
Technical indicators for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Optional


def sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average"""
    return prices.rolling(window=window).mean()


def ema(prices: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average"""
    return prices.ewm(span=window).mean()


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index
    
    Args:
        prices: Price series
        window: RSI period
    
    Returns:
        RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(prices: pd.Series, 
         fast_period: int = 12, 
         slow_period: int = 26, 
         signal_period: int = 9) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence)
    
    Args:
        prices: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period
    
    Returns:
        DataFrame with MACD, Signal, and Histogram
    """
    fast_ema = ema(prices, fast_period)
    slow_ema = ema(prices, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram
    })


def bollinger_bands(prices: pd.Series, 
                   window: int = 20, 
                   std_dev: float = 2) -> pd.DataFrame:
    """
    Bollinger Bands
    
    Args:
        prices: Price series
        window: Moving average period
        std_dev: Standard deviation multiplier
    
    Returns:
        DataFrame with Upper, Middle, and Lower bands
    """
    middle = sma(prices, window)
    std = prices.rolling(window=window).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return pd.DataFrame({
        'Upper': upper,
        'Middle': middle,
        'Lower': lower
    })


def stochastic(high: pd.Series, 
               low: pd.Series, 
               close: pd.Series, 
               k_period: int = 14, 
               d_period: int = 3) -> pd.DataFrame:
    """
    Stochastic Oscillator
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: %K period
        d_period: %D period
    
    Returns:
        DataFrame with %K and %D values
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return pd.DataFrame({
        'K': k_percent,
        'D': d_percent
    })


def atr(high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        window: int = 14) -> pd.Series:
    """
    Average True Range
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: ATR period
    
    Returns:
        ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_values = tr.rolling(window=window).mean()
    
    return atr_values


def williams_r(high: pd.Series, 
               low: pd.Series, 
               close: pd.Series, 
               window: int = 14) -> pd.Series:
    """
    Williams %R
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: Lookback period
    
    Returns:
        Williams %R values
    """
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    return williams_r


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume
    
    Args:
        close: Close price series
        volume: Volume series
    
    Returns:
        OBV values
    """
    price_change = close.diff()
    obv_values = volume.copy()
    obv_values[price_change < 0] = -volume[price_change < 0]
    obv_values[price_change == 0] = 0
    
    return obv_values.cumsum()


def momentum(prices: pd.Series, window: int = 10) -> pd.Series:
    """
    Price Momentum
    
    Args:
        prices: Price series
        window: Lookback period
    
    Returns:
        Momentum values
    """
    return prices - prices.shift(window) 