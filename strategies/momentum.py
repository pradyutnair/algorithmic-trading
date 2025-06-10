"""
Momentum Strategy using MACD and Moving Averages.

This strategy identifies trending markets and follows the momentum.
It uses MACD for momentum confirmation and moving averages for trend direction.
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from utils.indicators import macd, sma, ema


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy using MACD and Moving Averages.
    
    Entry Rules:
    - Buy when fast MA > slow MA AND MACD > Signal AND MACD histogram increasing
    - Sell when fast MA < slow MA AND MACD < Signal AND MACD histogram decreasing
    
    Exit Rules:
    - Close long when MACD crosses below signal or fast MA crosses below slow MA
    - Close short when MACD crosses above signal or fast MA crosses above slow MA
    """
    
    def __init__(self,
                 fast_ma: int = 12,
                 slow_ma: int = 26,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 **kwargs):
        """
        Initialize Momentum Strategy.
        
        Args:
            fast_ma: Fast moving average period
            slow_ma: Slow moving average period
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
        """
        super().__init__(**kwargs)
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on MACD and moving averages.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with signals
        """
        df = data.copy()
        
        # Calculate moving averages
        df['Fast_MA'] = ema(df['Close'], self.fast_ma)
        df['Slow_MA'] = ema(df['Close'], self.slow_ma)
        
        # Calculate MACD
        macd_data = macd(df['Close'], self.macd_fast, self.macd_slow, self.macd_signal)
        df['MACD'] = macd_data['MACD']
        df['MACD_Signal'] = macd_data['Signal']
        df['MACD_Histogram'] = macd_data['Histogram']
        
        # Calculate trend direction
        df['Trend'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, -1)
        
        # Calculate MACD momentum
        df['MACD_Momentum'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
        
        # Calculate histogram momentum (increasing/decreasing)
        df['Histogram_Change'] = df['MACD_Histogram'].diff()
        df['Histogram_Momentum'] = np.where(df['Histogram_Change'] > 0, 1, -1)
        
        # Generate signals
        df['Signal'] = 0.0
        
        # Long signals
        long_condition = (
            (df['Trend'] == 1) &  # Uptrend
            (df['MACD_Momentum'] == 1) &  # MACD above signal
            (df['Histogram_Momentum'] == 1)  # Histogram increasing
        )
        df.loc[long_condition, 'Signal'] = 1.0
        
        # Short signals
        short_condition = (
            (df['Trend'] == -1) &  # Downtrend
            (df['MACD_Momentum'] == -1) &  # MACD below signal
            (df['Histogram_Momentum'] == -1)  # Histogram decreasing
        )
        df.loc[short_condition, 'Signal'] = -1.0
        
        # Exit signals
        # MACD crossovers
        df['MACD_Cross'] = np.where(
            (df['MACD'].shift(1) < df['MACD_Signal'].shift(1)) & 
            (df['MACD'] > df['MACD_Signal']), 1,
            np.where(
                (df['MACD'].shift(1) > df['MACD_Signal'].shift(1)) & 
                (df['MACD'] < df['MACD_Signal']), -1, 0
            )
        )
        
        # MA crossovers
        df['MA_Cross'] = np.where(
            (df['Fast_MA'].shift(1) < df['Slow_MA'].shift(1)) & 
            (df['Fast_MA'] > df['Slow_MA']), 1,
            np.where(
                (df['Fast_MA'].shift(1) > df['Slow_MA'].shift(1)) & 
                (df['Fast_MA'] < df['Slow_MA']), -1, 0
            )
        )
        
        # Signal strength based on multiple confirmations
        df['Signal_Strength'] = (
            df['Trend'] + 
            df['MACD_Momentum'] + 
            df['Histogram_Momentum']
        ) / 3.0
        
        # Only take signals when all indicators align
        strong_signals = abs(df['Signal_Strength']) > 0.66
        df.loc[~strong_signals, 'Signal'] = 0.0
        
        # Adjust signal strength
        df['Signal'] = df['Signal'] * abs(df['Signal_Strength'])
        
        return df
    
    def get_strategy_params(self) -> dict:
        """Get strategy parameters."""
        return {
            'fast_ma': self.fast_ma,
            'slow_ma': self.slow_ma,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal
        } 