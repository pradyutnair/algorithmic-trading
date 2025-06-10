"""
Mean Reversion Strategy using Bollinger Bands.

This strategy is based on the principle that prices tend to revert to their mean over time.
It uses Bollinger Bands to identify overbought and oversold conditions.
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from utils.indicators import bollinger_bands, rsi


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using Bollinger Bands and RSI.
    
    Entry Rules:
    - Buy when price touches lower Bollinger Band AND RSI < 30
    - Sell when price touches upper Bollinger Band AND RSI > 70
    
    Exit Rules:
    - Close long position when price reaches middle Bollinger Band or RSI > 50
    - Close short position when price reaches middle Bollinger Band or RSI < 50
    """
    
    def __init__(self, 
                 bb_window: int = 20,
                 bb_std: float = 2.0,
                 rsi_window: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 **kwargs):
        """
        Initialize Mean Reversion Strategy.
        
        Args:
            bb_window: Bollinger Bands window
            bb_std: Bollinger Bands standard deviation multiplier
            rsi_window: RSI window
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
        """
        super().__init__(**kwargs)
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands and RSI.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with signals
        """
        df = data.copy()
        
        # Calculate indicators
        bb = bollinger_bands(df['Close'], self.bb_window, self.bb_std)
        df['BB_Upper'] = bb['Upper']
        df['BB_Middle'] = bb['Middle']
        df['BB_Lower'] = bb['Lower']
        
        df['RSI'] = rsi(df['Close'], self.rsi_window)
        
        # Calculate Bollinger Band position
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Generate signals
        df['Signal'] = 0.0
        
        # Buy signals (oversold conditions)
        buy_condition = (
            (df['Close'] <= df['BB_Lower']) & 
            (df['RSI'] <= self.rsi_oversold)
        )
        df.loc[buy_condition, 'Signal'] = 1.0
        
        # Sell signals (overbought conditions)
        sell_condition = (
            (df['Close'] >= df['BB_Upper']) & 
            (df['RSI'] >= self.rsi_overbought)
        )
        df.loc[sell_condition, 'Signal'] = -1.0
        
        # Exit signals (return to mean)
        # Close long positions
        close_long = (
            (df['Close'] >= df['BB_Middle']) | 
            (df['RSI'] >= 50)
        )
        
        # Close short positions
        close_short = (
            (df['Close'] <= df['BB_Middle']) | 
            (df['RSI'] <= 50)
        )
        
        # Apply exit logic (this is simplified - in practice you'd track positions)
        df['Exit_Long'] = close_long
        df['Exit_Short'] = close_short
        
        # Smooth signals to avoid whipsaws
        df['Signal_Smooth'] = df['Signal'].rolling(window=2).mean()
        
        return df
    
    def get_strategy_params(self) -> dict:
        """Get strategy parameters."""
        return {
            'bb_window': self.bb_window,
            'bb_std': self.bb_std,
            'rsi_window': self.rsi_window,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought
        } 