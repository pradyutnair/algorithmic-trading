"""
Pairs Trading Strategy for Market-Neutral Trading.

This strategy identifies two historically correlated assets and trades their spread.
When the spread deviates from its historical mean, it trades expecting mean reversion.
"""

from typing import Dict
import pandas as pd
import numpy as np
from scipy import stats
from .base_strategy import BaseStrategy
from utils.indicators import sma


class PairsTradingStrategy(BaseStrategy):
    """
    Pairs Trading Strategy using statistical arbitrage.
    
    The strategy:
    1. Calculates the spread between two assets
    2. Identifies when the spread deviates significantly from its mean
    3. Goes long the underperforming asset and short the overperforming asset
    4. Exits when the spread reverts to the mean
    """
    
    def __init__(self,
                 lookback_window: int = 60,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 correlation_threshold: float = 0.7,
                 **kwargs):
        """
        Initialize Pairs Trading Strategy.
        
        Args:
            lookback_window: Window for calculating mean and std of spread
            entry_threshold: Z-score threshold for entry (standard deviations)
            exit_threshold: Z-score threshold for exit
            correlation_threshold: Minimum correlation required for pair
        """
        super().__init__(**kwargs)
        self.lookback_window = lookback_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.correlation_threshold = correlation_threshold
        
    def calculate_spread(self, price1: pd.Series, price2: pd.Series) -> pd.DataFrame:
        """
        Calculate the spread between two price series.
        
        Args:
            price1: First asset price series
            price2: Second asset price series
            
        Returns:
            DataFrame with spread and related metrics
        """
        # Perform linear regression to find hedge ratio
        valid_data = pd.concat([price1, price2], axis=1).dropna()
        if len(valid_data) < self.lookback_window:
            return pd.DataFrame()
            
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_data.iloc[:, 1], valid_data.iloc[:, 0]
        )
        
        # Calculate spread using hedge ratio
        spread = price1 - slope * price2
        
        # Calculate rolling statistics
        spread_mean = spread.rolling(window=self.lookback_window).mean()
        spread_std = spread.rolling(window=self.lookback_window).std()
        
        # Calculate z-score
        z_score = (spread - spread_mean) / spread_std
        
        result = pd.DataFrame({
            'Price1': price1,
            'Price2': price2,
            'Spread': spread,
            'Spread_Mean': spread_mean,
            'Spread_Std': spread_std,
            'Z_Score': z_score,
            'Hedge_Ratio': slope,
            'Correlation': r_value
        })
        
        return result
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate trading signals for pairs trading.
        
        Args:
            data: Dictionary with two DataFrames for the pair
            
        Returns:
            DataFrame with signals for both assets
        """
        if len(data) != 2:
            raise ValueError("Pairs trading requires exactly 2 assets")
            
        symbols = list(data.keys())
        df1 = data[symbols[0]]
        df2 = data[symbols[1]]
        
        # Align data
        aligned_data = pd.concat([
            df1['Close'].rename('Price1'),
            df2['Close'].rename('Price2')
        ], axis=1).dropna()
        
        if len(aligned_data) < self.lookback_window:
            return pd.DataFrame()
        
        # Calculate spread metrics
        spread_data = self.calculate_spread(
            aligned_data['Price1'], 
            aligned_data['Price2']
        )
        
        # Check correlation requirement
        recent_correlation = aligned_data.rolling(
            window=self.lookback_window
        ).corr().iloc[0::2, 1].dropna()
        
        if recent_correlation.iloc[-1] < self.correlation_threshold:
            print(f"Warning: Correlation {recent_correlation.iloc[-1]:.2f} below threshold")
        
        # Generate signals
        signals = pd.DataFrame(index=spread_data.index)
        signals['Z_Score'] = spread_data['Z_Score']
        signals['Signal_Asset1'] = 0.0
        signals['Signal_Asset2'] = 0.0
        
        # Entry signals
        # When z-score > entry_threshold: spread is too high
        # Short asset1, long asset2
        long_spread_condition = spread_data['Z_Score'] > self.entry_threshold
        signals.loc[long_spread_condition, 'Signal_Asset1'] = -1.0
        signals.loc[long_spread_condition, 'Signal_Asset2'] = 1.0
        
        # When z-score < -entry_threshold: spread is too low  
        # Long asset1, short asset2
        short_spread_condition = spread_data['Z_Score'] < -self.entry_threshold
        signals.loc[short_spread_condition, 'Signal_Asset1'] = 1.0
        signals.loc[short_spread_condition, 'Signal_Asset2'] = -1.0
        
        # Exit signals (when spread reverts to mean)
        exit_condition = abs(spread_data['Z_Score']) < self.exit_threshold
        signals.loc[exit_condition, 'Signal_Asset1'] = 0.0
        signals.loc[exit_condition, 'Signal_Asset2'] = 0.0
        
        # Add metadata
        signals['Spread'] = spread_data['Spread']
        signals['Hedge_Ratio'] = spread_data['Hedge_Ratio']
        signals['Correlation'] = spread_data['Correlation']
        
        # Create results for both assets
        result = {}
        for i, symbol in enumerate(symbols):
            asset_signals = signals[[f'Signal_Asset{i+1}']].copy()
            asset_signals.columns = ['Signal']
            asset_signals['Price'] = spread_data[f'Price{i+1}']
            asset_signals['Z_Score'] = signals['Z_Score']
            asset_signals['Spread'] = signals['Spread']
            result[symbol] = asset_signals
            
        return result
    
    def check_pair_quality(self, price1: pd.Series, price2: pd.Series) -> Dict:
        """
        Check the quality of a trading pair.
        
        Args:
            price1: First asset price series
            price2: Second asset price series
            
        Returns:
            Dictionary with pair quality metrics
        """
        # Calculate correlation
        correlation = price1.corr(price2)
        
        # Calculate cointegration test (simplified)
        spread = price1 - price2
        adf_statistic = abs(spread.diff().mean() / spread.diff().std())
        
        # Calculate half-life of mean reversion
        spread_lagged = spread.shift(1)
        delta_spread = spread.diff()
        
        # Simple linear regression for half-life
        valid_data = pd.concat([delta_spread, spread_lagged], axis=1).dropna()
        if len(valid_data) > 10:
            slope = np.cov(valid_data.iloc[:, 0], valid_data.iloc[:, 1])[0, 1] / np.var(valid_data.iloc[:, 1])
            half_life = -np.log(2) / slope if slope < 0 else np.inf
        else:
            half_life = np.inf
        
        return {
            'correlation': correlation,
            'adf_statistic': adf_statistic,
            'half_life': half_life,
            'quality_score': correlation * min(1.0, 1.0 / max(1.0, half_life / 20))
        }
    
    def get_strategy_params(self) -> dict:
        """Get strategy parameters."""
        return {
            'lookback_window': self.lookback_window,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'correlation_threshold': self.correlation_threshold
        } 