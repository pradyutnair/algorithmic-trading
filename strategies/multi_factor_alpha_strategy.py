"""
Multi-Factor Alpha Strategy

A sophisticated quantitative strategy that combines multiple alpha factors:
1. Technical factors (momentum, mean reversion, volatility)
2. Fundamental factors (relative value, quality)
3. Market microstructure factors (volume, liquidity)
4. Cross-sectional ranking and portfolio construction
5. Risk-adjusted position sizing

Uses factor scoring, orthogonalization, and dynamic weighting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_strategy import BaseStrategy
from utils.indicators import *
from utils.risk_utils import *
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MultiFactorAlphaStrategy(BaseStrategy):
    """
    Advanced multi-factor strategy that:
    1. Calculates multiple alpha factors
    2. Combines factors using sophisticated weighting
    3. Performs cross-sectional ranking
    4. Implements risk controls and portfolio construction
    """
    
    def __init__(self,
                 lookback_periods: List[int] = [5, 10, 20, 60],
                 factor_decay: float = 0.05,
                 min_factor_score: float = 0.3,
                 max_position_weight: float = 0.15,
                 rebalance_frequency: int = 5,
                 **kwargs):
        """
        Initialize Multi-Factor Alpha Strategy.
        
        Args:
            lookback_periods: Different periods for factor calculation
            factor_decay: Decay rate for factor weights over time
            min_factor_score: Minimum factor score to generate signal
            max_position_weight: Maximum weight per position
            rebalance_frequency: Days between rebalancing
        """
        super().__init__(**kwargs)
        self.lookback_periods = lookback_periods
        self.factor_decay = factor_decay
        self.min_factor_score = min_factor_score
        self.max_position_weight = max_position_weight
        self.rebalance_frequency = rebalance_frequency
        
        # Factor tracking
        self.factor_weights = {}
        self.factor_performance = {}
        self.last_rebalance = None
        self.factor_scaler = RobustScaler()
        
    def calculate_technical_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical analysis factors.
        
        Returns:
            DataFrame with technical factors
        """
        factors = pd.DataFrame(index=data.index)
        
        # 1. Momentum Factors
        for period in [5, 10, 20, 60]:
            if len(data) > period:
                # Price momentum
                factors[f'momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
                
                # Risk-adjusted momentum
                returns = data['Close'].pct_change()
                vol = returns.rolling(period).std()
                factors[f'risk_adj_momentum_{period}'] = factors[f'momentum_{period}'] / vol
        
        # 2. Mean Reversion Factors
        # Short-term reversal
        factors['short_reversal'] = -data['Close'].pct_change().rolling(3).sum()
        
        # Distance from moving averages
        ma_20 = sma(data['Close'], 20)
        ma_50 = sma(data['Close'], 50)
        factors['ma_distance_20'] = (ma_20 - data['Close']) / data['Close']
        factors['ma_distance_50'] = (ma_50 - data['Close']) / data['Close']
        
        # Bollinger Band position
        bb_data = bollinger_bands(data['Close'])
        factors['bb_position'] = (data['Close'] - bb_data['Lower']) / (bb_data['Upper'] - bb_data['Lower'])
        factors['bb_mean_reversion'] = 0.5 - factors['bb_position']  # Distance from middle
        
        # 3. Volatility Factors
        returns = data['Close'].pct_change()
        
        # Realized volatility
        factors['volatility_10'] = returns.rolling(10).std() * np.sqrt(252)
        factors['volatility_30'] = returns.rolling(30).std() * np.sqrt(252)
        
        # Volatility ratio (recent vs historical)
        factors['vol_ratio'] = factors['volatility_10'] / factors['volatility_30']
        
        # GARCH-like volatility clustering
        factors['vol_clustering'] = returns.rolling(5).std() / returns.rolling(20).std()
        
        # 4. Technical Oscillators
        # RSI momentum
        rsi_vals = rsi(data['Close'], 14)
        factors['rsi_momentum'] = rsi_vals - 50  # Centered around 0
        factors['rsi_extreme'] = np.where(rsi_vals > 70, rsi_vals - 70, 
                                         np.where(rsi_vals < 30, 30 - rsi_vals, 0))
        
        # Stochastic
        if all(col in data.columns for col in ['High', 'Low']):
            stoch_data = stochastic(data['High'], data['Low'], data['Close'])
            factors['stoch_momentum'] = stoch_data['K'] - 50
        
        # Williams %R
        if all(col in data.columns for col in ['High', 'Low']):
            williams = williams_r(data['High'], data['Low'], data['Close'])
            factors['williams_momentum'] = williams + 50  # Centered around 0
        
        return factors
    
    def calculate_volume_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume and microstructure factors.
        
        Returns:
            DataFrame with volume factors
        """
        factors = pd.DataFrame(index=data.index)
        
        if 'Volume' not in data.columns:
            return factors
        
        # 1. Volume Momentum
        volume_ma_10 = data['Volume'].rolling(10).mean()
        volume_ma_30 = data['Volume'].rolling(30).mean()
        factors['volume_momentum'] = volume_ma_10 / volume_ma_30 - 1
        
        # 2. Price-Volume Relationship
        returns = data['Close'].pct_change()
        volume_change = data['Volume'].pct_change()
        
        # Volume-price correlation
        factors['vol_price_corr'] = returns.rolling(20).corr(volume_change)
        
        # On-Balance Volume momentum
        obv_vals = obv(data['Close'], data['Volume'])
        factors['obv_momentum'] = obv_vals / obv_vals.shift(20) - 1
        
        # 3. Volume Patterns
        # Accumulation/Distribution pattern
        money_flow_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        money_flow_volume = money_flow_multiplier * data['Volume']
        factors['accumulation'] = money_flow_volume.rolling(20).sum()
        
        # Volume Rate of Change
        factors['volume_roc'] = data['Volume'] / data['Volume'].shift(10) - 1
        
        # 4. Liquidity Proxies
        # Price impact (return per unit volume)
        dollar_volume = data['Close'] * data['Volume']
        factors['liquidity'] = 1 / (dollar_volume.rolling(20).mean() + 1e-8)  # Inverse liquidity
        
        # Amihud illiquidity measure
        abs_returns = abs(returns)
        factors['amihud_illiq'] = abs_returns / (dollar_volume + 1e-8)
        factors['amihud_illiq'] = factors['amihud_illiq'].rolling(20).mean()
        
        return factors
    
    def calculate_volatility_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced volatility factors.
        
        Returns:
            DataFrame with volatility factors
        """
        factors = pd.DataFrame(index=data.index)
        returns = data['Close'].pct_change()
        
        # 1. Volatility Term Structure
        vol_5 = returns.rolling(5).std() * np.sqrt(252)
        vol_20 = returns.rolling(20).std() * np.sqrt(252)
        vol_60 = returns.rolling(60).std() * np.sqrt(252)
        
        factors['vol_term_structure'] = (vol_5 - vol_60) / vol_20
        
        # 2. Volatility of Volatility
        factors['vol_of_vol'] = vol_20.rolling(20).std()
        
        # 3. Downside/Upside Volatility
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        upside_vol = positive_returns.rolling(20).std() * np.sqrt(252)
        downside_vol = negative_returns.rolling(20).std() * np.sqrt(252)
        
        factors['volatility_skew'] = (downside_vol - upside_vol) / vol_20
        
        # 4. GARCH-style volatility
        # Simple EWMA volatility
        alpha = 0.06  # Decay factor
        ewma_var = returns.ewm(alpha=alpha).var()
        factors['garch_volatility'] = np.sqrt(ewma_var * 252)
        
        # Volatility clustering strength
        vol_changes = vol_20.pct_change()
        factors['vol_clustering_strength'] = abs(vol_changes).rolling(10).mean()
        
        return factors
    
    def calculate_price_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price-based factors.
        
        Returns:
            DataFrame with price factors
        """
        factors = pd.DataFrame(index=data.index)
        
        # 1. Price Patterns
        # Gap analysis
        factors['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        factors['gap_persistence'] = factors['gap'].rolling(5).sum()
        
        # Intraday patterns
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Body vs shadow analysis
            body = abs(data['Close'] - data['Open'])
            total_range = data['High'] - data['Low']
            factors['body_ratio'] = body / (total_range + 1e-8)
            
            # Upper/lower shadow analysis
            upper_shadow = data['High'] - np.maximum(data['Open'], data['Close'])
            lower_shadow = np.minimum(data['Open'], data['Close']) - data['Low']
            factors['shadow_ratio'] = (upper_shadow - lower_shadow) / (total_range + 1e-8)
        
        # 2. Support/Resistance Levels
        # Rolling max/min analysis
        rolling_max = data['High'].rolling(20).max()
        rolling_min = data['Low'].rolling(20).min()
        
        factors['resistance_distance'] = (rolling_max - data['Close']) / data['Close']
        factors['support_distance'] = (data['Close'] - rolling_min) / data['Close']
        
        # 3. Fractal Analysis
        # Simple fractal dimension approximation
        price_changes = data['Close'].diff().abs()
        factors['fractal_dimension'] = price_changes.rolling(20).sum() / (data['Close'].rolling(20).max() - data['Close'].rolling(20).min())
        
        return factors
    
    def orthogonalize_factors(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        Remove correlations between factors using PCA and orthogonalization.
        
        Args:
            factors: Raw factors DataFrame
            
        Returns:
            Orthogonalized factors DataFrame
        """
        # Remove NaN values
        clean_factors = factors.dropna()
        if len(clean_factors) < 50:  # Need minimum data
            return factors
        
        # Standardize factors
        factor_names = clean_factors.columns
        scaled_factors = self.factor_scaler.fit_transform(clean_factors)
        scaled_df = pd.DataFrame(scaled_factors, index=clean_factors.index, columns=factor_names)
        
        # Apply PCA to reduce multicollinearity
        n_components = min(len(factor_names), int(len(clean_factors) * 0.8))
        pca = PCA(n_components=n_components)
        
        try:
            pca_factors = pca.fit_transform(scaled_df)
            
            # Create orthogonal factor names
            ortho_names = [f'factor_{i}' for i in range(n_components)]
            ortho_df = pd.DataFrame(pca_factors, index=clean_factors.index, columns=ortho_names)
            
            # Add explained variance as weights
            for i, variance_ratio in enumerate(pca.explained_variance_ratio_):
                ortho_df[f'factor_{i}'] *= np.sqrt(variance_ratio)  # Weight by explanatory power
            
            # Reindex to match original
            result = pd.DataFrame(0, index=factors.index, columns=ortho_names)
            result.loc[ortho_df.index] = ortho_df
            
            return result
            
        except Exception as e:
            print(f"PCA failed, using standardized factors: {e}")
            result = pd.DataFrame(0, index=factors.index, columns=factor_names)
            result.loc[scaled_df.index] = scaled_df
            return result
    
    def calculate_factor_scores(self, all_factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate composite factor scores across all assets.
        
        Args:
            all_factors: Dictionary of factors for each asset
            
        Returns:
            DataFrame with cross-sectional factor scores
        """
        if not all_factors:
            return pd.DataFrame()
        
        # Get common dates
        common_dates = None
        for factors in all_factors.values():
            if common_dates is None:
                common_dates = factors.index
            else:
                common_dates = common_dates.intersection(factors.index)
        
        if len(common_dates) == 0:
            return pd.DataFrame()
        
        # Combine all factors
        combined_scores = pd.DataFrame(index=common_dates)
        
        for symbol, factors in all_factors.items():
            # Get factors for common dates
            symbol_factors = factors.loc[common_dates]
            
            # Calculate composite score for this asset
            factor_score = symbol_factors.mean(axis=1, skipna=True)  # Simple average
            combined_scores[symbol] = factor_score
        
        # Cross-sectional ranking (rank within each date)
        cross_sectional_ranks = combined_scores.rank(axis=1, pct=True)
        
        # Convert ranks to signals (-1 to 1)
        signals = (cross_sectional_ranks - 0.5) * 2
        
        return signals
    
    def update_factor_performance(self, returns: pd.Series, factor_scores: pd.Series, symbol: str):
        """
        Update factor performance tracking for adaptive weighting.
        
        Args:
            returns: Asset returns
            factor_scores: Factor scores for the asset
            symbol: Asset symbol
        """
        if symbol not in self.factor_performance:
            self.factor_performance[symbol] = {
                'ic_history': [],  # Information coefficient history
                'return_history': [],
                'score_history': []
            }
        
        # Calculate information coefficient (correlation between factor scores and future returns)
        aligned_data = pd.concat([factor_scores, returns.shift(-1)], axis=1).dropna()
        
        if len(aligned_data) > 10:  # Need minimum data
            ic = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
            if not np.isnan(ic):
                self.factor_performance[symbol]['ic_history'].append(ic)
                
                # Keep only recent history
                if len(self.factor_performance[symbol]['ic_history']) > 50:
                    self.factor_performance[symbol]['ic_history'].pop(0)
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate multi-factor signals for all assets.
        
        Args:
            data: Dictionary of OHLCV DataFrames
            
        Returns:
            Dictionary of DataFrames with signals
        """
        if len(data) < 2:  # Need multiple assets for cross-sectional analysis
            return {symbol: df.assign(Signal=0.0) for symbol, df in data.items()}
        
        # Calculate factors for all assets
        all_factors = {}
        
        for symbol, df in data.items():
            if len(df) < max(self.lookback_periods):
                continue
                
            # Calculate different factor categories
            technical_factors = self.calculate_technical_factors(df)
            volume_factors = self.calculate_volume_factors(df)
            volatility_factors = self.calculate_volatility_factors(df)
            price_factors = self.calculate_price_factors(df)
            
            # Combine all factors
            combined_factors = pd.concat([
                technical_factors,
                volume_factors,
                volatility_factors,
                price_factors
            ], axis=1)
            
            # Orthogonalize factors
            ortho_factors = self.orthogonalize_factors(combined_factors)
            all_factors[symbol] = ortho_factors
        
        # Calculate cross-sectional factor scores
        factor_signals = self.calculate_factor_scores(all_factors)
        
        # Generate final signals
        results = {}
        
        for symbol, df in data.items():
            result_df = df.copy()
            
            if symbol in factor_signals.columns:
                # Get factor signals for this asset
                signals = factor_signals[symbol]
                
                # Apply minimum threshold
                signals = signals.where(abs(signals) > self.min_factor_score, 0)
                
                # Reindex to match original data
                aligned_signals = pd.Series(0.0, index=df.index)
                common_idx = aligned_signals.index.intersection(signals.index)
                aligned_signals.loc[common_idx] = signals.loc[common_idx]
                
                result_df['Signal'] = aligned_signals
                
                # Update factor performance tracking
                returns = df['Close'].pct_change()
                self.update_factor_performance(returns, signals, symbol)
                
            else:
                result_df['Signal'] = 0.0
            
            # Add factor information
            if symbol in all_factors:
                result_df['Factor_Count'] = all_factors[symbol].count(axis=1)
                result_df['Factor_Strength'] = abs(result_df['Signal'])
            
            results[symbol] = result_df
        
        return results
    
    def get_strategy_params(self) -> dict:
        """Get strategy parameters."""
        return {
            'lookback_periods': self.lookback_periods,
            'factor_decay': self.factor_decay,
            'min_factor_score': self.min_factor_score,
            'max_position_weight': self.max_position_weight,
            'rebalance_frequency': self.rebalance_frequency,
            'num_factors_tracked': len(self.factor_performance)
        } 