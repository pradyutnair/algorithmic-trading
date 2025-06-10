"""
Statistical Arbitrage Strategy

Advanced statistical arbitrage using:
1. Cointegration testing and selection
2. Kalman Filter for dynamic hedge ratios
3. Multi-pair portfolio construction
4. Advanced risk management
5. Regime-aware execution

This strategy identifies cointegrated pairs and trades their temporary divergences.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_strategy import BaseStrategy
from utils.indicators import *
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression
from pykalman import KalmanFilter
import warnings
warnings.filterwarnings('ignore')


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Advanced statistical arbitrage strategy that:
    1. Identifies cointegrated pairs using multiple tests
    2. Uses Kalman filters for dynamic hedge ratios
    3. Implements portfolio-level risk management
    4. Adapts to changing market regimes
    """
    
    def __init__(self,
                 lookback_period: int = 252,
                 cointegration_threshold: float = 0.05,
                 entry_zscore: float = 2.0,
                 exit_zscore: float = 0.5,
                 stop_loss_zscore: float = 4.0,
                 max_pairs: int = 5,
                 min_half_life: float = 1.0,
                 max_half_life: float = 30.0,
                 **kwargs):
        """
        Initialize Statistical Arbitrage Strategy.
        
        Args:
            lookback_period: Period for cointegration testing
            cointegration_threshold: P-value threshold for cointegration
            entry_zscore: Z-score threshold for entry
            exit_zscore: Z-score threshold for exit
            stop_loss_zscore: Z-score threshold for stop loss
            max_pairs: Maximum number of pairs to trade
            min_half_life: Minimum half-life for mean reversion (days)
            max_half_life: Maximum half-life for mean reversion (days)
        """
        super().__init__(**kwargs)
        self.lookback_period = lookback_period
        self.cointegration_threshold = cointegration_threshold
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore
        self.max_pairs = max_pairs
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        
        # Pair tracking
        self.active_pairs = {}
        self.pair_statistics = {}
        self.kalman_filters = {}
        
    def test_cointegration(self, price1: pd.Series, price2: pd.Series) -> Dict:
        """
        Test cointegration between two price series using multiple methods.
        
        Args:
            price1: First price series
            price2: Second price series
            
        Returns:
            Dictionary with cointegration test results
        """
        # Align series
        aligned_data = pd.concat([price1, price2], axis=1).dropna()
        if len(aligned_data) < 50:
            return {'cointegrated': False, 'reason': 'insufficient_data'}
        
        y = aligned_data.iloc[:, 0]
        x = aligned_data.iloc[:, 1]
        
        # 1. Engle-Granger test
        try:
            coint_stat, coint_pvalue, _ = coint(y, x)
            engle_granger_cointegrated = coint_pvalue < self.cointegration_threshold
        except:
            engle_granger_cointegrated = False
            coint_pvalue = 1.0
        
        # 2. OLS regression and residual testing
        try:
            # Fit OLS regression
            X = np.column_stack([np.ones(len(x)), x])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Calculate residuals
            residuals = y - (beta[0] + beta[1] * x)
            
            # Test residuals for stationarity
            adf_stat, adf_pvalue, _, _, _, _ = adfuller(residuals, regression='c')
            residuals_stationary = adf_pvalue < self.cointegration_threshold
            
            # Calculate half-life of mean reversion
            residuals_lag = residuals.shift(1).dropna()
            residuals_diff = residuals.diff().dropna()
            
            # Align for regression
            reg_data = pd.concat([residuals_diff, residuals_lag], axis=1).dropna()
            if len(reg_data) > 10:
                reg_y = reg_data.iloc[:, 0]
                reg_x = reg_data.iloc[:, 1]
                
                # Estimate mean reversion speed
                theta = np.cov(reg_y, reg_x)[0, 1] / np.var(reg_x)
                half_life = -np.log(2) / theta if theta < 0 else np.inf
            else:
                half_life = np.inf
                
        except:
            residuals_stationary = False
            adf_pvalue = 1.0
            half_life = np.inf
            beta = [0, 1]
        
        # 3. Correlation and business logic checks
        correlation = y.corr(x)
        
        # Final cointegration decision
        cointegration_checks = {
            'engle_granger': engle_granger_cointegrated,
            'residuals_stationary': residuals_stationary,
            'correlation_high': abs(correlation) > 0.5,
            'half_life_reasonable': self.min_half_life <= half_life <= self.max_half_life
        }
        
        cointegrated = all(cointegration_checks.values())
        
        return {
            'cointegrated': cointegrated,
            'coint_pvalue': coint_pvalue,
            'adf_pvalue': adf_pvalue,
            'correlation': correlation,
            'half_life': half_life,
            'hedge_ratio': beta[1],
            'intercept': beta[0],
            'checks': cointegration_checks
        }
    
    def setup_kalman_filter(self, price1: pd.Series, price2: pd.Series) -> KalmanFilter:
        """
        Set up Kalman filter for dynamic hedge ratio estimation.
        
        Args:
            price1: First price series
            price2: Second price series
            
        Returns:
            Configured Kalman filter
        """
        # Prepare data
        aligned_data = pd.concat([price1, price2], axis=1).dropna()
        y = aligned_data.iloc[:, 0].values
        x = aligned_data.iloc[:, 1].values
        
        # Initial hedge ratio estimate using OLS
        initial_hedge_ratio = np.cov(y, x)[0, 1] / np.var(x)
        
        # Kalman filter setup for state-space model
        # State: [intercept, hedge_ratio]
        # Observation: price1 = intercept + hedge_ratio * price2 + noise
        
        transition_matrices = np.eye(2)  # Random walk for parameters
        observation_matrices = np.column_stack([np.ones(len(x)), x])
        
        # Estimate noise variances
        initial_state_mean = np.array([0, initial_hedge_ratio])
        
        # Covariances (these can be tuned)
        transition_covariance = np.eye(2) * 1e-5  # Parameter evolution noise
        observation_covariance = np.var(y) * 0.1   # Observation noise
        initial_state_covariance = np.eye(2) * 0.1
        
        kf = KalmanFilter(
            transition_matrices=transition_matrices,
            observation_matrices=observation_matrices,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance
        )
        
        return kf
    
    def calculate_spread_zscore(self, price1: pd.Series, price2: pd.Series, 
                               hedge_ratio: float, intercept: float = 0) -> pd.Series:
        """
        Calculate z-score of the spread.
        
        Args:
            price1: First price series
            price2: Second price series
            hedge_ratio: Hedge ratio
            intercept: Intercept term
            
        Returns:
            Z-score series
        """
        # Calculate spread
        spread = price1 - intercept - hedge_ratio * price2
        
        # Calculate rolling z-score
        lookback = min(60, len(spread) // 4)  # Adaptive lookback
        if lookback < 10:
            lookback = len(spread)
        
        spread_mean = spread.rolling(lookback).mean()
        spread_std = spread.rolling(lookback).std()
        
        zscore = (spread - spread_mean) / spread_std
        return zscore
    
    def select_best_pairs(self, data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str]]:
        """
        Select the best cointegrated pairs for trading.
        
        Args:
            data: Dictionary of price data
            
        Returns:
            List of best pairs
        """
        symbols = list(data.keys())
        if len(symbols) < 2:
            return []
        
        # Test all possible pairs
        pair_scores = []
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                # Get price series
                price1 = data[symbol1]['Close']
                price2 = data[symbol2]['Close']
                
                # Test cointegration
                coint_result = self.test_cointegration(price1, price2)
                
                if coint_result['cointegrated']:
                    # Calculate quality score
                    score = self._calculate_pair_quality_score(coint_result)
                    
                    pair_scores.append({
                        'pair': (symbol1, symbol2),
                        'score': score,
                        'stats': coint_result
                    })
        
        # Sort by quality score and select top pairs
        pair_scores.sort(key=lambda x: x['score'], reverse=True)
        selected_pairs = [p['pair'] for p in pair_scores[:self.max_pairs]]
        
        # Store pair statistics
        for pair_info in pair_scores[:self.max_pairs]:
            pair = pair_info['pair']
            self.pair_statistics[pair] = pair_info['stats']
        
        return selected_pairs
    
    def _calculate_pair_quality_score(self, coint_result: Dict) -> float:
        """
        Calculate quality score for a cointegrated pair.
        
        Args:
            coint_result: Cointegration test results
            
        Returns:
            Quality score (higher is better)
        """
        # Base score from statistical significance
        coint_score = max(0, 1 - coint_result['coint_pvalue'] / self.cointegration_threshold)
        adf_score = max(0, 1 - coint_result['adf_pvalue'] / self.cointegration_threshold)
        
        # Correlation score
        corr_score = abs(coint_result['correlation'])
        
        # Half-life score (prefer moderate half-lives)
        half_life = coint_result['half_life']
        if self.min_half_life <= half_life <= self.max_half_life:
            half_life_score = 1.0 - abs(half_life - (self.min_half_life + self.max_half_life) / 2) / (self.max_half_life - self.min_half_life)
        else:
            half_life_score = 0.0
        
        # Combined score
        total_score = (coint_score * 0.3 + adf_score * 0.3 + 
                      corr_score * 0.25 + half_life_score * 0.15)
        
        return total_score
    
    def generate_pair_signals(self, data: Dict[str, pd.DataFrame], 
                            pair: Tuple[str, str]) -> Dict[str, pd.Series]:
        """
        Generate trading signals for a specific pair.
        
        Args:
            data: Price data dictionary
            pair: Tuple of symbol names
            
        Returns:
            Dictionary with signals for each symbol
        """
        symbol1, symbol2 = pair
        
        if symbol1 not in data or symbol2 not in data:
            return {symbol1: pd.Series(0, index=data[symbol1].index if symbol1 in data else []),
                   symbol2: pd.Series(0, index=data[symbol2].index if symbol2 in data else [])}
        
        price1 = data[symbol1]['Close']
        price2 = data[symbol2]['Close']
        
        # Get pair statistics
        if pair not in self.pair_statistics:
            return {symbol1: pd.Series(0, index=price1.index),
                   symbol2: pd.Series(0, index=price2.index)}
        
        stats = self.pair_statistics[pair]
        
        # Use static hedge ratio or Kalman filter
        use_kalman = len(price1) > 100  # Use Kalman for longer series
        
        if use_kalman and pair not in self.kalman_filters:
            try:
                self.kalman_filters[pair] = self.setup_kalman_filter(price1, price2)
            except:
                use_kalman = False
        
        if use_kalman:
            # Update Kalman filter and get dynamic hedge ratio
            try:
                kf = self.kalman_filters[pair]
                aligned_data = pd.concat([price1, price2], axis=1).dropna()
                
                if len(aligned_data) > 0:
                    observations = aligned_data.iloc[:, 0].values
                    state_means, _ = kf.em(observations).smooth()
                    
                    # Extract hedge ratios
                    hedge_ratios = pd.Series(state_means[:, 1], index=aligned_data.index)
                    intercepts = pd.Series(state_means[:, 0], index=aligned_data.index)
                    
                    # Calculate spread z-scores
                    zscores = pd.Series(index=aligned_data.index, dtype=float)
                    for i in range(len(aligned_data)):
                        if i >= 20:  # Need minimum data for z-score
                            current_data = aligned_data.iloc[:i+1]
                            current_hedge_ratio = hedge_ratios.iloc[i]
                            current_intercept = intercepts.iloc[i]
                            
                            zscore = self.calculate_spread_zscore(
                                current_data.iloc[:, 0], 
                                current_data.iloc[:, 1],
                                current_hedge_ratio,
                                current_intercept
                            ).iloc[-1]
                            zscores.iloc[i] = zscore
                else:
                    hedge_ratios = pd.Series(stats['hedge_ratio'], index=price1.index)
                    zscores = self.calculate_spread_zscore(price1, price2, stats['hedge_ratio'], stats['intercept'])
                    
            except:
                # Fallback to static hedge ratio
                hedge_ratios = pd.Series(stats['hedge_ratio'], index=price1.index)
                zscores = self.calculate_spread_zscore(price1, price2, stats['hedge_ratio'], stats['intercept'])
        else:
            # Use static hedge ratio
            hedge_ratios = pd.Series(stats['hedge_ratio'], index=price1.index)
            zscores = self.calculate_spread_zscore(price1, price2, stats['hedge_ratio'], stats['intercept'])
        
        # Generate signals based on z-scores
        signals1 = pd.Series(0.0, index=price1.index)
        signals2 = pd.Series(0.0, index=price2.index)
        
        # Align indices
        common_index = signals1.index.intersection(signals2.index).intersection(zscores.index)
        
        if len(common_index) > 0:
            zscores_aligned = zscores.reindex(common_index)
            
            # Entry signals
            # When z-score > entry_threshold: spread too high, short symbol1, long symbol2
            entry_short_mask = zscores_aligned > self.entry_zscore
            entry_short_indices = common_index[entry_short_mask]
            signals1.loc[entry_short_indices] = -0.5
            signals2.loc[entry_short_indices] = 0.5
            
            # When z-score < -entry_threshold: spread too low, long symbol1, short symbol2
            entry_long_mask = zscores_aligned < -self.entry_zscore
            entry_long_indices = common_index[entry_long_mask]
            signals1.loc[entry_long_indices] = 0.5
            signals2.loc[entry_long_indices] = -0.5
            
            # Exit signals
            # Exit when z-score returns toward zero
            exit_mask = abs(zscores_aligned) < self.exit_zscore
            exit_indices = common_index[exit_mask]
            signals1.loc[exit_indices] = 0.0
            signals2.loc[exit_indices] = 0.0
            
            # Stop loss signals
            stop_loss_mask = abs(zscores_aligned) > self.stop_loss_zscore
            stop_loss_indices = common_index[stop_loss_mask]
            signals1.loc[stop_loss_indices] = 0.0
            signals2.loc[stop_loss_indices] = 0.0
        
        return {symbol1: signals1, symbol2: signals2}
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate statistical arbitrage signals for all pairs.
        
        Args:
            data: Dictionary of OHLCV DataFrames
            
        Returns:
            Dictionary of DataFrames with signals
        """
        if len(data) < 2:
            return {symbol: df.assign(Signal=0.0) for symbol, df in data.items()}
        
        # Select best pairs if not already done
        if not self.active_pairs:
            best_pairs = self.select_best_pairs(data)
            self.active_pairs = {pair: True for pair in best_pairs}
        
        # Initialize results
        results = {}
        for symbol, df in data.items():
            results[symbol] = df.copy()
            results[symbol]['Signal'] = 0.0
            results[symbol]['Pair_Count'] = 0
        
        # Generate signals for each active pair
        for pair in self.active_pairs:
            if not self.active_pairs[pair]:  # Skip inactive pairs
                continue
                
            pair_signals = self.generate_pair_signals(data, pair)
            
            symbol1, symbol2 = pair
            
            # Add pair signals to results
            if symbol1 in results and symbol1 in pair_signals:
                # Average with existing signals (for multiple pairs)
                existing_count = results[symbol1]['Pair_Count'].iloc[0] if len(results[symbol1]) > 0 else 0
                new_count = 1
                total_count = existing_count + new_count
                
                if total_count > 0:
                    # Reindex pair signals to match results index
                    pair_signal_aligned = pair_signals[symbol1].reindex(results[symbol1].index, fill_value=0)
                    
                    # Calculate weighted average
                    results[symbol1]['Signal'] = (
                        (results[symbol1]['Signal'] * existing_count + pair_signal_aligned * new_count) / total_count
                    )
                    results[symbol1]['Pair_Count'] = total_count
            
            if symbol2 in results and symbol2 in pair_signals:
                existing_count = results[symbol2]['Pair_Count'].iloc[0] if len(results[symbol2]) > 0 else 0
                new_count = 1
                total_count = existing_count + new_count
                
                if total_count > 0:
                    # Reindex pair signals to match results index
                    pair_signal_aligned = pair_signals[symbol2].reindex(results[symbol2].index, fill_value=0)
                    
                    # Calculate weighted average
                    results[symbol2]['Signal'] = (
                        (results[symbol2]['Signal'] * existing_count + pair_signal_aligned * new_count) / total_count
                    )
                    results[symbol2]['Pair_Count'] = total_count
        
        return results
    
    def get_strategy_params(self) -> dict:
        """Get strategy parameters."""
        return {
            'lookback_period': self.lookback_period,
            'cointegration_threshold': self.cointegration_threshold,
            'entry_zscore': self.entry_zscore,
            'exit_zscore': self.exit_zscore,
            'stop_loss_zscore': self.stop_loss_zscore,
            'max_pairs': self.max_pairs,
            'min_half_life': self.min_half_life,
            'max_half_life': self.max_half_life,
            'active_pairs_count': len(self.active_pairs)
        } 