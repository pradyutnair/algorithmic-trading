"""
Adaptive Regime Strategy

This strategy dynamically adapts its behavior based on detected market regimes:
- High Volatility Regime: Mean reversion approach
- Low Volatility Regime: Momentum approach  
- Trending Regime: Trend following
- Range-bound Regime: Mean reversion with tight bands

Uses multiple timeframes and regime detection algorithms.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from .base_strategy import BaseStrategy
from utils.indicators import *
from scipy import stats
from sklearn.cluster import KMeans


class AdaptiveRegimeStrategy(BaseStrategy):
    """
    Advanced strategy that adapts to market regimes using:
    1. Volatility regime detection
    2. Trend strength measurement
    3. Multi-timeframe analysis
    4. Dynamic parameter adjustment
    """
    
    def __init__(self,
                 regime_lookback: int = 60,
                 volatility_threshold: float = 0.02,
                 trend_threshold: float = 0.05,
                 regime_confidence: float = 0.7,
                 **kwargs):
        """
        Initialize Adaptive Regime Strategy.
        
        Args:
            regime_lookback: Period for regime detection
            volatility_threshold: Threshold for high/low volatility regimes
            trend_threshold: Threshold for trending vs ranging markets
            regime_confidence: Minimum confidence for regime classification
        """
        super().__init__(**kwargs)
        self.regime_lookback = regime_lookback
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.regime_confidence = regime_confidence
        
        # Regime tracking
        self.current_regime = 'neutral'
        self.regime_history = []
        self.regime_signals = {}
        
    def detect_market_regime(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Detect current market regime using multiple indicators.
        
        Returns:
            Dictionary with regime probabilities
        """
        if len(data) < self.regime_lookback:
            return {'neutral': 1.0}
        
        recent_data = data.tail(self.regime_lookback)
        returns = recent_data['Close'].pct_change().dropna()
        
        # 1. Volatility Regime Detection
        volatility = returns.std() * np.sqrt(252)
        vol_regime = 'high_vol' if volatility > self.volatility_threshold else 'low_vol'
        
        # 2. Trend Detection using multiple methods
        # Method 1: Price vs Moving Averages
        price = recent_data['Close'].iloc[-1]
        ma_20 = recent_data['Close'].rolling(20).mean().iloc[-1]
        ma_50 = recent_data['Close'].rolling(50).mean().iloc[-1]
        
        trend_strength = abs(price - ma_50) / ma_50
        is_trending = trend_strength > self.trend_threshold
        
        # Method 2: ADX-like trend strength
        high_low = recent_data['High'] - recent_data['Low']
        high_close = abs(recent_data['High'] - recent_data['Close'].shift(1))
        low_close = abs(recent_data['Low'] - recent_data['Close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(14).mean().iloc[-1]
        
        # Directional movement
        up_move = recent_data['High'] - recent_data['High'].shift(1)
        down_move = recent_data['Low'].shift(1) - recent_data['Low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean().iloc[-1] / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean().iloc[-1] / atr
        
        adx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # 3. Market Microstructure Analysis
        # Analyze price action patterns
        body_size = abs(recent_data['Close'] - recent_data['Open'])
        wick_size = (recent_data['High'] - recent_data['Low']) - body_size
        wick_ratio = wick_size / (recent_data['High'] - recent_data['Low'])
        
        # High wick ratio suggests indecision/ranging
        avg_wick_ratio = wick_ratio.tail(10).mean()
        
        # 4. Momentum Persistence
        momentum_5 = (price / recent_data['Close'].iloc[-6] - 1) if len(recent_data) >= 6 else 0
        momentum_20 = (price / recent_data['Close'].iloc[-21] - 1) if len(recent_data) >= 21 else 0
        
        momentum_consistency = 1 if (momentum_5 > 0) == (momentum_20 > 0) else 0
        
        # 5. Volume Analysis
        if 'Volume' in recent_data.columns:
            volume_trend = recent_data['Volume'].rolling(5).mean().iloc[-1] / recent_data['Volume'].rolling(20).mean().iloc[-1]
            volume_confirmation = volume_trend > 1.2  # Volume supporting moves
        else:
            volume_confirmation = 0.5
        
        # Regime Classification Logic
        regime_scores = {
            'trending_bull': 0,
            'trending_bear': 0,
            'high_vol_reversion': 0,
            'low_vol_momentum': 0,
            'ranging': 0,
            'breakout': 0
        }
        
        # Trending Bull
        if (price > ma_20 > ma_50 and 
            momentum_20 > 0.02 and 
            adx > 25 and 
            volume_confirmation > 0.8):
            regime_scores['trending_bull'] = 0.8 + 0.2 * momentum_consistency
        
        # Trending Bear
        elif (price < ma_20 < ma_50 and 
              momentum_20 < -0.02 and 
              adx > 25 and 
              volume_confirmation > 0.8):
            regime_scores['trending_bear'] = 0.8 + 0.2 * momentum_consistency
        
        # High Volatility Mean Reversion
        elif volatility > self.volatility_threshold and avg_wick_ratio > 0.3:
            regime_scores['high_vol_reversion'] = 0.7 + 0.3 * (1 - momentum_consistency)
        
        # Low Volatility Momentum
        elif (volatility < self.volatility_threshold * 0.7 and 
              is_trending and 
              momentum_consistency > 0.5):
            regime_scores['low_vol_momentum'] = 0.6 + 0.4 * momentum_consistency
        
        # Ranging Market
        elif (not is_trending and 
              adx < 20 and 
              avg_wick_ratio > 0.25):
            regime_scores['ranging'] = 0.7 + 0.3 * (avg_wick_ratio - 0.25) / 0.25
        
        # Breakout Preparation
        elif (volatility < self.volatility_threshold * 0.5 and 
              volume_confirmation > 1.5):
            regime_scores['breakout'] = 0.6 + 0.4 * volume_confirmation / 2.0
        
        # Normalize scores
        total_score = sum(regime_scores.values())
        if total_score > 0:
            regime_scores = {k: v / total_score for k, v in regime_scores.items()}
        else:
            regime_scores['ranging'] = 1.0
        
        return regime_scores
    
    def generate_regime_signals(self, data: pd.DataFrame, regime: str) -> pd.DataFrame:
        """
        Generate trading signals based on the detected regime.
        
        Args:
            data: OHLCV data
            regime: Detected market regime
            
        Returns:
            DataFrame with regime-specific signals
        """
        df = data.copy()
        signals = pd.Series(0.0, index=df.index)
        
        if regime == 'trending_bull':
            signals = self._trend_following_signals(df, direction='bull')
        elif regime == 'trending_bear':
            signals = self._trend_following_signals(df, direction='bear')
        elif regime == 'high_vol_reversion':
            signals = self._volatility_mean_reversion_signals(df)
        elif regime == 'low_vol_momentum':
            signals = self._low_vol_momentum_signals(df)
        elif regime == 'ranging':
            signals = self._range_trading_signals(df)
        elif regime == 'breakout':
            signals = self._breakout_signals(df)
        
        df['Signal'] = signals
        df['Regime'] = regime
        
        return df
    
    def _trend_following_signals(self, df: pd.DataFrame, direction: str) -> pd.Series:
        """Trend following signals for trending markets."""
        signals = pd.Series(0.0, index=df.index)
        
        # Multi-timeframe trend confirmation
        ema_8 = ema(df['Close'], 8)
        ema_21 = ema(df['Close'], 21)
        ema_55 = ema(df['Close'], 55)
        
        # MACD for momentum confirmation
        macd_data = macd(df['Close'], fast_period=8, slow_period=21, signal_period=5)
        
        # Volume confirmation
        volume_ma = df['Volume'].rolling(20).mean() if 'Volume' in df.columns else pd.Series(1, index=df.index)
        volume_ratio = df['Volume'] / volume_ma if 'Volume' in df.columns else pd.Series(1, index=df.index)
        
        if direction == 'bull':
            # Bull trend conditions
            trend_aligned = (ema_8 > ema_21) & (ema_21 > ema_55)
            momentum_positive = macd_data['MACD'] > macd_data['Signal']
            volume_support = volume_ratio > 0.8
            
            # Entry on pullbacks to EMA21
            pullback_entry = (df['Close'] <= ema_21 * 1.02) & (df['Close'] >= ema_21 * 0.98)
            
            bull_signals = trend_aligned & momentum_positive & volume_support & pullback_entry
            signals[bull_signals] = 0.8
            
        else:  # bear
            # Bear trend conditions
            trend_aligned = (ema_8 < ema_21) & (ema_21 < ema_55)
            momentum_negative = macd_data['MACD'] < macd_data['Signal']
            volume_support = volume_ratio > 0.8
            
            # Entry on bounces to EMA21
            bounce_entry = (df['Close'] >= ema_21 * 0.98) & (df['Close'] <= ema_21 * 1.02)
            
            bear_signals = trend_aligned & momentum_negative & volume_support & bounce_entry
            signals[bear_signals] = -0.8
        
        return signals
    
    def _volatility_mean_reversion_signals(self, df: pd.DataFrame) -> pd.Series:
        """Mean reversion signals for high volatility periods."""
        signals = pd.Series(0.0, index=df.index)
        
        # Dynamic Bollinger Bands based on volatility
        returns = df['Close'].pct_change()
        rolling_vol = returns.rolling(20).std()
        
        # Adaptive standard deviation multiplier
        vol_percentile = rolling_vol.rolling(60).rank(pct=True)
        adaptive_std = 1.5 + vol_percentile  # 1.5 to 2.5 based on volatility regime
        
        bb_data = bollinger_bands(df['Close'], window=15, std_dev=adaptive_std.iloc[-1])
        
        # RSI with shorter period for volatile markets
        rsi_vals = rsi(df['Close'], window=10)
        
        # Z-score of price relative to recent range
        price_zscore = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
        
        # Mean reversion signals
        oversold = (df['Close'] < bb_data['Lower']) & (rsi_vals < 25) & (price_zscore < -1.5)
        overbought = (df['Close'] > bb_data['Upper']) & (rsi_vals > 75) & (price_zscore > 1.5)
        
        signals[oversold] = 0.9
        signals[overbought] = -0.9
        
        return signals
    
    def _low_vol_momentum_signals(self, df: pd.DataFrame) -> pd.Series:
        """Momentum signals for low volatility periods."""
        signals = pd.Series(0.0, index=df.index)
        
        # Momentum indicators
        momentum_10 = momentum(df['Close'], 10)
        momentum_20 = momentum(df['Close'], 20)
        
        # Rate of change
        roc_5 = df['Close'] / df['Close'].shift(5) - 1
        roc_10 = df['Close'] / df['Close'].shift(10) - 1
        
        # Momentum persistence
        momentum_consistent = (momentum_10 > 0) == (momentum_20 > 0)
        
        # Low volatility confirmation
        vol_20 = df['Close'].pct_change().rolling(20).std()
        low_vol = vol_20 < vol_20.rolling(60).quantile(0.3)
        
        # Momentum signals
        momentum_bull = (roc_5 > 0.01) & (roc_10 > 0.02) & momentum_consistent & low_vol
        momentum_bear = (roc_5 < -0.01) & (roc_10 < -0.02) & momentum_consistent & low_vol
        
        signals[momentum_bull] = 0.7
        signals[momentum_bear] = -0.7
        
        return signals
    
    def _range_trading_signals(self, df: pd.DataFrame) -> pd.Series:
        """Range trading signals for sideways markets."""
        signals = pd.Series(0.0, index=df.index)
        
        # Identify support and resistance levels
        rolling_high = df['High'].rolling(20).max()
        rolling_low = df['Low'].rolling(20).min()
        
        # Range boundaries
        range_top = rolling_high * 0.98
        range_bottom = rolling_low * 1.02
        
        # Oscillator for range trading
        stoch_data = stochastic(df['High'], df['Low'], df['Close'], k_period=10, d_period=3)
        
        # Range trading signals
        buy_range = (df['Close'] <= range_bottom) & (stoch_data['K'] < 20)
        sell_range = (df['Close'] >= range_top) & (stoch_data['K'] > 80)
        
        signals[buy_range] = 0.6
        signals[sell_range] = -0.6
        
        return signals
    
    def _breakout_signals(self, df: pd.DataFrame) -> pd.Series:
        """Breakout signals for compressed volatility periods."""
        signals = pd.Series(0.0, index=df.index)
        
        # Volatility compression detection
        atr_14 = atr(df['High'], df['Low'], df['Close'], 14)
        atr_percentile = atr_14.rolling(60).rank(pct=True)
        
        # Bollinger Band squeeze
        bb_data = bollinger_bands(df['Close'], window=20, std_dev=2.0)
        bb_width = (bb_data['Upper'] - bb_data['Lower']) / bb_data['Middle']
        bb_squeeze = bb_width < bb_width.rolling(60).quantile(0.2)
        
        # Volume buildup
        volume_buildup = df['Volume'].rolling(5).mean() > df['Volume'].rolling(20).mean() if 'Volume' in df.columns else pd.Series(True, index=df.index)
        
        # Breakout conditions
        compression = (atr_percentile < 0.3) & bb_squeeze & volume_buildup
        
        # Direction of breakout
        price_above_mid = df['Close'] > bb_data['Middle']
        recent_momentum = df['Close'] / df['Close'].shift(5) - 1
        
        # Breakout signals
        bullish_breakout = compression & price_above_mid & (recent_momentum > 0)
        bearish_breakout = compression & (~price_above_mid) & (recent_momentum < 0)
        
        signals[bullish_breakout] = 0.8
        signals[bearish_breakout] = -0.8
        
        return signals
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate adaptive signals based on regime detection.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with adaptive signals
        """
        if len(data) < self.regime_lookback:
            df = data.copy()
            df['Signal'] = 0.0
            df['Regime'] = 'insufficient_data'
            return df
        
        # Detect current regime
        regime_probs = self.detect_market_regime(data)
        
        # Select dominant regime
        dominant_regime = max(regime_probs.items(), key=lambda x: x[1])
        regime_name, regime_confidence = dominant_regime
        
        # Only trade if confidence is high enough
        if regime_confidence < self.regime_confidence:
            regime_name = 'neutral'
        
        # Generate regime-specific signals
        if regime_name == 'neutral':
            df = data.copy()
            df['Signal'] = 0.0
            df['Regime'] = 'neutral'
        else:
            df = self.generate_regime_signals(data, regime_name)
        
        # Add regime information
        df['Regime_Confidence'] = regime_confidence
        df['Regime_Probs'] = str(regime_probs)
        
        # Store regime history
        self.regime_history.append({
            'timestamp': data.index[-1],
            'regime': regime_name,
            'confidence': regime_confidence,
            'probabilities': regime_probs
        })
        
        return df
    
    def get_strategy_params(self) -> dict:
        """Get strategy parameters."""
        return {
            'regime_lookback': self.regime_lookback,
            'volatility_threshold': self.volatility_threshold,
            'trend_threshold': self.trend_threshold,
            'regime_confidence': self.regime_confidence,
            'current_regime': self.current_regime
        } 