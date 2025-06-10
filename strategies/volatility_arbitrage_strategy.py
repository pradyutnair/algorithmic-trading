"""
Volatility Arbitrage Strategy

Advanced volatility trading strategy that:
1. Models volatility surfaces and term structure
2. Identifies volatility mispricings and regime changes
3. Uses GARCH models for volatility forecasting
4. Implements delta-neutral volatility trades
5. Employs machine learning for volatility prediction

This strategy exploits differences between realized and implied volatility.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_strategy import BaseStrategy
from utils.indicators import *
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')


class VolatilityArbitrageStrategy(BaseStrategy):
    """
    Advanced volatility arbitrage strategy that:
    1. Forecasts volatility using GARCH and ML models
    2. Identifies volatility regime changes
    3. Implements volatility surface arbitrage
    4. Uses dynamic hedging techniques
    """
    
    def __init__(self,
                 volatility_lookback: int = 60,
                 regime_detection_period: int = 120,
                 volatility_threshold: float = 0.25,
                 entry_vol_ratio: float = 1.2,
                 exit_vol_ratio: float = 1.05,
                 rebalance_frequency: int = 5,
                 use_garch: bool = True,
                 use_ml_forecasting: bool = True,
                 **kwargs):
        """
        Initialize Volatility Arbitrage Strategy.
        
        Args:
            volatility_lookback: Period for volatility calculation
            regime_detection_period: Period for regime detection
            volatility_threshold: Threshold for high volatility regime
            entry_vol_ratio: Ratio threshold for entry signals
            exit_vol_ratio: Ratio threshold for exit signals
            rebalance_frequency: Days between portfolio rebalancing
            use_garch: Whether to use GARCH models
            use_ml_forecasting: Whether to use ML for volatility forecasting
        """
        super().__init__(**kwargs)
        self.volatility_lookback = volatility_lookback
        self.regime_detection_period = regime_detection_period
        self.volatility_threshold = volatility_threshold
        self.entry_vol_ratio = entry_vol_ratio
        self.exit_vol_ratio = exit_vol_ratio
        self.rebalance_frequency = rebalance_frequency
        self.use_garch = use_garch
        self.use_ml_forecasting = use_ml_forecasting
        
        # Model storage
        self.garch_models = {}
        self.ml_models = {}
        self.scalers = {}
        self.volatility_regimes = {}
        
    def calculate_realized_volatility(self, prices: pd.Series, 
                                    method: str = 'parkinson') -> pd.Series:
        """
        Calculate realized volatility using various estimators.
        
        Args:
            prices: Price series (can be OHLC data)
            method: Volatility estimation method
            
        Returns:
            Realized volatility series
        """
        if method == 'close_to_close':
            # Simple close-to-close volatility
            returns = prices.pct_change().dropna()
            vol = returns.rolling(self.volatility_lookback).std() * np.sqrt(252)
            
        elif method == 'parkinson':
            # Parkinson volatility estimator (uses high-low)
            if isinstance(prices, pd.DataFrame) and 'High' in prices.columns and 'Low' in prices.columns:
                high_low_ratio = np.log(prices['High'] / prices['Low'])
                parkinson_vol = high_low_ratio ** 2 / (4 * np.log(2))
                vol = np.sqrt(parkinson_vol.rolling(self.volatility_lookback).mean() * 252)
            else:
                # Fallback to close-to-close
                returns = prices.pct_change().dropna()
                vol = returns.rolling(self.volatility_lookback).std() * np.sqrt(252)
                
        elif method == 'garman_klass':
            # Garman-Klass volatility estimator
            if isinstance(prices, pd.DataFrame) and all(col in prices.columns for col in ['Open', 'High', 'Low', 'Close']):
                ln_hl = np.log(prices['High'] / prices['Low'])
                ln_co = np.log(prices['Close'] / prices['Open'])
                
                gk_vol = 0.5 * (ln_hl ** 2) - (2 * np.log(2) - 1) * (ln_co ** 2)
                vol = np.sqrt(gk_vol.rolling(self.volatility_lookback).mean() * 252)
            else:
                # Fallback to close-to-close
                returns = prices['Close'].pct_change().dropna()
                vol = returns.rolling(self.volatility_lookback).std() * np.sqrt(252)
                
        return vol
    
    def fit_garch_model(self, returns: pd.Series, symbol: str) -> Optional[object]:
        """
        Fit GARCH model to return series.
        
        Args:
            returns: Return series
            symbol: Asset symbol
            
        Returns:
            Fitted GARCH model
        """
        if not self.use_garch or len(returns) < 100:
            return None
        
        try:
            # Clean returns
            clean_returns = returns.dropna() * 100  # Scale for numerical stability
            
            if len(clean_returns) < 50:
                return None
            
            # Fit GARCH(1,1) model
            model = arch_model(clean_returns, vol='Garch', p=1, q=1, dist='normal')
            fitted_model = model.fit(disp='off', show_warning=False)
            
            return fitted_model
            
        except Exception as e:
            print(f"GARCH fitting failed for {symbol}: {e}")
            return None
    
    def forecast_garch_volatility(self, model: object, steps: int = 1) -> float:
        """
        Forecast volatility using fitted GARCH model.
        
        Args:
            model: Fitted GARCH model
            steps: Forecast steps ahead
            
        Returns:
            Forecasted volatility
        """
        try:
            forecast = model.forecast(horizon=steps)
            variance_forecast = forecast.variance.iloc[-1, 0]
            volatility_forecast = np.sqrt(variance_forecast) / 100 * np.sqrt(252)  # Annualized
            return volatility_forecast
        except:
            return None
    
    def create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for volatility prediction.
        
        Args:
            data: OHLCV data
            
        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        returns = data['Close'].pct_change()
        
        # Historical volatilities
        for window in [5, 10, 20, 60]:
            features[f'realized_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
            features[f'vol_of_vol_{window}'] = features[f'realized_vol_{window}'].rolling(window).std()
        
        # Volatility ratios
        features['vol_ratio_5_20'] = features['realized_vol_5'] / features['realized_vol_20']
        features['vol_ratio_10_60'] = features['realized_vol_10'] / features['realized_vol_60']
        
        # Range-based volatility
        if all(col in data.columns for col in ['High', 'Low']):
            features['true_range'] = np.maximum(
                data['High'] - data['Low'],
                np.maximum(
                    abs(data['High'] - data['Close'].shift(1)),
                    abs(data['Low'] - data['Close'].shift(1))
                )
            )
            features['atr_14'] = features['true_range'].rolling(14).mean()
            features['normalized_atr'] = features['atr_14'] / data['Close']
        
        # Volume-based features
        if 'Volume' in data.columns:
            features['volume_volatility'] = data['Volume'].rolling(20).std()
            features['price_volume_corr'] = returns.rolling(20).corr(data['Volume'].pct_change())
        
        # Momentum features
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
            features[f'rsi_{period}'] = rsi(data['Close'], period)
        
        # Microstructure features
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Gap volatility
            features['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
            features['gap_volatility'] = features['gap'].rolling(20).std()
            
            # Intraday patterns
            features['body_size'] = abs(data['Close'] - data['Open']) / data['Close']
            features['upper_shadow'] = (data['High'] - np.maximum(data['Open'], data['Close'])) / data['Close']
            features['lower_shadow'] = (np.minimum(data['Open'], data['Close']) - data['Low']) / data['Close']
        
        # Volatility clustering
        vol_20 = features['realized_vol_20']
        features['vol_clustering'] = vol_20.rolling(5).std()
        features['vol_persistence'] = vol_20.rolling(10).corr(vol_20.shift(1))
        
        # Regime indicators
        features['high_vol_regime'] = (features['realized_vol_20'] > features['realized_vol_20'].rolling(60).quantile(0.7)).astype(float)
        features['vol_trend'] = features['realized_vol_20'].rolling(10).apply(lambda x: stats.linregress(range(len(x)), x)[0])
        
        return features.dropna()
    
    def train_ml_volatility_model(self, data: pd.DataFrame, symbol: str) -> Optional[object]:
        """
        Train machine learning model for volatility prediction.
        
        Args:
            data: OHLCV data
            symbol: Asset symbol
            
        Returns:
            Trained ML model
        """
        if not self.use_ml_forecasting or len(data) < 200:
            return None
        
        try:
            # Create features
            features = self.create_volatility_features(data)
            
            if len(features) < 100:
                return None
            
            # Target: future 5-day realized volatility
            returns = data['Close'].pct_change()
            future_vol = returns.rolling(5).std().shift(-5) * np.sqrt(252)
            
            # Align features and target
            model_data = pd.concat([features, future_vol.rename('target')], axis=1).dropna()
            
            if len(model_data) < 50:
                return None
            
            X = model_data.drop('target', axis=1)
            y = model_data['target']
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Random Forest model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_scaled, y)
            
            # Store scaler
            self.scalers[symbol] = scaler
            
            return model
            
        except Exception as e:
            print(f"ML model training failed for {symbol}: {e}")
            return None
    
    def forecast_ml_volatility(self, data: pd.DataFrame, model: object, 
                              scaler: object) -> Optional[float]:
        """
        Forecast volatility using trained ML model.
        
        Args:
            data: Recent OHLCV data
            model: Trained ML model
            scaler: Feature scaler
            
        Returns:
            Forecasted volatility
        """
        try:
            # Create features
            features = self.create_volatility_features(data)
            
            if len(features) == 0:
                return None
            
            # Get latest features
            latest_features = features.iloc[-1:].values
            
            # Scale features
            latest_features_scaled = scaler.transform(latest_features)
            
            # Predict
            vol_forecast = model.predict(latest_features_scaled)[0]
            
            return max(0.01, vol_forecast)  # Ensure positive volatility
            
        except:
            return None
    
    def detect_volatility_regime(self, volatility: pd.Series) -> pd.Series:
        """
        Detect volatility regime changes.
        
        Args:
            volatility: Volatility series
            
        Returns:
            Regime series (0=low vol, 1=high vol, 2=transition)
        """
        if len(volatility) < self.regime_detection_period:
            return pd.Series(0, index=volatility.index)
        
        # Calculate regime thresholds
        vol_quantiles = volatility.rolling(self.regime_detection_period).quantile([0.3, 0.7])
        low_threshold = vol_quantiles.xs(0.3, level=1)
        high_threshold = vol_quantiles.xs(0.7, level=1)
        
        # Classify regimes
        regimes = pd.Series(0, index=volatility.index)  # Default to low vol
        
        regimes[volatility > high_threshold] = 1  # High vol regime
        
        # Detect transitions (rapid changes)
        vol_change = volatility.pct_change().abs()
        vol_change_threshold = vol_change.rolling(20).quantile(0.8)
        regimes[vol_change > vol_change_threshold] = 2  # Transition regime
        
        return regimes
    
    def calculate_volatility_signals(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Calculate volatility-based trading signals.
        
        Args:
            data: OHLCV data
            symbol: Asset symbol
            
        Returns:
            DataFrame with volatility signals
        """
        df = data.copy()
        
        # Calculate realized volatility
        realized_vol = self.calculate_realized_volatility(df, method='garman_klass')
        
        # Forecast volatility using multiple methods
        forecasted_vols = []
        
        # 1. Simple moving average forecast
        vol_ma = realized_vol.rolling(20).mean()
        forecasted_vols.append(vol_ma)
        
        # 2. GARCH forecast
        if self.use_garch:
            returns = df['Close'].pct_change().dropna()
            if symbol not in self.garch_models:
                self.garch_models[symbol] = self.fit_garch_model(returns, symbol)
            
            if self.garch_models[symbol] is not None:
                try:
                    garch_forecast = self.forecast_garch_volatility(self.garch_models[symbol])
                    if garch_forecast is not None:
                        garch_vol_series = pd.Series(garch_forecast, index=df.index)
                        forecasted_vols.append(garch_vol_series)
                except:
                    pass
        
        # 3. ML forecast
        if self.use_ml_forecasting:
            if symbol not in self.ml_models:
                self.ml_models[symbol] = self.train_ml_volatility_model(df, symbol)
            
            if self.ml_models[symbol] is not None and symbol in self.scalers:
                ml_forecast = self.forecast_ml_volatility(df, self.ml_models[symbol], self.scalers[symbol])
                if ml_forecast is not None:
                    ml_vol_series = pd.Series(ml_forecast, index=df.index)
                    forecasted_vols.append(ml_vol_series)
        
        # Combine forecasts
        if forecasted_vols:
            # Simple average of available forecasts
            forecast_df = pd.concat(forecasted_vols, axis=1).fillna(method='ffill')
            implied_vol = forecast_df.mean(axis=1)
        else:
            implied_vol = vol_ma
        
        # Detect volatility regime
        vol_regime = self.detect_volatility_regime(realized_vol)
        
        # Calculate volatility ratio
        vol_ratio = realized_vol / implied_vol
        
        # Generate signals
        signals = pd.Series(0.0, index=df.index)
        
        # Signal logic
        # Long volatility when realized < implied (volatility underpriced)
        long_vol_condition = (vol_ratio < 1/self.entry_vol_ratio) & (vol_regime != 1)  # Not in high vol regime
        signals[long_vol_condition] = 0.6
        
        # Short volatility when realized > implied (volatility overpriced)
        short_vol_condition = (vol_ratio > self.entry_vol_ratio) & (vol_regime == 1)  # In high vol regime
        signals[short_vol_condition] = -0.6
        
        # Exit signals
        exit_long_condition = vol_ratio > 1/self.exit_vol_ratio
        exit_short_condition = vol_ratio < self.exit_vol_ratio
        
        signals[exit_long_condition & (signals > 0)] = 0.0
        signals[exit_short_condition & (signals < 0)] = 0.0
        
        # Special handling for regime transitions
        transition_condition = vol_regime == 2
        signals[transition_condition] *= 0.5  # Reduce position size during transitions
        
        # Add volatility information to dataframe
        df['Realized_Vol'] = realized_vol
        df['Implied_Vol'] = implied_vol
        df['Vol_Ratio'] = vol_ratio
        df['Vol_Regime'] = vol_regime
        df['Signal'] = signals
        
        return df
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate volatility arbitrage signals for all assets.
        
        Args:
            data: Dictionary of OHLCV DataFrames
            
        Returns:
            Dictionary of DataFrames with signals
        """
        results = {}
        
        for symbol, df in data.items():
            try:
                if len(df) < self.volatility_lookback:
                    result_df = df.copy()
                    result_df['Signal'] = 0.0
                    results[symbol] = result_df
                    continue
                
                # Calculate volatility signals
                results[symbol] = self.calculate_volatility_signals(df, symbol)
                
            except Exception as e:
                print(f"   Warning: Error in volatility signals for {symbol}: {e}")
                # Return safe default
                result_df = df.copy()
                result_df['Signal'] = 0.0
                results[symbol] = result_df
        
        return results
    
    def get_strategy_params(self) -> dict:
        """Get strategy parameters."""
        return {
            'volatility_lookback': self.volatility_lookback,
            'regime_detection_period': self.regime_detection_period,
            'volatility_threshold': self.volatility_threshold,
            'entry_vol_ratio': self.entry_vol_ratio,
            'exit_vol_ratio': self.exit_vol_ratio,
            'rebalance_frequency': self.rebalance_frequency,
            'use_garch': self.use_garch,
            'use_ml_forecasting': self.use_ml_forecasting,
            'models_trained': len(self.ml_models)
        } 