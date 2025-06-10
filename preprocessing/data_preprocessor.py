"""
Data preprocessing utilities for financial data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from utils.data_utils import clean_data, resample_data
from utils.indicators import *
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessing for financial datasets.
    """
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.scaler_params = {}
        self.outlier_params = {}
        
    def preprocess_ohlcv_data(self, 
                             data: Dict[str, pd.DataFrame],
                             clean_missing: bool = True,
                             handle_outliers: bool = True,
                             add_technical_indicators: bool = True,
                             normalize_prices: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Comprehensive preprocessing of OHLCV data.
        
        Args:
            data: Dictionary of OHLCV DataFrames
            clean_missing: Whether to clean missing values
            handle_outliers: Whether to handle outliers
            add_technical_indicators: Whether to add technical indicators
            normalize_prices: Whether to normalize price data
            
        Returns:
            Dictionary of preprocessed DataFrames
        """
        processed_data = {}
        
        for symbol, df in data.items():
            print(f"Processing {symbol}...")
            
            # Start with a copy
            processed_df = df.copy()
            
            # 1. Clean missing values
            if clean_missing:
                processed_df = self._clean_missing_values(processed_df)
            
            # 2. Handle outliers
            if handle_outliers:
                processed_df = self._handle_outliers(processed_df, symbol)
            
            # 3. Add derived features
            processed_df = self._add_derived_features(processed_df)
            
            # 4. Add technical indicators
            if add_technical_indicators:
                processed_df = self._add_technical_indicators(processed_df)
            
            # 5. Normalize prices if requested
            if normalize_prices:
                processed_df = self._normalize_prices(processed_df, symbol)
            
            # 6. Final cleanup
            processed_df = processed_df.dropna()
            
            processed_data[symbol] = processed_df
            print(f"✅ {symbol} processed: {len(processed_df)} rows")
        
        return processed_data
    
    def _clean_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean missing values in OHLCV data."""
        # Forward fill first, then backward fill any remaining
        df_clean = df.fillna(method='ffill').fillna(method='bfill')
        
        # For any remaining NaN in Volume, fill with median
        if 'Volume' in df_clean.columns:
            df_clean['Volume'] = df_clean['Volume'].fillna(df_clean['Volume'].median())
        
        return df_clean
    
    def _handle_outliers(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Handle outliers in price data."""
        df_clean = df.copy()
        
        # Calculate z-scores for returns
        returns = df_clean['Close'].pct_change()
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        
        # Cap extreme outliers (z-score > 5)
        extreme_outliers = z_scores > 5
        
        if extreme_outliers.sum() > 0:
            print(f"   Found {extreme_outliers.sum()} extreme outliers in {symbol}")
            
            # Replace outlier returns with median return
            median_return = returns.median()
            prev_prices = df_clean['Close'].shift(1)
            
            # Calculate new prices based on median return
            outlier_indices = extreme_outliers[extreme_outliers].index
            for idx in outlier_indices:
                if idx in prev_prices.index and not pd.isna(prev_prices[idx]):
                    new_price = prev_prices[idx] * (1 + median_return)
                    df_clean.loc[idx, 'Close'] = new_price
                    df_clean.loc[idx, 'Open'] = new_price
                    df_clean.loc[idx, 'High'] = new_price
                    df_clean.loc[idx, 'Low'] = new_price
        
        # Store outlier parameters for later use
        self.outlier_params[symbol] = {
            'return_mean': returns.mean(),
            'return_std': returns.std(),
            'median_return': returns.median()
        }
        
        return df_clean
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features from OHLCV data."""
        df_enhanced = df.copy()
        
        # Price-based features
        df_enhanced['Returns'] = df_enhanced['Close'].pct_change()
        df_enhanced['Log_Returns'] = np.log(df_enhanced['Close'] / df_enhanced['Close'].shift(1))
        df_enhanced['High_Low_Pct'] = (df_enhanced['High'] - df_enhanced['Low']) / df_enhanced['Close']
        df_enhanced['Open_Close_Pct'] = (df_enhanced['Close'] - df_enhanced['Open']) / df_enhanced['Open']
        
        # Volume features
        if 'Volume' in df_enhanced.columns:
            df_enhanced['Volume_MA_Ratio'] = df_enhanced['Volume'] / df_enhanced['Volume'].rolling(20).mean()
            df_enhanced['Price_Volume'] = df_enhanced['Close'] * df_enhanced['Volume']
        
        # Volatility features
        df_enhanced['Volatility_5d'] = df_enhanced['Returns'].rolling(5).std()
        df_enhanced['Volatility_20d'] = df_enhanced['Returns'].rolling(20).std()
        
        # Price momentum features
        df_enhanced['Price_Change_1d'] = df_enhanced['Close'] / df_enhanced['Close'].shift(1) - 1
        df_enhanced['Price_Change_5d'] = df_enhanced['Close'] / df_enhanced['Close'].shift(5) - 1
        df_enhanced['Price_Change_20d'] = df_enhanced['Close'] / df_enhanced['Close'].shift(20) - 1
        
        return df_enhanced
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data."""
        df_technical = df.copy()
        
        # Moving averages
        df_technical['SMA_10'] = sma(df_technical['Close'], 10)
        df_technical['SMA_20'] = sma(df_technical['Close'], 20)
        df_technical['SMA_50'] = sma(df_technical['Close'], 50)
        df_technical['EMA_12'] = ema(df_technical['Close'], 12)
        df_technical['EMA_26'] = ema(df_technical['Close'], 26)
        
        # RSI
        df_technical['RSI_14'] = rsi(df_technical['Close'], 14)
        
        # MACD
        macd_data = macd(df_technical['Close'])
        df_technical['MACD'] = macd_data['MACD']
        df_technical['MACD_Signal'] = macd_data['Signal']
        df_technical['MACD_Histogram'] = macd_data['Histogram']
        
        # Bollinger Bands
        bb_data = bollinger_bands(df_technical['Close'])
        df_technical['BB_Upper'] = bb_data['Upper']
        df_technical['BB_Middle'] = bb_data['Middle']
        df_technical['BB_Lower'] = bb_data['Lower']
        df_technical['BB_Width'] = (bb_data['Upper'] - bb_data['Lower']) / bb_data['Middle']
        df_technical['BB_Position'] = (df_technical['Close'] - bb_data['Lower']) / (bb_data['Upper'] - bb_data['Lower'])
        
        # Average True Range
        if all(col in df_technical.columns for col in ['High', 'Low', 'Close']):
            df_technical['ATR_14'] = atr(df_technical['High'], df_technical['Low'], df_technical['Close'])
        
        # Williams %R
        if all(col in df_technical.columns for col in ['High', 'Low', 'Close']):
            df_technical['Williams_R'] = williams_r(df_technical['High'], df_technical['Low'], df_technical['Close'])
        
        # On-Balance Volume
        if 'Volume' in df_technical.columns:
            df_technical['OBV'] = obv(df_technical['Close'], df_technical['Volume'])
        
        # Price momentum
        df_technical['Momentum_10'] = momentum(df_technical['Close'], 10)
        
        return df_technical
    
    def _normalize_prices(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize price data to [0, 1] range."""
        df_normalized = df.copy()
        
        price_columns = ['Open', 'High', 'Low', 'Close']
        available_price_cols = [col for col in price_columns if col in df_normalized.columns]
        
        if available_price_cols:
            # Use min-max normalization
            price_min = df_normalized[available_price_cols].min().min()
            price_max = df_normalized[available_price_cols].max().max()
            
            for col in available_price_cols:
                df_normalized[f'{col}_Normalized'] = (df_normalized[col] - price_min) / (price_max - price_min)
            
            # Store normalization parameters
            self.scaler_params[symbol] = {
                'price_min': price_min,
                'price_max': price_max
            }
        
        return df_normalized
    
    def create_features_for_ml(self, 
                              data: Dict[str, pd.DataFrame],
                              target_column: str = 'Returns',
                              lookback_periods: List[int] = [1, 5, 10, 20],
                              include_technical: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create feature matrix for machine learning.
        
        Args:
            data: Dictionary of preprocessed DataFrames
            target_column: Target variable column name
            lookback_periods: Periods to create lagged features
            include_technical: Whether to include technical indicators
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        feature_dfs = []
        target_series = []
        
        for symbol, df in data.items():
            # Create lagged features
            symbol_features = pd.DataFrame(index=df.index)
            
            # Add current period features
            feature_cols = ['Returns', 'Volatility_5d', 'Volatility_20d', 'Volume_MA_Ratio']
            if include_technical:
                feature_cols.extend(['RSI_14', 'MACD', 'BB_Position', 'Williams_R'])
            
            available_cols = [col for col in feature_cols if col in df.columns]
            
            for col in available_cols:
                symbol_features[f'{symbol}_{col}'] = df[col]
                
                # Add lagged versions
                for lag in lookback_periods:
                    symbol_features[f'{symbol}_{col}_lag_{lag}'] = df[col].shift(lag)
            
            # Add symbol identifier
            symbol_features[f'{symbol}_symbol'] = 1
            
            feature_dfs.append(symbol_features)
            
            # Add target variable (forward-looking)
            if target_column in df.columns:
                target_series.append(df[target_column].shift(-1))  # Next period return
        
        # Combine all features
        if feature_dfs:
            combined_features = pd.concat(feature_dfs, axis=1).fillna(0)
            combined_target = pd.concat(target_series, axis=0)
            
            # Align features and target
            common_index = combined_features.index.intersection(combined_target.index)
            combined_features = combined_features.loc[common_index]
            combined_target = combined_target.loc[common_index]
            
            # Remove rows with any NaN
            mask = ~(combined_features.isna().any(axis=1) | combined_target.isna())
            combined_features = combined_features[mask]
            combined_target = combined_target[mask]
            
            return combined_features, combined_target
        
        return pd.DataFrame(), pd.Series()
    
    def prepare_for_strategy(self, 
                            data: Dict[str, pd.DataFrame],
                            strategy_type: str = 'mean_reversion') -> Dict[str, pd.DataFrame]:
        """
        Prepare data specifically for a strategy type.
        
        Args:
            data: Dictionary of preprocessed DataFrames
            strategy_type: Type of strategy ('mean_reversion', 'momentum', 'pairs_trading')
            
        Returns:
            Dictionary of strategy-ready DataFrames
        """
        strategy_data = {}
        
        for symbol, df in data.items():
            strategy_df = df.copy()
            
            if strategy_type == 'mean_reversion':
                # Ensure we have required indicators for mean reversion
                required_cols = ['BB_Upper', 'BB_Lower', 'BB_Middle', 'RSI_14']
                if all(col in strategy_df.columns for col in required_cols):
                    strategy_data[symbol] = strategy_df
                else:
                    print(f"Warning: {symbol} missing required indicators for mean reversion")
            
            elif strategy_type == 'momentum':
                # Ensure we have required indicators for momentum
                required_cols = ['MACD', 'MACD_Signal', 'EMA_12', 'EMA_26']
                if all(col in strategy_df.columns for col in required_cols):
                    strategy_data[symbol] = strategy_df
                else:
                    print(f"Warning: {symbol} missing required indicators for momentum")
            
            elif strategy_type == 'pairs_trading':
                # For pairs trading, we need clean price data
                required_cols = ['Close', 'Returns']
                if all(col in strategy_df.columns for col in required_cols):
                    strategy_data[symbol] = strategy_df
                else:
                    print(f"Warning: {symbol} missing required data for pairs trading")
            
            else:
                # Default: include all data
                strategy_data[symbol] = strategy_df
        
        return strategy_data
    
    def get_preprocessing_summary(self) -> str:
        """
        Get summary of preprocessing steps applied.
        
        Returns:
            Summary string
        """
        summary = """
# 🔧 Data Preprocessing Summary

## ✅ Steps Applied:
1. **Missing Value Treatment**: Forward fill → Backward fill → Median imputation
2. **Outlier Detection**: Z-score based (threshold: 5.0)
3. **Derived Features**: Returns, volatility, price ratios
4. **Technical Indicators**: MA, RSI, MACD, Bollinger Bands, etc.
5. **Data Validation**: Remove remaining NaN values

## 📊 Features Created:
- **Price Features**: Returns, log returns, price changes
- **Volume Features**: Volume ratios, price-volume
- **Volatility Features**: Rolling volatility (5d, 20d)
- **Technical Indicators**: 15+ technical indicators
- **Momentum Features**: Multi-period momentum

## ⚠️ Quality Checks:
- Outlier detection and correction
- Missing value imputation
- Data type validation
- Index alignment
"""
        
        if self.outlier_params:
            summary += f"\n## 🚨 Outliers Handled:\n"
            for symbol, params in self.outlier_params.items():
                summary += f"- **{symbol}**: Mean return: {params['return_mean']:.4f}, Std: {params['return_std']:.4f}\n"
        
        return summary 