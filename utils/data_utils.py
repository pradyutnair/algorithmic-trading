"""
Data utility functions for fetching and processing financial data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def fetch_stock_data(symbol: str, 
                    start_date: str, 
                    end_date: str, 
                    interval: str = '1d') -> Optional[pd.DataFrame]:
    """
    Fetch stock data from Yahoo Finance for a single symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d', '1h', '5m', etc.)
    
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    try:
        # Clean the symbol to avoid parsing issues
        symbol = symbol.strip().upper()
        
        # Download data using yfinance
        ticker = yf.Ticker(symbol)
        # Try multiple approaches to fetch data
        try:
            # First try with start/end dates
            df = ticker.history(start=start_date, end=end_date, interval=interval, auto_adjust=True)
            
            # If empty, try with period instead
            if df.empty:
                # Calculate period in years
                from datetime import datetime
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                years = (end_dt - start_dt).days / 365.25
                
                if years >= 2:
                    df = ticker.history(period="2y", interval=interval, auto_adjust=True)
                elif years >= 1:
                    df = ticker.history(period="1y", interval=interval, auto_adjust=True)
                else:
                    df = ticker.history(period="6mo", interval=interval, auto_adjust=True)
                
                # Filter to requested date range if we got data
                if not df.empty and start_date and end_date:
                    df = df.loc[start_date:end_date]
                    
        except Exception as fetch_error:
            # Fallback to period-based fetch
            df = ticker.history(period="1y", interval=interval, auto_adjust=True)
        
        if df.empty:
            print(f"No data found for {symbol}")
            return None
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            print(f"Missing required columns for {symbol}")
            return None
        
        # Clean data
        df = df.dropna()
        
        if len(df) < 50:  # Reduced minimum data requirement
            print(f"Insufficient data for {symbol}: only {len(df)} records")
            return None
        return df
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None


def fetch_multiple_stocks(symbols: List[str], 
                         start_date: str, 
                         end_date: str, 
                         interval: str = '1d') -> Dict[str, pd.DataFrame]:
    """
    Fetch stock data from Yahoo Finance for multiple symbols.
    
    Args:
        symbols: List of stock symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d', '1h', '5m', etc.)
    
    Returns:
        Dictionary with symbol as key and DataFrame as value
    """
    data = {}
    
    for symbol in symbols:
        df = fetch_stock_data(symbol, start_date, end_date, interval)
        if df is not None:
            data[symbol] = df
    
    return data


def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        method: 'simple' or 'log'
    
    Returns:
        Returns series
    """
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError("Method must be 'simple' or 'log'")


def resample_data(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Resample OHLCV data to different frequency.
    
    Args:
        df: OHLCV DataFrame
        frequency: Target frequency ('D', 'W', 'M', etc.)
    
    Returns:
        Resampled DataFrame
    """
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    return df.resample(frequency).agg(ohlc_dict).dropna()


def align_dataframes(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Align multiple DataFrames to have the same date index.
    
    Args:
        dfs: List of DataFrames to align
    
    Returns:
        List of aligned DataFrames
    """
    # Find common date range
    start_date = max([df.index.min() for df in dfs])
    end_date = min([df.index.max() for df in dfs])
    
    # Align all DataFrames
    aligned_dfs = []
    for df in dfs:
        aligned_df = df.loc[start_date:end_date]
        aligned_dfs.append(aligned_df)
    
    return aligned_dfs


def clean_data(df: pd.DataFrame, 
               fill_method: str = 'forward', 
               remove_outliers: bool = True,
               outlier_threshold: float = 3.0) -> pd.DataFrame:
    """
    Clean financial data by handling missing values and outliers.
    
    Args:
        df: Input DataFrame
        fill_method: Method to fill missing values ('forward', 'backward', 'interpolate')
        remove_outliers: Whether to remove outliers
        outlier_threshold: Z-score threshold for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Handle missing values
    if fill_method == 'forward':
        df_clean = df_clean.fillna(method='ffill')
    elif fill_method == 'backward':
        df_clean = df_clean.fillna(method='bfill')
    elif fill_method == 'interpolate':
        df_clean = df_clean.interpolate()
    
    # Remove outliers
    if remove_outliers:
        for column in df_clean.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / df_clean[column].std())
            df_clean = df_clean[z_scores < outlier_threshold]
    
    return df_clean


def split_data(df: pd.DataFrame, 
               train_ratio: float = 0.7, 
               val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        train_ratio: Ratio for training data
        val_ratio: Ratio for validation data
    
    Returns:
        Train, validation, and test DataFrames
    """
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:train_size + val_size]
    test_data = df.iloc[train_size + val_size:]
    
    return train_data, val_data, test_data 