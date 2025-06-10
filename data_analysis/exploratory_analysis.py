"""
Exploratory Data Analysis for Financial Markets

This module provides tools for analyzing financial data before developing strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class MarketAnalyzer:
    """
    Comprehensive market data analysis tool.
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Initialize the analyzer with market data.
        
        Args:
            data: Dictionary of stock DataFrames
        """
        self.data = data
        self.returns_data = self._calculate_returns()
        
    def _calculate_returns(self) -> Dict[str, pd.Series]:
        """Calculate returns for all assets."""
        returns = {}
        for symbol, df in self.data.items():
            returns[symbol] = df['Close'].pct_change().dropna()
        return returns
    
    def basic_statistics(self) -> pd.DataFrame:
        """
        Calculate basic statistics for all assets.
        
        Returns:
            DataFrame with basic statistics
        """
        stats_list = []
        
        for symbol, df in self.data.items():
            returns = self.returns_data[symbol]
            
            stats = {
                'Symbol': symbol,
                'Start_Date': df.index[0].strftime('%Y-%m-%d'),
                'End_Date': df.index[-1].strftime('%Y-%m-%d'),
                'Total_Days': len(df),
                'Avg_Price': df['Close'].mean(),
                'Price_Range': df['High'].max() - df['Low'].min(),
                'Avg_Volume': df['Volume'].mean(),
                'Daily_Return_Mean': returns.mean(),
                'Daily_Return_Std': returns.std(),
                'Annualized_Return': returns.mean() * 252,
                'Annualized_Volatility': returns.std() * np.sqrt(252),
                'Skewness': returns.skew(),
                'Kurtosis': returns.kurtosis(),
                'Min_Return': returns.min(),
                'Max_Return': returns.max(),
            }
            stats_list.append(stats)
        
        return pd.DataFrame(stats_list)
    
    def correlation_analysis(self) -> pd.DataFrame:
        """
        Calculate correlation matrix between assets.
        
        Returns:
            Correlation matrix DataFrame
        """
        # Create returns DataFrame
        returns_df = pd.DataFrame(self.returns_data)
        return returns_df.corr()
    
    def plot_price_charts(self, figsize: tuple = (15, 10)):
        """
        Plot price charts for all assets.
        
        Args:
            figsize: Figure size tuple
        """
        n_assets = len(self.data)
        n_cols = 2
        n_rows = (n_assets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes]
        
        for i, (symbol, df) in enumerate(self.data.items()):
            row = i // n_cols
            col = i % n_cols
            
            ax = axes[row][col]
            ax.plot(df.index, df['Close'], label=f'{symbol} Close', linewidth=1)
            ax.set_title(f'{symbol} Price Chart')
            ax.set_ylabel('Price ($)')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide empty subplots
        for i in range(n_assets, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row][col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_returns_distribution(self, figsize: tuple = (15, 8)):
        """
        Plot returns distribution for all assets.
        
        Args:
            figsize: Figure size tuple
        """
        n_assets = len(self.returns_data)
        n_cols = 3
        n_rows = (n_assets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes]
        
        for i, (symbol, returns) in enumerate(self.returns_data.items()):
            row = i // n_cols
            col = i % n_cols
            
            ax = axes[row][col]
            ax.hist(returns, bins=50, alpha=0.7, density=True)
            ax.set_title(f'{symbol} Returns Distribution')
            ax.set_xlabel('Daily Returns')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            
            # Add normal distribution overlay
            mu, sigma = returns.mean(), returns.std()
            x = np.linspace(returns.min(), returns.max(), 100)
            normal_dist = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            ax.plot(x, normal_dist, 'r-', linewidth=2, label='Normal Distribution')
            ax.legend()
        
        # Hide empty subplots
        for i in range(n_assets, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row][col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, figsize: tuple = (10, 8)):
        """
        Plot correlation heatmap.
        
        Args:
            figsize: Figure size tuple
        """
        corr_matrix = self.correlation_analysis()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Asset Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def volatility_analysis(self) -> pd.DataFrame:
        """
        Analyze volatility patterns.
        
        Returns:
            DataFrame with volatility metrics
        """
        vol_stats = []
        
        for symbol, returns in self.returns_data.items():
            # Rolling volatility
            rolling_vol = returns.rolling(30).std() * np.sqrt(252)
            
            stats = {
                'Symbol': symbol,
                'Mean_Volatility': rolling_vol.mean(),
                'Volatility_Std': rolling_vol.std(),
                'Min_Volatility': rolling_vol.min(),
                'Max_Volatility': rolling_vol.max(),
                'Current_Volatility': rolling_vol.iloc[-1] if len(rolling_vol) > 0 else np.nan,
                'Volatility_Percentile_50': rolling_vol.quantile(0.5),
                'Volatility_Percentile_95': rolling_vol.quantile(0.95),
            }
            vol_stats.append(stats)
        
        return pd.DataFrame(vol_stats)
    
    def market_regime_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze different market regimes (bull, bear, sideways).
        
        Returns:
            Dictionary with regime analysis for each asset
        """
        regime_analysis = {}
        
        for symbol, df in self.data.items():
            # Calculate rolling returns
            returns = self.returns_data[symbol]
            rolling_returns = returns.rolling(60).mean() * 252  # 60-day annualized returns
            
            # Define regimes
            bull_threshold = 0.10  # 10% annualized return
            bear_threshold = -0.10  # -10% annualized return
            
            regimes = pd.Series('Sideways', index=rolling_returns.index)
            regimes[rolling_returns > bull_threshold] = 'Bull'
            regimes[rolling_returns < bear_threshold] = 'Bear'
            
            # Calculate regime statistics
            regime_stats = []
            for regime in ['Bull', 'Bear', 'Sideways']:
                mask = regimes == regime
                regime_returns = returns[mask]
                
                if len(regime_returns) > 0:
                    stats = {
                        'Regime': regime,
                        'Days': len(regime_returns),
                        'Percentage': len(regime_returns) / len(returns) * 100,
                        'Avg_Return': regime_returns.mean(),
                        'Volatility': regime_returns.std(),
                        'Sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0,
                        'Best_Day': regime_returns.max(),
                        'Worst_Day': regime_returns.min(),
                    }
                    regime_stats.append(stats)
            
            regime_analysis[symbol] = pd.DataFrame(regime_stats)
        
        return regime_analysis
    
    def technical_levels_analysis(self) -> Dict[str, Dict]:
        """
        Identify key technical levels (support, resistance).
        
        Returns:
            Dictionary with technical analysis for each asset
        """
        technical_analysis = {}
        
        for symbol, df in self.data.items():
            prices = df['Close']
            
            # Calculate support and resistance levels
            recent_high = prices.rolling(252).max().iloc[-1]  # 1-year high
            recent_low = prices.rolling(252).min().iloc[-1]   # 1-year low
            current_price = prices.iloc[-1]
            
            # Calculate moving averages
            ma_20 = prices.rolling(20).mean().iloc[-1]
            ma_50 = prices.rolling(50).mean().iloc[-1]
            ma_200 = prices.rolling(200).mean().iloc[-1]
            
            # Price relative to moving averages
            price_vs_ma20 = (current_price - ma_20) / ma_20 * 100
            price_vs_ma50 = (current_price - ma_50) / ma_50 * 100
            price_vs_ma200 = (current_price - ma_200) / ma_200 * 100
            
            analysis = {
                'current_price': current_price,
                'recent_high': recent_high,
                'recent_low': recent_low,
                'price_from_high': (current_price - recent_high) / recent_high * 100,
                'price_from_low': (current_price - recent_low) / recent_low * 100,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'ma_200': ma_200,
                'price_vs_ma20_pct': price_vs_ma20,
                'price_vs_ma50_pct': price_vs_ma50,
                'price_vs_ma200_pct': price_vs_ma200,
                'trend_short': 'Bullish' if price_vs_ma20 > 0 else 'Bearish',
                'trend_medium': 'Bullish' if price_vs_ma50 > 0 else 'Bearish',
                'trend_long': 'Bullish' if price_vs_ma200 > 0 else 'Bearish',
            }
            
            technical_analysis[symbol] = analysis
        
        return technical_analysis
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            Formatted analysis report string
        """
        basic_stats = self.basic_statistics()
        vol_stats = self.volatility_analysis()
        corr_matrix = self.correlation_analysis()
        technical_levels = self.technical_levels_analysis()
        
        report = f"""
# 📊 Market Analysis Report

## 📈 Portfolio Overview
- **Number of Assets**: {len(self.data)}
- **Analysis Period**: {basic_stats['Start_Date'].iloc[0]} to {basic_stats['End_Date'].iloc[0]}
- **Total Trading Days**: {basic_stats['Total_Days'].mean():.0f} days average

## 🎯 Performance Summary

### Top Performers (Annualized Return)
{basic_stats.nlargest(3, 'Annualized_Return')[['Symbol', 'Annualized_Return']].to_string(index=False)}

### Highest Volatility
{vol_stats.nlargest(3, 'Mean_Volatility')[['Symbol', 'Mean_Volatility']].to_string(index=False)}

### Most Correlated Pairs
"""
        
        # Find most correlated pairs
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                symbol1 = corr_matrix.columns[i]
                symbol2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                corr_pairs.append((symbol1, symbol2, correlation))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for pair in corr_pairs[:3]:
            report += f"\n{pair[0]} - {pair[1]}: {pair[2]:.3f}"
        
        report += f"""

## 🚨 Risk Assessment

### High Risk Assets (Volatility > 30%)
"""
        high_vol_assets = vol_stats[vol_stats['Mean_Volatility'] > 0.30]
        if len(high_vol_assets) > 0:
            report += high_vol_assets[['Symbol', 'Mean_Volatility']].to_string(index=False)
        else:
            report += "None"
        
        report += f"""

## 📊 Technical Analysis Summary

### Current Trends
"""
        for symbol, analysis in technical_levels.items():
            report += f"\n**{symbol}**: Short: {analysis['trend_short']}, Medium: {analysis['trend_medium']}, Long: {analysis['trend_long']}"
        
        report += f"""

## 🎯 Strategy Recommendations

### Mean Reversion Candidates
- Look for assets with high volatility but stable fundamentals
- Best candidates: Assets trading near support levels

### Momentum Candidates  
- Look for assets in strong trends with good volume
- Best candidates: Assets above all moving averages

### Pairs Trading Candidates
- Look for highly correlated assets with temporary divergence
- Best pairs: Assets with correlation > 0.7

## ⚠️ Risk Warnings
1. High correlation between assets reduces diversification benefits
2. High volatility assets require smaller position sizes
3. Consider market regime when selecting strategies
"""
        
        return report 