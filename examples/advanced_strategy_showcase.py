#!/usr/bin/env python3
"""
Advanced Strategy Showcase

Demonstrates the sophisticated trading strategies available in the hackathon toolkit:
1. Adaptive Regime Strategy - Adapts to market conditions
2. Multi-Factor Alpha Strategy - Combines multiple alpha sources
3. Statistical Arbitrage Strategy - Pairs trading with cointegration
4. Volatility Arbitrage Strategy - Exploits volatility mispricings

This example shows how to implement and compare these strategies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Import strategies
from strategies.adaptive_regime_strategy import AdaptiveRegimeStrategy
from strategies.multi_factor_alpha_strategy import MultiFactorAlphaStrategy
from strategies.statistical_arbitrage_strategy import StatisticalArbitrageStrategy
from strategies.volatility_arbitrage_strategy import VolatilityArbitrageStrategy

# Import utilities
from utils.data_utils import fetch_stock_data, calculate_returns
from backtesting.backtest_engine import BacktestEngine
from backtesting.performance_metrics import PerformanceMetrics


def main():
    """
    Main function to showcase advanced strategies.
    """
    print("🚀 Advanced Strategy Showcase for Hackathon")
    print("=" * 60)
    
    # Configuration - More conservative for demo
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Fewer symbols
    start_date = '2023-01-01'  # Shorter, more recent period
    end_date = '2024-01-01'
    initial_capital = 100000
    
    print(f"📊 Fetching data for {len(symbols)} symbols...")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"💰 Initial Capital: ${initial_capital:,}")
    
    # Step 1: Fetch Data
    try:
        data = {}
        
        # Test connection first
        print("   Testing connection...")
        test_data = fetch_stock_data('AAPL', '2023-01-01', '2023-06-01')  # 5 months of data
        if test_data is None:
            print("❌ Cannot connect to data source. Check internet connection.")
            return
        print("   ✅ Connection successful")
        
        for symbol in symbols:
            print(f"   Fetching {symbol}...")
            stock_data = fetch_stock_data(symbol, start_date, end_date)
            if stock_data is not None and len(stock_data) > 50:  # Reduced minimum requirement
                data[symbol] = stock_data
                print(f"   ✅ {symbol}: {len(stock_data)} records")
            else:
                print(f"   ⚠️ Insufficient data for {symbol}")
        
        if len(data) < 2:  # Reduced minimum requirement
            print("❌ Not enough valid data. Need at least 2 symbols.")
            print("💡 Try running examples/test_data_fetch.py to debug")
            return
            
        print(f"✅ Successfully loaded data for {len(data)} symbols")
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return
    
    # Step 2: Initialize Advanced Strategies
    print("\n⚙️ Initializing Advanced Strategies...")
    
    strategies = {
        'Adaptive Regime': AdaptiveRegimeStrategy(
            initial_capital=initial_capital,
            regime_lookback=60,
            volatility_threshold=0.02,
            trend_threshold=0.05,
            regime_confidence=0.7
        ),
        
        'Multi-Factor Alpha': MultiFactorAlphaStrategy(
            initial_capital=initial_capital,
            lookback_periods=[5, 10, 20, 60],
            factor_decay=0.05,
            min_factor_score=0.3,
            max_position_weight=0.15
        ),
        
        'Statistical Arbitrage': StatisticalArbitrageStrategy(
            initial_capital=initial_capital,
            lookback_period=252,
            cointegration_threshold=0.05,
            entry_zscore=2.0,
            exit_zscore=0.5,
            max_pairs=3
        ),
        
        'Volatility Arbitrage': VolatilityArbitrageStrategy(
            initial_capital=initial_capital,
            volatility_lookback=60,
            entry_vol_ratio=1.2,
            exit_vol_ratio=1.05,
            use_garch=True,
            use_ml_forecasting=True
        )
    }
    
    print(f"✅ Initialized {len(strategies)} strategies")
    
    # Step 3: Generate Signals for Each Strategy
    print("\n📡 Generating Signals...")
    strategy_results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"   Processing {strategy_name}...")
        try:
            if strategy_name in ['Multi-Factor Alpha', 'Statistical Arbitrage', 'Volatility Arbitrage']:
                # These strategies need multiple assets as a dictionary
                signals = strategy.generate_signals(data)
            else:
                # Single asset strategies
                signals = {}
                for symbol, df in data.items():
                    signal_df = strategy.generate_signals(df)
                    signals[symbol] = signal_df
            
            strategy_results[strategy_name] = signals
            print(f"   ✅ {strategy_name} signals generated")
            
        except Exception as e:
            print(f"   ❌ Error generating signals for {strategy_name}: {e}")
            strategy_results[strategy_name] = None
    
    # Step 4: Backtest Each Strategy
    print("\n🔄 Running Backtests...")
    backtest_results = {}
    
    for strategy_name, signals in strategy_results.items():
        if signals is None:
            continue
            
        print(f"   Backtesting {strategy_name}...")
        try:
            # Simple backtest simulation
            if isinstance(signals, dict):
                # Multi-asset strategy - combine signals
                all_returns = []
                for symbol, signal_df in signals.items():
                    if symbol in data and 'Signal' in signal_df.columns:
                        # Calculate simple returns based on signals
                        price_data = data[symbol]['Close']
                        signal_data = signal_df['Signal']
                        
                        # Align data
                        aligned_prices = price_data.reindex(signal_data.index, method='ffill')
                        returns = aligned_prices.pct_change() * signal_data.shift(1)
                        returns = returns.dropna()
                        
                        if len(returns) > 0:
                            all_returns.append(returns)
                
                if all_returns:
                    # Combine returns (equal weight)
                    combined_returns = pd.concat(all_returns, axis=1).mean(axis=1)
                    backtest_results[strategy_name] = {'returns': combined_returns}
            else:
                # Single asset strategy
                if 'Signal' in signals.columns and len(data) > 0:
                    symbol = list(data.keys())[0]
                    price_data = data[symbol]['Close']
                    signal_data = signals['Signal']
                    
                    # Align data and calculate returns
                    aligned_prices = price_data.reindex(signal_data.index, method='ffill')
                    returns = aligned_prices.pct_change() * signal_data.shift(1)
                    returns = returns.dropna()
                    
                    if len(returns) > 0:
                        backtest_results[strategy_name] = {'returns': returns}
            
            print(f"   ✅ {strategy_name} backtest completed")
            
        except Exception as e:
            print(f"   ❌ Backtest failed for {strategy_name}: {e}")
    
    # Step 5: Performance Analysis
    print("\n📈 Performance Analysis")
    print("=" * 40)
    
    if not backtest_results:
        print("❌ No successful backtests to analyze")
        return
    
    # Calculate performance metrics
    performance_summary = {}
    
    for strategy_name, result in backtest_results.items():
        try:
            if isinstance(result, dict) and 'returns' in result:
                returns = result['returns']
            else:
                returns = result.get('returns', pd.Series())
            
            if len(returns) > 0:
                # Calculate key metrics directly
                total_return = (1 + returns).prod() - 1
                annual_return = (1 + returns.mean()) ** 252 - 1
                volatility = returns.std() * np.sqrt(252)
                
                # Sharpe ratio calculation
                sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
                
                # Max drawdown calculation
                cum_returns = (1 + returns).cumprod()
                running_max = cum_returns.expanding().max()
                drawdown = (cum_returns - running_max) / running_max
                max_drawdown = drawdown.min()
                
                win_rate = (returns > 0).mean()
                
                performance_summary[strategy_name] = {
                    'Total Return': f"{total_return:.2%}",
                    'Annual Return': f"{annual_return:.2%}",
                    'Volatility': f"{volatility:.2%}",
                    'Sharpe Ratio': f"{sharpe_ratio:.3f}",
                    'Max Drawdown': f"{max_drawdown:.2%}",
                    'Win Rate': f"{win_rate:.2%}",
                    'Total Trades': len(returns[returns != 0])
                }
        except Exception as e:
            print(f"   ❌ Error calculating metrics for {strategy_name}: {e}")
    
    # Display performance table
    if performance_summary:
        print("\n📊 Strategy Performance Comparison:")
        print("-" * 80)
        
        # Create performance DataFrame
        df_performance = pd.DataFrame(performance_summary).T
        print(df_performance.to_string())
        
        # Step 6: Visualization
        print("\n📊 Creating Visualizations...")
        
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Advanced Trading Strategies Performance Analysis', fontsize=16)
            
            # Plot 1: Cumulative Returns
            ax1 = axes[0, 0]
            for strategy_name, result in backtest_results.items():
                if isinstance(result, dict) and 'returns' in result:
                    returns = result['returns']
                    cum_returns = (1 + returns).cumprod()
                    ax1.plot(cum_returns.index, cum_returns.values, 
                            label=strategy_name, linewidth=2)
            
            ax1.set_title('Cumulative Returns Comparison')
            ax1.set_ylabel('Cumulative Return')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Risk-Return Scatter
            ax2 = axes[0, 1]
            risk_return_data = []
            for strategy_name, result in backtest_results.items():
                if isinstance(result, dict) and 'returns' in result:
                    returns = result['returns']
                    annual_return = (1 + returns.mean()) ** 252 - 1
                    volatility = returns.std() * np.sqrt(252)
                    risk_return_data.append((volatility, annual_return, strategy_name))
            
            if risk_return_data:
                volatilities, annual_returns, names = zip(*risk_return_data)
                scatter = ax2.scatter(volatilities, annual_returns, s=100, alpha=0.7)
                
                # Add labels
                for i, name in enumerate(names):
                    ax2.annotate(name, (volatilities[i], annual_returns[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
                
                ax2.set_xlabel('Volatility (Annual)')
                ax2.set_ylabel('Annual Return')
                ax2.set_title('Risk-Return Profile')
                ax2.grid(True, alpha=0.3)
            
            # Plot 3: Drawdown Analysis
            ax3 = axes[1, 0]
            for strategy_name, result in backtest_results.items():
                if isinstance(result, dict) and 'returns' in result:
                    returns = result['returns']
                    cum_returns = (1 + returns).cumprod()
                    rolling_max = cum_returns.expanding().max()
                    drawdown = (cum_returns - rolling_max) / rolling_max
                    ax3.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, label=strategy_name)
            
            ax3.set_title('Drawdown Analysis')
            ax3.set_ylabel('Drawdown')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Performance Metrics Heatmap
            ax4 = axes[1, 1]
            if performance_summary:
                # Prepare data for heatmap
                metrics_for_heatmap = {}
                for strategy, metrics in performance_summary.items():
                    metrics_for_heatmap[strategy] = {
                        'Sharpe Ratio': float(metrics['Sharpe Ratio']),
                        'Win Rate': float(metrics['Win Rate'].strip('%')) / 100,
                        'Total Trades': min(metrics['Total Trades'] / 100, 1)  # Normalize
                    }
                
                heatmap_df = pd.DataFrame(metrics_for_heatmap).T
                sns.heatmap(heatmap_df, annot=True, cmap='RdYlGn', ax=ax4, 
                           cbar_kws={'label': 'Normalized Score'})
                ax4.set_title('Performance Metrics Heatmap')
            
            plt.tight_layout()
            plt.savefig('advanced_strategies_performance.png', dpi=300, bbox_inches='tight')
            print("✅ Performance chart saved as 'advanced_strategies_performance.png'")
            plt.show()
            
        except Exception as e:
            print(f"❌ Visualization error: {e}")
    
    # Step 7: Strategy Insights
    print("\n🧠 Strategy Insights:")
    print("=" * 40)
    
    insights = {
        'Adaptive Regime': "Adapts to market conditions automatically. Best for volatile markets with regime changes.",
        'Multi-Factor Alpha': "Combines multiple alpha sources with cross-sectional ranking. Strong for stock selection.",
        'Statistical Arbitrage': "Market-neutral pairs trading. Lower risk but requires good pair selection.",
        'Volatility Arbitrage': "Exploits volatility mispricings. Uses GARCH and ML for sophisticated vol forecasting."
    }
    
    for strategy_name, insight in insights.items():
        if strategy_name in backtest_results:
            print(f"\n📌 {strategy_name}:")
            print(f"   {insight}")
            
            # Add strategy-specific parameters
            if strategy_name in strategies:
                params = strategies[strategy_name].get_strategy_params()
                print(f"   Key Parameters: {params}")
    
    # Final recommendations
    print("\n🎯 Hackathon Recommendations:")
    print("=" * 40)
    print("1. 🔥 For quick results: Use Adaptive Regime Strategy")
    print("2. 🧬 For sophistication: Multi-Factor Alpha Strategy")  
    print("3. 🛡️ For risk management: Statistical Arbitrage")
    print("4. ⚡ For vol trading: Volatility Arbitrage Strategy")
    print("\n💡 Tip: Combine multiple strategies for diversification!")
    print("💡 Tip: Tune parameters based on your specific dataset!")
    print("💡 Tip: Add your own alpha factors to Multi-Factor strategy!")
    
    print(f"\n🏆 Advanced Strategy Showcase Complete!")


if __name__ == "__main__":
    main() 