#!/usr/bin/env python3
"""
Simple Backtest Example

This script demonstrates how to:
1. Fetch data
2. Initialize a strategy
3. Run a backtest
4. Analyze results

Perfect for getting started quickly!
"""

import sys
import os

from strategies.momentum import MomentumStrategy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import our modules
from utils.data_utils import fetch_stock_data
from strategies.mean_reversion import MeanReversionStrategy
from backtesting.backtest_engine import BacktestEngine


def main():
    """Run a simple backtest example."""
    
    print("🚀 Starting Simple Backtest Example")
    print("=" * 50)
    
    # Step 1: Fetch Data
    print("📊 Fetching stock data...")
    symbols = ['AAPL', 'MSFT']
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    
    try:
        data = fetch_stock_data(symbols, start_date, end_date)
        print(f"✅ Successfully fetched data for {len(data)} symbols")
        
        for symbol, df in data.items():
            print(f"   {symbol}: {len(df)} days of data")
    
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        print("💡 Tip: Make sure you have internet connection and try again")
        return
    
    # Step 2: Initialize Strategy
    print("\n⚙️ Initializing Mean Reversion Strategy...")
    # strategy = MeanReversionStrategy(
    #     initial_capital=100000,
    #     bb_window=15,
    #     bb_std=1.5,
    #     rsi_window=14,
    #     rsi_oversold=25,
    #     rsi_overbought=75
    # )
    strategy = MomentumStrategy(
        initial_capital=10000,
        fast_ma=10,
        slow_ma=20,
        macd_fast=10,
        macd_slow=20,
        macd_signal=9
    )
    print(f"✅ Strategy initialized with ${strategy.initial_capital:,} capital")
    
    # Step 3: Run Backtest
    print("\n🔄 Running backtest...")
    engine = BacktestEngine(strategy, data)
    
    try:
        results = engine.run_backtest()
        print("✅ Backtest completed successfully!")
    
    except Exception as e:
        print(f"❌ Error during backtesting: {e}")
        return
    
    # Step 4: Display Results
    print("\n📈 Backtest Results")
    print("=" * 30)
    
    metrics = results['metrics']
    
    # Key performance metrics
    print(f"📊 PERFORMANCE SUMMARY")
    print(f"   Total Return:      {metrics.get('total_return', 0):.2%}")
    print(f"   Annualized Return: {metrics.get('annualized_return', 0):.2%}")
    print(f"   Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"   Max Drawdown:      {metrics.get('max_drawdown', 0):.2%}")
    
    print(f"\n📋 TRADING STATISTICS")
    print(f"   Total Trades:      {metrics.get('total_trades', 0)}")
    print(f"   Win Rate:          {metrics.get('win_rate', 0):.2%}")
    print(f"   Profit Factor:     {metrics.get('profit_factor', 0):.2f}")
    
    print(f"\n⚠️  RISK METRICS")
    print(f"   Value at Risk:     {metrics.get('var_95', 0):.2%}")
    print(f"   Expected Shortfall: {metrics.get('expected_shortfall', 0):.2%}")
    
    # Step 5: Create Visualizations
    print("\n📊 Generating plots...")
    try:
        engine.plot_results()
        print("✅ Plots generated successfully!")
    except Exception as e:
        print(f"⚠️  Could not generate plots: {e}")
    
    # Step 6: Strategy Analysis
    print("\n🔍 Strategy Analysis")
    print("=" * 30)
    
    portfolio_df = results['portfolio_values']
    
    # Calculate some additional metrics
    final_value = portfolio_df['portfolio_value'].iloc[-1]
    initial_value = portfolio_df['portfolio_value'].iloc[0]
    total_return = (final_value - initial_value) / initial_value
    
    print(f"📈 Portfolio Performance:")
    print(f"   Initial Value:  ${initial_value:,.2f}")
    print(f"   Final Value:    ${final_value:,.2f}")
    print(f"   Profit/Loss:    ${final_value - initial_value:,.2f}")
    
    # Risk assessment
    if metrics.get('sharpe_ratio', 0) > 1.0:
        print(f"\n✅ Good risk-adjusted performance (Sharpe > 1.0)")
    else:
        print(f"\n⚠️  Low risk-adjusted performance (Sharpe < 1.0)")
    
    if abs(metrics.get('max_drawdown', 0)) < 0.15:
        print(f"✅ Acceptable maximum drawdown (< 15%)")
    else:
        print(f"⚠️  High maximum drawdown (> 15%)")
    
    # Step 7: Next Steps
    print(f"\n🎯 Next Steps for Improvement")
    print("=" * 35)
    print("1. 🔧 Parameter Optimization:")
    print("   - Try different Bollinger Band periods (15, 25)")
    print("   - Adjust RSI thresholds (25/75, 35/65)")
    print("   - Test different standard deviations (1.5, 2.5)")
    
    print("\n2. 📊 Enhanced Testing:")
    print("   - Test on different time periods")
    print("   - Add more stocks to the universe")
    print("   - Compare with buy-and-hold strategy")
    
    print("\n3. ⚠️  Risk Management:")
    print("   - Implement stop-loss orders")
    print("   - Add position sizing rules")
    print("   - Consider correlation between stocks")
    
    print("\n4. 🔍 Strategy Variants:")
    print("   - Try momentum strategy")
    print("   - Implement pairs trading")
    print("   - Add machine learning features")
    
    print(f"\n🎉 Example completed! Check the plots and try modifying parameters.")


if __name__ == "__main__":
    main() 