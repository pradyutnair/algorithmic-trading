#!/usr/bin/env python3
"""
Quick Strategy Demo

A simplified demo that focuses on one advanced strategy to ensure it works.
Perfect for quick testing and hackathon demos.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import strategy
from strategies.adaptive_regime_strategy import AdaptiveRegimeStrategy

# Import utilities  
from utils.data_utils import fetch_stock_data
from backtesting.backtest_engine import BacktestEngine
from backtesting.performance_metrics import PerformanceMetrics


def main():
    """
    Quick demo of the Adaptive Regime Strategy.
    """
    print("🚀 Quick Strategy Demo - Adaptive Regime Strategy")
    print("=" * 55)
    
    # Configuration - Simple and reliable
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01' 
    initial_capital = 100000
    
    print(f"📊 Symbol: {symbol}")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"💰 Capital: ${initial_capital:,}")
    
    # Step 1: Fetch Data
    print("\n📡 Fetching Data...")
    try:
        data = fetch_stock_data(symbol, start_date, end_date)
        
        if data is None:
            print("❌ Failed to fetch data. Trying with different dates...")
            # Try with more recent dates
            data = fetch_stock_data(symbol, '2023-06-01', '2023-12-31')
            
        if data is None:
            print("❌ Could not fetch any data. Check:")
            print("  - Internet connection")
            print("  - Symbol validity")
            print("  - Date ranges")
            return
            
        print(f"✅ Got {len(data)} records")
        print(f"📈 Date range: {data.index.min().date()} to {data.index.max().date()}")
        print(f"📊 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return
    
    # Step 2: Initialize Strategy
    print("\n⚙️ Initializing Adaptive Regime Strategy...")
    strategy = AdaptiveRegimeStrategy(
        initial_capital=initial_capital,
        regime_lookback=40,  # Shorter for demo data
        volatility_threshold=0.02,
        trend_threshold=0.05,
        regime_confidence=0.6  # Lower threshold for demo
    )
    print("✅ Strategy initialized")
    
    # Step 3: Generate Signals
    print("\n🧠 Generating Signals...")
    try:
        signals_df = strategy.generate_signals(data)
        
        # Count signals
        signal_counts = {
            'Total Days': len(signals_df),
            'Signal Days': len(signals_df[signals_df['Signal'] != 0]),
            'Buy Signals': len(signals_df[signals_df['Signal'] > 0]),
            'Sell Signals': len(signals_df[signals_df['Signal'] < 0]),
            'Regimes Detected': len(signals_df['Regime'].unique()) if 'Regime' in signals_df.columns else 0
        }
        
        print("✅ Signals generated:")
        for key, value in signal_counts.items():
            print(f"   {key}: {value}")
        
        # Show regime distribution
        if 'Regime' in signals_df.columns:
            regime_counts = signals_df['Regime'].value_counts()
            print("\n📊 Regime Distribution:")
            for regime, count in regime_counts.items():
                percentage = (count / len(signals_df)) * 100
                print(f"   {regime}: {count} days ({percentage:.1f}%)")
                
    except Exception as e:
        print(f"❌ Error generating signals: {e}")
        return
    
    # Step 4: Backtest
    print("\n🔄 Running Backtest...")
    try:
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=0.001,  # 0.1%
            slippage=0.0005   # 0.05%
        )
        
        result = engine.run_backtest(signals_df, symbol)
        
        if result is None:
            print("❌ Backtest failed")
            return
            
        print("✅ Backtest completed")
        
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        return
    
    # Step 5: Performance Analysis
    print("\n📈 Performance Analysis")
    print("=" * 30)
    
    try:
        returns = result['returns']
        metrics = PerformanceMetrics()
        
        # Calculate metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = metrics.sharpe_ratio(returns)
        max_drawdown = metrics.max_drawdown((1 + returns).cumprod())
        win_rate = (returns > 0).mean()
        
        # Display results
        print(f"📊 Total Return: {total_return:.2%}")
        print(f"📈 Annual Return: {annual_return:.2%}")
        print(f"📉 Volatility: {volatility:.2%}")
        print(f"⚡ Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"📉 Max Drawdown: {max_drawdown:.2%}")
        print(f"🎯 Win Rate: {win_rate:.2%}")
        print(f"📝 Total Trades: {len(returns[returns != 0])}")
        
        # Benchmark comparison
        buy_hold_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
        print(f"\n🏆 Strategy vs Buy & Hold:")
        print(f"   Strategy: {total_return:.2%}")
        print(f"   Buy & Hold: {buy_hold_return:.2%}")
        print(f"   Outperformance: {(total_return - buy_hold_return):.2%}")
        
    except Exception as e:
        print(f"❌ Performance analysis error: {e}")
        return
    
    # Step 6: Visualization
    print("\n📊 Creating Visualization...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Adaptive Regime Strategy - {symbol}', fontsize=16)
        
        # Plot 1: Price and Signals
        ax1 = axes[0, 0]
        ax1.plot(data.index, data['Close'], label='Price', alpha=0.7)
        
        # Highlight signal points
        buy_signals = signals_df[signals_df['Signal'] > 0]
        sell_signals = signals_df[signals_df['Signal'] < 0]
        
        if len(buy_signals) > 0:
            ax1.scatter(buy_signals.index, buy_signals['Close'], 
                       color='green', marker='^', s=50, alpha=0.8, label='Buy')
        if len(sell_signals) > 0:
            ax1.scatter(sell_signals.index, sell_signals['Close'], 
                       color='red', marker='v', s=50, alpha=0.8, label='Sell')
        
        ax1.set_title('Price and Trading Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Returns
        ax2 = axes[0, 1]
        cum_returns = (1 + returns).cumprod()
        cum_bh = data['Close'] / data['Close'].iloc[0]
        
        ax2.plot(cum_returns.index, cum_returns.values, label='Strategy', linewidth=2)
        ax2.plot(data.index, cum_bh.values, label='Buy & Hold', alpha=0.7)
        ax2.set_title('Cumulative Returns')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        ax3 = axes[1, 0]
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        ax3.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax3.set_title('Strategy Drawdown')
        ax3.set_ylabel('Drawdown')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Regime Timeline
        ax4 = axes[1, 1]
        if 'Regime' in signals_df.columns:
            regime_numeric = pd.Categorical(signals_df['Regime']).codes
            ax4.plot(signals_df.index, regime_numeric, drawstyle='steps-post')
            ax4.set_title('Market Regime Timeline')
            ax4.set_ylabel('Regime')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Regime Data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Regime Data Not Available')
        
        plt.tight_layout()
        plt.savefig('quick_strategy_demo_results.png', dpi=300, bbox_inches='tight')
        print("✅ Chart saved as 'quick_strategy_demo_results.png'")
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
    
    # Final Summary
    print("\n🎯 Strategy Summary:")
    print("=" * 30)
    print("✅ Adaptive Regime Strategy successfully demonstrated!")
    print(f"📊 Processed {len(data)} days of {symbol} data")
    print(f"🧠 Detected {signal_counts['Regimes Detected']} different market regimes")
    print(f"📈 Generated {signal_counts['Signal Days']} trading signals")
    print(f"🏆 {'Outperformed' if total_return > buy_hold_return else 'Underperformed'} buy & hold")
    
    print("\n💡 Next Steps for Hackathon:")
    print("1. 🔧 Tune parameters for your specific dataset")
    print("2. 🎨 Enhance visualizations for presentation")
    print("3. 📊 Add more symbols for portfolio testing")
    print("4. 🚀 Try other advanced strategies for comparison")
    print("5. 🏆 Combine strategies for ensemble approach")
    
    print(f"\n🚀 Ready for hackathon success! 🎉")


if __name__ == "__main__":
    main() 