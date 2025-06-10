"""
Backtesting engine for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from strategies.base_strategy import BaseStrategy
from .performance_metrics import PerformanceMetrics


class BacktestEngine:
    """
    Backtesting engine for trading strategies.
    """
    
    def __init__(self, 
                 strategy: BaseStrategy,
                 data: Dict[str, pd.DataFrame],
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
        """
        Initialize the backtesting engine.
        
        Args:
            strategy: Trading strategy to backtest
            data: Dictionary of OHLCV data for each asset
            start_date: Start date for backtesting
            end_date: End date for backtesting
        """
        self.strategy = strategy
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.results = {}
        
    def prepare_data(self) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for backtesting.
        
        Returns:
            Prepared data dictionary
        """
        prepared_data = {}
        
        for symbol, df in self.data.items():
            # Filter by date range if specified
            if self.start_date or self.end_date:
                mask = pd.Series(True, index=df.index)
                if self.start_date:
                    mask &= df.index >= self.start_date
                if self.end_date:
                    mask &= df.index <= self.end_date
                df = df[mask]
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Data for {symbol} missing required columns")
            
            prepared_data[symbol] = df.copy()
        
        return prepared_data
    
    def run_backtest(self) -> Dict:
        """
        Run the backtest simulation.
        
        Returns:
            Backtest results dictionary
        """
        data = self.prepare_data()
        
        # Get all unique dates and sort them
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        all_dates = sorted(all_dates)
        
        # Initialize tracking variables
        portfolio_values = []
        positions_history = []
        trades_history = []
        
        # Main backtesting loop
        for date in all_dates:
            # Get current prices
            current_prices = {}
            current_data = {}
            
            for symbol, df in data.items():
                if date in df.index:
                    current_prices[symbol] = df.loc[date, 'Close']
                    # Get historical data up to current date
                    historical_data = df.loc[:date]
                    current_data[symbol] = historical_data
            
            if not current_prices:
                continue
            
            # Generate signals for each asset
            signals = {}
            for symbol, historical_df in current_data.items():
                if len(historical_df) < 50:  # Need minimum data for indicators
                    continue
                    
                try:
                    signal_data = self.strategy.generate_signals(historical_df)
                    if not signal_data.empty:
                        # Get the latest signal
                        latest_signal = signal_data['Signal'].iloc[-1] if 'Signal' in signal_data.columns else 0
                        signals[symbol] = latest_signal
                except Exception as e:
                    print(f"Error generating signals for {symbol} on {date}: {e}")
                    signals[symbol] = 0
            
            # Execute trades based on signals
            for symbol, signal in signals.items():
                if symbol in current_prices:
                    self.strategy.execute_trade(
                        symbol=symbol,
                        signal=signal,
                        price=current_prices[symbol],
                        timestamp=date
                    )
            
            # Update portfolio value
            self.strategy.update_portfolio_value(current_prices)
            
            # Record current state
            portfolio_values.append({
                'date': date,
                'portfolio_value': self.strategy.portfolio_value,
                'cash': self.strategy.cash
            })
            
            positions_history.append({
                'date': date,
                **self.strategy.positions.copy()
            })
        
        # Create results DataFrames
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df.set_index('date', inplace=True)
        
        positions_df = pd.DataFrame(positions_history)
        positions_df.set_index('date', inplace=True)
        
        trades_df = pd.DataFrame(self.strategy.trades) if self.strategy.trades else pd.DataFrame()
        
        # Calculate performance metrics
        performance = PerformanceMetrics(portfolio_df, trades_df)
        metrics = performance.calculate_all_metrics()
        
        # Store results
        self.results = {
            'portfolio_values': portfolio_df,
            'positions': positions_df,
            'trades': trades_df,
            'metrics': metrics,
            'strategy_params': self.strategy.get_strategy_params() if hasattr(self.strategy, 'get_strategy_params') else {}
        }
        
        return self.results
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot backtest results.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.results:
            raise ValueError("No results to plot. Run backtest first.")
        
        portfolio_df = self.results['portfolio_values']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        axes[0, 0].plot(portfolio_df.index, portfolio_df['portfolio_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Drawdown
        cumulative_returns = portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].iloc[0]
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        
        axes[0, 1].fill_between(portfolio_df.index, drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown (%)')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True)
        
        # Returns distribution
        returns = portfolio_df['portfolio_value'].pct_change().dropna()
        axes[1, 0].hist(returns, bins=50, alpha=0.7)
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Daily Returns')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # Monthly returns heatmap (simplified)
        monthly_returns = returns.resample('M').sum() * 100
        if len(monthly_returns) > 1:
            axes[1, 1].plot(monthly_returns.index, monthly_returns.values, marker='o')
            axes[1, 1].set_title('Monthly Returns (%)')
            axes[1, 1].set_ylabel('Monthly Return (%)')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor monthly returns', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_summary(self) -> str:
        """
        Get a summary of backtest results.
        
        Returns:
            Summary string
        """
        if not self.results:
            return "No results available. Run backtest first."
        
        metrics = self.results['metrics']
        
        summary = f"""
=== BACKTEST SUMMARY ===

Strategy: {self.strategy.__class__.__name__}
Period: {self.results['portfolio_values'].index[0]} to {self.results['portfolio_values'].index[-1]}

PERFORMANCE METRICS:
- Total Return: {metrics.get('total_return', 0):.2%}
- Annualized Return: {metrics.get('annualized_return', 0):.2%}
- Annualized Volatility: {metrics.get('annualized_volatility', 0):.2%}
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
- Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}
- Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}

TRADING STATISTICS:
- Total Trades: {metrics.get('total_trades', 0)}
- Win Rate: {metrics.get('win_rate', 0):.2%}
- Average Trade: {metrics.get('avg_trade_return', 0):.2%}
- Best Trade: {metrics.get('best_trade', 0):.2%}
- Worst Trade: {metrics.get('worst_trade', 0):.2%}

RISK METRICS:
- Value at Risk (5%): {metrics.get('var_95', 0):.2%}
- Expected Shortfall: {metrics.get('expected_shortfall', 0):.2%}
"""
        
        return summary 