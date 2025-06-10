"""
Configuration file for algorithmic trading system.
"""

import os
from typing import Dict, List

# Data Configuration
DATA_CONFIG = {
    'default_start_date': '2020-01-01',
    'default_end_date': '2023-12-31',
    'default_interval': '1d',
    'data_source': 'yahoo',  # yahoo, alpha_vantage, etc.
}

# Trading Configuration
TRADING_CONFIG = {
    'initial_capital': 100000,
    'commission_rate': 0.001,  # 0.1%
    'slippage_rate': 0.001,    # 0.1%
    'max_position_size': 0.1,  # 10% of portfolio
    'max_portfolio_risk': 0.02, # 2% risk per trade
}

# Risk Management
RISK_CONFIG = {
    'max_drawdown_limit': 0.20,  # 20%
    'stop_loss_percentage': 0.05, # 5%
    'position_size_method': 'fixed_percentage',  # fixed_percentage, kelly, risk_parity
    'max_positions': 10,
    'correlation_limit': 0.7,
}

# Strategy Parameters
STRATEGY_PARAMS = {
    'mean_reversion': {
        'bb_window': 20,
        'bb_std': 2.0,
        'rsi_window': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
    },
    'momentum': {
        'fast_ma': 12,
        'slow_ma': 26,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
    },
    'pairs_trading': {
        'lookback_window': 60,
        'entry_threshold': 2.0,
        'exit_threshold': 0.5,
        'correlation_threshold': 0.7,
    }
}

# Asset Universe
ASSET_UNIVERSE = {
    'large_cap_stocks': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'META', 'NVDA', 'JPM', 'JNJ', 'V'
    ],
    'stable_stocks': [
        'KO', 'PG', 'JNJ', 'WMT', 'MCD',
        'VZ', 'T', 'XOM', 'CVX', 'PFE'
    ],
    'tech_stocks': [
        'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA',
        'CRM', 'ORCL', 'ADBE', 'NFLX', 'AMD'
    ],
    'etfs': [
        'SPY', 'QQQ', 'IWM', 'VTI', 'VEA',
        'VWO', 'AGG', 'TLT', 'GLD', 'VNQ'
    ]
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'train_ratio': 0.7,
    'validation_ratio': 0.15,
    'test_ratio': 0.15,
    'walk_forward_analysis': False,
    'out_of_sample_months': 6,
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'min_sharpe_ratio': 0.5,
    'max_drawdown': 0.25,
    'min_win_rate': 0.45,
    'min_profit_factor': 1.2,
    'min_trades': 10,
}

# Hackathon Specific
HACKATHON_CONFIG = {
    'recommended_strategies': ['mean_reversion', 'momentum'],
    'max_complexity_score': 3,  # 1-5 scale
    'time_limit_hours': 24,
    'presentation_time_minutes': 10,
    'focus_on_metrics': ['sharpe_ratio', 'max_drawdown', 'total_return'],
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_to_file': True,
    'log_file': 'trading_system.log',
}

# File Paths
PATHS = {
    'data_dir': 'data/',
    'results_dir': 'results/',
    'logs_dir': 'logs/',
    'models_dir': 'models/',
    'plots_dir': 'plots/',
}

def get_config(section: str = None) -> Dict:
    """
    Get configuration for a specific section or all configurations.
    
    Args:
        section: Configuration section name
        
    Returns:
        Configuration dictionary
    """
    all_configs = {
        'data': DATA_CONFIG,
        'trading': TRADING_CONFIG,
        'risk': RISK_CONFIG,
        'strategies': STRATEGY_PARAMS,
        'assets': ASSET_UNIVERSE,
        'backtest': BACKTEST_CONFIG,
        'performance': PERFORMANCE_THRESHOLDS,
        'hackathon': HACKATHON_CONFIG,
        'logging': LOGGING_CONFIG,
        'paths': PATHS,
    }
    
    if section:
        return all_configs.get(section, {})
    return all_configs

def get_strategy_params(strategy_name: str) -> Dict:
    """Get parameters for a specific strategy."""
    return STRATEGY_PARAMS.get(strategy_name, {})

def get_asset_universe(category: str = None) -> List[str]:
    """Get asset universe for a specific category."""
    if category:
        return ASSET_UNIVERSE.get(category, [])
    
    # Return all assets
    all_assets = []
    for assets in ASSET_UNIVERSE.values():
        all_assets.extend(assets)
    return list(set(all_assets))  # Remove duplicates 