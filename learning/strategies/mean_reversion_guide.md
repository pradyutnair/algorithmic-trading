# 📈 Mean Reversion Strategy - Complete Guide

## 🎯 Strategy Overview

Mean reversion is based on the principle that asset prices tend to return to their long-term average over time. When prices deviate significantly from the mean, there's a statistical tendency for them to "revert" back.

### Core Principle
> "What goes up must come down, what goes down must come up"

## 📊 Theoretical Foundation

### Statistical Basis
- **Central Limit Theorem**: Price movements around the mean follow normal distribution
- **Ornstein-Uhlenbeck Process**: Mathematical model for mean-reverting processes
- **Half-Life**: Time it takes for price to revert halfway back to mean

### Why It Works
1. **Market Overreactions**: News causes temporary price overshoots
2. **Profit Taking**: Traders lock in gains, causing reversals
3. **Value Investors**: Buy "cheap" stocks, sell "expensive" ones
4. **Technical Support/Resistance**: Price levels act as magnets

## 🛠️ Implementation Details

### Key Indicators Used

#### 1. Bollinger Bands
```python
# Calculate Bollinger Bands
middle_band = price.rolling(20).mean()  # 20-day moving average
std_dev = price.rolling(20).std()       # Standard deviation
upper_band = middle_band + (2 * std_dev)
lower_band = middle_band - (2 * std_dev)
```

**Signals:**
- **Buy**: Price touches lower band (oversold)
- **Sell**: Price touches upper band (overbought)
- **Exit**: Price returns to middle band

#### 2. RSI (Relative Strength Index)
```python
# RSI calculation
delta = price.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
```

**Signals:**
- **Buy**: RSI < 30 (oversold)
- **Sell**: RSI > 70 (overbought)
- **Exit**: RSI returns to 50 (neutral)

### Entry Rules (ALL must be true)
```python
# Long Entry
if (price <= bollinger_lower_band and 
    rsi <= 30 and
    volume > average_volume):
    signal = BUY

# Short Entry  
if (price >= bollinger_upper_band and
    rsi >= 70 and
    volume > average_volume):
    signal = SELL
```

### Exit Rules
```python
# Exit Long Position
if (price >= bollinger_middle_band or 
    rsi >= 50 or
    stop_loss_hit):
    signal = CLOSE_LONG

# Exit Short Position
if (price <= bollinger_middle_band or
    rsi <= 50 or
    stop_loss_hit):
    signal = CLOSE_SHORT
```

## ⚙️ Parameter Optimization

### Bollinger Bands Parameters
- **Window**: 10-50 days (default: 20)
  - Shorter = More signals, more noise
  - Longer = Fewer signals, more reliable
- **Standard Deviations**: 1.5-3.0 (default: 2.0)
  - Lower = More signals, earlier entries
  - Higher = Fewer signals, extreme moves only

### RSI Parameters
- **Window**: 7-21 days (default: 14)
  - Shorter = More sensitive, more signals
  - Longer = Less sensitive, fewer false signals
- **Thresholds**: 
  - Conservative: 20/80
  - Standard: 30/70
  - Aggressive: 40/60

## 📈 Market Conditions

### When Mean Reversion Works Best

#### ✅ Favorable Conditions
1. **Range-bound markets**: Clear support/resistance levels
2. **Low volatility periods**: VIX < 20
3. **Mature, stable companies**: Large-cap stocks
4. **Normal market conditions**: No major news/events
5. **High-frequency oscillations**: Intraday reversals

#### ❌ Unfavorable Conditions
1. **Strong trending markets**: Bull/bear runs
2. **High volatility**: VIX > 30
3. **Breaking news**: Earnings, FDA approvals, etc.
4. **Market crashes**: Panic selling continues
5. **Momentum stocks**: Growth companies in strong trends

### Best Asset Classes
- **Individual Stocks**: Large-cap, dividend-paying
- **ETFs**: Sector ETFs, broad market ETFs
- **Forex**: Major currency pairs (EUR/USD, GBP/USD)
- **Commodities**: Gold, oil (with careful analysis)

## 📊 Performance Expectations

### Realistic Targets
- **Annual Return**: 8-15%
- **Sharpe Ratio**: 0.8-1.5
- **Maximum Drawdown**: 10-20%
- **Win Rate**: 55-65%
- **Average Holding Period**: 3-10 days

### Risk Metrics
- **Value at Risk (5%)**: -2% to -4%
- **Maximum Single Trade Loss**: -5%
- **Correlation to Market**: 0.3-0.7

## 🚨 Risk Management

### Position Sizing
```python
# Kelly Criterion for optimal position size
win_rate = 0.60
avg_win = 0.03
avg_loss = 0.02
kelly = (win_rate * (1 + avg_win/avg_loss) - 1) / (avg_win/avg_loss)
position_size = min(kelly, 0.25) * portfolio_value  # Cap at 25%
```

### Stop Losses
- **Technical**: Below recent swing low/high
- **Percentage**: 5-8% from entry
- **Time-based**: Close after 10 days regardless
- **Volatility-based**: 2x ATR (Average True Range)

### Portfolio Rules
- **Maximum positions**: 5-8 simultaneous trades
- **Sector limits**: Max 30% in any sector
- **Correlation limits**: Avoid highly correlated positions

## 🔍 Example Implementation

### Complete Strategy Code
```python
class MeanReversionStrategy:
    def __init__(self):
        self.bb_window = 20
        self.bb_std = 2.0
        self.rsi_window = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
    def generate_signals(self, data):
        # Calculate indicators
        bb = bollinger_bands(data['Close'], self.bb_window, self.bb_std)
        rsi_values = rsi(data['Close'], self.rsi_window)
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Long signals
        long_condition = (
            (data['Close'] <= bb['Lower']) &
            (rsi_values <= self.rsi_oversold)
        )
        signals[long_condition] = 1
        
        # Short signals  
        short_condition = (
            (data['Close'] >= bb['Upper']) &
            (rsi_values >= self.rsi_overbought)
        )
        signals[short_condition] = -1
        
        return signals
```

### Backtesting Example
```python
# Load data
data = fetch_stock_data(['AAPL'], '2020-01-01', '2023-12-31')

# Initialize strategy
strategy = MeanReversionStrategy()

# Run backtest
engine = BacktestEngine(strategy, data)
results = engine.run_backtest()

# Analyze results
print(f"Total Return: {results['metrics']['total_return']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
```

## 🎯 Hackathon Tips

### Quick Implementation Checklist
- [ ] Choose 3-5 liquid stocks (AAPL, MSFT, JNJ, KO, PG)
- [ ] Use 20-day Bollinger Bands with 2 std dev
- [ ] Set RSI to 14-day with 30/70 thresholds
- [ ] Include 0.1% transaction costs
- [ ] Backtest on 2+ years of data
- [ ] Validate on recent out-of-sample period

### Common Mistakes to Avoid
1. **Over-optimization**: Don't curve-fit to historical data
2. **Ignoring transaction costs**: Include realistic fees
3. **No risk management**: Always use stop losses
4. **Single stock testing**: Test on multiple assets
5. **Look-ahead bias**: Don't use future information

### Presentation Points
- Explain why mean reversion works (behavioral finance)
- Show performance across different market conditions
- Highlight risk management features
- Compare to buy-and-hold benchmark
- Discuss strategy limitations honestly

## 📚 Further Reading

### Academic Papers
- "Do Stock Prices Mean Revert? Evidence from Relative Prices" (Poterba & Summers)
- "Mean Reversion in Stock Prices: Evidence and Implications" (Fama & French)

### Books
- "Quantitative Portfolio Theory" by Markowitz
- "Evidence-Based Technical Analysis" by Aronson

### Practical Resources
- QuantConnect tutorials on mean reversion
- Zipline documentation for backtesting
- Alpha Architect blog posts on factor investing

---

**Remember**: Mean reversion is a statistical tendency, not a guarantee. Always use proper risk management and understand that markets can trend longer than you can stay solvent! 