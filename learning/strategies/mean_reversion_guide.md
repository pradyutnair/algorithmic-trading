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
*Bollinger Bands* are used to identify periods when prices are unusually high or low relative to recent history. They help spot overbought (upper band) and oversold (lower band) conditions, making them ideal for mean reversion signals.

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

#### ❓ What is VIX?

The **VIX** (Volatility Index), often called the "fear gauge," measures the market's expectation of volatility over the next 30 days, derived from S&P 500 index options. A low VIX indicates calm, stable markets, while a high VIX signals uncertainty and expected large price swings.

##### How is VIX Calculated?
VIX is calculated using the prices of a wide range of S&P 500 index options. The formula involves estimating the expected volatility by averaging the weighted prices of out-of-the-money put and call options, resulting in an annualized percentage volatility figure.

While the exact calculation is complex and handled by the CBOE, you can approximate VIX using implied volatility from S&P 500 options. For research or backtesting, you can fetch historical VIX data using:

```python
import yfinance as yf

# Download historical VIX data
vix = yf.download('^VIX', start='2020-01-01', end='2023-12-31')
print(vix.head())
```
*This code fetches the VIX index from Yahoo Finance for analysis or filtering market regimes in your strategy.*

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
```python
risk_per_trade = 0.01  # 1% of portfolio
stop_loss_pct = 0.05   # 5% stop loss
position_size = (portfolio_value * risk_per_trade) / stop_loss_pct
```

#### 📊 How to Calculate Win Rate, Average Win, and Average Loss

- **Win Rate:**  
  The percentage of trades that were profitable.  
  `win_rate = (number of winning trades) / (total number of trades)`

- **Average Win:**  
  The average profit from your winning trades.  
  `avg_win = (sum of profits from winning trades) / (number of winning trades)`

- **Average Loss:**  
  The average loss from your losing trades.  
  `avg_loss = (sum of losses from losing trades) / (number of losing trades)`

You obtain these values by running a **backtest** of your strategy on historical data. Most backtesting libraries (like Backtrader, Zipline, or QuantConnect) will provide these statistics automatically.

---

### 🧪 How to Choose Backtest Values

Before deploying your strategy, always run a backtest. To choose appropriate backtest values:

- **Parameter Ranges:**  
  Start with standard values from literature (e.g., 20-day Bollinger Bands, 14-day RSI) and test a range (e.g., Bollinger window 10–50, RSI window 7–21).
- **Asset Selection:**  
  Use highly liquid assets to ensure realistic execution (e.g., AAPL, MSFT, S&P 500 ETFs).
- **Time Period:**  
  Backtest over multiple years and include different market regimes (bull, bear, sideways).
- **Transaction Costs:**  
  Include realistic fees and slippage to avoid overestimating performance.
- **Out-of-Sample Testing:**  
  After optimizing on one period, validate your strategy on a separate, unseen period to check robustness.

> **Tip:** Avoid overfitting by not optimizing too many parameters and always test on out-of-sample data.

---

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

> **Note:** "Long Entry" refers to buying the asset in anticipation of a price increase. It does **not** imply a long-term investment—mean reversion trades are typically short-term, and the holding period can range from minutes (in high-frequency trading) to several days.

---

### ⚡️ High-Frequency Trading (HFT) Considerations

For high-frequency trading, mean reversion strategies require special attention to:

- **Execution Speed:** Milliseconds matter. Use low-latency infrastructure and colocated servers if possible.
- **Transaction Costs:** HFT strategies generate many trades, so minimizing fees and slippage is critical.
- **Liquidity:** Focus on highly liquid assets (e.g., large-cap stocks, major forex pairs) to ensure fast order execution.
- **Robust Signal Filtering:** Use stricter entry/exit criteria or combine multiple indicators to reduce false signals.
- **Risk Controls:** Implement tight stop-losses and position limits to avoid large losses from rare but extreme price moves.
- **Backtesting with Realistic Assumptions:** Simulate order book dynamics, latency, and partial fills to ensure your strategy is viable in real markets.

> **Most Important for HFT:**  
> - Minimize latency and transaction costs  
> - Trade only the most liquid instruments  
> - Use robust, fast-executing risk management  
> - Continuously monitor and adapt to changing market microstructure

---

## 🛠️ Key Concepts & Definitions

### Momentum
**Momentum** measures the speed and magnitude of recent price changes. It helps identify the strength of a trend.

**Formula (N-day momentum):**
```python
momentum = price / price.shift(N) - 1
```
- Example: If N=20, this gives the percentage change over the last 20 periods.
- **Interpretation:** Positive momentum means price is rising; negative means falling.

---

### Support Level
A **support level** is a price point where a falling asset tends to stop declining and may bounce back up, due to increased buying interest.

**How to calculate:**
- Often set as the recent lowest price over a lookback window (e.g., 20 days).
- **Formula (pandas):**
  ```python
  support_level = price.rolling(window=20).min()
  ```

---

### Resistance Level
A **resistance level** is a price point where a rising asset tends to stop climbing and may reverse down, due to increased selling interest.

**How to calculate:**
- Often set as the recent highest price over a lookback window (e.g., 20 days).
- **Formula (pandas):**
  ```python
  resistance_level = price.rolling(window=20).max()
  ```

---

### Volatility
**Volatility** measures how much the price of an asset fluctuates over time. It is a key factor in determining market conditions and strategy suitability.

**How to measure volatility (pandas example):**
```python
volatility = price.pct_change().rolling(window=20).std()
```
- This calculates the rolling standard deviation of daily returns over 20 periods.

#### Low Volatility
- **Definition:** Price changes are small and steady; the market is calm.
- **Implications:** 
  - Favors mean reversion and range-bound strategies.
  - Easier to predict support/resistance levels.
  - Lower risk, but also lower profit potential.
- **Example:** VIX < 20, stock prices move within a tight range.

#### High Volatility
- **Definition:** Price changes are large and frequent; the market is jumpy or uncertain.
- **Implications:** 
  - Favors breakout and momentum strategies.
  - Higher risk, but also higher potential rewards.
  - Mean reversion can be riskier, as prices may overshoot more often.
- **Example:** VIX > 30, stock prices swing widely in short periods.

---

### Volume
**Volume** is the total number of shares or contracts traded for an asset during a specific period. It measures trading activity and liquidity.

- **High volume:** Indicates strong interest and liquidity, making it easier to enter/exit trades.
- **Low volume:** Indicates weak interest and low liquidity, which can lead to slippage and unreliable price signals.

**How to access volume (pandas example):**
```python
import yfinance as yf
data = yf.download('AAPL', start='2024-01-01', end='2024-06-01')
print(data['Volume'].head())
```

---

### MAD (Mean Absolute Deviation)
**MAD** is the average of the absolute differences between each price and the mean price.

- **Use:** Measures volatility or dispersion; helps understand how much prices deviate from the average.

**Formula (pandas):**
```python
mad = price.mad()
```

---

### MACD (Moving Average Convergence Divergence)
**MACD** is a momentum indicator that shows the relationship between two moving averages (usually 12- and 26-period EMAs).

**Formula:**
```python
macd = price.ewm(span=12).mean() - price.ewm(span=26).mean()
signal = macd.ewm(span=9).mean()
```
- **Use:** Identifies trend direction, momentum, and possible buy/sell signals when the MACD crosses above/below the signal line.

---

### SMA (Simple Moving Average)
**SMA** is the average price over a set period. Used to smooth out price data and identify trends.

**Formula (pandas):**
```python
sma = price.rolling(window=20).mean()
```

---

### EMA (Exponential Moving Average)
**EMA** is like SMA, but gives more weight to recent prices. Reacts faster to price changes.

**Formula (pandas):**
```python
ema = price.ewm(span=20, adjust=False).mean()
```

---

### RSI (Relative Strength Index)
**RSI** measures the speed and change of price movements. Values above 70 = overbought, below 30 = oversold.

**Formula (pandas):**
```python
delta = price.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
```

---

### ATR (Average True Range)
**ATR** measures market volatility by averaging the true range over a set period.

**Formula (pandas):**
```python
high_low = high - low
high_close = (high - close.shift()).abs()
low_close = (low - close.shift()).abs()
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
atr = tr.rolling(window=14).mean()
```

---

### Bollinger Bands
**Bollinger Bands** plot bands two standard deviations above and below a moving average. Used to identify overbought/oversold conditions.

**Formula (pandas):**
```python
middle_band = price.rolling(20).mean()
std_dev = price.rolling(20).std()
upper_band = middle_band + (2 * std_dev)
lower_band = middle_band - (2 * std_dev)
```

---

### VWAP (Volume Weighted Average Price)
**VWAP** is the average price weighted by volume. Used by institutional traders to assess trade quality.

---

### ADX (Average Directional Index)
**ADX** measures the strength of a trend (not direction).

---

### Stochastic Oscillator
**Stochastic Oscillator** compares a particular closing price to a range of its prices over a certain period. Used to identify overbought/oversold conditions.

---

*These concepts and indicators are essential tools for analyzing trends, momentum, volatility, and potential entry/exit points in trading strategies.*