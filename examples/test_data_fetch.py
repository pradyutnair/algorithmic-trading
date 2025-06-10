#!/usr/bin/env python3
"""
Test Data Fetching

Simple script to test and debug data fetching issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import fetch_stock_data, fetch_multiple_stocks
import yfinance as yf
from datetime import datetime, timedelta

def test_single_stock():
    """Test fetching a single stock."""
    print("🧪 Testing single stock fetch...")
    
    # Test with a simple, reliable symbol
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    
    print(f"Fetching {symbol} from {start_date} to {end_date}")
    
    data = fetch_stock_data(symbol, start_date, end_date)
    
    if data is not None:
        print(f"✅ Success! Got {len(data)} rows of data")
        print(f"Columns: {list(data.columns)}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Sample data:\n{data.head()}")
        return True
    else:
        print("❌ Failed to fetch data")
        return False

def test_yfinance_directly():
    """Test yfinance directly to see what's happening."""
    print("\n🔍 Testing yfinance directly...")
    
    try:
        import yfinance as yf
        print(f"YFinance version: {yf.__version__}")
        
        ticker = yf.Ticker('AAPL')
        info = ticker.info
        print(f"Ticker info available: {len(info) > 0}")
        
        hist = ticker.history(period='1mo')
        print(f"History data shape: {hist.shape}")
        print(f"History columns: {list(hist.columns)}")
        
        if not hist.empty:
            print("✅ YFinance working correctly")
            return True
        else:
            print("❌ YFinance returned empty data")
            return False
            
    except Exception as e:
        print(f"❌ YFinance error: {e}")
        return False

def test_multiple_stocks():
    """Test fetching multiple stocks."""
    print("\n📊 Testing multiple stock fetch...")
    
    # Use a smaller, more reliable set
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2023-06-01'
    end_date = '2023-12-31'
    
    print(f"Fetching {symbols} from {start_date} to {end_date}")
    
    data = fetch_multiple_stocks(symbols, start_date, end_date)
    
    print(f"Successfully fetched data for {len(data)} symbols:")
    for symbol, df in data.items():
        print(f"  {symbol}: {len(df)} rows")
    
    return len(data) > 0

def test_date_ranges():
    """Test different date ranges."""
    print("\n📅 Testing different date ranges...")
    
    # Recent data (should work)
    recent_data = fetch_stock_data('AAPL', '2023-01-01', '2023-12-31')
    print(f"Recent data (2023): {'✅ Success' if recent_data is not None else '❌ Failed'}")
    
    # Very recent data
    very_recent = fetch_stock_data('AAPL', '2024-01-01', '2024-06-01')
    print(f"Very recent data (2024): {'✅ Success' if very_recent is not None else '❌ Failed'}")
    
    # Historical data
    historical = fetch_stock_data('AAPL', '2020-01-01', '2021-01-01')
    print(f"Historical data (2020): {'✅ Success' if historical is not None else '❌ Failed'}")

def main():
    """Main test function."""
    print("🚀 Data Fetching Debug Session")
    print("=" * 50)
    
    # Test 1: Direct yfinance test
    yf_works = test_yfinance_directly()
    
    # Test 2: Single stock
    single_works = test_single_stock()
    
    # Test 3: Date ranges
    test_date_ranges()
    
    # Test 4: Multiple stocks (only if single works)
    if single_works:
        multiple_works = test_multiple_stocks()
    
    print("\n" + "=" * 50)
    print("🏁 Debug Summary:")
    print(f"YFinance Direct: {'✅' if yf_works else '❌'}")
    print(f"Single Stock: {'✅' if single_works else '❌'}")
    
    if single_works:
        print("\n💡 Data fetching is working! The issue might be:")
        print("  1. Date range too aggressive (2022-2024 might hit API limits)")
        print("  2. Too many symbols requested at once")
        print("  3. Network connectivity issues")
        print("\n🔧 Recommended fixes for showcase:")
        print("  - Use shorter date range (1 year max)")
        print("  - Fetch fewer symbols (4-5 max)")
        print("  - Add retry logic")
        print("  - Use more recent dates")
    else:
        print("\n❌ Data fetching has issues. Check:")
        print("  - Internet connection")
        print("  - YFinance library version")
        print("  - API rate limits")

if __name__ == "__main__":
    main() 