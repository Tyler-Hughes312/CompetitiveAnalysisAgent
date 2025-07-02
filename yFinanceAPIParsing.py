import yfinance as yf
import pandas as pd
import os
from redis_cache import get_cache

class YFinanceAPIParsing:
    def __init__(self, company_list):
        self.company_list = company_list if company_list is not None else []
        self.cache = get_cache()

    def get_data(self, ticker):
        # Calculate date range for last 12 months (only historical data, not future)
        from datetime import datetime, timedelta
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)  # Today at midnight
        start_date = end_date - timedelta(days=365)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print(f"Fetching stock data for {ticker} from {start_str} to {end_str}")
        
        # Check cache first
        cached_data = self.cache.get("yfinance_data", ticker, start_str, end_str)
        if cached_data is not None:
            print(f"Using cached data for {ticker}")
            print(f"Cached data range: {cached_data['Date'].min()} to {cached_data['Date'].max()}")
            return cached_data
        
        # If not in cache, fetch from API
        print(f"Fetching fresh data for {ticker} from yFinance API (last 12 months)")
        data = yf.download(ticker, start=start_str, end=end_str)
        if data is None or data.empty:
            print(f"No data for {ticker}")
            return None
        
        data.reset_index(inplace=True)
        
        print(f"Fetched data range: {data['Date'].min()} to {data['Date'].max()}")
        
        # Cache the data for 1 hour (3600 seconds)
        self.cache.set("yfinance_data", data, ticker, start_str, end_str, ttl_seconds=3600)
        
        return data

    def select_ticker(self, output_folder='yfinance_data'):
        os.makedirs(output_folder, exist_ok=True)
        for company in self.company_list:
            df = self.get_data(company)
            if df is not None:
                # Keep only Date, Open, and Close columns
                df = df[['Date', 'Open', 'Close']]
                csv_path = os.path.join(output_folder, f"{company}_historical_data.csv")
                df.to_csv(csv_path, index=False)
                print(f"Saved {company} data to {csv_path}")

