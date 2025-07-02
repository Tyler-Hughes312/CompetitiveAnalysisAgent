import os
from competitionIdentification import CompetitionIdentifier
from analysis import StockDataAnalyzer
from yFinanceAPIParsing import YFinanceAPIParsing
from simple_news_stock_csv import SimpleNewsStockCSV
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
import yaml
from redis_cache import get_cache
from datetime import datetime, timedelta

load_dotenv()

with open("agent_config.yaml") as f:
    config = yaml.safe_load(f)

DATA_FOLDER = 'yfinance_data'

def delete_existing_csvs():
    if os.path.exists(DATA_FOLDER):
        for filename in os.listdir(DATA_FOLDER):
            if filename.endswith('_historical_data.csv'):
                os.remove(os.path.join(DATA_FOLDER, filename))

def create_csvs_from_cache(tickers, output_folder='yfinance_data'):
    """Create CSV files from cached data if available"""
    cache = get_cache()
    os.makedirs(output_folder, exist_ok=True)
    
    cached_tickers = []
    api_tickers = []
    
    print("Checking cache for existing data...")
    
    # Calculate the correct date range (last 12 months from today, historical only)
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)  # Today at midnight
    start_date = end_date - timedelta(days=365)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Looking for cached data from {start_str} to {end_str}")
    
    for ticker in tickers:
        # Check if data exists in cache with the correct date range
        cached_data = cache.get("yfinance_data", ticker, start_str, end_str)
        if cached_data is not None:
            print(f"Found cached data for {ticker}, creating CSV...")
            print(f"Cached data range: {cached_data['Date'].min()} to {cached_data['Date'].max()}")
            # Keep only Date, Open, and Close columns
            cached_data = cached_data[['Date', 'Open', 'Close']]
            csv_path = os.path.join(output_folder, f"{ticker}_historical_data.csv")
            cached_data.to_csv(csv_path, index=False)
            print(f"Saved {ticker} data from cache to {csv_path}")
            cached_tickers.append(ticker)
        else:
            print(f"No cached data found for {ticker}, will fetch from API")
            api_tickers.append(ticker)
    
    return cached_tickers, api_tickers

def plot_news_stock_correlation(csv_path):
    """Plot news impact correlation with stock price"""
    try:
        # Load the CSV data
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter out rows without news
        news_df = df.dropna(subset=['News_Headline'])
        
        if news_df.empty:
            print("No news articles found in the data.")
            return
        
        print(f"Found {len(news_df)} days with news articles")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'News-Stock Correlation Analysis', fontsize=16)
        
        # 1. Stock price over time with news events
        ax1.plot(df['Date'], df['Close'], label='Stock Price', alpha=0.7)
        ax1.scatter(news_df['Date'], news_df['Close'], 
                   c=news_df['News_Ranking'], cmap='viridis', 
                   s=50, alpha=0.8, label='News Events')
        ax1.set_title('Stock Price with News Events')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price ($)')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Add colorbar for news ranking
        scatter = ax1.scatter([], [], c=[], cmap='viridis')
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('News Source Ranking (1-100)')
        
        # 2. News ranking distribution
        ax2.hist(news_df['News_Ranking'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Distribution of News Source Rankings')
        ax2.set_xlabel('News Source Ranking')
        ax2.set_ylabel('Frequency')
        ax2.axvline(news_df['News_Ranking'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {news_df["News_Ranking"].mean():.1f}')
        ax2.legend()
        
        # 3. Price change vs News ranking
        news_df['Price_Change'] = news_df['Close'].pct_change() * 100
        ax3.scatter(news_df['News_Ranking'], news_df['Price_Change'], alpha=0.7)
        ax3.set_title('Price Change vs News Source Ranking')
        ax3.set_xlabel('News Source Ranking')
        ax3.set_ylabel('Price Change (%)')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax3.axvline(x=70, color='green', linestyle='--', alpha=0.5, label='High Quality (70+)')
        ax3.legend()
        
        # 4. Correlation analysis
        correlation = news_df['News_Ranking'].corr(news_df['Price_Change'].dropna())
        ax4.text(0.1, 0.8, f'Correlation Analysis', fontsize=14, fontweight='bold')
        ax4.text(0.1, 0.7, f'Correlation: {correlation:.3f}', fontsize=12)
        ax4.text(0.1, 0.6, f'Total news days: {len(news_df)}', fontsize=12)
        ax4.text(0.1, 0.5, f'Avg news ranking: {news_df["News_Ranking"].mean():.1f}', fontsize=12)
        ax4.text(0.1, 0.4, f'Avg price change: {news_df["Price_Change"].mean():.2f}%', fontsize=12)
        ax4.text(0.1, 0.3, f'High quality news: {len(news_df[news_df["News_Ranking"] >= 70])}', fontsize=12)
        ax4.text(0.1, 0.2, f'Low quality news: {len(news_df[news_df["News_Ranking"] < 50])}', fontsize=12)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\nCorrelation Analysis Summary:")
        print(f"Correlation coefficient: {correlation:.3f}")
        print(f"Total days with news: {len(news_df)}")
        print(f"Average news ranking: {news_df['News_Ranking'].mean():.1f}")
        print(f"Average price change on news days: {news_df['Price_Change'].mean():.2f}%")
        
        # High vs Low quality news analysis
        high_quality = news_df[news_df['News_Ranking'] >= 70]
        low_quality = news_df[news_df['News_Ranking'] < 50]
        
        if not high_quality.empty:
            print(f"High quality news days: {len(high_quality)}")
            print(f"  Avg price change: {high_quality['Price_Change'].mean():.2f}%")
        
        if not low_quality.empty:
            print(f"Low quality news days: {len(low_quality)}")
            print(f"  Avg price change: {low_quality['Price_Change'].mean():.2f}%")
        
    except Exception as e:
        print(f"Error plotting correlation: {e}")

def main():
    while True:
        print("\nOptions:")
        print("1. Enter a new company and fetch competitors' data")
        print("2. Plot competition closing prices")
        print("3. Predict future (simple trend) for a ticker")
        print("4. Joint Linear Regression Forecast for all companies (and save to CSV)")
        print("5. Predictive Model (LSTM) for all companies (and save to CSV)")
        print("6. Manage Redis Cache")
        print("7. News-Stock Analysis and Correlation")
        print("8. Exit")
        choice = input("Select an option (1-8): ").strip()

        if choice == '1':
            confirm = input("This will delete all existing CSVs. Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                company_name = input("Enter the company name to search for competitors: ")
                our_company_ticker = input("Enter your company's ticker symbol: ")
                
                # Get competitors (this will use cache if available)
                identifier = CompetitionIdentifier()
                tickers = identifier.get_top_competitors(company_name)
                
                if not isinstance(tickers, list) or not tickers or any('error' in str(tickers).lower() for tickers in tickers):
                    print(f"Error: Failed to get competitors for {company_name}. Agent returned: {tickers}")
                    continue
                
                tickers.append(our_company_ticker.upper())
                print(f"Getting data for tickers: {tickers}")
                
                # Delete existing CSVs first
                delete_existing_csvs()
                
                # Check cache and create CSVs from cache where possible
                cached_tickers, api_tickers = create_csvs_from_cache(tickers)
                
                # For tickers not in cache, fetch from API
                if api_tickers:
                    print(f"Fetching fresh data for: {api_tickers}")
                    yfinance_parser = YFinanceAPIParsing(api_tickers)
                    yfinance_parser.select_ticker(output_folder=DATA_FOLDER)
                else:
                    print("All data was found in cache!")
                
                print("Data processing completed.")
        elif choice == '2':
            analyzer = StockDataAnalyzer()
            tickers = analyzer.get_tickers()
            print(f"Available tickers: {tickers}")
            ticker = input("Enter ticker to plot (or 'all' for all): ").strip().upper()
            if ticker == 'ALL':
                for t in tickers:
                    analyzer.plot_ticker(t)
            elif ticker in tickers:
                analyzer.plot_ticker(ticker)
            else:
                print("Ticker not found.")
        elif choice == '3':
            analyzer = StockDataAnalyzer()
            tickers = analyzer.get_tickers()
            print(f"Available tickers: {tickers}")
            ticker = input("Enter ticker to predict (or 'all' for all): ").strip().upper()
            days_ahead = 21  # ~1 month of business days
            # 1. Plot company open/close for 1 month into the future
            if ticker == 'ALL':
                for t in tickers:
                    print(f"\n--- {t} ---")
                    analyzer.predict_linear_regression(t, days_ahead=days_ahead, plot=True)
            elif ticker in tickers:
                analyzer.predict_linear_regression(ticker, days_ahead=days_ahead, plot=True)
            else:
                print("Ticker not found.")
                return
            # 2. Plot market average prediction (joint model)
            print("\nPlotting joint market average prediction for all companies...")
            analyzer.joint_linear_regression_forecast(days_ahead=days_ahead, plot=True)
        elif choice == '4':
            analyzer = StockDataAnalyzer()
            days_ahead = 21  # ~1 month
            print("Running joint linear regression forecast for all companies...")
            forecast_df = analyzer.joint_linear_regression_forecast(days_ahead=days_ahead, plot=True)
            csv_path = os.path.join(DATA_FOLDER, 'joint_linear_regression_forecast.csv')
            forecast_df.to_csv(csv_path)
            print(f"Joint linear regression forecast saved to {csv_path}")
        elif choice == '5':
            analyzer = StockDataAnalyzer()
            days_ahead = 21  # ~1 month
            print("Running LSTM predictive model for all companies...")
            tickers = analyzer.get_tickers()
            # For each ticker, get LSTM predictions
            results = {}
            for ticker in tickers:
                preds = analyzer.predict_lstm(ticker, days_ahead=days_ahead, plot=False)
                if preds is not None:
                    results[ticker] = preds
            # Build DataFrame
            future_dates = pd.date_range(analyzer.dataframes[tickers[0]]['Date'].iloc[-1], periods=days_ahead+1, freq='B')[1:]
            lstm_df = pd.DataFrame(results, index=future_dates)
            lstm_df['Market_Avg'] = lstm_df.mean(axis=1)
            # Save to CSV
            csv_path = os.path.join(DATA_FOLDER, 'lstm_predictive_forecast.csv')
            lstm_df.to_csv(csv_path)
            print(f"LSTM predictive forecast saved to {csv_path}")
            # Plot last 3 months + prediction for all companies
            plt.figure(figsize=(14, 7))
            for ticker in tickers:
                hist = analyzer.dataframes[ticker]
                hist = hist.dropna(subset=['Date', 'Close'])
                hist['Date'] = pd.to_datetime(hist['Date'])
                last_3mo = hist[hist['Date'] >= (hist['Date'].max() - pd.Timedelta(days=90))]
                plt.plot(last_3mo['Date'], last_3mo['Close'], label=f"{ticker} (history)")
                # Append prediction
                pred_dates = list(future_dates)
                pred_vals = lstm_df[ticker].values
                plt.plot(pred_dates, pred_vals, linestyle='--', label=f"{ticker} (pred)")
            plt.plot(lstm_df.index, lstm_df['Market_Avg'], label='Market_Avg (pred)', color='black', linewidth=2, linestyle=':')
            plt.title('LSTM Predictive Model: Last 3 Months + Next Month for All Companies and Market Average')
            plt.xlabel('Date')
            plt.ylabel('Predicted Close Price')
            plt.legend(loc='upper left', ncol=2)
            plt.tight_layout()
            plt.show()
        elif choice == '6':
            cache = get_cache()
            print("\nRedis Cache Management:")
            print("1. View cache status")
            print("2. Clear all cache")
            print("3. Clear yFinance data cache only")
            print("4. Clear competitor data cache only")
            print("5. Back to main menu")
            cache_choice = input("Select cache option (1-5): ").strip()
            
            if cache_choice == '1':
                print(f"Redis connection: {'Connected' if cache.redis_client else 'Not connected'}")
                if cache.redis_client:
                    try:
                        info = cache.redis_client.info()
                        if info:
                            print(f"Redis server: {info.get('redis_version', 'Unknown')}")
                            print(f"Connected clients: {info.get('connected_clients', 'Unknown')}")
                            print(f"Used memory: {info.get('used_memory_human', 'Unknown')}")
                        else:
                            print("Could not retrieve Redis info")
                    except Exception as e:
                        print(f"Error getting Redis info: {e}")
            elif cache_choice == '2':
                confirm = input("Clear ALL cache data? (y/n): ").strip().lower()
                if confirm == 'y':
                    cache.clear_all()
                    print("All cache cleared.")
            elif cache_choice == '3':
                confirm = input("Clear yFinance data cache? (y/n): ").strip().lower()
                if confirm == 'y':
                    cache.clear_all("yfinance_data")
                    print("yFinance data cache cleared.")
            elif cache_choice == '4':
                confirm = input("Clear competitor data cache? (y/n): ").strip().lower()
                if confirm == 'y':
                    cache.clear_all("competitor_tickers")
                    print("Competitor data cache cleared.")
            elif cache_choice == '5':
                continue
            else:
                print("Invalid cache option.")
        elif choice == '7':
            print("\n=== News-Stock Analysis and Correlation ===")
            
            # Get company info
            company_name = input("Enter company name: ").strip()
            ticker = input("Enter ticker symbol: ").strip().upper()
            
            print(f"\nCreating news-stock analysis for {company_name} ({ticker})...")
            
            # Create the news-stock CSV
            news_analyzer = SimpleNewsStockCSV()
            csv_path = news_analyzer.create_simple_csv(company_name, ticker)
            
            if csv_path and os.path.exists(csv_path):
                print(f"✅ News-stock CSV created: {csv_path}")
                
                # Automatically analyze the correlation
                print("\nGenerating correlation analysis...")
                plot_news_stock_correlation(csv_path)
                
            else:
                print("❌ Failed to create news-stock CSV")
                print("Please check your API configuration and try again.")
                
        elif choice == '8':
            print("Exiting.")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()
