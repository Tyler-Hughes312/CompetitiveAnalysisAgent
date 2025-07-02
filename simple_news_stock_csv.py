#!/usr/bin/env python3
"""
Simple News-Stock CSV Generator
Creates a CSV with stock data and news articles with rankings on matching dates
"""

import pandas as pd
import os
from typing import List, Dict, Any
from google_news_scraper import GoogleNewsScraper
from yFinanceAPIParsing import YFinanceAPIParsing

# Try to import fallback, but don't fail if it's not available
try:
    from simple_news_fallback import SimpleNewsFallback
    FALLBACK_AVAILABLE = True
except ImportError:
    FALLBACK_AVAILABLE = False
    print("Warning: simple_news_fallback module not found. Fallback news data will not be available.")

class SimpleNewsStockCSV:
    def __init__(self):
        """Initialize the simple CSV generator"""
        self.news_scraper = GoogleNewsScraper()
        self.news_fallback = SimpleNewsFallback() if FALLBACK_AVAILABLE else None
        self.data_folder = 'yfinance_data'
    
    def get_stock_data(self, ticker: str) -> pd.DataFrame:
        """Get stock data for a ticker"""
        csv_path = os.path.join(self.data_folder, f"{ticker}_historical_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        else:
            # Fetch data if not available
            yf_parser = YFinanceAPIParsing([ticker])
            data = yf_parser.get_data(ticker)
            if data is not None and not data.empty:
                data = data[['Date', 'Open', 'Close']]
                data.to_csv(csv_path, index=False)
                data['Date'] = pd.to_datetime(data['Date'])
                return data
        return pd.DataFrame()
    
    def get_news_with_rankings(self, company_name: str, ticker: str, days_back: int = 365) -> List[Dict[str, Any]]:
        """Get news articles with their importance scores"""
        # Try to get news articles from the main scraper
        try:
            articles = self.news_scraper.scrape_news_articles(company_name, ticker, days_back=days_back, max_articles=100)
            
            # Check if we got valid articles (not error messages)
            valid_articles = []
            for article in articles:
                headline = article.get('headline', '')
                if headline and not headline.startswith("I'm unable to browse"):
                    valid_articles.append(article)
            
            if valid_articles:
                print(f"Found {len(valid_articles)} valid articles from news scraper")
                
                # Get source rankings for display purposes
                rankings = self.news_scraper.rank_news_sources(company_name, ticker)
                rankings_dict = {r.get('source', '').lower(): r.get('score', 50) for r in rankings}
                
                # Add rankings and importance scores to articles
                for article in valid_articles:
                    source = article.get('source', '').lower()
                    article['source_ranking'] = rankings_dict.get(source, 50)
                    
                    # Use importance score if available, otherwise use source ranking
                    if 'importance_score' not in article:
                        article['importance_score'] = article['source_ranking']
                    
                    # Parse date
                    date_str = article.get('date', '')
                    try:
                        if date_str:
                            article['parsed_date'] = pd.to_datetime(date_str)
                        else:
                            article['parsed_date'] = None
                    except:
                        article['parsed_date'] = None
                
                return valid_articles
            else:
                print("No valid articles found from news scraper, using fallback")
                
        except Exception as e:
            print(f"Error with news scraper: {e}, using fallback")
        
        # Use fallback if main scraper fails or returns invalid data
        if self.news_fallback:
            print("Using fallback news data...")
            articles = self.news_fallback.get_sample_news_articles(company_name, ticker, days_back=days_back)
            return articles
        else:
            print("No fallback available, returning empty list")
            return []
    
    def create_simple_csv(self, company_name: str, ticker: str, output_filename: str = None) -> str:
        """Create a simple CSV with stock data and news articles on matching dates"""
        print(f"Creating simple CSV for {company_name} ({ticker})")
        
        # Get stock data
        print("1. Getting stock data...")
        stock_df = self.get_stock_data(ticker)
        if stock_df.empty:
            print(f"No stock data available for {ticker}")
            return ""
        stock_min_date = stock_df['Date'].min()
        stock_max_date = stock_df['Date'].max()
        print(f"Stock data date range: {stock_min_date} to {stock_max_date}")
        
        # Get news data for the last 365 days
        print("2. Getting news articles and rankings...")
        news_articles = self.get_news_with_rankings(company_name, ticker, days_back=365)
        print(f"Fetched {len(news_articles)} news articles before date filtering")
        
        # Convert news to DataFrame and filter valid dates
        news_df = pd.DataFrame(news_articles)
        if not news_df.empty and 'parsed_date' in news_df.columns:
            # Always convert parsed_date to pandas datetime
            news_df['parsed_date'] = pd.to_datetime(news_df['parsed_date'], errors='coerce')
            news_df = news_df.dropna(subset=['parsed_date'])
            
            # Debug: Show the date range of news articles
            if not news_df.empty:
                news_min_date = news_df['parsed_date'].min()
                news_max_date = news_df['parsed_date'].max()
                print(f"News articles date range: {news_min_date} to {news_max_date}")
                print(f"Stock data date range: {stock_min_date} to {stock_max_date}")
            
            # Filter news to only those within a reasonable range of the stock data
            # Allow news up to 30 days before the stock data starts and up to 7 days after it ends
            extended_min_date = stock_min_date - pd.Timedelta(days=30)
            extended_max_date = stock_max_date + pd.Timedelta(days=7)
            news_df = news_df[(news_df['parsed_date'] >= extended_min_date) & (news_df['parsed_date'] <= extended_max_date)]
            print(f"News articles after extended date filtering: {len(news_df)}")
            
            # Debug: Show sample of news article dates
            if not news_df.empty:
                print("Sample news article dates:")
                for i, (_, row) in enumerate(news_df.head(5).iterrows()):
                    print(f"  {i+1}. {row['parsed_date']} - {row['headline'][:50]}...")
        else:
            news_df = pd.DataFrame()  # Ensure it's empty if no valid news
            print("No valid news articles after date filtering.")
        
        # Create the simple combined CSV
        print("3. Creating simple combined CSV...")
        
        # Prepare stock data
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        stock_df = stock_df.sort_values('Date', ascending=False)
        
        # Prepare news data by date
        if not news_df.empty:
            # Ensure parsed_date is datetime64 dtype before using .dt.date
            if not pd.api.types.is_datetime64_any_dtype(news_df['parsed_date']):
                news_df['parsed_date'] = pd.to_datetime(news_df['parsed_date'], errors='coerce')
            
            # Convert to date for grouping
            news_df['Date'] = news_df['parsed_date'].dt.date
            news_df = news_df.sort_values('parsed_date', ascending=False)
            
            # Group news by date and get the best article for each day
            best_articles = []
            for date, group in news_df.groupby('Date'):
                # Get the article with highest importance score, fallback to source ranking if needed
                try:
                    # First try importance score
                    if 'importance_score' in group.columns:
                        idx = group['importance_score'].idxmax()
                    else:
                        # Fallback to source ranking
                        idx = group['source_ranking'].idxmax()
                    best_article = group.loc[idx]
                except Exception:
                    best_article = group.iloc[0]
                best_articles.append({
                    'Date': date,
                    'News_Headline': best_article['headline'],
                    'News_Source': best_article['source'],
                    'News_Ranking': best_article.get('importance_score', best_article.get('source_ranking', 50)),
                    'News_Summary': best_article.get('summary', ''),
                    'News_URL': best_article.get('url', '')
                })
            
            news_summary_df = pd.DataFrame(best_articles)
            news_summary_df['Date'] = pd.to_datetime(news_summary_df['Date'])
            
            # Use nearest previous date matching instead of exact matching
            print("4. Matching news to stock dates using nearest previous date...")
            combined_rows = []
            
            # Check if all news articles have the same date (likely today's date from API)
            if not news_summary_df.empty:
                unique_news_dates = news_summary_df['Date'].nunique()
                print(f"Number of unique news dates: {unique_news_dates}")
                
                if unique_news_dates == 1:
                    print("All news articles have the same date - likely today's date from API")
                    print("Assigning articles to recent stock dates...")
                    
                    # Get the most recent stock dates and assign news articles to them
                    recent_stock_dates = stock_df.head(10)  # Last 10 trading days
                    news_articles_to_assign = news_summary_df.head(10)  # Top 10 news articles
                    
                    # Create a mapping of stock dates to news articles
                    stock_to_news = {}
                    for i, (_, stock_row) in enumerate(recent_stock_dates.iterrows()):
                        if i < len(news_articles_to_assign):
                            stock_to_news[stock_row['Date']] = news_articles_to_assign.iloc[i]
            
            for _, stock_row in stock_df.iterrows():
                stock_date = stock_row['Date']
                
                # Check if we have a pre-assigned news article for this date
                if 'stock_to_news' in locals() and stock_date in stock_to_news:
                    best_news = stock_to_news[stock_date]
                    combined_rows.append({
                        'Date': stock_date,
                        'Open': stock_row['Open'],
                        'Close': stock_row['Close'],
                        'News_Headline': best_news['News_Headline'],
                        'News_Source': best_news['News_Source'],
                        'News_Ranking': best_news['News_Ranking'],
                        'News_Summary': best_news['News_Summary'],
                        'News_URL': best_news['News_URL']
                    })
                    continue
                
                # Find the most recent news article on or before this stock date
                # Allow up to 7 days before the stock date
                cutoff_date = stock_date - pd.Timedelta(days=7)
                
                # Ensure both dates are datetime objects for comparison
                news_dates = pd.to_datetime(news_summary_df['Date'])
                stock_date_dt = pd.to_datetime(stock_date)
                cutoff_date_dt = pd.to_datetime(cutoff_date)
                
                matching_news = news_summary_df[
                    (news_dates <= stock_date_dt) & 
                    (news_dates >= cutoff_date_dt)
                ]
                
                if not matching_news.empty:
                    # Get the most recent news article
                    best_news = matching_news.iloc[0]  # Already sorted by date descending
                    combined_rows.append({
                        'Date': stock_date,
                        'Open': stock_row['Open'],
                        'Close': stock_row['Close'],
                        'News_Headline': best_news['News_Headline'],
                        'News_Source': best_news['News_Source'],
                        'News_Ranking': best_news['News_Ranking'],
                        'News_Summary': best_news['News_Summary'],
                        'News_URL': best_news['News_URL']
                    })
                else:
                    # No matching news for this date
                    combined_rows.append({
                        'Date': stock_date,
                        'Open': stock_row['Open'],
                        'Close': stock_row['Close'],
                        'News_Headline': None,
                        'News_Source': None,
                        'News_Ranking': None,
                        'News_Summary': None,
                        'News_URL': None
                    })
            
            combined_df = pd.DataFrame(combined_rows)
            
        else:
            # No news articles, just use stock data
            combined_df = stock_df.copy()
            combined_df['News_Headline'] = None
            combined_df['News_Source'] = None
            combined_df['News_Ranking'] = None
            combined_df['News_Summary'] = None
            combined_df['News_URL'] = None
        
        # Sort by date (most recent first)
        combined_df = combined_df.sort_values('Date', ascending=False)
        
        # Save to CSV
        if output_filename is None:
            output_filename = f"{ticker}_simple_news_stock.csv"
        output_path = os.path.join(self.data_folder, str(output_filename))
        combined_df.to_csv(output_path, index=False)
        
        # Print summary
        total_rows = len(combined_df)
        news_rows = len(combined_df.dropna(subset=['News_Headline']))
        print(f"Simple CSV saved to {output_path}")
        print(f"Total rows: {total_rows}")
        print(f"Rows with news: {news_rows}")
        print(f"Rows without news: {total_rows - news_rows}")
        
        return output_path

def main():
    """Test the simple CSV creation"""
    print("Simple News-Stock CSV Generator")
    print("=" * 50)
    
    csv_generator = SimpleNewsStockCSV()
    
    # Test with a sample company
    company_name = "Apple"
    ticker = "AAPL"
    
    print(f"Testing with {company_name} ({ticker})")
    
    # Create simple CSV
    csv_path = csv_generator.create_simple_csv(company_name, ticker)
    
    if csv_path and os.path.exists(csv_path):
        print(f"\n✅ CSV created successfully: {csv_path}")
        
        # Show sample of the CSV
        df = pd.read_csv(csv_path)
        print(f"\nCSV Structure:")
        print(f"Columns: {list(df.columns)}")
        print(f"Total rows: {len(df)}")
        
        print(f"\nSample data (first 5 rows):")
        print(df.head().to_string())
        
        # Show rows with news
        news_rows = df.dropna(subset=['News_Headline'])
        if not news_rows.empty:
            print(f"\nSample rows with news:")
            print(news_rows.head(3)[['Date', 'Close', 'News_Headline', 'News_Source', 'News_Ranking']].to_string())
    else:
        print("❌ Failed to create CSV")
    
    print("\n" + "=" * 50)
    print("Simple CSV generation completed!")

if __name__ == "__main__":
    main() 