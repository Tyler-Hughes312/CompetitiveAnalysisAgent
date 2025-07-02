#!/usr/bin/env python3
"""
Google News Scraper using Google Search API
Fetches news articles and ranks sources from investor perspective
"""

import os
import yaml
import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from redis_cache import get_cache
from dotenv import load_dotenv

load_dotenv()

class GoogleNewsScraper:
    def __init__(self):
        """Initialize the Google news scraper"""
        self.cache = get_cache()
        self.config = self._load_config()
        self.api_key = os.getenv('GOOGLE_SEARCH_API_KEY') or self.config.get('web_search', {}).get('google_search_api_key', '')
        self.engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID') or self.config.get('web_search', {}).get('google_search_engine_id', '')
        
        if not self.api_key:
            print("Warning: Google Search API key not found. Please set GOOGLE_SEARCH_API_KEY in .env file")
        if not self.engine_id:
            print("Warning: Google Search Engine ID not found. Please set GOOGLE_SEARCH_ENGINE_ID in .env file")
    
    def _load_config(self) -> dict:
        """Load configuration from agent_config.yaml"""
        config_path = os.path.join(os.path.dirname(__file__), "agent_config.yaml")
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return {}
    
    def get_news_sources(self, company_name: str, ticker: str = None) -> List[str]:
        """Get list of relevant news sources for a company"""
        cache_key = f"news_sources_{company_name}_{ticker}"
        cached_sources = self.cache.get("news_sources", company_name, ticker)
        if cached_sources is not None:
            print(f"Using cached news sources for {company_name}")
            return cached_sources
        
        print(f"Fetching fresh news sources for {company_name}")
        
        # Default list of reliable financial news sources
        sources = [
            "Bloomberg", "Reuters", "CNBC", "MarketWatch", "Yahoo Finance",
            "The Wall Street Journal", "Financial Times", "Barron's",
            "Investor's Business Daily", "Seeking Alpha", "Motley Fool",
            "Forbes", "Fortune", "Business Insider", "TechCrunch",
            "The New York Times", "Los Angeles Times", "USA Today",
            "Associated Press", "Dow Jones", "S&P Global", "Morningstar"
        ]
        
        # Cache for 24 hours since news sources don't change often
        self.cache.set("news_sources", sources, company_name, ticker, ttl_seconds=86400)
        
        return sources
    
    def search_google_news(self, query: str, max_results: int = 10, date_restrict: str = 'd30') -> List[Dict[str, Any]]:
        """Search for news using Google Search API"""
        if not self.api_key or not self.engine_id:
            print("Google Search API not configured")
            return []
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': self.api_key,
            'cx': self.engine_id,
            'q': query,
            'num': min(max_results, 10),  # Google API limit is 10 per request
            'dateRestrict': date_restrict,  # Configurable date restriction
            'sort': 'date'  # Sort by date
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            items = data.get('items', [])
            
            articles = []
            for item in items:
                article = {
                    'headline': item.get('title', ''),
                    'url': item.get('link', ''),
                    'summary': item.get('snippet', ''),
                    'source': self._extract_source_from_url(item.get('link', '')),
                    'date': self._extract_date_from_item(item)
                }
                articles.append(article)
            
            return articles
            
        except requests.exceptions.RequestException as e:
            print(f"Error searching Google News: {e}")
            return []
        except Exception as e:
            print(f"Error parsing Google Search response: {e}")
            return []
    
    def _extract_source_from_url(self, url: str) -> str:
        """Extract news source from URL"""
        if not url:
            return "Unknown"
        
        # Common news domains
        domain_mapping = {
            'bloomberg.com': 'Bloomberg',
            'reuters.com': 'Reuters',
            'cnbc.com': 'CNBC',
            'marketwatch.com': 'MarketWatch',
            'finance.yahoo.com': 'Yahoo Finance',
            'wsj.com': 'The Wall Street Journal',
            'ft.com': 'Financial Times',
            'barrons.com': "Barron's",
            'investors.com': "Investor's Business Daily",
            'seekingalpha.com': 'Seeking Alpha',
            'fool.com': 'Motley Fool',
            'forbes.com': 'Forbes',
            'fortune.com': 'Fortune',
            'businessinsider.com': 'Business Insider',
            'techcrunch.com': 'TechCrunch',
            'nytimes.com': 'The New York Times',
            'latimes.com': 'Los Angeles Times',
            'usatoday.com': 'USA Today',
            'ap.org': 'Associated Press',
            'dowjones.com': 'Dow Jones',
            'spglobal.com': 'S&P Global',
            'morningstar.com': 'Morningstar'
        }
        
        # Extract domain from URL
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain_mapping.get(domain, domain.title())
        except:
            return "Unknown"
    
    def _extract_date_from_item(self, item: Dict[str, Any]) -> str:
        """Extract date from Google Search result item"""
        # Try to get date from metatags
        metatags = item.get('pagemap', {}).get('metatags', [{}])[0]
        
        # Check various date fields
        date_fields = ['article:published_time', 'date', 'pubdate', 'publishdate']
        for field in date_fields:
            if field in metatags:
                date_str = metatags[field]
                try:
                    # Parse and format the date
                    parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    return parsed_date.strftime('%Y-%m-%d')
                except:
                    continue
        
        # If no date found, return today's date as a reasonable default
        return datetime.now().strftime('%Y-%m-%d')
    
    def extract_date_with_llm(self, article: Dict[str, Any]) -> str:
        """Use Ambivo assistant LLM to extract publication date from article content"""
        try:
            from ambivo_agents import AssistantAgent
            import yaml
            from datetime import datetime, timedelta
            
            config_path = os.path.join(os.path.dirname(__file__), "agent_config.yaml")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assistant, _ = AssistantAgent.create(
                user_id="Tyler_Hughes",
                session_metadata={
                    "project": "news_date_extraction",
                    "llm_provider": "anthropic"
                },
                config=config
            )
            
            headline = article.get('headline', '')
            summary = article.get('summary', '')
            url = article.get('url', '')
            current_date = article.get('date', '')
            
            # Only use LLM if the current date looks suspicious (all same date)
            if current_date == datetime.now().strftime('%Y-%m-%d'):
                prompt = f"""
                Analyze this news article and estimate the publication date based on the content.
                
                Headline: {headline}
                Summary: {summary}
                URL: {url}
                
                Look for any clues about when this article was published:
                - Specific dates mentioned
                - Time references (yesterday, last week, etc.)
                - Event references that might indicate timing
                - Seasonal or contextual clues
                
                If you can make a reasonable estimate, return the date in YYYY-MM-DD format.
                If you cannot make any reasonable estimate, return "UNKNOWN".
                
                Be conservative - only return a date if you have some confidence.
                """
                
                import asyncio
                for method in ["ask", "chat", "query", "complete", "get_response"]:
                    if hasattr(assistant, method):
                        func = getattr(assistant, method)
                        if asyncio.iscoroutinefunction(func):
                            response = asyncio.run(func(prompt))
                        else:
                            response = func(prompt)
                        if response and 'UNKNOWN' not in response.upper():
                            import re
                            date_pattern = r'\d{4}-\d{2}-\d{2}'
                            match = re.search(date_pattern, response)
                            if match:
                                extracted_date = match.group(0)
                                # Validate that the date is reasonable
                                try:
                                    extracted_dt = datetime.strptime(extracted_date, '%Y-%m-%d')
                                    today_dt = datetime.now()
                                    if extracted_dt <= today_dt and extracted_dt >= (today_dt - timedelta(days=365)):
                                        return extracted_date
                                except:
                                    pass
                        break
            else:
                # Current date looks reasonable, keep it
                return current_date
                
        except Exception as e:
            print(f"Error using LLM for date extraction: {e}")
        
        # Return the original date if LLM fails or isn't needed
        return article.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    def calculate_article_importance(self, article: Dict[str, Any]) -> float:
        """Calculate the importance score of an article based on multiple factors"""
        score = 0.0
        
        # Source credibility (0-50 points)
        source = article.get('source', '').lower()
        source_scores = {
            'bloomberg': 50, 'reuters': 50, 'wsj': 48, 'ft': 47, 'cnbc': 45,
            'marketwatch': 42, 'yahoo finance': 40, 'barrons': 45, 'investors': 43,
            'seeking alpha': 35, 'motley fool': 38, 'forbes': 40, 'fortune': 42,
            'business insider': 35, 'techcrunch': 37, 'nytimes': 45, 'latimes': 40,
            'usatoday': 38, 'ap': 45, 'dow jones': 46, 'sp global': 44, 'morningstar': 43
        }
        score += source_scores.get(source, 25)  # Default score for unknown sources
        
        # Headline importance indicators (0-30 points)
        headline = article.get('headline', '').lower()
        importance_keywords = [
            'earnings', 'quarterly', 'financial results', 'revenue', 'profit', 'loss',
            'stock price', 'market cap', 'valuation', 'acquisition', 'merger', 'buyout',
            'ceo', 'executive', 'leadership', 'strategy', 'expansion', 'growth',
            'ipo', 'dividend', 'buyback', 'split', 'guidance', 'forecast'
        ]
        for keyword in importance_keywords:
            if keyword in headline:
                score += 2  # 2 points per important keyword, max 30
        
        # Summary length and content (0-20 points)
        summary = article.get('summary', '')
        if len(summary) > 100:  # Longer summaries often indicate more detailed articles
            score += 10
        if any(keyword in summary.lower() for keyword in importance_keywords):
            score += 10
        
        return min(score, 100)  # Cap at 100

    def scrape_news_articles(self, company_name: str, ticker: str = None, 
                           days_back: int = 365, max_articles: int = 100) -> List[Dict[str, Any]]:
        """Scrape news articles using Google Search API - get biggest article per day over the last year"""
        cache_key = f"news_articles_{company_name}_{ticker}_{days_back}"
        cached_articles = self.cache.get("news_articles", company_name, ticker, days_back)
        if cached_articles is not None:
            print(f"Using cached news articles for {company_name}")
            return cached_articles
        
        print(f"Searching for news articles about {company_name} over the last {days_back} days")
        
        all_articles = []
        
        # Create comprehensive search queries for different time periods
        search_queries = [
            f'"{company_name}" earnings financial results',
            f'"{company_name}" {ticker} stock price market',
            f'"{company_name}" business news strategy',
            f'"{company_name}" quarterly results revenue',
            f'"{company_name}" acquisition merger deal',
            f'"{company_name}" ceo executive leadership',
            f'"{company_name}" expansion growth investment'
        ]
        
        # Search in different time periods to cover the full year
        time_periods = [
            ('d7', 10),    # Last 7 days - more results
            ('d30', 15),   # Last 30 days - more results  
            ('m3', 20),    # Last 3 months
            ('m6', 25),    # Last 6 months
            ('y1', 30)     # Last year
        ]
        
        for date_restrict, max_per_query in time_periods:
            for query in search_queries:
                if len(all_articles) >= max_articles:
                    break
                
                query_articles = self.search_google_news(query, max_results=max_per_query, date_restrict=date_restrict)
                all_articles.extend(query_articles)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article['url'] not in seen_urls:
                seen_urls.add(article['url'])
                unique_articles.append(article)
        
        # Limit to max_articles
        unique_articles = unique_articles[:max_articles]
        
        # Use LLM to improve dates only when needed
        print("Improving article dates with LLM when needed...")
        for article in unique_articles:
            current_date = article.get('date', '')
            if current_date == datetime.now().strftime('%Y-%m-%d'):
                print(f"Improving date for: {article['headline'][:50]}...")
                improved_date = self.extract_date_with_llm(article)
                article['date'] = improved_date
                print(f"  Improved date: {improved_date}")
        
        # Calculate importance scores for all articles
        print("Calculating article importance scores...")
        for article in unique_articles:
            article['importance_score'] = self.calculate_article_importance(article)
        
        # Group articles by date and select the most important one for each day
        print("Selecting most important article per day...")
        articles_by_date = {}
        for article in unique_articles:
            date = article.get('date', '')
            if date and date != "UNKNOWN":
                if date not in articles_by_date:
                    articles_by_date[date] = []
                articles_by_date[date].append(article)
        
        # Select the most important article for each date
        best_articles = []
        for date, day_articles in articles_by_date.items():
            # Sort by importance score (highest first)
            day_articles.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
            best_article = day_articles[0]  # Get the most important article
            best_articles.append(best_article)
        
        # Sort by date (most recent first)
        best_articles.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        print(f"Selected {len(best_articles)} most important articles (one per day)")
        if best_articles:
            date_range = f"{best_articles[-1].get('date')} to {best_articles[0].get('date')}"
            print(f"Article date range: {date_range}")
        
        # Cache for 2 hours since news articles change frequently
        self.cache.set("news_articles", best_articles, company_name, ticker, days_back, ttl_seconds=7200)
        
        return best_articles
    
    def rank_news_sources(self, company_name: str, ticker: str = None) -> List[Dict[str, Any]]:
        """Rank news sources by credibility for investors"""
        cache_key = f"source_rankings_{company_name}_{ticker}"
        cached_rankings = self.cache.get("source_rankings", company_name, ticker)
        if cached_rankings is not None:
            print(f"Using cached source rankings for {company_name}")
            return cached_rankings
        
        print(f"Ranking news sources for {company_name}")
        
        # Predefined rankings for major financial news sources
        source_rankings = {
            "Bloomberg": 95,
            "Reuters": 94,
            "The Wall Street Journal": 93,
            "Financial Times": 92,
            "Barron's": 90,
            "Investor's Business Daily": 87,
            "CNBC": 88,
            "MarketWatch": 85,
            "Yahoo Finance": 82,
            "Dow Jones": 91,
            "S&P Global": 89,
            "Morningstar": 86,
            "Seeking Alpha": 75,
            "Motley Fool": 78,
            "Forbes": 80,
            "Fortune": 83,
            "Business Insider": 70,
            "TechCrunch": 72,
            "The New York Times": 88,
            "Los Angeles Times": 85,
            "USA Today": 80,
            "Associated Press": 90
        }
        
        # Convert to list format
        rankings = []
        for source, score in source_rankings.items():
            rankings.append({
                'source': source,
                'score': score,
                'justification': f"{source} is a {self._get_credibility_level(score)} source for financial news"
            })
        
        # Sort by score (highest first)
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        # Cache for 24 hours since rankings don't change often
        self.cache.set("source_rankings", rankings, company_name, ticker, ttl_seconds=86400)
        
        return rankings
    
    def _get_credibility_level(self, score: int) -> str:
        """Get credibility level description based on score"""
        if score >= 90:
            return "highly credible"
        elif score >= 80:
            return "very credible"
        elif score >= 70:
            return "credible"
        elif score >= 60:
            return "moderately credible"
        else:
            return "less credible"

def main():
    """Test the Google news scraper"""
    print("=== Google News Scraper Test ===\n")
    
    scraper = GoogleNewsScraper()
    
    # Test with Amazon
    company_name = "Amazon"
    ticker = "AMZN"
    
    print(f"Testing with {company_name} ({ticker})")
    
    # Test news sources
    print("\n1. Testing news sources...")
    sources = scraper.get_news_sources(company_name, ticker)
    print(f"Found {len(sources)} news sources")
    
    # Test news articles
    print("\n2. Testing news articles...")
    articles = scraper.scrape_news_articles(company_name, ticker, days_back=365, max_articles=100)
    print(f"Found {len(articles)} news articles")
    
    if articles:
        print("\nSample articles:")
        for i, article in enumerate(articles[:3]):
            print(f"{i+1}. {article['headline']}")
            print(f"   Source: {article['source']}")
            print(f"   Date: {article['date']}")
            print(f"   URL: {article['url']}")
            print()
    
    # Test source rankings
    print("\n3. Testing source rankings...")
    rankings = scraper.rank_news_sources(company_name, ticker)
    print(f"Found {len(rankings)} source rankings")
    
    if rankings:
        print("\nTop 5 sources:")
        for i, rank in enumerate(rankings[:5]):
            print(f"{i+1}. {rank['source']}: {rank['score']}")
    
    print("\n✅ Google news scraper test completed!")

if __name__ == "__main__":
    main() 