#!/usr/bin/env python3
"""
News Scraper and Investor Sentiment Analysis
Uses Ambivo web scraping agent to gather news and rank sources from investor perspective
"""

import os
import yaml
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from ambivo_agents import AssistantAgent
from redis_cache import get_cache

class NewsScraper:
    def __init__(self, user_id="Tyler_Hughes"):
        """Initialize the news scraper with Ambivo agents"""
        self.user_id = user_id
        self.cache = get_cache()
        self.config = self._load_config()
        self.assistant, _ = AssistantAgent.create(
            user_id=user_id,
            session_metadata={
                "project": "news_scraping",
                "llm_provider": "anthropic"
            },
            config=self.config
        )
    
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
        
        prompt = f"""
        List the top 15 most important news sources for investors researching {company_name} ({ticker if ticker else 'N/A'}).
        Focus on sources that provide:
        1. Financial news and analysis
        2. Company earnings and financial reports
        3. Market analysis and stock performance
        4. Industry trends and competitive analysis
        5. Regulatory news and compliance updates
        
        Return only the news source names as a comma-separated list.
        Examples: Bloomberg, Reuters, CNBC, MarketWatch, Yahoo Finance, etc.
        """
        
        try:
            import asyncio
            for method in ["ask", "chat", "query", "complete", "get_response"]:
                if hasattr(self.assistant, method):
                    func = getattr(self.assistant, method)
                    if asyncio.iscoroutinefunction(func):
                        response = asyncio.run(func(prompt))
                    else:
                        response = func(prompt)
                    
                    sources = [s.strip() for s in response.split(',') if s.strip()]
                    
                    # Cache for 24 hours since news sources don't change often
                    self.cache.set("news_sources", sources, company_name, ticker, ttl_seconds=86400)
                    
                    return sources
        except Exception as e:
            print(f"Error getting news sources: {e}")
            return []
        
        return []
    
    def scrape_news_articles(self, company_name: str, ticker: str = None, 
                           days_back: int = 365, max_articles: int = 100) -> List[Dict[str, Any]]:
        """Scrape news articles from various sources"""
        cache_key = f"news_articles_{company_name}_{ticker}_{days_back}"
        cached_articles = self.cache.get("news_articles", company_name, ticker, days_back)
        if cached_articles is not None:
            print(f"Using cached news articles for {company_name}")
            return cached_articles
        
        print(f"Scraping fresh news articles for {company_name}")
        
        # Get news sources
        sources = self.get_news_sources(company_name, ticker)
        if not sources:
            print("No news sources found")
            return []
        
        articles = []
        search_terms = [company_name]
        if ticker:
            search_terms.append(ticker)
        
        for source in sources[:5]:  # Limit to top 5 sources to avoid rate limiting
            try:
                # Create search query for the source
                search_query = f"{company_name} {ticker if ticker else ''} news"
                
                # Use web scraping agent to search and scrape
                scrape_prompt = f"""
                Search for recent news articles about {company_name} ({ticker if ticker else 'N/A'}) 
                from {source} in the last {days_back} days.
                
                Focus on articles that would be relevant to investors, such as:
                - Earnings reports and financial performance
                - Stock price movements and market analysis
                - Strategic announcements and business developments
                - Industry trends and competitive positioning
                - Regulatory news and compliance updates
                
                For each article found, extract:
                1. Headline
                2. Publication date
                3. Source/author
                4. Brief summary (2-3 sentences)
                5. URL (if available)
                
                Return the results in a structured format.
                """
                
                import asyncio
                for method in ["ask", "chat", "query", "complete", "get_response"]:
                    if hasattr(self.assistant, method):
                        func = getattr(self.assistant, method)
                        if asyncio.iscoroutinefunction(func):
                            response = asyncio.run(func(scrape_prompt))
                        else:
                            response = func(scrape_prompt)
                        
                        # Parse the response to extract article information
                        parsed_articles = self._parse_articles_response(response, source)
                        articles.extend(parsed_articles)
                        break
                
            except Exception as e:
                print(f"Error scraping from {source}: {e}")
                continue
        
        # Limit to max_articles
        articles = articles[:max_articles]
        
        # Cache for 2 hours since news articles change frequently
        self.cache.set("news_articles", articles, company_name, ticker, days_back, ttl_seconds=7200)
        
        return articles
    
    def _parse_articles_response(self, response: str, source: str) -> List[Dict[str, Any]]:
        """Parse the response from the web scraping agent to extract article information"""
        articles = []
        
        try:
            # Simple parsing - look for patterns in the response
            lines = response.split('\n')
            current_article = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_article:
                        articles.append(current_article)
                        current_article = {}
                    continue
                
                # Try to identify article components
                if line.lower().startswith(('headline:', 'title:', 'article:')):
                    current_article['headline'] = line.split(':', 1)[1].strip()
                elif line.lower().startswith(('date:', 'published:', 'time:')):
                    current_article['date'] = line.split(':', 1)[1].strip()
                elif line.lower().startswith(('summary:', 'description:', 'content:')):
                    current_article['summary'] = line.split(':', 1)[1].strip()
                elif line.lower().startswith(('url:', 'link:', 'source:')):
                    current_article['url'] = line.split(':', 1)[1].strip()
                elif line.lower().startswith(('author:', 'by:', 'writer:')):
                    current_article['author'] = line.split(':', 1)[1].strip()
                else:
                    # If no clear pattern, assume it's a headline if we don't have one
                    if 'headline' not in current_article:
                        current_article['headline'] = line
            
            # Add the last article if exists
            if current_article:
                articles.append(current_article)
            
            # Add source to each article
            for article in articles:
                article['source'] = source
                
        except Exception as e:
            print(f"Error parsing articles response: {e}")
        
        return articles
    
    def rank_news_sources(self, company_name: str, ticker: str = None) -> List[Dict[str, Any]]:
        """Rank news sources from 1-100 based on investor value"""
        cache_key = f"source_rankings_{company_name}_{ticker}"
        cached_rankings = self.cache.get("source_rankings", company_name, ticker)
        if cached_rankings is not None:
            print(f"Using cached source rankings for {company_name}")
            return cached_rankings
        
        print(f"Ranking news sources for {company_name}")
        
        sources = self.get_news_sources(company_name, ticker)
        if not sources:
            return []
        
        ranking_prompt = f"""
        Rank the following news sources from 1-100 based on their value to investors researching {company_name} ({ticker if ticker else 'N/A'}).
        
        Consider these factors:
        1. **Financial Focus** (25 points): How well does the source cover financial news, earnings, and market analysis?
        2. **Accuracy & Reliability** (25 points): How trustworthy and accurate is the information?
        3. **Timeliness** (20 points): How quickly does the source report breaking news?
        4. **Depth of Analysis** (20 points): How comprehensive and insightful is the coverage?
        5. **Investor Relevance** (10 points): How specifically relevant is it to investors vs. general news?
        
        News sources to rank:
        {', '.join(sources)}
        
        For each source, provide:
        1. Source name
        2. Overall score (1-100)
        3. Brief justification for the score
        4. Key strengths
        5. Key weaknesses
        
        Return the results in a structured format.
        """
        
        try:
            import asyncio
            for method in ["ask", "chat", "query", "complete", "get_response"]:
                if hasattr(self.assistant, method):
                    func = getattr(self.assistant, method)
                    if asyncio.iscoroutinefunction(func):
                        response = asyncio.run(func(ranking_prompt))
                    else:
                        response = func(ranking_prompt)
                    
                    rankings = self._parse_rankings_response(response, sources)
                    
                    # Cache for 24 hours since rankings don't change often
                    self.cache.set("source_rankings", rankings, company_name, ticker, ttl_seconds=86400)
                    
                    return rankings
        except Exception as e:
            print(f"Error ranking news sources: {e}")
            return []
        
        return []
    
    def _parse_rankings_response(self, response: str, sources: List[str]) -> List[Dict[str, Any]]:
        """Parse the response to extract source rankings"""
        rankings = []
        
        try:
            lines = response.split('\n')
            current_ranking = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_ranking:
                        rankings.append(current_ranking)
                        current_ranking = {}
                    continue
                
                # Try to identify ranking components
                if any(source.lower() in line.lower() for source in sources):
                    # This line contains a source name
                    current_ranking['source'] = line.split(':')[0].strip()
                elif 'score:' in line.lower() or 'rating:' in line.lower():
                    try:
                        score_text = line.split(':')[1].strip()
                        score = int(''.join(filter(str.isdigit, score_text)))
                        current_ranking['score'] = min(100, max(1, score))
                    except:
                        pass
                elif 'justification:' in line.lower() or 'reason:' in line.lower():
                    current_ranking['justification'] = line.split(':', 1)[1].strip()
                elif 'strengths:' in line.lower():
                    current_ranking['strengths'] = line.split(':', 1)[1].strip()
                elif 'weaknesses:' in line.lower():
                    current_ranking['weaknesses'] = line.split(':', 1)[1].strip()
            
            # Add the last ranking if exists
            if current_ranking:
                rankings.append(current_ranking)
            
            # Ensure all sources have rankings
            ranked_sources = {r.get('source', '').lower() for r in rankings}
            for source in sources:
                if source.lower() not in ranked_sources:
                    rankings.append({
                        'source': source,
                        'score': 50,  # Default score
                        'justification': 'Default ranking',
                        'strengths': 'Standard financial news source',
                        'weaknesses': 'Limited specialized coverage'
                    })
            
            # Sort by score (highest first)
            rankings.sort(key=lambda x: x.get('score', 0), reverse=True)
            
        except Exception as e:
            print(f"Error parsing rankings response: {e}")
        
        return rankings
    
    def get_investor_sentiment_analysis(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the sentiment of news articles from an investor perspective"""
        if not articles:
            return {}
        
        cache_key = f"sentiment_analysis_{len(articles)}"
        cached_sentiment = self.cache.get("sentiment_analysis", len(articles))
        if cached_sentiment is not None:
            print("Using cached sentiment analysis")
            return cached_sentiment
        
        print("Performing fresh sentiment analysis")
        
        # Prepare articles for analysis
        articles_text = []
        for article in articles:
            text = f"Headline: {article.get('headline', '')}\n"
            text += f"Summary: {article.get('summary', '')}\n"
            text += f"Source: {article.get('source', '')}\n"
            articles_text.append(text)
        
        analysis_prompt = f"""
        Analyze the sentiment of the following news articles from an investor's perspective for {len(articles)} articles.
        
        For each article, provide:
        1. **Sentiment Score** (1-100): 1=Very Negative, 50=Neutral, 100=Very Positive
        2. **Investor Impact**: How this news affects investment decisions
        3. **Risk Level**: Low/Medium/High risk implications
        4. **Key Themes**: Main topics and themes discussed
        5. **Action Items**: What investors should watch for
        
        Articles:
        {chr(10).join(articles_text)}
        
        Provide a comprehensive analysis focusing on investment implications.
        """
        
        try:
            import asyncio
            for method in ["ask", "chat", "query", "complete", "get_response"]:
                if hasattr(self.assistant, method):
                    func = getattr(self.assistant, method)
                    if asyncio.iscoroutinefunction(func):
                        response = asyncio.run(func(analysis_prompt))
                    else:
                        response = func(analysis_prompt)
                    
                    sentiment_analysis = {
                        'overall_sentiment': self._extract_overall_sentiment(response),
                        'risk_assessment': self._extract_risk_assessment(response),
                        'key_themes': self._extract_key_themes(response),
                        'action_items': self._extract_action_items(response),
                        'analysis_text': response
                    }
                    
                    # Cache for 1 hour since sentiment can change
                    self.cache.set("sentiment_analysis", sentiment_analysis, len(articles), ttl_seconds=3600)
                    
                    return sentiment_analysis
        except Exception as e:
            print(f"Error performing sentiment analysis: {e}")
            return {}
        
        return {}
    
    def _extract_overall_sentiment(self, response: str) -> int:
        """Extract overall sentiment score from response"""
        try:
            # Look for sentiment score patterns
            import re
            score_match = re.search(r'sentiment.*?(\d+)', response.lower())
            if score_match:
                return int(score_match.group(1))
        except:
            pass
        return 50  # Default neutral score
    
    def _extract_risk_assessment(self, response: str) -> str:
        """Extract risk assessment from response"""
        if 'high risk' in response.lower():
            return 'High'
        elif 'medium risk' in response.lower():
            return 'Medium'
        elif 'low risk' in response.lower():
            return 'Low'
        return 'Medium'  # Default
    
    def _extract_key_themes(self, response: str) -> List[str]:
        """Extract key themes from response"""
        themes = []
        try:
            # Simple extraction - look for theme indicators
            lines = response.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['theme:', 'topic:', 'focus:', 'key:']):
                    theme = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                    if theme:
                        themes.append(theme)
        except:
            pass
        return themes[:5]  # Limit to top 5 themes
    
    def _extract_action_items(self, response: str) -> List[str]:
        """Extract action items from response"""
        actions = []
        try:
            # Simple extraction - look for action indicators
            lines = response.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['action:', 'watch:', 'monitor:', 'consider:']):
                    action = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                    if action:
                        actions.append(action)
        except:
            pass
        return actions[:5]  # Limit to top 5 actions

def main():
    """Test the news scraper functionality"""
    print("News Scraper and Investor Sentiment Analysis")
    print("=" * 60)
    
    # Test with a sample company
    company_name = "Apple"
    ticker = "AAPL"
    
    scraper = NewsScraper()
    
    print(f"Testing with {company_name} ({ticker})")
    
    # 1. Get news sources
    print("\n1. Getting news sources...")
    sources = scraper.get_news_sources(company_name, ticker)
    print(f"Found {len(sources)} news sources: {sources[:5]}...")
    
    # 2. Rank news sources
    print("\n2. Ranking news sources...")
    rankings = scraper.rank_news_sources(company_name, ticker)
    print("Top 5 ranked sources:")
    for i, ranking in enumerate(rankings[:5]):
        print(f"  {i+1}. {ranking.get('source', 'Unknown')}: {ranking.get('score', 0)}/100")
    
    # 3. Scrape news articles
    print("\n3. Scraping news articles...")
    articles = scraper.scrape_news_articles(company_name, ticker, days_back=7, max_articles=10)
    print(f"Found {len(articles)} articles")
    
    # 4. Analyze sentiment
    if articles:
        print("\n4. Analyzing investor sentiment...")
        sentiment = scraper.get_investor_sentiment_analysis(articles)
        print(f"Overall sentiment: {sentiment.get('overall_sentiment', 'N/A')}/100")
        print(f"Risk level: {sentiment.get('risk_assessment', 'N/A')}")
        print(f"Key themes: {sentiment.get('key_themes', [])}")
    
    print("\n" + "=" * 60)
    print("News scraper test completed!")

if __name__ == "__main__":
    main() 