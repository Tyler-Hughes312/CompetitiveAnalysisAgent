from ambivo_agents import AssistantAgent
import os
from yFinanceAPIParsing import YFinanceAPIParsing
from dotenv import load_dotenv
import yaml
from redis_cache import get_cache

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

def load_agent_config():
    config_path = os.path.join(os.path.dirname(__file__), "agent_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

class CompetitionIdentifier:
    def __init__(self, user_id="Tyler_Hughes"):
        config = load_agent_config()
        self.assistant, _ = AssistantAgent.create(
            user_id=user_id,
            session_metadata={
                "project": "competition_identification",
                "llm_provider": "anthropic"
            },
            config=config  # Pass the dict, not config_path
        )
        self.cache = get_cache()

    def get_top_competitors(self, company_name, top_n=10):
        import asyncio
        
        # Check cache first
        cache_key = f"competitors_{company_name}_{top_n}"
        cached_tickers = self.cache.get("competitor_tickers", company_name, top_n)
        if cached_tickers is not None:
            print(f"Using cached competitor list for {company_name}")
            return cached_tickers
        
        # If not in cache, ask the LLM
        print(f"Fetching fresh competitor list for {company_name} from LLM")
        prompt = f"List the stock ticker symbols (not company names) for the largest {top_n} public competitors to the company '{company_name}'. Return only the ticker symbols as a comma-separated list."
        for method in ["ask", "chat", "query", "complete", "get_response"]:
            if hasattr(self.assistant, method):
                func = getattr(self.assistant, method)
                if asyncio.iscoroutinefunction(func):
                    response = asyncio.run(func(prompt))
                else:
                    response = func(prompt)
                tickers = [c.strip().upper() for c in response.split(',') if c.strip()]
                
                # Cache the result for 24 hours (86400 seconds) since competitor lists don't change often
                self.cache.set("competitor_tickers", tickers, company_name, top_n, ttl_seconds=86400)
                
                return tickers
        raise AttributeError("No supported method found on AssistantAgent. Tried: ask, chat, query, complete, get_response.")

    def get_competitor_data(self, company_name, our_company_ticker, top_n=10, output_folder='yfinance_data'):
        tickers = self.get_top_competitors(company_name)
        if not isinstance(tickers, list) or not tickers or any('error' in str(tickers).lower() for tickers in tickers):
            print(f"Error: Failed to get competitors for {company_name}. Agent returned: {tickers}")
            return
        tickers.append(our_company_ticker.upper())
        print(f"Getting data for tickers: {tickers}")
        yfinance_parser = YFinanceAPIParsing(tickers)
        yfinance_parser.select_ticker(output_folder=output_folder)
        return tickers


if __name__ == "__main__":
    company_name = input("Enter the company name to search for competitors: ")
    our_company_ticker = input("Enter your company's ticker symbol: ")
    identifier = CompetitionIdentifier()
    identifier.get_competitor_data(company_name, our_company_ticker)
