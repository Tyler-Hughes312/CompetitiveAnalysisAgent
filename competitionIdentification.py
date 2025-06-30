from ambivo_agents import AssistantAgent
import os
from yFinanceAPIParsing import YFinanceAPIParsing
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key (ensure this is set securely in production)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")


class CompetitionIdentifier:
    def __init__(self, user_id="Tyler_Hughes"):
        self.assistant, _ = AssistantAgent.create(
            user_id=user_id,
            session_metadata={"project": "competition_identification"},
            llm_provider="openai"
        )

    def get_top_competitors(self, company_name, top_n=10):
        prompt = f"List the largest {top_n} competitors to the company '{company_name}'. Return only the company names as a comma-separated list."
        response = self.assistant.run(prompt)
        competitors = [c.strip() for c in response.split(',') if c.strip()]
        return competitors

    def get_competitor_data(self, company_name, our_company_ticker, top_n=10, output_folder='yfinance_data'):
        competitors = self.get_top_competitors(company_name, top_n=top_n)
        # Add our company to the list
        all_companies = competitors + [our_company_ticker]
        print(f"Getting data for: {all_companies}")
        yfinance_parser = YFinanceAPIParsing(all_companies)
        yfinance_parser.select_ticker(output_folder=output_folder)
        return all_companies


if __name__ == "__main__":
    company_name = input("Enter the company name to search for competitors: ")
    our_company_ticker = input("Enter your company's ticker symbol: ")
    identifier = CompetitionIdentifier()
    identifier.get_competitor_data(company_name, our_company_ticker)
