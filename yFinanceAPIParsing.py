import yfinance as yf
import pandas as pd
import os

class YFinanceAPIParsing:
    def __init__(self, company_list):
        self.company_list = company_list if company_list is not None else []

    def get_data(self, ticker):
        data = yf.download(ticker, start='2024-01-01', end='2025-01-01')
        if data.empty:
            print(f"No data for {ticker}")
            return None
        data.reset_index(inplace=True)
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

