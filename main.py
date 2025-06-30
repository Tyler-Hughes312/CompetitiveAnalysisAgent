import os
from competitionIdentification import CompetitionIdentifier
from analysis import StockDataAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
import yaml

load_dotenv()

with open("agent_config.yaml") as f:
    config = yaml.safe_load(f)

DATA_FOLDER = 'yfinance_data'

def delete_existing_csvs():
    if os.path.exists(DATA_FOLDER):
        for filename in os.listdir(DATA_FOLDER):
            if filename.endswith('_historical_data.csv'):
                os.remove(os.path.join(DATA_FOLDER, filename))

def main():
    while True:
        print("\nOptions:")
        print("1. Enter a new company and fetch competitors' data")
        print("2. Plot competition closing prices")
        print("3. Predict future (simple trend) for a ticker")
        print("4. Joint Linear Regression Forecast for all companies (and save to CSV)")
        print("5. Predictive Model (LSTM) for all companies (and save to CSV)")
        print("6. Exit")
        choice = input("Select an option (1-6): ").strip()

        if choice == '1':
            confirm = input("This will delete all existing CSVs. Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                delete_existing_csvs()
                company_name = input("Enter the company name to search for competitors: ")
                our_company_ticker = input("Enter your company's ticker symbol: ")
                identifier = CompetitionIdentifier()
                identifier.get_competitor_data(company_name, our_company_ticker)
                print("Data fetched and saved.")
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
            print("Exiting.")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()
