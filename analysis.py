import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np


class StockDataAnalyzer:
    def __init__(self, data_folder='yfinance_data'):
        self.data_folder = data_folder
        self.dataframes = {}
        self._load_data()

    def _load_data(self):
        # Load all CSVs in the data folder
        for filename in os.listdir(self.data_folder):
            if filename.endswith('_historical_data.csv'):
                ticker = filename.split('_')[0]
                filepath = os.path.join(self.data_folder, filename)
                df = pd.read_csv(filepath, usecols=['Date', 'Open', 'Close'])
                # Convert Open/Close to numeric, coerce errors to NaN
                df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                # Drop rows where Open or Close is NaN
                df = df.dropna(subset=['Open', 'Close'])
                self.dataframes[ticker] = df

    def get_tickers(self):
        return list(self.dataframes.keys())

    def get_summary(self, ticker):
        df = self.dataframes.get(ticker)
        if df is None:
            return None
        return {
            'start_date': df['Date'].iloc[0],
            'end_date': df['Date'].iloc[-1],
            'num_days': len(df),
            'mean_open': df['Open'].mean(),
            'mean_close': df['Close'].mean(),
            'max_open': df['Open'].max(),
            'min_open': df['Open'].min(),
            'max_close': df['Close'].max(),
            'min_close': df['Close'].min(),
        }

    def compare_mean_close(self):
        # Returns a dict of ticker: mean_close
        return {ticker: df['Close'].mean() for ticker, df in self.dataframes.items()}

    def plot_ticker(self, ticker):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed. Please install it to plot.")
            return
        df = self.dataframes.get(ticker)
        if df is not None:
            # Clean the data: drop rows with missing Date, ensure Date is str
            df = df.dropna(subset=['Date'])
            df = df[df['Date'].apply(lambda x: isinstance(x, str))]
            if df.empty:
                print(f"No valid date data to plot for {ticker}.")
                return
            # Convert Date to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            # Plot all daily data, but label x-axis by month
            plt.figure(figsize=(12, 6))
            plt.plot(df['Date'], df['Open'], label=f'{ticker} Open', alpha=0.7)
            plt.plot(df['Date'], df['Close'], label=f'{ticker} Close', alpha=0.7)
            plt.title(f'{ticker} Daily Open/Close Prices')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            # Set x-ticks to the first day of each month
            months = pd.date_range(df['Date'].min(), df['Date'].max(), freq='MS')
            plt.xticks(months, [d.strftime('%Y-%m') for d in months], rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print(f"No data found for ticker {ticker}.")

    def predict_linear_regression(self, ticker, days_ahead=7, plot=True):
        """
        Predict future closing prices using a simple linear regression model (PyTorch).
        Returns predicted values for the next `days_ahead` days.
        """
        df = self.dataframes.get(ticker)
        if df is None or len(df) < 2:
            print(f"Not enough data for ticker {ticker}.")
            return None
        # Use only the Close price
        close_prices = df['Close'].values.astype(np.float32)
        X = np.arange(len(close_prices)).reshape(-1, 1).astype(np.float32)
        y = close_prices.reshape(-1, 1)
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        # Define a simple linear regression model with bias (y-intercept)
        model = nn.Linear(1, 1, bias=True)  # bias=True ensures y-intercept is learned from data
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        # Train
        for epoch in range(300):
            model.train()
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
        # Predict future
        future_X = np.arange(len(close_prices) + days_ahead).reshape(-1, 1).astype(np.float32)
        future_X_tensor = torch.from_numpy(future_X)
        with torch.no_grad():
            predicted = model(future_X_tensor).numpy().flatten()
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(pd.to_datetime(df['Date']), close_prices, label='Actual Close')
            future_dates = pd.date_range(df['Date'].iloc[-1], periods=days_ahead+1, freq='B')[1:]
            all_dates = pd.to_datetime(df['Date']).tolist() + list(future_dates)
            plt.plot(all_dates, predicted, label='Linear Regression Prediction', linestyle='--')
            plt.title(f'{ticker} Close Price Prediction (Linear Regression)')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.tight_layout()
            plt.show()
        return predicted[-days_ahead:]

    def predict_lstm(self, ticker, days_ahead=7, plot=True):
        """
        Predict future closing prices using an optimized LSTM model (PyTorch).
        Returns predicted values for the next `days_ahead` days.
        """
        import random
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        df = self.dataframes.get(ticker)
        if df is None or len(df) < 30:
            print(f"Not enough data for ticker {ticker} (need at least 30 days).")
            return None
        close_prices = df['Close'].values.astype(np.float32)
        # Normalize
        mean = close_prices.mean()
        std = close_prices.std()
        norm_prices = (close_prices - mean) / std
        seq_len = 20
        X, y = [], []
        for i in range(len(norm_prices) - seq_len):
            X.append(norm_prices[i:i+seq_len])
            y.append(norm_prices[i+seq_len])
        X = np.array(X)
        y = np.array(y)
        # Validation split
        val_split = int(len(X) * 0.8)
        X_train, X_val = X[:val_split], X[val_split:]
        y_train, y_val = y[:val_split], y[val_split:]
        X_train_tensor = torch.from_numpy(X_train).unsqueeze(-1)
        y_train_tensor = torch.from_numpy(y_train)
        X_val_tensor = torch.from_numpy(X_val).unsqueeze(-1)
        y_val_tensor = torch.from_numpy(y_val)
        class LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(64, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out.squeeze()
        model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        # Train with early stopping
        for epoch in range(500):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_tensor)
            loss = criterion(output, y_train_tensor)
            loss.backward()
            optimizer.step()
            # Validation
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_tensor)
                val_loss = criterion(val_output, y_val_tensor)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break
        # Load best model
        model.load_state_dict(best_model_state)
        # Predict future
        model.eval()
        last_seq = torch.from_numpy(norm_prices[-seq_len:]).unsqueeze(0).unsqueeze(-1)
        preds = []
        for _ in range(days_ahead):
            with torch.no_grad():
                pred = model(last_seq).item()
            preds.append(pred)
            last_seq = torch.cat([last_seq[:, 1:, :], torch.tensor([[[pred]]])], dim=1)
        preds = np.array(preds) * std + mean
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(df['Date'], close_prices, label='Actual Close')
            future_dates = pd.date_range(df['Date'].iloc[-1], periods=days_ahead+1, freq='B')[1:]
            all_dates = pd.to_datetime(df['Date']).tolist() + list(future_dates)
            plt.plot(all_dates[-(days_ahead+len(close_prices)):], np.concatenate([close_prices, preds]), label='LSTM Prediction', linestyle='--')
            plt.title(f'{ticker} Close Price Prediction (LSTM)')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.tight_layout()
            plt.show()
        return preds

    def compare_all_companies(self, days_ahead=7, model='linear'):
        """
        Create a DataFrame comparing all companies' current and predicted future prices, growth, and market average for each day.
        model: 'linear' or 'lstm'
        Returns: pd.DataFrame
        """
        tickers = self.get_tickers()
        results = []
        # Collect all historical dates
        all_dates = set()
        for ticker in tickers:
            df = self.dataframes[ticker]
            all_dates.update(pd.to_datetime(df['Date']))
        all_dates = sorted(list(all_dates))
        # Build a DataFrame with all tickers' historical close prices
        hist_df = pd.DataFrame(index=all_dates)
        for ticker in tickers:
            df = self.dataframes[ticker]
            df = df.dropna(subset=['Date', 'Close'])
            df['Date'] = pd.to_datetime(df['Date'])
            hist_df[ticker] = df.set_index('Date')['Close']
        # Compute market average for each day
        hist_df['Market_Avg'] = hist_df.mean(axis=1)
        # Predict future prices for each ticker
        future_dates = pd.date_range(hist_df.index[-1], periods=days_ahead+1, freq='B')[1:]
        for ticker in tickers:
            if model == 'linear':
                preds = self.predict_linear_regression(ticker, days_ahead=days_ahead, plot=False)
            else:
                preds = self.predict_lstm(ticker, days_ahead=days_ahead, plot=False)
            # Current price is last available close
            current_price = hist_df[ticker].dropna().iloc[-1]
            # Growth = (future - current) / current
            for i, pred in enumerate(preds):
                results.append({
                    'Ticker': ticker,
                    'Date': future_dates[i],
                    'Current_Close': current_price,
                    'Predicted_Close': pred,
                    'Growth_%': 100 * (pred - current_price) / current_price
                })
        # Add market average for future dates
        for i, date in enumerate(future_dates):
            market_pred = np.mean([r['Predicted_Close'] for r in results if r['Date'] == date])
            results.append({
                'Ticker': 'Market_Avg',
                'Date': date,
                'Current_Close': np.nan,
                'Predicted_Close': market_pred,
                'Growth_%': np.nan
            })
        # Combine historical and future into one DataFrame
        future_df = pd.DataFrame(results)
        # For historical, melt hist_df to long format
        hist_long = hist_df.reset_index().melt(id_vars=['index'], var_name='Ticker', value_name='Close')
        hist_long = hist_long.rename(columns={'index': 'Date'})
        hist_long['Type'] = 'Historical'
        future_df['Type'] = 'Predicted'
        future_df = future_df.rename(columns={'Predicted_Close': 'Close'})
        combined = pd.concat([
            hist_long[['Date', 'Ticker', 'Close', 'Type']],
            future_df[['Date', 'Ticker', 'Close', 'Type', 'Current_Close', 'Growth_%']]
        ], ignore_index=True)
        return combined

    def joint_linear_regression_forecast(self, days_ahead=7, plot=True):
        """
        Jointly fit a linear regression model to all companies' close prices to forecast future prices for each company.
        Returns a DataFrame with predicted prices for each ticker and the market average.
        """
        tickers = self.get_tickers()
        # Build a DataFrame with all tickers' historical close prices
        all_dates = set()
        for ticker in tickers:
            df = self.dataframes[ticker]
            all_dates.update(pd.to_datetime(df['Date']))
        all_dates = sorted(list(all_dates))
        hist_df = pd.DataFrame(index=all_dates)
        for ticker in tickers:
            df = self.dataframes[ticker]
            df = df.dropna(subset=['Date', 'Close'])
            df['Date'] = pd.to_datetime(df['Date'])
            hist_df[ticker] = df.set_index('Date')['Close']
        # Align all tickers (forward fill)
        hist_df = hist_df.ffill()
        # Prepare X (time) and Y (all tickers)
        X = np.arange(len(hist_df)).reshape(-1, 1).astype(np.float32)
        Y = hist_df[tickers].values.astype(np.float32)
        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        # Joint linear regression: one model, multiple outputs
        model = nn.Linear(1, len(tickers))
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for epoch in range(400):
            model.train()
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, Y_tensor)
            loss.backward()
            optimizer.step()
        # Predict future
        future_X = np.arange(len(hist_df) + days_ahead).reshape(-1, 1).astype(np.float32)
        future_X_tensor = torch.from_numpy(future_X)
        with torch.no_grad():
            predicted = model(future_X_tensor).numpy()
        # Build result DataFrame
        all_dates_full = list(hist_df.index) + list(pd.date_range(hist_df.index[-1], periods=days_ahead+1, freq='B')[1:])
        pred_df = pd.DataFrame(predicted, columns=tickers, index=all_dates_full)
        pred_df['Market_Avg'] = pred_df[tickers].mean(axis=1)
        if plot:
            plt.figure(figsize=(14, 7))
            for ticker in tickers:
                plt.plot(pred_df.index, pred_df[ticker], label=f'{ticker} (predicted)', linestyle='--')
                plt.plot(hist_df.index, hist_df[ticker], label=f'{ticker} (actual)')
            plt.plot(pred_df.index, pred_df['Market_Avg'], label='Market_Avg (predicted)', color='black', linewidth=2, linestyle=':')
            plt.title('Joint Linear Regression Forecast for All Companies')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend(loc='upper left', ncol=2)
            plt.tight_layout()
            plt.show()
        return pred_df

# Example usage:
# analyzer = StockDataAnalyzer()
# print(analyzer.get_tickers())
# print(analyzer.get_summary('AMZN'))
# analyzer.plot_ticker('AMZN')
# analyzer.predict_linear_regression('AMZN', days_ahead=30)
# analyzer.predict_lstm('AMZN', days_ahead=30)
# comparison_df = analyzer.compare_all_companies(days_ahead=30, model='linear')
# print(comparison_df)
# joint_forecast_df = analyzer.joint_linear_regression_forecast(days_ahead=30)
# print(joint_forecast_df)
