import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

class StockDataset:
    """
    Loads stock data from Yahoo Finance, scales it, and creates input sequences.
    Ensures single-level columns: Date, Open, High, Low, Close, Volume.
    """
    def __init__(self, 
                 ticker, 
                 start_date="2018-01-01", 
                 end_date="2022-12-31", 
                 window_size=30, 
                 save_local=False, 
                 data_dir="data/raw"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.save_local = save_local
        self.data_dir = data_dir

        # Download and prepare data
        self.data = self.download_data()
        self.scaler, self.scaled_data = self.scale_data(self.data)

    def download_data(self):
        """
        Download OHLCV data from Yahoo Finance and return a DataFrame
        with single-level columns: Date, Open, High, Low, Close, Volume.
        """
        # Pass a single ticker string instead of a list to reduce multi-level columns.
        df = yf.download(
            self.ticker, 
            start=self.start_date, 
            end=self.end_date
        )

        # Move the datetime index to a normal column
        df.reset_index(inplace=True)

        # If yfinance still returns a MultiIndex, flatten it:
        if isinstance(df.columns, pd.MultiIndex):
            # Convert multi-index columns into a single flat index, e.g. ("Close","AAPL") -> "Close_AAPL"
            df.columns = df.columns.to_flat_index()
            df.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col 
                          for col in df.columns]

        # At this point, you may see columns like: "Date", "Close_AAPL", "Open_AAPL", ...
        # Let's rename them to remove the ticker suffix if present.
        def strip_ticker_suffix(col: str):
            """
            If a column name ends with '_XXXX' where XXXX might be the ticker,
            split on '_' and keep only the first part (e.g. 'Close_AAPL' -> 'Close').
            Otherwise return the original col name.
            """
            parts = col.split("_")
            if len(parts) > 1 and parts[-1].upper() == self.ticker.upper():
                return "_".join(parts[:-1])  # drop the last part
            else:
                return col

        # Apply to all columns
        df.columns = [strip_ticker_suffix(col) for col in df.columns]

        # Typically from yfinance we want: Date, Open, High, Low, Close, Volume (and maybe Adj Close)
        # If there's an "Adj Close" column, you can rename or drop it as needed.
        if "Adj Close" in df.columns:
            df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)

        # Now let's check the columns we need
        expected_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing expected columns {missing}. "
                f"Got columns: {df.columns.tolist()}"
            )

        # Optionally save the raw data
        if self.save_local:
            os.makedirs(self.data_dir, exist_ok=True)
            raw_file = os.path.join(self.data_dir, f"{self.ticker}_raw.csv")
            df.to_csv(raw_file, index=False)

        return df

    def scale_data(self, df):
        """
        Scale numeric columns (Open/High/Low/Close/Volume) with MinMaxScaler.
        Returns (scaler, scaled_values).
        """
        scaler = MinMaxScaler()
        # We'll select only the numeric columns
        numeric_data = df[["Open", "High", "Low", "Close", "Volume"]].values
        scaled_values = scaler.fit_transform(numeric_data)
        return scaler, scaled_values

    def create_sequences(self):
        """
        Create (X, y) for training:
        - X is a window of length self.window_size
        - y is the next day's Close
        """
        X, y = [], []
        close_idx = 3  # 'Close' is the 4th column among [Open, High, Low, Close, Volume]

        for i in range(len(self.scaled_data) - self.window_size):
            seq_x = self.scaled_data[i : i + self.window_size]
            seq_y = self.scaled_data[i + self.window_size, close_idx]  # Next day's Close
            X.append(seq_x)
            y.append(seq_y)

        return np.array(X), np.array(y)
