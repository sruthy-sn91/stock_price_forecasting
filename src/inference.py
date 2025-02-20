import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import StockDataset
from src.model import TimeSeriesTransformer

def run_inference(ticker="AAPL"):
    # Hyperparameters: must match what was used in training
    window_size = 30
    model_load_path = f"models/{ticker}_transformer.pth"

    # Load data
    dataset = StockDataset(ticker=ticker, window_size=window_size)
    X, y = dataset.create_sequences()  # entire dataset
    close_idx = 3  # 'Close' is 4th column in [Open, High, Low, Close, Volume]

    # Last sample in X is the latest window
    last_sequence = X[-1:]  # shape: (1, window_size, num_features)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(num_features=X.shape[2], d_model=64, nhead=4, num_layers=2)
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model.eval().to(device)

    # Run inference
    last_sequence_tensor = torch.from_numpy(last_sequence).float().to(device)
    with torch.no_grad():
        pred_scaled = model(last_sequence_tensor).cpu().numpy()  # shape: (1,1)
    
    # Reverse scaling for the 'Close' price
    # We can do it by building a dummy array
    # with the scaled prediction in the 'Close' column index,
    # then applying inverse_transform from the original scaler.
    dummy_row = np.zeros((1, 5))  # [Open, High, Low, Close, Volume]
    dummy_row[0, close_idx] = pred_scaled[0, 0]

    inverted_pred = dataset.scaler.inverse_transform(dummy_row)[0, close_idx]
    #print(f"Next day predicted close price for {ticker}: {inverted_pred:.2f}")
    return inverted_pred
