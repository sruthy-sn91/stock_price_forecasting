import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from src.dataset import StockDataset
from src.model import TimeSeriesTransformer

def run_training(ticker="AAPL"):
    # Hyperparameters
    window_size = 30
    batch_size = 32
    epochs = 20
    learning_rate = 1e-3
    model_save_path = f"models/{ticker}_transformer.pth"

    # 1. Prepare Dataset
    dataset = StockDataset(ticker=ticker, window_size=window_size, save_local=True)
    X, y = dataset.create_sequences()

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                      test_size=0.2, 
                                                      shuffle=False)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).float()

    # DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(num_features=X.shape[2], d_model=64, nhead=4, num_layers=2)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Training Loop
    best_val_loss = float("inf")
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_features.size(0)
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device).unsqueeze(1)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item() * batch_features.size(0)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  [*] Model saved at epoch {epoch+1} with val_loss {avg_val_loss:.4f}")

    print("Training complete.")

