
import argparse
from src import train, inference

def main():
    parser = argparse.ArgumentParser(description="Transformer Stock Forecasting")
    parser.add_argument("--mode", type=str, required=True, 
                        help="Mode to run: 'train' or 'inference'")
    parser.add_argument("--ticker", type=str, default="AAPL",
                        help="Stock ticker symbol (e.g., 'AAPL', 'MSFT')")
    args = parser.parse_args()
    
    if args.mode == "train":
        train.run_training(args.ticker)
    elif args.mode == "inference":
        inference.run_inference(args.ticker)
    else:
        raise ValueError("Unknown mode. Use 'train' or 'inference'.")

if __name__ == "__main__":
    main()
