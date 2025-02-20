import streamlit as st
import pandas as pd
import os
import plotly.express as px
import torch

# Import modules from our src/ directory
from src.dataset import StockDataset
from src.train import run_training
from src.inference import run_inference

def main():
    st.title("Transformer-Based Stock Forecasting")

    # Sidebar for user inputs
    st.sidebar.header("1. Select Ticker & Date Range")
    ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL):", value="AAPL")
    start_date = st.sidebar.date_input("Start date:", value=pd.to_datetime("2018-01-01"))
    end_date = st.sidebar.date_input("End date:", value=pd.to_datetime("2022-12-31"))
    
    if st.sidebar.button("Fetch Data"):
        # Download data from Yahoo Finance using StockDataset
        dataset = StockDataset(
            ticker=ticker, 
            start_date=str(start_date), 
            end_date=str(end_date),
            window_size=30, 
            save_local=True  # optionally save local CSV
        )
        
        # Display the head of the data
        st.write(f"## Raw Data for {ticker}")
        st.dataframe(dataset.data.head(10))
        
        # Plot closing price
        fig = px.line(dataset.data, x="Date", y="Close", title="AAPL Closing Price")
        st.plotly_chart(fig)
        
        st.success("Data fetched and displayed successfully.")

    # Option to run training
    st.sidebar.header("2. Train Model")
    train_button = st.sidebar.button("Train Transformer Model")
    if train_button:
        with st.spinner("Training the model..."):
            run_training(ticker=ticker)  # uses the default hyperparameters from train.py
        st.success("Training complete! Model weights saved in 'models' directory.")

    # Option to run inference
    st.sidebar.header("3. Run Inference")
    predict_button = st.sidebar.button("Predict Next Day's Close")

    if predict_button:
        with st.spinner("Running inference..."):
            # IMPORTANT: capture the return value
            predicted_price = run_inference(ticker=ticker)

        # Now predicted_price is in the same scope
        st.success(f"The predicted next day’s close for {ticker} is **${predicted_price:.2f}**")

if __name__ == "__main__":
    main()
