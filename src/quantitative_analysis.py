import pandas as pd
import os
import talib
import matplotlib.pyplot as plt

# Define the path to the stock data folder
data_path = 'C:/Users/neba/Documents/Kifiya/yfinance_data/yfinance_data'

# List of stock data files
stock_files = [
    'AAPL_historical_data.csv',
    'AMZN_historical_data.csv',
    'GOOG_historical_data.csv',
    'META_historical_data.csv',
    'MSFT_historical_data.csv',
    'NVDA_historical_data.csv',
    'TSLA_historical_data.csv'
]

# Load each stock data file into a pandas DataFrame
stock_data = {}
for file in stock_files:
    stock_name = os.path.splitext(file)[0]
    stock_data[stock_name] = pd.read_csv(os.path.join(data_path, file))

# Display the first few rows of each DataFrame
for stock_name, df in stock_data.items():
    print(f"\n{stock_name} Data:\n", df.head())

# Apply TA-Lib indicators
for stock_name, df in stock_data.items():
    # Convert Date to datetime and set as index for time series analysis
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Calculate moving averages
    df['SMA'] = talib.SMA(df['Close'], timeperiod=30)
    df['EMA'] = talib.EMA(df['Close'], timeperiod=30)

    # Calculate RSI (Relative Strength Index)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

    # Calculate MACD (Moving Average Convergence Divergence)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # Display the modified DataFrame for verification
    print(f"\n{stock_name} Data with Indicators:\n", df[['Close', 'SMA', 'EMA', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']].head())

    # Save the modified data to a new CSV file
    output_file = f"{stock_name}_with_indicators.csv"
    df.to_csv(os.path.join(data_path, output_file))
    print(f"Saved {stock_name} data with indicators to {output_file}")

    # Plot the stock data and indicators
    plt.figure(figsize=(12, 6))

    # Plot Close Price with SMA and EMA
    plt.subplot(2, 1, 1)
    plt.plot(df['Close'], label='Close Price', color='blue')
    plt.plot(df['SMA'], label='SMA (30)', color='orange')
    plt.plot(df['EMA'], label='EMA (30)', color='green')
    plt.title(f'{stock_name} - Close Price, SMA, and EMA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Plot RSI
    plt.subplot(2, 1, 2)
    plt.plot(df['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--')  # Overbought threshold
    plt.axhline(30, color='green', linestyle='--')  # Oversold threshold
    plt.title(f'{stock_name} - RSI')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()
