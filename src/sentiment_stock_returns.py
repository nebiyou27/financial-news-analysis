import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load news data
news_file_path = r'C:\Users\neba\Documents\raw_analyst_ratings.csv'
try:
    news_data = pd.read_csv(news_file_path)
    # Strip any leading/trailing spaces from column names
    news_data.columns = news_data.columns.str.strip()
except PermissionError:
    print(f"Permission denied while trying to read the file at {news_file_path}. Please check the file permissions.")
    exit()  # Exit the script if the file cannot be loaded
except FileNotFoundError:
    print(f"The file at {news_file_path} was not found. Please check the file path.")
    exit()  # Exit the script if the file is not found
except Exception as e:
    print(f"Error loading news data: {e}")
    exit()  # Exit the script if any other error occurs

# Normalize the 'date' column in news data and remove timezone information
news_data['Date'] = pd.to_datetime(news_data['date'], errors='coerce').dt.tz_localize(None)

# Check the first few rows to verify the data
print(news_data[['Date', 'headline', 'url', 'publisher', 'stock']].head())

# List of stock files
stock_files = {
    'AAPL': r'C:\Users\neba\Documents\Kifiya\yfinance_data\yfinance_data\AAPL_historical_data.csv',
    'AMZN': r'C:\Users\neba\Documents\Kifiya\yfinance_data\yfinance_data\AMZN_historical_data.csv',
    'GOOG': r'C:\Users\neba\Documents\Kifiya\yfinance_data\yfinance_data\GOOG_historical_data.csv',
    'META': r'C:\Users\neba\Documents\Kifiya\yfinance_data\yfinance_data\META_historical_data.csv',
    'MSFT': r'C:\Users\neba\Documents\Kifiya\yfinance_data\yfinance_data\MSFT_historical_data.csv',
    'NVDA': r'C:\Users\neba\Documents\Kifiya\yfinance_data\yfinance_data\NVDA_historical_data.csv',
    'TSLA': r'C:\Users\neba\Documents\Kifiya\yfinance_data\yfinance_data\TSLA_historical_data.csv'
}

# Load all stock data into a dictionary
stock_data = {}
for stock, file in stock_files.items():
    try:
        stock_data[stock] = pd.read_csv(file)
    except Exception as e:
        print(f"Error loading stock data for {stock}: {e}")

# Normalize dates in stock data and remove timezone information
for stock in stock_data:
    stock_data[stock]['Date'] = pd.to_datetime(stock_data[stock]['Date'], errors='coerce').dt.tz_localize(None)
    stock_data[stock] = stock_data[stock].sort_values('Date')

# Function to calculate sentiment score
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

# Apply sentiment analysis to news headlines
news_data['Sentiment'] = news_data['headline'].apply(get_sentiment)

# Display sentiment scores
print(news_data[['Date', 'headline', 'Sentiment']].head())

# Group news data by date and calculate average sentiment
daily_sentiment = news_data.groupby('Date')['Sentiment'].mean().reset_index()
daily_sentiment.rename(columns={'Sentiment': 'Avg_Sentiment'}, inplace=True)

# Display aggregated sentiment scores
print(daily_sentiment.head())

# Calculate daily returns and merge with sentiment
for stock in stock_data:
    stock_data[stock]['Daily_Return'] = stock_data[stock]['Close'].pct_change() * 100
    # Merge with daily sentiment
    stock_data[stock] = pd.merge(stock_data[stock], daily_sentiment, on='Date', how='inner')
    print(f"{stock} data with daily returns and sentiment:")
    print(stock_data[stock][['Date', 'Close', 'Daily_Return', 'Avg_Sentiment']].head())

# Calculate correlation for each stock
for stock in stock_data:
    correlation = stock_data[stock]['Daily_Return'].corr(stock_data[stock]['Avg_Sentiment'])
    print(f"Correlation between sentiment and daily returns for {stock}: {correlation:.4f}")

# Plot sentiment vs stock return for one stock (e.g., TSLA)
stock_name = 'TSLA'
plt.scatter(stock_data[stock_name]['Avg_Sentiment'], stock_data[stock_name]['Daily_Return'])
plt.xlabel('Average Sentiment Score')
plt.ylabel('Daily Stock Return (%)')
plt.title(f'Sentiment vs Stock Return for {stock_name}')
plt.show()
