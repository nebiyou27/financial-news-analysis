import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load the data
data = pd.read_csv('C:/Users/neba/Documents/raw_analyst_ratings.csv')

# Display the first few rows
print(data.head())

# Descriptive Statistics
# Length of headlines
data['headline_length'] = data['headline'].apply(len)

# Descriptive statistics for headline lengths
print(data['headline_length'].describe())

# Count articles per publisher
publisher_counts = data['publisher'].value_counts()
print(publisher_counts)

# Sentiment Analysis
# Function to get sentiment polarity
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply sentiment analysis
data['sentiment'] = data['headline'].apply(get_sentiment)
print(data[['headline', 'sentiment']].head())

# Time Series Analysis
# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d %H:%M:%S", errors='coerce')

# Drop rows with invalid dates (if any)
data = data.dropna(subset=['date'])

# Plot the number of articles over time
data.set_index('date').resample('D').size().plot(title='Articles Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.show()

# Publisher Analysis
# Top publishers
print(publisher_counts.head())
