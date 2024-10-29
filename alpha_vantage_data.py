import requests
import pandas as pd

def fetch_live_data(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full'
    response = requests.get(url)
    data = response.json()

    # Check if the response contains an error message
    if "Error Message" in data:
        raise ValueError(f"Error fetching data for symbol {symbol}: {data['Error Message']}")
    if "Information" in data:
        raise ValueError(f"Information: {data['Information']}")

    # Extract time series data
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Unexpected data format for symbol {symbol}")

    time_series = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume"
    })
    df.index.name = 'Date'
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    return df
