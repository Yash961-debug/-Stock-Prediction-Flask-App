# Import necessary libraries for Flask, data handling, and modeling
from flask import Flask, request, jsonify, render_template
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import yfinance as yf
# from yahoo_fin import stock_info as si # Removed direct import to handle AttributeError more cleanly
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress specific warnings from matplotlib and other libraries if desired
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
warnings.filterwarnings("ignore", category=RuntimeWarning) # For potential tkinter warnings
warnings.filterwarnings("ignore", category=FutureWarning) # For potential pandas/statsmodels future warnings

# Initialize Flask app
app = Flask(__name__)

# Define headers for web scraping to mimic a browser request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# --- Stock Prediction Functions (Adapted from your Notebook) ---

def search_for_ticker(query, headers):
    """
    Searches for a stock ticker based on the provided query.
    Returns a list of dictionaries with 'symbol', 'longname', 'exchange' for matching quotes.
    Returns an empty list if no results are found.
    """
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}&lang=en-US&region=US&quotesCount=10&newsCount=0&enableFuzzyQuery=false&quotesQueryId=src_quotes_query_string&multiQuoteQueryId=src_quotes_multiquote_query_string&newsQueryId=src_news_query_string&enableEnhancedSearching=false&enableFuzzyMatching=false&enableResearchReports=false&enableTrendings=false&enableFaves=false"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        quotes = data.get("quotes", [])
        
        # Prepare a list of relevant information for each quote
        options = []
        for quote in quotes:
            symbol = quote.get("symbol")
            longname = quote.get("longname", "")
            exchange = quote.get("exchange", "")
            if symbol: # Only add if a symbol exists
                options.append({
                    "symbol": symbol,
                    "longname": longname,
                    "exchange": exchange
                })
        return options
            
    except requests.exceptions.RequestException as e:
        print(f"Error during ticker search: {e}")
        return [] # Return empty list on error

def get_historical_data(ticker):
    """Fetches historical stock data using yfinance."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2) # Fetch last 2 years of data
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None
        return data['Close'].dropna()
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return None

def get_market_breadth(headers):
    """Fetches US market breadth from Yahoo Finance."""
    try:
        url = "https://finance.yahoo.com/markets/us"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        return 0.5 # Neutral market breadth
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch US market breadth: {e}")
        return 0.5 # Default to neutral

def get_sentiment(ticker, headers):
    """
    Attempts to fetch news sentiment for a given ticker.
    This version tries to use yfinance for news, and gracefully handles
    issues with yahoo_fin.stock_info.get_news by returning neutral sentiment.
    """
    sentiment_score = 0.0 # Default to neutral

    try:
        # First, try to import yahoo_fin.stock_info dynamically
        import importlib
        stock_info = importlib.import_module('yahoo_fin.stock_info')

        # Check if 'get_news' attribute exists before calling it
        if hasattr(stock_info, 'get_news'):
            headlines = stock_info.get_news(ticker)
            if headlines and not headlines.empty:
                for title in headlines['title']:
                    title_lower = title.lower()
                    if 'rise' in title_lower or 'gain' in title_lower or 'up' in title_lower:
                        sentiment_score += 0.1
                    if 'fall' in title_lower or 'drop' in title_lower or 'down' in title_lower:
                        sentiment_score -= 0.1
                sentiment_score = np.clip(sentiment_score, -1, 1)
                # print(f"Sentiment for {ticker}: {sentiment_score:.2f} based on {len(headlines)} headlines (via yahoo_fin).")
                return sentiment_score
            else:
                # print(f"No headlines found for {ticker} via yahoo_fin. Returning neutral sentiment.")
                pass # Proceed to yfinance if no headlines
        else:
            # print("yahoo_fin.stock_info.get_news attribute not found. Falling back to yfinance news.")
            pass # Proceed to yfinance if attribute not found

    except ImportError:
        # print("yahoo_fin not installed. Falling back to yfinance for news.")
        pass # Proceed to yfinance if not installed
    except Exception as e:
        # print(f"-> Error fetching news from yahoo_fin for {ticker}: {e}. Falling back to yfinance news.")
        pass # Proceed to yfinance on other yahoo_fin errors

    # Fallback to yfinance.Ticker for news if yahoo_fin fails or is not preferred
    try:
        ticker_obj = yf.Ticker(ticker)
        news_items = ticker_obj.news
        
        if news_items:
            for item in news_items:
                title = item.get('title', '').lower()
                if 'rise' in title or 'gain' in title or 'up' in title:
                    sentiment_score += 0.1
                if 'fall' in title or 'drop' in title or 'down' in title:
                    sentiment_score -= 0.1
            sentiment_score = np.clip(sentiment_score, -1, 1)
            print(f"Sentiment for {ticker}: {sentiment_score:.2f} based on {len(news_items)} headlines (via yfinance).")
        else:
            print(f"No news found for {ticker} via yfinance. Returning neutral sentiment.")

    except Exception as e:
        print(f"-> Error fetching news from yfinance for {ticker}: {e}. Returning neutral sentiment.")
        sentiment_score = 0.0 # Ensure neutral sentiment on any failure

    return sentiment_score


def predict_stock_price(ticker, headers):
    """
    Predicts stock price using a hybrid ARIMAX-LSTM model.
    """
    historical_data = get_historical_data(ticker)
    if historical_data is None or historical_data.empty:
        return None, None

    # Ensure data is sorted by date
    historical_data = historical_data.sort_index()

    # --- FIX: Ensure DatetimeIndex has frequency for statsmodels ARIMA ---
    # Reindex to a daily frequency to explicitly set it, filling missing dates if any
    # This also helps with the ValueError: "A date index has been provided..."
    historical_data = historical_data.asfreq('D') # Set frequency to daily
    historical_data = historical_data.fillna(method='ffill').fillna(method='bfill') # Fill NaNs after reindexing

    # Feature Engineering for external features
    market_breadth = get_market_breadth(headers)
    sentiment_score = get_sentiment(ticker, headers)

    exog_data = pd.DataFrame(index=historical_data.index)
    exog_data['market_breadth'] = market_breadth
    exog_data['sentiment'] = sentiment_score

    print("Fitting ARIMAX model with external features...")
    try:
        model_arima = ARIMA(historical_data, exog=exog_data, order=(5,1,0))
        model_arima_fit = model_arima.fit()
        arima_residuals = model_arima_fit.resid

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_residuals = scaler.fit_transform(arima_residuals.values.reshape(-1, 1))

        X, y = [], []
        timesteps = 10 # Number of previous days to consider for LSTM prediction
        for i in range(len(scaled_residuals) - timesteps):
            X.append(scaled_residuals[i:(i + timesteps), 0])
            y.append(scaled_residuals[i + timesteps, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        if X.shape[0] == 0:
            print("Not enough data to train LSTM after differencing. Skipping LSTM.")
            # Ensure future_exog_data is created correctly for ARIMA-only forecast
            future_exog_data_for_arima = pd.DataFrame(index=pd.date_range(start=historical_data.index[-1] + timedelta(days=1), periods=5))
            future_exog_data_for_arima['market_breadth'] = market_breadth
            future_exog_data_for_arima['sentiment'] = sentiment_score
            forecast_arima = model_arima_fit.predict(start=len(historical_data), end=len(historical_data) + 4, exog=future_exog_data_for_arima)
            return historical_data, forecast_arima.values
            
        print("Training LSTM model on ARIMAX residuals...")
        model_lstm = Sequential()
        model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model_lstm.add(LSTM(units=50))
        model_lstm.add(Dense(units=1))
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        model_lstm.fit(X, y, epochs=50, batch_size=32, verbose=0)

        last_timesteps = scaled_residuals[-timesteps:]
        future_residuals_scaled = []
        current_batch = last_timesteps.reshape((1, timesteps, 1))

        for i in range(5): # Forecast 5 days
            future_pred_scaled = model_lstm.predict(current_batch, verbose=0)[0]
            future_residuals_scaled.append(future_pred_scaled)
            current_batch = np.append(current_batch[:, 1:, :], [[future_pred_scaled]], axis=1)

        future_residuals = scaler.inverse_transform(np.array(future_residuals_scaled).reshape(-1, 1)).flatten()

        # Create future external features for ARIMAX forecast
        future_exog_data = pd.DataFrame(index=pd.date_range(start=historical_data.index[-1] + timedelta(days=1), periods=5, freq='D'))
        future_exog_data['market_breadth'] = market_breadth
        future_exog_data['sentiment'] = sentiment_score

        arima_forecast = model_arima_fit.predict(start=len(historical_data), end=len(historical_data) + 4, exog=future_exog_data)

        final_forecast = arima_forecast.values + future_residuals

        return historical_data, final_forecast

    except Exception as e:
        print(f"Error during model training or prediction for {ticker}: {e}")
        return historical_data, None


def plot_predictions(ticker, historical_data, forecast):
    """
    Plots historical data and future predictions.
    Saves the plot to a BytesIO object and encodes it to base64.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(12, 6))

    plt.plot(historical_data.index, historical_data, label=f'{ticker} Historical Price', color='blue', linewidth=1.5)

    if forecast is not None:
        last_date = historical_data.index[-1]
        # Ensure forecast_dates generation respects the frequency if historical_data is sparse
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(forecast), freq='D')
        plt.plot(forecast_dates, forecast, label=f'{ticker} 5-Day Forecast', color='orange', linestyle='--', marker='o', markersize=5)

    plt.title(f'{ticker} Stock Price Prediction', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles stock prediction requests.
    - If 'selected_ticker' is present, it proceeds with prediction for that ticker.
    - If only 'stock_name' is present, it searches for tickers and returns options.
    """
    stock_name = request.form.get('stock_name')
    selected_ticker = request.form.get('selected_ticker')

    if not stock_name and not selected_ticker:
        return jsonify(
            success=False, 
            message="Please enter a stock or index name."
        )
    
    if selected_ticker:
        # User has selected a ticker from the options, proceed with prediction
        ticker_to_predict = selected_ticker
    else:
        # Initial search query, find matching tickers
        options = search_for_ticker(stock_name, HEADERS)
        
        if not options:
            return jsonify(
                success=False,
                message=f"No results found for '{stock_name}'. Please try another term.",
                type="no_options"
            )
        elif len(options) == 1:
            # If only one option, proceed directly with prediction
            ticker_to_predict = options[0]['symbol']
            print(f"Only one option found: {ticker_to_predict}. Proceeding with prediction.")
        else:
            # If multiple options, return them to the frontend for selection
            print(f"Multiple options found for '{stock_name}'. Returning options to frontend.")
            return jsonify(
                success=True,
                type="options",
                options=options,
                original_query=stock_name # Send original query back for context if needed
            )

    # Proceed with prediction using ticker_to_predict
    historical_data, forecast = predict_stock_price(ticker_to_predict, HEADERS)
    if forecast is not None:
        forecast_list = [f"{val:.2f}" for val in forecast]
        plot_base64 = plot_predictions(ticker_to_predict, historical_data, forecast)
        
        return jsonify(
            success=True,
            type="prediction", # Indicate that this is a final prediction
            ticker=ticker_to_predict,
            forecast=forecast_list,
            plot_image=plot_base64
        )
    else:
        return jsonify(
            success=False, 
            message=f"Could not generate forecast for '{ticker_to_predict}'. Check ticker or data availability or try a different ticker.",
            type="prediction_failed"
        )

if __name__ == '__main__':
    app.run(debug=True, port=8000)
