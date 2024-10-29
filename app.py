import os
import numpy as np
from flask import Flask, render_template, request
from lstm_model import preprocess_data, create_sequences, train_lstm_model
from alpha_vantage_data import fetch_live_data
from tensorflow.keras.models import load_model
from database import store_prediction, query_db
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Load your Alpha Vantage API key from environment variable
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

if not API_KEY:
    raise ValueError("Please set the ALPHA_VANTAGE_API_KEY environment variable in the .env file.")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    symbol = None

    if request.method == 'POST':
        symbol = request.form.get('symbol', '').upper().strip()

        if not symbol:
            error = "Please enter a valid stock symbol."
            return render_template('index.html', prediction=None, error=error)

        try:
            # Fetch live stock data using Alpha Vantage
            df = fetch_live_data(symbol, API_KEY)
            if df.empty:
                raise ValueError(f"No data found for symbol: {symbol}")

            # Preprocess data
            scaled_data, scaler = preprocess_data(df)

            # Create sequences
            sequence_length = 60
            X, y = create_sequences(scaled_data, sequence_length=sequence_length)

            # Check if model exists; if not, train a new one
            model_path = 'lstm_model.h5'
            if os.path.exists(model_path):
                model = load_model(model_path)
            else:
                input_shape = (X.shape[1], X.shape[2])
                model = train_lstm_model(X, y, input_shape)

            # Make prediction
            last_sequence = scaled_data[-sequence_length:]
            last_sequence = np.reshape(last_sequence, (1, sequence_length, last_sequence.shape[1]))  # Adjust shape to match indicators
            predicted_price = model.predict(last_sequence)
            
            # Create a placeholder for the full feature set to inverse transform
            full_feature_set = np.zeros((predicted_price.shape[0], scaled_data.shape[1]))
            full_feature_set[:, 0] = predicted_price[:, 0]
            predicted_price = scaler.inverse_transform(full_feature_set)[:, 0]
            
            prediction = round(float(predicted_price[0]), 2)

            # Store the prediction in the database
            current_date = datetime.now().strftime('%Y-%m-%d')
            store_prediction(symbol, prediction, current_date)

        except Exception as e:
            error = str(e)
            return render_template('index.html', prediction=None, error=error)

    return render_template('index.html', prediction=prediction, error=error, symbol=symbol)

@app.route('/history')
def history():
    predictions = query_db('SELECT symbol, predicted_price, timestamp FROM predictions ORDER BY timestamp DESC')
    return render_template('history.html', predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)
