import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

def preprocess_data(df):
    # Ensure all data is numeric and handle non-numeric values
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close'])

    # Add technical indicators
    df['SMA'] = ta.sma(df['close'], length=20)  # Simple Moving Average
    df['EMA'] = ta.ema(df['close'], length=20)  # Exponential Moving Average
    df['RSI'] = ta.rsi(df['close'], length=14)  # Relative Strength Index

    # Drop rows with NaN values (due to indicator calculations)
    df = df.dropna()

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['close', 'SMA', 'EMA', 'RSI']])
    return scaled_data, scaler

def create_sequences(data, sequence_length=60):
    # preparing sequences for LSTM model
    sequences = []
    targets = []
    for i in range(sequence_length, len(data)):
        sequences.append(data[i-sequence_length:i, :])
        targets.append(data[i, 0])
    X = np.array(sequences)
    y = np.array(targets)
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
    return X, y

def build_bidirectional_lstm_model(input_shape, units=50, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=units, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(units=units, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(units=units, return_sequences=False)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_lstm_model(X, y, input_shape, epochs=50, batch_size=32):
    model = build_bidirectional_lstm_model(input_shape)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model.save('lstm_model.h5')
    return model
