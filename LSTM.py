import numpy as np
import pandas as pd
from binance.client import Client
import talib  # For technical indicators
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Binance API keys (you will need to get your own keys from Binance)
api_key = 'your_api_key'
api_secret = 'your_api_secret'

# Initialize Binance client
client = Client(api_key, api_secret, tld='us')

# Get millisecond-level historical data for Bitcoin (BTC/USDT pair)
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 Jan, 2021", "31 Dec, 2021")

# Convert the kline data into a DataFrame
btc_data = pd.DataFrame(klines, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base', 'Taker_buy_quote', 'Ignore'])

# Convert prices to float
btc_data['Close'] = btc_data['Close'].astype(float)
btc_data['Open'] = btc_data['Open'].astype(float)
btc_data['High'] = btc_data['High'].astype(float)
btc_data['Low'] = btc_data['Low'].astype(float)
btc_data['Volume'] = btc_data['Volume'].astype(float)

# Set timestamp as index
btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'], unit='ms')
btc_data.set_index('timestamp', inplace=True)

# Feature engineering for high-frequency data
btc_data['Return'] = np.log(btc_data['Close'] / btc_data['Close'].shift(1))
btc_data['SMA_5'] = btc_data['Close'].rolling(window=5).mean()
btc_data['EMA_5'] = btc_data['Close'].ewm(span=5, adjust=False).mean()
btc_data['RSI'] = talib.RSI(btc_data['Close'], timeperiod=14)
btc_data['BB_upper'], btc_data['BB_middle'], btc_data['BB_lower'] = talib.BBANDS(btc_data['Close'], timeperiod=20)

# Drop NaN values after technical indicators calculation
btc_data.dropna(inplace=True)

# Define target variable: 1 if next tick's return is positive, 0 otherwise
btc_data['Target'] = np.where(btc_data['Return'].shift(-1) > 0, 1, 0)

# Drop NaN from target column
btc_data.dropna(inplace=True)

# Prepare the feature set
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'EMA_5', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower']
X = btc_data[features].values
y = btc_data['Target'].values

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape the data for LSTM (samples, timesteps, features)
# We will use a window size (timesteps) of 60 (i.e., the past 60 minutes) to predict the next price movement
window_size = 60
X_lstm = []
y_lstm = []

for i in range(window_size, len(X_scaled)):
    X_lstm.append(X_scaled[i-window_size:i])  # Past window_size time steps
    y_lstm.append(y[i])  # Target is the next time step

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# Train/test split (80/20) ensuring no shuffling to preserve time order
split = int(0.8 * len(X_lstm))
X_train, X_test = X_lstm[:split], X_lstm[split:]
y_train, y_test = y_lstm[:split], y_lstm[split:]

# Build the LSTM model
model = Sequential()

# LSTM layers with dropout to prevent overfitting
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))

# Output layer (binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Predict probabilities for AUC calculation
y_pred_prob = model.predict(X_test)

# Predict classes for accuracy calculation
y_pred_class = (y_pred_prob > 0.5).astype(int)

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_prob)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_class)

# Print the results
print(f"AUC: {auc:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Training and Test Accuracy for LSTM Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
