import numpy as np
import pandas as pd
from binance.client import Client
import talib  # For technical indicators
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

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

# Use TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

auc_scores = []
accuracy_scores = []

# Function to build the MLP model
def create_model(input_dim):
    model = Sequential()

    # Input layer with regularization and batch normalization
    model.add(Dense(512, input_dim=input_dim, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Hidden layers
    model.add(Dense(256, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(128, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    optimizer = Adam(learning_rate=0.00001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# TimeSeriesSplit cross-validation
for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Create a new instance of the model for each fold
    model = create_model(input_dim=X_train.shape[1])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)

    # Predict probabilities for ROC AUC calculation
    y_pred_prob = model.predict(X_test)

    # Predict classes for accuracy calculation
    y_pred_class = (y_pred_prob > 0.5).astype(int)

    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred_prob)
    auc_scores.append(auc)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_class)
    accuracy_scores.append(accuracy)

    print(f"AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")

# Print the average AUC and Accuracy
print(f'Average AUC: {np.mean(auc_scores):.4f}')
print(f'Average Accuracy: {np.mean(accuracy_scores) * 100:.2f}%')
