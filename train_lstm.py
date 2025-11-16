import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

print("Starting LSTM model training...")

# --- 1. Load Prepared Data ---
try:
    df = pd.read_csv('cash_flow_prepared.csv', parse_dates=['date'], index_col='date')
except FileNotFoundError:
    print("Error: 'cash_flow_prepared.csv' not found. Did you run preprocess.py?")
    exit()

# --- 2. Scale the Data ---
# Neural networks need data scaled between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# We also need a separate scaler for *just* the target variable ('cash_flow')
# This is so we can "inverse transform" our predictions back to real dollars
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(df[['cash_flow']])

# --- 3. Create Time-Series Sequences ---
# We will use the last 'n_steps' days to predict the next day

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        # Find the end of this sequence
        end_ix = i + n_steps
        # Check if we are beyond the dataset
        if end_ix > len(data) - 1:
            break
        # Gather input (X) and output (y) parts of the sequence
        # X is all features for 'n_steps' days
        # y is *only* the 'cash_flow' (at index 0) for the day *after* the sequence
        seq_x, seq_y = data[i:end_ix, :], data[end_ix, 0] 
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

n_steps = 30  # We'll use 30 days of history to predict the next day
X_seq, y_seq = create_sequences(scaled_data, n_steps)

# --- 4. Split Data into Train and Test ---
# We'll use the same 180-day split as before
# We need to calculate the split point based on the *original* df length
test_periods = 180
train_split_index = len(df) - test_periods - n_steps

X_train, X_test = X_seq[:train_split_index], X_seq[train_split_index:]
y_train, y_test = y_seq[:train_split_index], y_seq[train_split_index:]

print(f"Training sequences: {len(X_train)}")
print(f"Testing sequences: {len(X_test)}")

# --- 5. Build the LSTM Model ---
print("Building LSTM model...")
# The shape of our input is (30 days, 3 features)
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(n_steps, X_seq.shape[2])))
model.add(Dense(units=1)) # Output layer: 1 neuron to predict the one 'cash_flow' value

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# --- 6. Train the Model ---
print("Training model... (This will take a few minutes)")
# We'll train for 50 epochs (cycles through the data)
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# --- 7. Save the Model ---
model.save('lstm_model.keras') # Keras models are saved this way
print("Model saved to 'lstm_model.keras'")
# We also save the target scaler, we need it for predictions
joblib.dump(target_scaler, 'lstm_target_scaler.joblib')
print("Scaler saved to 'lstm_target_scaler.joblib'")

# --- 8. Make Predictions and Save ---
print("Generating forecasts...")
# Get the model's predictions (which are still scaled 0-1)
test_predictions_scaled = model.predict(X_test)

# Inverse transform the predictions to get real dollar amounts
test_predictions = target_scaler.inverse_transform(test_predictions_scaled)

# Inverse transform the actuals to get real dollar amounts
y_test_real = target_scaler.inverse_transform(y_test.reshape(-1, 1))

# Save forecasts to a CSV
forecast_df = pd.DataFrame({
    # We have to get the correct dates from the original dataframe
    'date': df.index[-len(y_test_real):], 
    'actual_cash_flow': y_test_real.flatten(),
    'forecast_cash_flow': test_predictions.flatten()
})
forecast_df.to_csv('lstm_forecasts.csv', index=False)
print(f"Forecasts saved to 'lstm_forecasts.csv'")

print("\n--- Phase 4 Complete! ---")