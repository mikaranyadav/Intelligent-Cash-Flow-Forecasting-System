import pandas as pd
import pmdarima as pm
from pmdarima import model_selection
import joblib

print("Starting ARIMA model training...")

# --- 1. Load Prepared Data ---
try:
    df = pd.read_csv('cash_flow_prepared.csv', parse_dates=['date'], index_col='date')
except FileNotFoundError:
    print("Error: 'cash_flow_prepared.csv' not found. Did you run preprocess.py?")
    exit()

# --- 2. Define Variables and Split Data ---
# Our target variable (what we want to predict)
target = 'cash_flow'
# Our external "exogenous" features
features = ['inflation_rate', 'interest_rate']

# We'll use the last 180 days (approx. 6 months) for testing
test_periods = 180
train_df = df.iloc[:-test_periods]
test_df = df.iloc[-test_periods:]

print(f"Training data size: {len(train_df)} days")
print(f"Testing data size: {len(test_df)} days")

# --- 3. Find Best Model with Auto-ARIMA ---
print("Running auto_arima... (This may take a few minutes)")

# This is the "brain" of our process
# It will automatically test different models to find the best one.
# m=7 hints that there might be a 7-day (weekly) seasonal pattern.
auto_model = pm.auto_arima(
    y=train_df[target],             # Our target data
    X=train_df[features],           # Our external indicators
    m=7,                            # We hint at a weekly seasonality
    seasonal=True,                  # We want to check for seasonal patterns
    stepwise=True,                  # Speeds up the search
    suppress_warnings=True,         # Hides convergence warnings
    error_action='ignore'           # Skips models that fail
)

print("\n--- Auto-ARIMA Model Summary ---")
print(auto_model.summary())

# --- 4. Train the Final Model ---
# auto_arima already returns a fitted model, but if we needed to re-fit:
# auto_model.fit(y=train_df[target], X=train_df[features])

# --- 5. Make Forecasts on the Test Set ---
print(f"\nGenerating forecasts for {len(test_df)} periods...")
forecasts, conf_int = auto_model.predict(
    n_periods=len(test_df),
    X=test_df[features],            # We must provide the *future* indicators
    return_conf_int=True
)

# --- 6. Save Model and Forecasts ---

# Save the trained model to a file
joblib.dump(auto_model, 'arima_model.joblib')
print(f"Model saved to 'arima_model.joblib'")

# Save forecasts to a CSV file for review
forecast_df = pd.DataFrame({
    'date': test_df.index,
    'actual_cash_flow': test_df[target].values,  # Add .values
    'forecast_cash_flow': forecasts.values       # Add .values
})
forecast_df.to_csv('arima_forecasts.csv', index=False)
print(f"Forecasts saved to 'arima_forecasts.csv'")

print("\n--- Phase 3 Complete! ---")