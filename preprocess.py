import pandas as pd
import numpy as np

# --- 1. Load Datasets ---
print("Loading datasets...")
try:
    cash_flow_df = pd.read_csv('historical_transactions.csv')
    econ_df = pd.read_csv('economic_indicators.csv')
except FileNotFoundError:
    print("Error: Make sure 'historical_transactions.csv' and 'economic_indicators.csv' are in the same folder.")
    exit()

# --- 2. Process Cash Flow Data ---
print("Processing cash flow data...")
# Convert 'date' to datetime objects
cash_flow_df['date'] = pd.to_datetime(cash_flow_df['date'], dayfirst=True)
# Set 'date' as the index
cash_flow_df = cash_flow_df.set_index('date')
# Ensure it's sorted by date
cash_flow_df = cash_flow_df.sort_index()

# --- 3. Process and Upsample Economic Data ---
print("Processing and upsampling economic data...")
# Create a 'month_year' key to match the CSV
econ_df['month_year_dt'] = pd.to_datetime(econ_df['month_year'], format='%Y-%m')
econ_df = econ_df.set_index('month_year_dt')

# Create a new daily index that matches our cash flow data
daily_index = cash_flow_df.index

# Re-index the economic data to this new daily index
# This will create NaNs (missing values) for all the new days
daily_econ_df = econ_df.reindex(daily_index)

# Use forward-fill (ffill) to fill the missing values
# This applies the month's indicator to every day of that month
daily_econ_df = daily_econ_df.ffill()

# Drop the helper column
daily_econ_df = daily_econ_df.drop(columns=['month_year'])

# --- 4. Merge DataFrames ---
print("Merging dataframes...")
# Now we can join the two DataFrames on their common index (the date)
df_prepared = cash_flow_df.join(daily_econ_df)

# Check for any remaining NaNs (e.g., from the very start)
df_prepared = df_prepared.dropna()

# --- 5. Save the Prepared Data ---
df_prepared.to_csv('cash_flow_prepared.csv')

print("\n--- Phase 2 Complete! ---")
print("Merged data saved to 'cash_flow_prepared.csv'.")
print("\nFirst 5 rows of prepared data:")
print(df_prepared.head())
print("\nData Info:")
df_prepared.info()