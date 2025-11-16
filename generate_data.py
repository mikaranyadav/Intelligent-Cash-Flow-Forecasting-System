import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

print("Starting data generation...")

# --- 1. Generate Historical Transaction Data ---

# We'll create 3 years of daily data (approx. 1095 days)
base_date = datetime(2022, 1, 1)
n_days = 365 * 3
date_range = [base_date + timedelta(days=x) for x in range(n_days)]

transactions = []

# Create a baseline daily cash flow
base_cash_flow = 100000

for date in date_range:
    # --- Create seasonal patterns ---
    
    # 1. Weekly pattern (e.g., lower activity on weekends)
    day_of_week = date.weekday()
    if day_of_week in [5, 6]: # Saturday, Sunday
        weekly_factor = 0.5
    else:
        weekly_factor = 1.0
        
    # 2. Monthly pattern (e.g., big payouts for payroll on 25th)
    if date.day == 25:
        monthly_factor = -2.0 # Big outflow (payroll)
    elif date.day == 1:
        monthly_factor = 1.5 # Big inflow (customer payments)
    else:
        monthly_factor = 1.0
        
    # --- Add random noise ---
    random_noise = random.uniform(0.8, 1.2)
    
    # --- Calculate final cash flow ---
    # We combine base, factors, and noise
    cash_flow = (base_cash_flow * weekly_factor * monthly_factor * random_noise) + random.randint(-5000, 5000)
    
    # Add a slight upward trend over time
    trend_factor = (date - base_date).days / 365 * 0.05 # 5% growth per year
    cash_flow *= (1 + trend_factor)
    
    transactions.append({
        'date': date,
        'cash_flow': int(cash_flow)
    })

trans_df = pd.DataFrame(transactions)
trans_df.to_csv('historical_transactions.csv', index=False)
print("Generated historical_transactions.csv")

# --- 2. Generate External Economic Indicators ---

# This data is monthly
month_range = pd.date_range(start=base_date, periods=36, freq='MS') # 36 months

indicators = []
base_inflation = 2.0
base_interest_rate = 1.5

for date in month_range:
    # Simulate slight random changes
    base_inflation = max(1.0, base_inflation + random.uniform(-0.3, 0.3))
    base_interest_rate = max(0.5, base_interest_rate + random.uniform(-0.1, 0.1))
    
    indicators.append({
        'month_year': date.strftime('%Y-%m'),
        'inflation_rate': round(base_inflation, 2),
        'interest_rate': round(base_interest_rate, 2)
    })

econ_df = pd.DataFrame(indicators)
econ_df.to_csv('economic_indicators.csv', index=False)
print("Generated economic_indicators.csv")

print("--- Phase 1 Complete! ---")