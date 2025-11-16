import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import numpy as np

# --- 1. Load All Assets (Cache for performance) ---

@st.cache_data
def load_data(csv_path):
    """Loads a CSV file with date parsing."""
    try:
        df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
        return df
    except Exception as e:
        st.error(f"Error loading {csv_path}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_forecast_data(csv_path):
    """Loads a forecast CSV file."""
    try:
        df = pd.read_csv(csv_path, parse_dates=['date'])
        return df
    except Exception as e:
        st.error(f"Error loading {csv_path}: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_arima_model():
    """Loads the saved ARIMA model."""
    try:
        model = joblib.load('arima_model.joblib')
        return model
    except FileNotFoundError:
        st.error("ARIMA model file 'arima_model.joblib' not found.")
        return None

@st.cache_resource
def load_lstm_model_and_scaler():
    """Loads the saved LSTM model and its scaler."""
    try:
        model = load_model('lstm_model.keras')
        scaler = joblib.load('lstm_target_scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("LSTM model ('lstm_model.keras') or scaler ('lstm_target_scaler.joblib') not found.")
        return None, None
    except Exception as e:
        st.error(f"Error loading LSTM model: {e}")
        return None, None

# --- Load all data and models ---
df_prepared = load_data('cash_flow_prepared.csv')
arima_forecasts_df = load_forecast_data('arima_forecasts.csv')
lstm_forecasts_df = load_forecast_data('lstm_forecasts.csv')

arima_model = load_arima_model()
lstm_model, lstm_scaler = load_lstm_model_and_scaler()

# --- 2. Build the Dashboard UI ---

st.set_page_config(layout="wide")
st.title("💰 Intelligent Cash-Flow Forecasting System")
st.markdown("A dashboard for Finance & Controlling to compare forecasting models.")

# --- 3. Historical Data Section ---
st.header("Historical Cash Flow (3 Years)")
st.markdown("This chart shows the daily cash flow, including external economic factors. You can see the weekly/monthly patterns and the upward trend.")

# Plot historical data
if not df_prepared.empty:
    fig_hist = px.line(df_prepared, y='cash_flow', title='Daily Cash Flow')
    fig_hist.add_scatter(x=df_prepared.index, y=df_prepared['inflation_rate'], name='Inflation Rate', yaxis='y2')
    fig_hist.add_scatter(x=df_prepared.index, y=df_prepared['interest_rate'], name='Interest Rate', yaxis='y2')
    
    # Create a secondary y-axis for the economic indicators
    fig_hist.update_layout(
        yaxis_title='Cash Flow ($)',
        yaxis2=dict(
            title='Rate (%)',
            overlaying='y',
            side='right'
        )
    )
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.warning("Could not load historical data.")

st.markdown("---")

# --- 4. Model Comparison Section ---
st.header("Forecasting Model Performance (On 180-Day Test Set)")
st.markdown("Here we compare the performance of our two models against the *actual* cash flow from the test period.")

# Create two columns for the models
col1, col2 = st.columns(2)

# --- ARIMA Model Column ---
with col1:
    st.subheader("1. ARIMA Model")
    if not arima_forecasts_df.empty:
        # Calculate MAE
        mae = mean_absolute_error(arima_forecasts_df['actual_cash_flow'], arima_forecasts_df['forecast_cash_flow'])
        st.metric(label="Mean Absolute Error (MAE)", value=f"${mae:,.2f}")
        st.markdown(f"On average, the model's forecast was off by ${mae:,.2f}.")
        
        # Plot ARIMA
        fig_arima = px.line(arima_forecasts_df, x='date', y='actual_cash_flow', title='ARIMA vs. Actual')
        fig_arima.add_scatter(x=arima_forecasts_df['date'], y=arima_forecasts_df['forecast_cash_flow'], name='ARIMA Forecast')
        st.plotly_chart(fig_arima, use_container_width=True)
        
        with st.expander("Model Explainability (ARIMA)"):
            st.markdown("""
            **How does this model work?**
            The ARIMA (SARIMAX) model is a statistical model that learns from:
            1.  **Past Values:** It assumes tomorrow's value is related to yesterday's value.
            2.  **Seasonality:** It automatically detected the 7-day (weekly) pattern in the data.
            3.  **External Factors (Explainability):** This model's predictions were directly influenced by the `inflation_rate` and `interest_rate`. This is the "X" in SARIMAX.
            """)
            if arima_model:
                st.text("ARIMA Model Parameters found by auto_arima:")
                st.code(f"ARIMA{arima_model.order} Seasonal{arima_model.seasonal_order}")

# --- LSTM Model Column ---
with col2:
    st.subheader("2. LSTM Neural Network")
    if not lstm_forecasts_df.empty:
        # Calculate MAE
        mae = mean_absolute_error(lstm_forecasts_df['actual_cash_flow'], lstm_forecasts_df['forecast_cash_flow'])
        st.metric(label="Mean Absolute Error (MAE)", value=f"${mae:,.2f}")
        st.markdown(f"On average, the model's forecast was off by ${mae:,.2f}.")
        
        # Plot LSTM
        fig_lstm = px.line(lstm_forecasts_df, x='date', y='actual_cash_flow', title='LSTM vs. Actual')
        fig_lstm.add_scatter(x=lstm_forecasts_df['date'], y=lstm_forecasts_df['forecast_cash_flow'], name='LSTM Forecast')
        st.plotly_chart(fig_lstm, use_container_width=True)
        
        with st.expander("Model Explainability (LSTM)"):
            st.markdown("""
            **How does this model work?**
            The LSTM is a deep learning model that acts like a "black box," but it learns much more complex patterns.
            1.  **Time-Series "Memory":** It "remembers" patterns from the last 30 days (`n_steps`) to predict the next day.
            2.  **Complex Relationships (Explainability):** Unlike ARIMA, the LSTM learns *non-linear* relationships between all three features (`cash_flow`, `inflation_rate`, `interest_rate`) at once. It might learn, for example, that high interest rates *only* matter if inflation is *also* high.
            """)
            if lstm_model:
                st.text("LSTM Model Structure:")
                # We can't easily print the summary here, but we can describe it.
                st.code("LSTM(units=50) -> Dense(units=1)")