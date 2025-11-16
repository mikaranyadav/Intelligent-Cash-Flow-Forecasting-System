# Intelligent Cash-Flow Forecasting System

This project builds a complete time-series forecasting system to predict corporate cash flow. It uses both classical statistics (ARIMA) and deep learning (LSTM) to create accurate models, which are then presented in an interactive Streamlit dashboard.

This project fulfills all requirements of a case study, including model development, integration of external economic data, and a dashboard with model explainability.

## 📈 Dashboard Preview

(https://github.com/mikaranyadav/Intelligent-Cash-Flow-Forecasting-System/blob/main/Dashboard1.png
https://github.com/mikaranyadav/Intelligent-Cash-Flow-Forecasting-System/blob/main/Dashboard2.png
https://github.com/mikaranyadav/Intelligent-Cash-Flow-Forecasting-System/blob/main/Dashboard3.png

## 🛠️ Tech Stack
* **Core:** Python, Pandas, NumPy
* **Data Simulation:** Faker
* **ARIMA Model:** `statsmodels` & `pmdarima` (for auto-ARIMA)
* **LSTM Model:** `TensorFlow (Keras)` & `scikit-learn`
* **Dashboard:** Streamlit & Plotly

## 📂 Project Structure

* `generate_data.py`: Script 1 - Simulates 3 years of daily cash flow and monthly economic data.
* `preprocess.py`: Script 2 - Cleans, merges, and prepares the data for modeling.
* `train_arima.py`: Script 3 - Trains the SARIMAX model (with economic data) and saves it.
* `train_lstm.py`: Script 4 - Trains the LSTM neural network (with economic data) and saves it.
* `dashboard.py`: Script 5 - The final Streamlit dashboard application.
* `.gitignore`: Tells Git to ignore large files like data (`.csv`) and models (`.joblib`, `.keras`).
* `requirements.txt`: A list of all Python libraries needed to run this project.

## 🚀 How to Run This Project

1.  **Clone or download** this repository.
2.  **Create a virtual environment** and install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the full pipeline** in order. The dashboard will *not* work until the models are trained.
    ```bash
    # Step 1: Create the raw data
    python generate_data.py

    # Step 2: Prepare the data for modeling
    python preprocess.py

    # Step 3: Train the ARIMA model
    python train_arima.py

    # Step 4: Train the LSTM model
    python train_lstm.py
    ```
4.  **Run the dashboard!**
    ```bash
    streamlit run dashboard.py

    ```


