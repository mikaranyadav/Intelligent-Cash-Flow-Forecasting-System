# Intelligent Cash-Flow Forecasting System

This project builds a complete time-series forecasting system to predict corporate cash flow. It uses both classical statistics (ARIMA) and deep learning (LSTM) to create accurate models, which are then presented in an interactive Streamlit dashboard.

This project fulfills all requirements of a case study, including model development, integration of external economic data, and a dashboard with model explainability.


## 💡 What's the Goal?

The goal of this project is to build an "intelligent crystal ball" for a company's finance department. Its job is to **predict how much cash the company will have in the near future**.

Just like a weather forecast helps you decide if you need an umbrella, a cash-flow forecast helps a company decide if it has enough money to pay its employees, invest in new projects, or if it needs to be more careful with its spending.

## 🔮 How Does It Work?

We built a system that works in three simple steps:

### Step 1: Learn from the Past
A company's cash flow isn't random. It has patterns. For example:
* **Weekly:** Less cash comes in on weekends.
* **Monthly:** A lot of cash goes out on the 25th for payroll.

Our system first learns all these past patterns from **3 years of historical transaction data**.

### Step 2: Build Two "Brains" to Predict the Future
We don't rely on just one prediction. We build two different types of "forecasters" and compare them.

1.  **The Statistical Expert (ARIMA):** This model is like a seasoned accountant. It's a "statistical" model that is fantastic at finding and continuing known patterns, like the weekly and monthly cycles. We also feed it **external economic data** (like inflation rates), and it uses these factors in its calculation.

2.  **The Creative Genius (LSTM):** This model is a "neural network," a type of Artificial Intelligence. It's like a creative detective that can find hidden, complex connections that the simpler model might miss. For example, it might learn that interest rates *only* matter when inflation is *also* high. It learns these deep, non-obvious patterns from the past 30 days of activity to predict the next day.

### Step 3: Show the Results on a Dashboard
All this data is useless if it's hidden in a spreadsheet. We built an **interactive dashboard** (a simple, one-page website) that shows the results.

This dashboard lets a manager see:
* The historical data and patterns.
* The predictions from *both* models plotted against the *actual* cash flow.
* A simple explanation of *how* the models are making their decisions.

## 🎯 What's the Final Result?

The final product is a **data-driven decision-making tool**. Instead of just guessing, a finance manager can now look at the dashboard and see two reliable forecasts for the company's future cash flow, allowing them to make smarter, more confident decisions.

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



