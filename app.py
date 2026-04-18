import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
bitcoin = pd.read_csv("data/bitcoin_cleaned.csv", parse_dates=['Date'], index_col='Date')

st.title("📈 Cryptocurrency Time-Series Dashboard")
st.write("Developed by **Jagathishwaran** — Amdox Technologies Internship Project")

# Sidebar navigation
menu = st.sidebar.selectbox("Select Option", 
                            ["Dataset Overview", 
                             "EDA",
                             "Technical Indicators",
                             "Volatility Analysis",
                             "Model Forecasts",
                             "Model Evaluation"])

# -------------------------------
# 1. DATASET OVERVIEW
# -------------------------------
if menu == "Dataset Overview":
    st.header("📄 Dataset Overview")
    
    st.write(bitcoin.head())
    st.write(bitcoin.describe())

    st.subheader("Close Price Trend")
    st.line_chart(bitcoin['Close'])

# -------------------------------
# 2. EDA
# -------------------------------
if menu == "EDA":
    st.header("🔍 Exploratory Data Analysis")

    num_cols = ['Open','High','Low','Close','Volume']

    st.subheader("Histograms")
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(bitcoin[col], kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(bitcoin[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -------------------------------
# 3. TECHNICAL INDICATORS
# -------------------------------
if menu == "Technical Indicators":
    st.header("📊 Technical Indicators")

    bitcoin['MA7'] = bitcoin['Close'].rolling(7).mean()
    bitcoin['MA30'] = bitcoin['Close'].rolling(30).mean()

    st.subheader("Moving Averages (MA7 & MA30)")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(bitcoin['Close'], label='Close')
    ax.plot(bitcoin['MA7'], label='MA7')
    ax.plot(bitcoin['MA30'], label='MA30')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Daily Returns")
    returns = bitcoin['Close'].pct_change()
    st.line_chart(returns)

# -------------------------------
# 4. MODEL FORECASTS
# -------------------------------
if menu == "Model Forecasts":
    st.header("📈 Forecasts from Models")

    st.subheader("ARIMA Forecast")
    arima = pd.read_csv("data/arima_forecast.csv")
    st.line_chart(arima.set_index("Date"))

    st.subheader("Prophet Forecast")
    prophet = pd.read_csv("data/prophet_forecast.csv")
    st.line_chart(prophet[['ds', 'yhat']].set_index('ds'))

    st.subheader("LSTM Forecast")
    lstm = pd.read_csv("data/lstm_forecast.csv")
    st.line_chart(lstm)

# -------------------------------
# 5. MODEL EVALUATION
# -------------------------------
if menu == "Model Evaluation":
    st.header("📉 Model Evaluation Metrics")

    metrics = pd.read_csv("data/model_metrics.csv")
    st.dataframe(metrics)

    best_model = metrics.loc[metrics['MAPE (%)'].idxmin(), 'Model']
    st.success(f"🏆 Best Model: **{best_model}** (Lowest MAPE)")
# -----------------------------------------------
# 6. VOLATILITY ANALYSIS
# -----------------------------------------------
if menu == "Volatility Analysis":
    st.header("⚡ Volatility Analysis")
    st.write("""
    Volatility measures how much Bitcoin's price fluctuates over time. 
    High volatility means higher risk and higher profit potential. 
    This section computes and visualizes 7-day, 30-day, and annualized volatility.
    """)

    # Load data
    bitcoin = pd.read_csv("data/bitcoin_cleaned.csv", parse_dates=['Date'], index_col='Date')

    # Calculate returns
    bitcoin['Returns'] = bitcoin['Close'].pct_change()

    # Rolling volatility
    bitcoin['Volatility_7'] = bitcoin['Returns'].rolling(7).std()
    bitcoin['Volatility_30'] = bitcoin['Returns'].rolling(30).std()

    # Annualized volatility
    bitcoin['Annualized_Volatility'] = bitcoin['Volatility_30'] * (365**0.5)

    # Plot 7-day vs 30-day volatility
    st.subheader("7-Day vs 30-Day Rolling Volatility")
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(bitcoin['Volatility_7'], label="7-Day Volatility", color="orange")
    ax.plot(bitcoin['Volatility_30'], label="30-Day Volatility", color="red")
    ax.legend()
    st.pyplot(fig)

    # Plot annualized volatility
    st.subheader("Annualized Volatility")
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(bitcoin['Annualized_Volatility'], color="purple")
    st.pyplot(fig)

    # Volatility clusters
    st.subheader("Volatility Clusters (Squared Returns)")
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(bitcoin['Returns']**2, color="green")
    st.pyplot(fig)
