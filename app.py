# =====================================================
# IMPORTS
# =====================================================
import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from textblob import TextBlob

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Cryptocurrency Analytics Dashboard",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================
if "run_app" not in st.session_state:
    st.session_state.run_app = False

# =====================================================
# SIDEBAR – USER CONTROLS
# =====================================================
st.sidebar.title("⚙️ Analysis Controls")

symbol = st.sidebar.text_input("Cryptocurrency Symbol (Yahoo Finance)", "BTC-USD")
forecast_days = st.sidebar.slider("Forecast Period (Days)", 7, 60, 30)

if st.sidebar.button("🚀 Run Full Analysis"):
    st.session_state.run_app = True

# =====================================================
# DATA LOADER
# =====================================================
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, period="2y")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

# =====================================================
# MAIN DASHBOARD
# =====================================================
if st.session_state.run_app:

    df = load_data(symbol)

    if df.empty or len(df) < 150:
        st.error("❌ Insufficient historical data available")
        st.stop()

    # =================================================
    # FEATURE ENGINEERING
    # =================================================
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(20).std()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + rs))

    # =================================================
    # DASHBOARD TITLE
    # =================================================
    st.title("📈 Cryptocurrency Time Series Analysis & Forecasting")
    st.caption(
        "Technical Analysis • Forecasting Models • Market Sentiment • Risk Metrics • Trading Signals"
    )

    # =================================================
    # SECTION 1: MARKET OVERVIEW
    # =================================================
    st.header("📊 Market Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
    c2.metric("Daily Return", f"{df['Return'].iloc[-1]:.2%}")
    c3.metric("Market Volatility", f"{df['Volatility'].iloc[-1]:.2%}")
    c4.metric("RSI Value", f"{df['RSI'].iloc[-1]:.1f}")

    # =================================================
    # SECTION 2: PRICE ACTION
    # =================================================
    st.header("🕯️ Price Action & Candlestick Analysis")

    fig = go.Figure(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    ))
    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # SECTION 3: TREND ANALYSIS
    # =================================================
    st.header("📈 Trend Analysis Using Moving Averages")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Closing Price"))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name="20-Day SMA"))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name="50-Day SMA"))
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # SECTION 4: MOMENTUM ANALYSIS
    # =================================================
    st.header("📊 Momentum Analysis – Relative Strength Index (RSI)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI"))
    fig.add_hline(y=70, line_dash="dash", annotation_text="Overbought Zone")
    fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold Zone")
    fig.update_layout(template="plotly_dark", yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # SECTION 5: VOLATILITY & RISK
    # =================================================
    st.header("⚠️ Market Risk & Volatility Analysis")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Volatility'], name="Rolling Volatility"))
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # SECTION 6: MARKET REGIME IDENTIFICATION
    # =================================================
    st.header("🌍 Market Regime Identification")

    price = df['Close'].iloc[-1]
    sma50 = df['SMA_50'].iloc[-1]
    vol = df['Volatility'].iloc[-1]

    if price > sma50 and vol < df['Volatility'].mean():
        regime = "🟢 Bullish Market Regime"
    elif price < sma50 and vol > df['Volatility'].mean():
        regime = "🔴 Bearish Market Regime"
    else:
        regime = "🟡 Sideways / Consolidation Phase"

    st.subheader(regime)

    # =================================================
    # SECTION 7: PRICE FORECASTING MODELS
    # =================================================
    st.header("🔮 Price Forecasting Using Time Series Models")

    tab1, tab2, tab3 = st.tabs([
        "Prophet (Trend & Seasonality)",
        "ARIMA (Statistical Model)",
        "LSTM (Deep Learning Model)"
    ])

    with tab1:
        df_p = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        df_p['ds'] = pd.to_datetime(df_p['ds']).dt.tz_localize(None)

        model = Prophet()
        model.fit(df_p)

        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Historical Price"))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecasted Price"))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        series = df['Close'].dropna()
        arima = ARIMA(series, order=(5, 1, 0))
        arima_fit = arima.fit()
        arima_forecast = arima_fit.forecast(steps=forecast_days)

        future_dates = pd.date_range(
            df['Date'].iloc[-1],
            periods=forecast_days + 1,
            freq='D'
        )[1:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historical Price"))
        fig.add_trace(go.Scatter(x=future_dates, y=arima_forecast, name="ARIMA Forecast"))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)

        X, y = [], []
        for i in range(60, len(scaled)):
            X.append(scaled[i-60:i, 0])
            y.append(scaled[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        lstm = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            LSTM(50),
            Dense(1)
        ])
        lstm.compile(optimizer='adam', loss='mse')
        lstm.fit(X, y, epochs=5, batch_size=32, verbose=0)

        last_seq = scaled[-60:].reshape(1, 60, 1)
        preds = []

        for _ in range(forecast_days):
            pred = lstm.predict(last_seq, verbose=0)
            preds.append(pred[0, 0])
            last_seq = np.concatenate(
                (last_seq[:, 1:, :], pred.reshape(1, 1, 1)),
                axis=1
            )

        preds = scaler.inverse_transform(
            np.array(preds).reshape(-1, 1)
        ).flatten()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historical Price"))
        fig.add_trace(go.Scatter(x=future_dates, y=preds, name="LSTM Forecast"))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # SECTION 8: MARKET SENTIMENT ANALYSIS
    # =================================================
    st.header("📰 Market Sentiment Analysis (News & Social Media)")

    with st.form("sentiment_form"):
        text = st.text_area("Paste Cryptocurrency News Headlines or Tweets")
        analyze = st.form_submit_button("Analyze Sentiment")

    polarity = 0
    if analyze and text.strip():
        polarity = TextBlob(text).sentiment.polarity

        if polarity > 0.1:
            st.success(f"📈 Overall Market Sentiment: Positive ({polarity:.2f})")
        elif polarity < -0.1:
            st.error(f"📉 Overall Market Sentiment: Negative ({polarity:.2f})")
        else:
            st.warning(f"⚖️ Overall Market Sentiment: Neutral ({polarity:.2f})")

    # =================================================
    # SECTION 9: SMART TRADING SIGNAL
    # =================================================
    st.header("🤖 Intelligent Buy / Sell / Hold Signal")

    signal = "HOLD"
    reasons = []

    if df['RSI'].iloc[-1] < 30:
        reasons.append("RSI indicates oversold conditions")
    if df['RSI'].iloc[-1] > 70:
        reasons.append("RSI indicates overbought conditions")
    if price > sma50:
        reasons.append("Price is above long-term trend (SMA 50)")
    if polarity > 0.1:
        reasons.append("Positive market sentiment")
    if polarity < -0.1:
        reasons.append("Negative market sentiment")

    if "RSI indicates oversold conditions" in reasons and "Positive market sentiment" in reasons:
        signal = "🟢 BUY"
    elif "RSI indicates overbought conditions" in reasons and "Negative market sentiment" in reasons:
        signal = "🔴 SELL"

    st.subheader(signal)
    st.write("**Signal Rationale:**")
    for r in reasons:
        st.write("•", r)

    # =================================================
    # SECTION 10: STRATEGY BACKTESTING
    # =================================================
    st.header("📉 Strategy Backtesting & Performance Evaluation")

    df['Position'] = np.where(df['Close'] > df['SMA_20'], 1, 0)
    df['Strategy_Return'] = df['Position'].shift(1) * df['Return']
    df['Equity'] = (1 + df['Strategy_Return']).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Equity'], name="Strategy Equity Curve"))
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    max_dd = (df['Equity'] / df['Equity'].cummax() - 1).min()
    sharpe = df['Strategy_Return'].mean() / df['Strategy_Return'].std() * np.sqrt(252)

    c1, c2 = st.columns(2)
    c1.metric("Maximum Drawdown", f"{max_dd:.2%}")
    c2.metric("Sharpe Ratio", f"{sharpe:.2f}")

    # =================================================
    # SECTION 11: AI-GENERATED MARKET SUMMARY
    # =================================================
    st.header("🧠 AI-Generated Market Summary")

    st.info(
        f"The cryptocurrency is currently in a **{regime}**. "
        f"The RSI value of **{df['RSI'].iloc[-1]:.1f}** indicates momentum conditions. "
        f"Volatility analysis suggests "
        f"{'elevated risk levels' if vol > df['Volatility'].mean() else 'relatively stable market conditions'}. "
        f"Based on combined technical indicators, sentiment signals, and risk metrics, "
        f"the recommended action is **{signal}**."
    )

    # =================================================
    # SECTION 12: DATA EXPLORER
    # =================================================
    st.header("📁 Historical Data Explorer")

    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇️ Download Dataset (CSV)",
        df.to_csv(index=False),
        f"{symbol}_historical_data.csv"
    )

else:
    st.info("👈 Select parameters from the sidebar and click **Run Full Analysis**")
