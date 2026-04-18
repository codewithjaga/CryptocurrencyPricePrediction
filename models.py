from prophet import Prophet
import pandas as pd
import numpy as np

def prophet_forecast(df, days):
    # --- STRICT CLEANING FOR PROPHET ---
    df_p = df[['Date', 'Close']].copy()

    # Rename columns
    df_p.columns = ['ds', 'y']

    # Convert datetime (remove timezone if any)
    df_p['ds'] = pd.to_datetime(df_p['ds']).dt.tz_localize(None)

    # Convert price to numeric FLOAT (CRITICAL FIX)
    df_p['y'] = pd.to_numeric(df_p['y'], errors='coerce')

    # Remove NaN rows
    df_p = df_p.dropna()

    # Ensure 1-D float series
    df_p['y'] = df_p['y'].astype(float)

    # Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(df_p)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    return forecast
