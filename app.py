import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
from datetime import datetime, timedelta
import numpy as np

# Model paths
LR_MODEL_PATH = "models/lr_model.pkl"
XGB_MODEL_PATH = "models/xgb_model.pkl"
# Cached loaders
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/cleaned_data.csv", parse_dates=["Date"])
    df["Hour"] = df["Date"].dt.hour
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    return df

@st.cache_resource
def load_model(model_name):
    if model_name == "Linear Regression":
        return joblib.load(LR_MODEL_PATH)
    elif model_name == "XGBoost":
        return joblib.load(XGB_MODEL_PATH)
    else:
        return None

def make_predictions(model, features):
    return model.predict(features)

def future_dataframe(n_days):
    future_dates = pd.date_range(start=datetime.now(), periods=n_days * 24, freq="H")
    df = pd.DataFrame({"Date": future_dates})
    df["Hour"] = df["Date"].dt.hour
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    # Placeholder: using average weather for future
    df["Temperature"] = 30
    df["Humidity"] = 40
    df["WindSpeed"] = 2
    return df

def main():
    st.set_page_config(page_title="Delhi Electricity Demand Forecast", layout="wide")
    st.title("Electricity Demand Forecast – Delhi")
    st.markdown("Select a model and explore demand insights and forecasts.")

    # Load cleaned data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Missing file: data/processed/cleaned_data.csv")
        return

    feature_cols = ["Temperature", "Humidity", "WindSpeed", "Hour", "DayOfWeek", "Month"]
    features = df[feature_cols]
    true_demand = df["Demand"]

    # Model selection
    model_option = st.selectbox("Choose Forecasting Model", ["Linear Regression", "XGBoost"])
    model = load_model(model_option)

    if model:
        df["Predicted Demand"] = make_predictions(model, features)

        # Stats
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Peak Demand (MW)", f"{df['Demand'].max():.2f}")
        col2.metric("Minimum Demand (MW)", f"{df['Demand'].min():.2f}")
        col3.metric("Average Demand (MW)", f"{df['Demand'].mean():.2f}")
        peak_time = df.loc[df['Demand'].idxmax(), 'Date']
        col4.metric("Peak Time", peak_time.strftime("%Y-%m-%d %H:%M"))

        # Line Chart
        st.subheader("Actual vs Predicted Demand")
        fig = px.line(df, x="Date", y=["Demand", "Predicted Demand"], labels={"value": "MW"}, title="Demand Comparison")
        st.plotly_chart(fig, use_container_width=True)

        # Forecast future demand
        st.subheader("Future Forecast")
        n_days = st.slider("Days to forecast", min_value=1, max_value=7, value=3)
        future_df = future_dataframe(n_days)
        future_features = future_df[feature_cols]
        future_df["Forecast Demand"] = make_predictions(model, future_features)
        fig_future = px.line(future_df, x="Date", y="Forecast Demand", title=f"{n_days}-Day Forecast")
        st.plotly_chart(fig_future, use_container_width=True)

        # Manual prediction
        st.subheader("Manual Date Prediction")
        input_date = st.date_input("Choose a date")
        input_hour = st.slider("Hour (0–23)", 0, 23, 12)
        temperature = st.number_input("Temperature (°C)", 10.0, 50.0, 30.0)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
        wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 15.0, 2.0)

        if st.button("Predict"):
            dt = datetime.combine(input_date, datetime.min.time()) + timedelta(hours=input_hour)
            input_data = pd.DataFrame([{
                "Temperature": temperature,
                "Humidity": humidity,
                "WindSpeed": wind_speed,
                "Hour": dt.hour,
                "DayOfWeek": dt.weekday(),
                "Month": dt.month
            }])
            predicted = make_predictions(model, input_data)[0]
            st.success(f"Predicted Demand for {dt.strftime('%Y-%m-%d %H:%M')} is {predicted:.2f} MW")

        # Show raw data
        with st.expander("View Raw Data"):
            st.dataframe(df.tail(100))
    else:
        st.warning("Please select a valid model.")

if __name__ == "__main__":
    main()