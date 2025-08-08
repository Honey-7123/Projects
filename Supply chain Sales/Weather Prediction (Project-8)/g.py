import streamlit as st
import plotly.express as px
import pandas as pd
from prepare_data import load_and_preprocess_data
from forecast_model import train_prophet
from mongo_utils import save_forecast

# Page setup
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
st.title("🌫️ Air Quality Forecast Dashboard")

# Sidebar input
st.sidebar.header("🔧 Forecast Settings")
forecast_days = st.sidebar.slider("Select number of days to forecast", min_value=3, max_value=14, value=7)

# Load historical data
df = load_and_preprocess_data()

# Metrics summary
latest_value = df["CO_concentration"].iloc[-1]
avg_value = df["CO_concentration"].mean()
max_value = df["CO_concentration"].max()
min_value = df["CO_concentration"].min()

col1, col2, col3 = st.columns(3)
col1.metric("📍 Latest CO", f"{latest_value:.2f} mg/m³")
col2.metric("📊 Average CO", f"{avg_value:.2f} mg/m³")
col3.metric("📈 Max CO", f"{max_value:.2f} mg/m³")

# Historical chart
st.subheader("📊 CO Concentration Over Time")
fig = px.line(
    df,
    x="DateTime",
    y="CO_concentration",
    labels={"DateTime": "Date", "CO_concentration": "CO Level (mg/m³)"},
    title="Actual CO Levels"
)
st.plotly_chart(fig, use_container_width=True)

# Forecast section
st.subheader(f"🔮 Forecast for Next {forecast_days} Day(s)")
forecast = train_prophet(periods=forecast_days)
forecast_data = forecast.to_dict(orient="records")

# Save forecast to MongoDB
try:
    save_forecast(forecast_data)
    st.success("✅ Forecast saved to MongoDB.")
except Exception as e:
    st.error(f"❌ MongoDB Error: {e}")

# Forecast plot
fig2 = px.line(
    forecast,
    x="ds",
    y="yhat",
    labels={"ds": "Date", "yhat": "Forecasted CO Level (mg/m³)"},
    title="Forecasted CO Levels"
)
fig2.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound")
fig2.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound")
st.plotly_chart(fig2, use_container_width=True)

# Raw forecast data option
with st.expander("📋 Show Forecast Data Table"):
    st.dataframe(forecast.tail(forecast_days), use_container_width=True)
