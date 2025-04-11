import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import datetime
from streamlit_lottie import st_lottie
import plotly.express as px
from final_rf import RainPredictionSystem
# Set Streamlit page configuration
st.set_page_config(page_title="Rainfall Prediction", layout="wide")

# Load the enhanced prediction system from the saved PKL file
prediction_system = joblib.load("full_rain_prediction_system.pkl")

sheet_id = "1JXxsLUMpwkubsGcZ1iGEQuKhQxvrLB92JehDwILTB_4"
sheet_name = "Sheet1"  # Change this if your sheet has a different name
file_path = f"https://docs.google.com/spreadsheets/d/1JXxsLUMpwkubsGcZ1iGEQuKhQxvrLB92JehDwILTB_4/gviz/tq?tqx=out:csv&sheet=11_april_data_log"

try:
    data = pd.read_csv(file_path)
except Exception as e:
    st.error(f"Failed to load data from Google Sheets. Error: {e}")
    st.stop()

# Load Lottie Animation
with open("Cloud.json", "r", encoding="utf-8") as f:
    lottie_animation = json.load(f)

# Title and animation section
col1, col2 = st.columns([2, 2])
with col1:
    st.title("Rainfall Prediction App")
    st.subheader("Real-Time Weather Forecast and Rainfall Prediction")
    st.write("""
    This app provides real-time weather predictions using sensor-based IoT data and machine learning models. 
    It forecasts temperature, humidity, and atmospheric pressure while predicting the possibility of rainfall for the next five days. 
    Stay updated with the latest weather trends and make informed decisions for outdoor activities and planning.
    """)

with col2:
    st_lottie(lottie_animation, height=320, key="weather_animation")

# Forecast section
st.title("📅 5-Day Weather Forecast")
prediction_system.load_and_prepare_data()
prediction_system.predict_future()

future_df = prediction_system.future_df
future_predictions = prediction_system.future_predictions
last_time = pd.to_datetime(data["time"].iloc[-1])

cols = st.columns(5)
for i, col in enumerate(cols):
    next_day = last_time + pd.Timedelta(days=i + 1)
    temperature = float(future_df.iloc[i]["temperature"])
    humidity = float(future_df.iloc[i]["humidity"])
    rain_result = "🌧️ Rain" if future_predictions[i] == 1 else "☀️ No Rain"
    
    col.metric(
        label=next_day.strftime("%A, %b %d"),
        value=f"{temperature:.1f}°C",
        delta=f"{humidity:.1f}% Humidity"
    )
    col.write(rain_result)


# Sidebar: Manual Rain Check with trend-aware prediction
st.sidebar.title("🌦️ Manual Rain Check")
st.sidebar.write("Enter values to check if it will rain today:")

temp = st.sidebar.number_input("Temperature (°C)", value=25.0)
humidity = st.sidebar.number_input("Humidity (%)", value=60.0)
pressure = st.sidebar.number_input("Pressure (hPa)", value=1013.0)
dew_point = st.sidebar.number_input("Dew Point (°C)", value=15.0)

if st.sidebar.button("Check Prediction"):
    try:
        result, chance = prediction_system.predict_manual_input(temp, humidity, pressure, dew_point)
        st.sidebar.success(f"{'🌧️ Rain Expected' if result == 1 else '☀️ No Rain'} ({chance})")
    except Exception as e:
        st.sidebar.error(f"❌ Error occurred: {e}")


# 📊 5-Day Forecast Graph
st.title("📈 Predicted Weather Trends (Next 5 Days)")

# Use future_df already predicted
plot_future = future_df[["temperature", "humidity", "pressure"]].copy()
plot_future["date"] = [last_time + pd.Timedelta(days=i+1) for i in range(len(plot_future))]

# Melt the DataFrame for Plotly
plot_future = plot_future.melt(id_vars="date", var_name="Metric", value_name="Value")

# Create the multi-line line chart
fig = px.line(
    plot_future,
    x="date",
    y="Value",
    color="Metric",
    title="Predicted Temperature, Humidity, and Pressure (Next 5 Days)",
    labels={"date": "Date", "Value": "Predicted Value"},
    markers=True
)

# Beautify it
fig.update_traces(line=dict(width=1.5))  # Thin lines
fig.update_layout(
    title_font_size=22,
    title_x=0.5,
    legend_title_text="Metrics",
    template="plotly_white",
    hovermode="x unified",
    font=dict(size=14),
    height=500
)

# Show in Streamlit
st.plotly_chart(fig, use_container_width=True)



# About section
st.title("📌 About This App")
st.write("""
This web-based tool monitors and predicts weather using physical sensors deployed in the environment.
It collects and processes temperature, humidity, pressure, and dew point data for short-term weather forecasting.

**Features:**
- 5-day automated weather prediction using sensor readings
- Manual rainfall check using custom weather input
- Visual graphs to interpret weather patterns
- Developed using embedded system components and Python

**Tech Stack (Non-AI Focused):**
- Python (Streamlit, Pandas, NumPy)
- ESP8266 for Wi-Fi transmission
- DHT22 (Temperature & Humidity Sensor)
- BMP180 (Pressure Sensor)
- Rain Sensor for binary rainfall status
- Excel/Google Sheets for data storage and visualization

Created as a **final-year engineering project** to demonstrate how IoT and machine learning can enhance everyday weather forecasting.
""")
