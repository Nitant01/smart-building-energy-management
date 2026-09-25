import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from optimizer import optimize_building


# -------------------------------
# Load models and data
# -------------------------------

df = pd.read_csv("building_data.csv")

occupancy_model = joblib.load("occupancy_model.pkl")
energy_model = joblib.load("energy_model.pkl")


# -------------------------------
# Page configuration
# -------------------------------

st.set_page_config(
    page_title="Smart Building AI",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 AI-Based Smart Building Energy Management System")

st.write(
    "Comfort-aware AI system for predicting occupancy, "
    "monitoring energy consumption and reducing unnecessary energy usage."
)


# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.header("Building Conditions")

hour = st.sidebar.slider(
    "Hour",
    min_value=8,
    max_value=18,
    value=12
)

temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=20.0,
    max_value=40.0,
    value=29.0
)

humidity = st.sidebar.slider(
    "Humidity (%)",
    min_value=30.0,
    max_value=90.0,
    value=60.0
)

light_level = st.sidebar.slider(
    "Light Level (lux)",
    min_value=100,
    max_value=800,
    value=400
)

current_ac = st.sidebar.selectbox(
    "Current AC Status",
    [0, 1],
    format_func=lambda x: "OFF" if x == 0 else "ON"
)

current_lights = st.sidebar.selectbox(
    "Current Lighting",
    [0, 1],
    format_func=lambda x: "OFF" if x == 0 else "ON"
)


# -------------------------------
# Occupancy Prediction
# -------------------------------

occupancy_input = pd.DataFrame({
    "day": [15],
    "hour": [hour],
    "temperature": [temperature],
    "humidity": [humidity],
    "light_level": [light_level]
})

predicted_occupancy = occupancy_model.predict(
    occupancy_input
)[0]

predicted_occupancy = max(
    0,
    min(40, round(predicted_occupancy))
)


# -------------------------------
# Energy Prediction - Current
# -------------------------------

energy_input = pd.DataFrame({
    "hour": [hour],
    "occupancy": [predicted_occupancy],
    "temperature": [temperature],
    "humidity": [humidity],
    "light_level": [light_level],
    "ac": [current_ac],
    "lights": [current_lights]
})

current_energy = energy_model.predict(
    energy_input
)[0]


# -------------------------------
# Optimization
# -------------------------------

result = optimize_building(
    predicted_occupancy,
    temperature,
    humidity,
    current_ac,
    current_lights
)

optimized_ac = result["optimized_ac"]
optimized_lights = result["optimized_lights"]


# -------------------------------
# Optimized Energy Prediction
# -------------------------------

optimized_input = pd.DataFrame({
    "hour": [hour],
    "occupancy": [predicted_occupancy],
    "temperature": [temperature],
    "humidity": [humidity],
    "light_level": [light_level],
    "ac": [optimized_ac],
    "lights": [optimized_lights]
})

optimized_energy = energy_model.predict(
    optimized_input
)[0]

optimized_energy = max(0, optimized_energy)


# -------------------------------
# Energy Saving
# -------------------------------

energy_saved = current_energy - optimized_energy

if current_energy > 0:
    saving_percentage = (
        energy_saved / current_energy
    ) * 100
else:
    saving_percentage = 0


# -------------------------------
# Dashboard metrics
# -------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Predicted Occupancy",
    f"{predicted_occupancy} people"
)

col2.metric(
    "Current Energy",
    f"{current_energy:.2f} kWh"
)

col3.metric(
    "Optimized Energy",
    f"{optimized_energy:.2f} kWh"
)

col4.metric(
    "Potential Saving",
    f"{saving_percentage:.1f}%"
)


# -------------------------------
# Current Building Status
# -------------------------------

st.subheader("Current Building Status")

col1, col2, col3, col4 = st.columns(4)

col1.write(f"🌡️ Temperature: **{temperature:.1f} °C**")
col2.write(f"💧 Humidity: **{humidity:.1f}%**")
col3.write(
    f"❄️ AC: **{'ON' if current_ac else 'OFF'}**"
)
col4.write(
    f"💡 Lights: **{'ON' if current_lights else 'OFF'}**"
)


# -------------------------------
# AI Recommendation
# -------------------------------

st.subheader("🤖 AI Recommendation")

st.info(
    result["recommendation"]
)

st.write(
    f"Recommended AC: "
    f"**{'ON' if optimized_ac else 'OFF'}**"
)

st.write(
    f"Recommended Lighting: "
    f"**{'ON' if optimized_lights else 'OFF'}**"
)


# -------------------------------
# Energy Comparison
# -------------------------------

st.subheader("Energy Comparison")

comparison = pd.DataFrame({
    "System": [
        "Current Building",
        "AI Optimized Building"
    ],
    "Energy (kWh)": [
        current_energy,
        optimized_energy
    ]
})

fig = px.bar(
    comparison,
    x="System",
    y="Energy (kWh)",
    title="Current vs AI-Optimized Energy Consumption"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# -------------------------------
# Historical Energy
# -------------------------------

st.subheader("Historical Energy Consumption")

historical = (
    df.groupby("hour")["energy"]
    .mean()
    .reset_index()
)

fig2 = px.line(
    historical,
    x="hour",
    y="energy",
    markers=True,
    title="Average Energy Consumption by Hour"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)


# -------------------------------
# Project Innovation
# -------------------------------

st.subheader("💡 Our Innovation")

st.write(
    "The system uses predictive occupancy and energy models "
    "to recommend HVAC and lighting changes while considering "
    "temperature and occupancy conditions. The goal is to "
    "reduce energy consumption without compromising occupant comfort."
)