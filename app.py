import streamlit as st
import pandas as pd
import plotly.express as px

from simulation import (
    simulate_building_conditions,
    check_comfort,
    CLIMATE_PROFILES
)

from optimizer import (
    optimize_energy,
    comfort_aware_optimization
)


st.set_page_config(
    page_title="Smart Building Energy Management",
    page_icon="🏢",
    layout="wide"
)


st.title(
    "🏢 AI-Based Smart Building Energy Management System"
)

st.write(
    "Occupancy-aware energy prediction and comfort-aware "
    "optimization using Indian building energy data."
)


# ============================================================
# LOAD IIT-D DATA
# ============================================================

DATA_FILE = (
    "iitd_processed/"
    "iitd_energy_occupancy.csv"
)

PIPELINE_FILE = (
    "iitd_processed/"
    "iitd_pipeline_results.csv.gz"
)

SUMMARY_FILE = (
    "iitd_processed/"
    "iitd_building_summary.csv"
)


data = pd.read_csv(
    DATA_FILE,
    parse_dates=["timestamp"]
)

pipeline = pd.read_csv(
    PIPELINE_FILE,
    parse_dates=["timestamp"]
)

building_summary = pd.read_csv(
    SUMMARY_FILE
)


data = data.sort_values(
    ["building", "timestamp"]
).reset_index(drop=True)

pipeline = pipeline.sort_values(
    ["building", "timestamp"]
).reset_index(drop=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Building & Simulation Controls"
)


# Building selection

buildings = sorted(
    pipeline["building"].dropna().unique()
)

selected_building = st.sidebar.selectbox(
    "Building",
    buildings
)


# Select building data

building_pipeline = pipeline[
    pipeline["building"] == selected_building
].copy()

building_pipeline = building_pipeline.sort_values(
    "timestamp"
)


# Date selection

available_dates = sorted(
    building_pipeline["timestamp"]
    .dt.date
    .unique()
)

selected_date = st.sidebar.selectbox(
    "Date",
    available_dates
)


# Data for selected date

occupancy_day = building_pipeline[
    building_pipeline["timestamp"].dt.date
    == selected_date
].copy()


if occupancy_day.empty:

    st.error(
        "No data is available for the selected date."
    )

    st.stop()


# Time selection

time_values = (
    occupancy_day["timestamp"]
    .dt.strftime("%H:%M")
    .tolist()
)

selected_time = st.sidebar.selectbox(
    "Time",
    time_values
)


selected_rows = occupancy_day[
    occupancy_day["timestamp"].dt.strftime("%H:%M")
    == selected_time
]


if selected_rows.empty:

    st.error(
        "No record is available for the selected time."
    )

    st.stop()


selected_row = selected_rows.iloc[0]


# ============================================================
# CLIMATE ZONE
# ============================================================

climate_zone = st.sidebar.selectbox(
    "Indian Climate Zone",
    list(CLIMATE_PROFILES.keys()),
    index=list(CLIMATE_PROFILES.keys()).index(
        "Composite"
    )
)


st.sidebar.caption(
    "Climate conditions shown in the dashboard are simulated."
)


# ============================================================
# HVAC / LIGHTING CONTROLS
# ============================================================

hvac_mode = st.sidebar.selectbox(
    "HVAC Mode",
    ["Low", "Medium", "High"],
    index=1
)


lighting_mode = st.sidebar.selectbox(
    "Lighting Mode",
    ["Low", "Medium", "High"],
    index=1
)


# ============================================================
# HISTORICAL / ML VALUES
# ============================================================

actual_power = float(
    selected_row["power"]
)

actual_occupancy = float(
    selected_row["occupancy_count"]
)

predicted_occupancy = float(
    selected_row["predicted_occupancy"]
)

predicted_energy = float(
    selected_row["predicted_energy"]
)


# ============================================================
# SIMULATED BUILDING CONDITIONS
# ============================================================

hour = selected_row["timestamp"].hour

simulated_conditions = simulate_building_conditions(
    occupancy=predicted_occupancy,
    hour=hour,
    climate_zone=climate_zone
)


temperature = simulated_conditions[
    "temperature"
]

humidity = simulated_conditions[
    "humidity"
]

daylight = simulated_conditions[
    "daylight"
]

simulated_hvac = simulated_conditions[
    "hvac_status"
]

simulated_lighting = simulated_conditions[
    "lighting_status"
]


# ============================================================
# COMFORT CHECK
# ============================================================

comfort_ok = check_comfort(
    temperature=temperature,
    humidity=humidity
)


# ============================================================
# OPTIMIZATION
# ============================================================

optimization_result = comfort_aware_optimization(
    predicted_energy=predicted_energy,
    occupancy=predicted_occupancy,
    temperature=temperature,
    humidity=humidity,
    hvac_mode=hvac_mode,
    lighting_mode=lighting_mode
)


optimized_energy = float(
    optimization_result["optimized_energy"]
)

total_saving = float(
    optimization_result["total_saving"]
)

saving_percent = float(
    optimization_result["saving_percent"]
)

recommendation = optimization_result[
    "recommendation"
]

recommended_hvac = optimization_result[
    "recommended_hvac"
]

recommended_lighting = optimization_result[
    "recommended_lighting"
]

comfort_status = optimization_result[
    "comfort_status"
]


# ============================================================
# HEADER INFORMATION
# ============================================================

st.subheader(
    f"{selected_building} Building"
)

st.write(
    f"Selected timestamp: "
    f"**{selected_row['timestamp']}**"
)

st.caption(
    "Historical energy and occupancy values are from the IIT-D "
    "Indian building dataset. Sensor and climate conditions shown "
    "below are simulated for prototype demonstration."
)


# ============================================================
# CORE ENERGY METRICS
# ============================================================

st.subheader(
    "Core Energy Metrics"
)

col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Actual Energy",
    f"{actual_power:,.0f} W"
)


col2.metric(
    "Predicted Energy",
    f"{predicted_energy:,.0f} W"
)


col3.metric(
    "Optimized Energy",
    f"{optimized_energy:,.0f} W"
)


col4.metric(
    "Potential Saving",
    f"{saving_percent:.2f}%"
)


# ============================================================
# OCCUPANCY
# ============================================================

st.subheader(
    "Occupancy Prediction"
)

col1, col2, col3 = st.columns(3)


col1.metric(
    "Actual Occupancy",
    f"{actual_occupancy:.0f}"
)


col2.metric(
    "Predicted Occupancy",
    f"{predicted_occupancy:.0f}"
)


occupancy_difference = (
    predicted_occupancy
    - actual_occupancy
)


col3.metric(
    "Prediction Difference",
    f"{occupancy_difference:+.0f}"
)


# ============================================================
# SIMULATED BUILDING CONDITIONS
# ============================================================

st.subheader(
    "Simulated Building Conditions"
)

col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Temperature",
    f"{temperature:.1f} °C"
)


col2.metric(
    "Humidity",
    f"{humidity:.0f}%"
)


col3.metric(
    "Daylight",
    f"{daylight:.0f}%"
)


col4.metric(
    "Climate Zone",
    climate_zone
)


st.caption(
    "SIMULATED: Temperature, humidity, daylight and climate "
    "conditions are generated by the prototype and are not "
    "measured IIT-D sensor data."
)


# ============================================================
# SIMULATED EQUIPMENT CONDITIONS
# ============================================================

st.subheader(
    "Simulated HVAC & Lighting Conditions"
)

col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Simulated HVAC Requirement",
        simulated_hvac
    )


with col2:

    st.metric(
        "Simulated Lighting Requirement",
        simulated_lighting
    )


st.caption(
    "SIMULATED: HVAC and lighting states are inferred from "
    "occupancy, daylight and the selected climate scenario."
)


# ============================================================
# COMFORT STATUS
# ============================================================

st.subheader(
    "Comfort Status"
)


if comfort_ok:

    st.success(
        "Simulated comfort condition is within the target range."
    )

else:

    st.warning(
        "Simulated comfort condition is outside the target range."
    )


st.write(
    f"Temperature range: **22–26 °C**"
)

st.write(
    f"Humidity range: **30–70% RH**"
)

st.info(
    comfort_status
)

st.caption(
    "The comfort condition is a simulated prototype constraint. "
    "The IIT-D dataset does not provide measured thermal comfort."
)


# ============================================================
# AI RECOMMENDATION
# ============================================================

st.subheader(
    "AI Recommendation"
)

st.info(
    recommendation
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Recommended HVAC",
        recommended_hvac
    )


with col2:

    st.metric(
        "Recommended Lighting",
        recommended_lighting
    )


# ============================================================
# BASELINE VS OPTIMIZED ENERGY
# ============================================================

st.subheader(
    "Baseline vs AI-Optimized Energy"
)


comparison = pd.DataFrame(
    {
        "Scenario": [
            "Predicted Baseline",
            "AI Optimized"
        ],
        "Energy (W)": [
            predicted_energy,
            optimized_energy
        ]
    }
)


fig_comparison = px.bar(
    comparison,
    x="Scenario",
    y="Energy (W)",
    title="Predicted Baseline vs Optimized Energy"
)


st.plotly_chart(
    fig_comparison,
    width="stretch"
)


col1, col2, col3 = st.columns(3)


col1.metric(
    "Predicted Baseline",
    f"{predicted_energy:,.0f} W"
)


col2.metric(
    "Optimized",
    f"{optimized_energy:,.0f} W"
)


col3.metric(
    "Potential Reduction",
    f"{total_saving:,.0f} W"
)


st.caption(
    "Optimization saving is a simulated potential reduction based "
    "on predefined HVAC and lighting energy-share assumptions."
)


# ============================================================
# ENERGY SAVING PROJECTION
# ============================================================

st.subheader(
    "Energy Saving Projection"
)


operating_hours = st.slider(
    "Estimated operating hours per day",
    min_value=1,
    max_value=24,
    value=10
)


baseline_kwh_per_hour = (
    predicted_energy / 1000
)

optimized_kwh_per_hour = (
    optimized_energy / 1000
)

saving_kwh_per_hour = (
    total_saving / 1000
)


daily_saving_kwh = (
    saving_kwh_per_hour
    * operating_hours
)

monthly_saving_kwh = (
    daily_saving_kwh
    * 30
)

annual_saving_kwh = (
    daily_saving_kwh
    * 365
)


col1, col2, col3 = st.columns(3)


col1.metric(
    "Daily Potential",
    f"{daily_saving_kwh:.2f} kWh"
)


col2.metric(
    "Monthly Potential",
    f"{monthly_saving_kwh:.2f} kWh"
)


col3.metric(
    "Annual Potential",
    f"{annual_saving_kwh:.2f} kWh"
)


st.caption(
    "Illustrative projection based on the selected operating-hour "
    "assumption. These are not measured annual savings."
)



# ============================================================
# BUILDING ENERGY INTENSITY (EUI)
# ============================================================

st.subheader("Building Energy Intensity")

st.caption(
    "Illustrative EUI calculation based on the selected scenario, "
    "assumed operating hours, and user-defined building area."
)

col1, col2 = st.columns(2)

with col1:
    building_area = st.number_input(
        "Building Area (m²)",
        min_value=100.0,
        max_value=1000000.0,
        value=5000.0,
        step=100.0
    )

with col2:
    annual_operating_hours = st.number_input(
        "Annual Operating Hours",
        min_value=100.0,
        max_value=8760.0,
        value=3000.0,
        step=100.0
    )

# Convert predicted and optimized power from W to kW
baseline_power_kw = predicted_energy / 1000.0
optimized_power_kw = optimized_energy / 1000.0

# Annual energy
baseline_annual_kwh = baseline_power_kw * annual_operating_hours
optimized_annual_kwh = optimized_power_kw * annual_operating_hours

# Energy intensity
baseline_eui = baseline_annual_kwh / building_area
optimized_eui = optimized_annual_kwh / building_area

eui_reduction = baseline_eui - optimized_eui

if baseline_eui > 0:
    eui_reduction_percent = (
        eui_reduction / baseline_eui
    ) * 100
else:
    eui_reduction_percent = 0.0

# Display EUI metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Baseline EUI",
        f"{baseline_eui:.2f} kWh/m²/year"
    )

with col2:
    st.metric(
        "Optimized EUI",
        f"{optimized_eui:.2f} kWh/m²/year"
    )

with col3:
    st.metric(
        "Potential EUI Reduction",
        f"{eui_reduction:.2f} kWh/m²/year",
        f"{eui_reduction_percent:.2f}%"
    )

st.info(
    "Baseline = predicted building energy before optimization. "
    "Optimized = simulated energy after the selected HVAC/lighting "
    "optimization scenario. Building area and operating hours are "
    "user-defined assumptions for illustrative EUI estimation."
)






# ============================================================
# OCCUPANCY TREND
# ============================================================

st.subheader(
    "Actual vs Predicted Occupancy"
)


occupancy_plot = occupancy_day[
    [
        "timestamp",
        "occupancy_count",
        "predicted_occupancy"
    ]
].copy()


occupancy_plot = occupancy_plot.rename(
    columns={
        "occupancy_count": "Actual Occupancy",
        "predicted_occupancy": "Predicted Occupancy"
    }
)


fig_occupancy = px.line(
    occupancy_plot,
    x="timestamp",
    y=[
        "Actual Occupancy",
        "Predicted Occupancy"
    ],
    title="Occupancy Prediction for Selected Day"
)


st.plotly_chart(
    fig_occupancy,
    width="stretch"
)


# ============================================================
# ENERGY TREND
# ============================================================

st.subheader(
    "Actual vs Predicted Energy"
)


energy_plot = occupancy_day[
    [
        "timestamp",
        "power",
        "predicted_energy"
    ]
].copy()


energy_plot = energy_plot.rename(
    columns={
        "power": "Actual Power",
        "predicted_energy": "Predicted Energy"
    }
)


fig_energy = px.line(
    energy_plot,
    x="timestamp",
    y=[
        "Actual Power",
        "Predicted Energy"
    ],
    title="Energy Prediction for Selected Day"
)


st.plotly_chart(
    fig_energy,
    width="stretch"
)


# ============================================================
# OPTIMIZED ENERGY TREND
# ============================================================

st.subheader(
    "Predicted vs Optimized Energy"
)


energy_scenario = energy_plot.copy()


energy_scenario["Optimized Energy"] = (
    energy_scenario["Predicted Energy"]
    * (1 - saving_percent / 100)
)


fig_optimized = px.line(
    energy_scenario,
    x="timestamp",
    y=[
        "Predicted Energy",
        "Optimized Energy"
    ],
    title="Predicted vs Optimized Energy"
)


st.plotly_chart(
    fig_optimized,
    width="stretch"
)


# ============================================================
# BUILDING-WISE ENERGY IMPACT
# ============================================================

st.subheader(
    "Building-Wise Energy Impact"
)


summary_plot = building_summary.copy()


summary_plot["Actual Energy"] = (
    summary_plot["actual_energy"]
)

summary_plot["Optimized Energy"] = (
    summary_plot["optimized_energy"]
)


fig_buildings = px.bar(
    summary_plot,
    x="building",
    y=[
        "Actual Energy",
        "Optimized Energy"
    ],
    barmode="group",
    title="Actual vs Optimized Energy by Building"
)


st.plotly_chart(
    fig_buildings,
    width="stretch"
)


# ============================================================
# VALIDATED MODEL PERFORMANCE
# ============================================================

st.subheader(
    "Validated AI Model Performance"
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Occupancy R²",
    "0.9940"
)


col2.metric(
    "Occupancy MAE",
    "3.56 people"
)


col3.metric(
    "Energy R²",
    "0.9853"
)


col4.metric(
    "Energy MAE",
    "798.96 W"
)


st.caption(
    "These metrics come from the validated IIT-D test pipeline "
    "using predicted occupancy for energy prediction."
)


# ============================================================
# VALIDATED OPTIMIZATION SCENARIOS
# ============================================================

st.subheader(
    "Optimization Scenario Reference"
)


scenario_data = pd.DataFrame(
    {
        "HVAC": [
            "High",
            "High",
            "High",
            "Medium",
            "Medium",
            "Medium",
            "Low",
            "Low",
            "Low"
        ],
        "Lighting": [
            "High",
            "Medium",
            "Low",
            "High",
            "Medium",
            "Low",
            "High",
            "Medium",
            "Low"
        ],
        "Potential Saving": [
            "0.00%",
            "2.25%",
            "4.50%",
            "5.00%",
            "7.25%",
            "9.50%",
            "10.00%",
            "12.25%",
            "14.50%"
        ]
    }
)


st.table(
    scenario_data
)


st.caption(
    "These percentages follow the prototype's predefined HVAC "
    "and lighting energy-share assumptions."
)


# ============================================================
# PEAK DEMAND
# ============================================================

st.subheader(
    "Peak Demand"
)


actual_peak = float(
    pipeline["power"].max()
)

predicted_peak = float(
    pipeline["predicted_energy"].max()
)

interactive_optimized_peak = (
    predicted_peak
    * (1 - saving_percent / 100)
)

peak_reduction = (
    predicted_peak
    - interactive_optimized_peak
)

peak_reduction_percent = (
    peak_reduction
    / predicted_peak
    * 100
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Actual Peak",
    f"{actual_peak:,.0f} W"
)


col2.metric(
    "Predicted Peak",
    f"{predicted_peak:,.0f} W"
)


col3.metric(
    "Optimized Peak",
    f"{interactive_optimized_peak:,.0f} W"
)


col4.metric(
    "Peak Reduction",
    f"{peak_reduction_percent:.2f}%"
)


st.caption(
    "Optimized peak is an interactive scenario calculation using "
    "the selected optimization percentage."
)


# ============================================================
# RETROFIT / DEPLOYMENT ESTIMATE
# ============================================================

st.subheader(
    "Illustrative Deployment & Retrofit Estimate"
)


col1, col2 = st.columns(2)


with col1:

    building_area = st.number_input(
        "Building area (m²)",
        min_value=500,
        max_value=100000,
        value=5000,
        step=500
    )


with col2:

    sensor_count = st.number_input(
        "Estimated sensor points",
        min_value=10,
        max_value=5000,
        value=100,
        step=10
    )


sensor_cost = 2500.0

gateway_cost = 25000.0

software_cost = 100000.0


hardware_cost = (
    sensor_count
    * sensor_cost
)

total_deployment_cost = (
    hardware_cost
    + gateway_cost
    + software_cost
)


st.write(
    f"Estimated sensor deployment: "
    f"**₹{hardware_cost:,.0f}**"
)

st.write(
    f"Estimated gateway cost: "
    f"**₹{gateway_cost:,.0f}**"
)

st.write(
    f"Estimated software/integration cost: "
    f"**₹{software_cost:,.0f}**"
)

st.write(
    f"**Illustrative total deployment cost: "
    f"₹{total_deployment_cost:,.0f}**"
)


# ============================================================
# PAYBACK
# ============================================================

electricity_tariff = st.number_input(
    "Electricity tariff (₹/kWh)",
    min_value=1.0,
    max_value=30.0,
    value=8.0,
    step=0.5
)


annual_cost_saving = (
    annual_saving_kwh
    * electricity_tariff
)


if annual_cost_saving > 0:

    payback_years = (
        total_deployment_cost
        / annual_cost_saving
    )

else:

    payback_years = 0.0


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Illustrative Annual Saving",
        f"₹{annual_cost_saving:,.0f}"
    )


with col2:

    if payback_years > 0:

        st.metric(
            "Illustrative Payback",
            f"{payback_years:.1f} years"
        )

    else:

        st.metric(
            "Illustrative Payback",
            "N/A"
        )


st.caption(
    "Deployment cost, electricity tariff and payback are illustrative "
    "prototype assumptions. Actual values require building survey, "
    "vendor quotations and BMS integration assessment."
)


# ============================================================
# BUILDING TYPOLOGIES
# ============================================================

st.subheader(
    "Supported Building Typologies"
)


st.write(
    "The validated IIT-D dataset contains seven building "
    "typologies used by this prototype:"
)


typology_data = pd.DataFrame(
    {
        "Building Type": [
            "Academic",
            "Boys Hostel",
            "Dining",
            "Facilities",
            "Girls Hostel",
            "Lecture",
            "Library"
        ]
    }
)


st.table(
    typology_data
)


# ============================================================
# CLIMATE SCALABILITY
# ============================================================

st.subheader(
    "Scalability Across Indian Climate Conditions"
)


st.write(
    "The simulation layer allows the same decision-support "
    "architecture to be demonstrated under different Indian "
    "climate profiles."
)


climate_data = pd.DataFrame(
    {
        "Climate Profile": [
            "Hot & Dry",
            "Warm & Humid",
            "Composite",
            "Temperate",
            "Cold"
        ],
        "Prototype Use": [
            "High cooling demand scenario",
            "High humidity scenario",
            "Mixed temperature scenario",
            "Moderate climate scenario",
            "Low-temperature scenario"
        ]
    }
)


st.table(
    climate_data
)


# ============================================================
# FUTURE IOT / BMS DEPLOYMENT
# ============================================================

st.subheader(
    "Future IoT / BMS Integration"
)


st.write(
    """
    **Future deployment architecture**
    
    IoT Sensors → Edge / Cloud AI → Optimization Engine →
    Building Management System → HVAC & Lighting Controllers
    """
)


st.write(
    "In a real deployment, measured temperature, humidity, "
    "occupancy, lighting, HVAC and energy data could replace "
    "the simulated inputs used by this prototype."
)


# ============================================================
# DATA SOURCE AND LIMITATIONS
# ============================================================

st.subheader(
    "Dataset & Prototype Scope"
)


st.write(
    "The prediction models are validated using the Indian "
    "IIT-D building energy and occupancy dataset."
)


st.write(
    "The dataset provides historical electrical energy and "
    "occupancy information for seven building types."
)


st.write(
    "Temperature, humidity, daylight, HVAC state and lighting "
    "state shown in this dashboard are simulated inputs used "
    "to demonstrate the future IoT/BMS deployment concept."
)


st.warning(
    "This prototype does not claim measured post-retrofit "
    "energy savings, measured thermal comfort improvement, "
    "automatic HVAC control or automatic lighting control."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Based Smart Building Energy Management System | "
    "IIT-D Indian Building Dataset + Simulated IoT/BMS Layer"
)