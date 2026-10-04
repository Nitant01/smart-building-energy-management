import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from optimizer import optimize_energy


INPUT_FILE = "iitd_processed/iitd_energy_occupancy.csv"
OUTPUT_FOLDER = "iitd_processed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    ["building", "timestamp"]
).reset_index(drop=True)


# ============================================================
# TIME FEATURES
# ============================================================

df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["day_of_month"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

df["hour_decimal"] = (
    df["hour"] +
    df["minute"] / 60
)

df["hour_sin"] = np.sin(
    2 * np.pi * df["hour_decimal"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour_decimal"] / 24
)

df["day_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["day_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)


# ============================================================
# OCCUPANCY FEATURES
# ============================================================

occupancy_group = df.groupby(
    "building"
)["occupancy_count"]

df["occupancy_lag_1"] = (
    occupancy_group.shift(1)
)

df["occupancy_lag_6"] = (
    occupancy_group.shift(6)
)

df["occupancy_lag_144"] = (
    occupancy_group.shift(144)
)

df["occupancy_rolling_6"] = (
    occupancy_group
    .shift(1)
    .rolling(6)
    .mean()
    .reset_index(level=0, drop=True)
)

df["occupancy_rolling_144"] = (
    occupancy_group
    .shift(1)
    .rolling(144)
    .mean()
    .reset_index(level=0, drop=True)
)


# ============================================================
# ENERGY FEATURES
# ============================================================

energy_group = df.groupby(
    "building"
)["power"]

df["energy_lag_1"] = (
    energy_group.shift(1)
)

df["energy_lag_6"] = (
    energy_group.shift(6)
)

df["energy_lag_144"] = (
    energy_group.shift(144)
)

df["energy_rolling_6"] = (
    energy_group
    .shift(1)
    .rolling(6)
    .mean()
    .reset_index(level=0, drop=True)
)

df["energy_rolling_144"] = (
    energy_group
    .shift(1)
    .rolling(144)
    .mean()
    .reset_index(level=0, drop=True)
)


df = df.dropna().reset_index(drop=True)


# ============================================================
# OCCUPANCY MODEL
# ============================================================

occupancy_bundle = joblib.load(
    "iitd_processed/occupancy_model.pkl"
)

occupancy_model = occupancy_bundle["model"]

occupancy_features = occupancy_bundle["features"]


occupancy_input = pd.get_dummies(
    df,
    columns=["building"],
    dtype=float
)


for column in occupancy_features:

    if column not in occupancy_input.columns:

        occupancy_input[column] = 0


X_occupancy = occupancy_input[
    occupancy_features
]


df["predicted_occupancy"] = np.maximum(
    occupancy_model.predict(
        X_occupancy
    ),
    0
)


# ============================================================
# OCCUPANCY VALIDATION
# ============================================================

occupancy_mae = mean_absolute_error(
    df["occupancy_count"],
    df["predicted_occupancy"]
)

occupancy_rmse = np.sqrt(
    mean_squared_error(
        df["occupancy_count"],
        df["predicted_occupancy"]
    )
)

occupancy_r2 = r2_score(
    df["occupancy_count"],
    df["predicted_occupancy"]
)


# ============================================================
# ENERGY MODEL
# ============================================================

energy_bundle = joblib.load(
    "iitd_processed/iitd_energy_model.pkl"
)

energy_model = energy_bundle["model"]

energy_features = energy_bundle["features"]


# ============================================================
# ORACLE ENERGY PREDICTION
# Uses actual occupancy
# ============================================================

oracle_input = df.copy()

oracle_input = pd.get_dummies(
    oracle_input,
    columns=["building"],
    dtype=float
)


for column in energy_features:

    if column not in oracle_input.columns:

        oracle_input[column] = 0


X_oracle = oracle_input[
    energy_features
]


df["energy_prediction_actual_occupancy"] = np.maximum(
    energy_model.predict(
        X_oracle
    ),
    0
)


# ============================================================
# END-TO-END ENERGY PREDICTION
# Uses predicted occupancy
# ============================================================

end_to_end_input = df.copy()

end_to_end_input[
    "occupancy_count"
] = end_to_end_input[
    "predicted_occupancy"
]


end_to_end_input = pd.get_dummies(
    end_to_end_input,
    columns=["building"],
    dtype=float
)


for column in energy_features:

    if column not in end_to_end_input.columns:

        end_to_end_input[column] = 0


X_end_to_end = end_to_end_input[
    energy_features
]


df["predicted_energy"] = np.maximum(
    energy_model.predict(
        X_end_to_end
    ),
    0
)


# ============================================================
# ENERGY VALIDATION
# ============================================================

actual_energy = df["power"]

oracle_energy = (
    df["energy_prediction_actual_occupancy"]
)

pipeline_energy = (
    df["predicted_energy"]
)


oracle_mae = mean_absolute_error(
    actual_energy,
    oracle_energy
)

oracle_rmse = np.sqrt(
    mean_squared_error(
        actual_energy,
        oracle_energy
    )
)

oracle_r2 = r2_score(
    actual_energy,
    oracle_energy
)


pipeline_mae = mean_absolute_error(
    actual_energy,
    pipeline_energy
)

pipeline_rmse = np.sqrt(
    mean_squared_error(
        actual_energy,
        pipeline_energy
    )
)

pipeline_r2 = r2_score(
    actual_energy,
    pipeline_energy
)


# ============================================================
# OPTIMIZATION
# ============================================================

optimization_results = []


for _, row in df.iterrows():

    result = optimize_energy(
        predicted_energy=row["predicted_energy"],
        occupancy=row["predicted_occupancy"],
        hvac_mode="Medium",
        lighting_mode="Medium"
    )

    optimization_results.append(result)


optimization_df = pd.DataFrame(
    optimization_results
)


df = pd.concat(
    [
        df.reset_index(drop=True),
        optimization_df.reset_index(drop=True)
    ],
    axis=1
)


# ============================================================
# BUILDING SUMMARY
# ============================================================

building_summary = (
    df.groupby("building")
    .agg(
        actual_energy=("power", "sum"),

        predicted_energy=(
            "predicted_energy",
            "sum"
        ),

        optimized_energy=(
            "optimized_energy",
            "sum"
        ),

        potential_saving=(
            "total_saving",
            "sum"
        ),

        average_actual_occupancy=(
            "occupancy_count",
            "mean"
        ),

        average_predicted_occupancy=(
            "predicted_occupancy",
            "mean"
        )
    )
    .reset_index()
)


building_summary[
    "saving_percent"
] = (
    building_summary["potential_saving"]
    /
    building_summary["predicted_energy"]
) * 100


# ============================================================
# PEAK DEMAND
# ============================================================

actual_peak = df["power"].max()

predicted_peak = df[
    "predicted_energy"
].max()

optimized_peak = df[
    "optimized_energy"
].max()


peak_reduction = (
    (
        predicted_peak -
        optimized_peak
    )
    /
    predicted_peak
) * 100


# ============================================================
# OVERALL SAVING
# ============================================================

total_predicted = (
    df["predicted_energy"].sum()
)

total_optimized = (
    df["optimized_energy"].sum()
)

total_saving = (
    df["total_saving"].sum()
)

saving_percent = (
    total_saving /
    total_predicted
) * 100


# ============================================================
# SAVE PIPELINE RESULTS
# ============================================================

df[
    [
        "timestamp",
        "building",
        "power",
        "occupancy_count",
        "predicted_occupancy",
        "energy_prediction_actual_occupancy",
        "predicted_energy",
        "optimized_energy",
        "total_saving",
        "saving_percent",
        "recommended_hvac",
        "recommended_lighting",
        "recommendation"
    ]
].to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "iitd_pipeline_results.csv"
    ),
    index=False
)


building_summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "iitd_building_summary.csv"
    ),
    index=False
)


# ============================================================
# SAVE VALIDATION METRICS
# ============================================================

metrics = pd.DataFrame([
    {
        "metric": "Occupancy MAE",
        "value": occupancy_mae,
        "unit": "people"
    },
    {
        "metric": "Occupancy RMSE",
        "value": occupancy_rmse,
        "unit": "people"
    },
    {
        "metric": "Occupancy R2",
        "value": occupancy_r2,
        "unit": ""
    },
    {
        "metric": "Energy MAE - Actual Occupancy",
        "value": oracle_mae,
        "unit": "W"
    },
    {
        "metric": "Energy RMSE - Actual Occupancy",
        "value": oracle_rmse,
        "unit": "W"
    },
    {
        "metric": "Energy R2 - Actual Occupancy",
        "value": oracle_r2,
        "unit": ""
    },
    {
        "metric": "Energy MAE - Predicted Occupancy",
        "value": pipeline_mae,
        "unit": "W"
    },
    {
        "metric": "Energy RMSE - Predicted Occupancy",
        "value": pipeline_rmse,
        "unit": "W"
    },
    {
        "metric": "Energy R2 - Predicted Occupancy",
        "value": pipeline_r2,
        "unit": ""
    },
    {
        "metric": "Actual Peak Power",
        "value": actual_peak,
        "unit": "W"
    },
    {
        "metric": "Predicted Peak Power",
        "value": predicted_peak,
        "unit": "W"
    },
    {
        "metric": "Optimized Peak Power",
        "value": optimized_peak,
        "unit": "W"
    },
    {
        "metric": "Peak Reduction",
        "value": peak_reduction,
        "unit": "%"
    },
    {
        "metric": "Potential Energy Saving",
        "value": total_saving,
        "unit": "W-equivalent aggregate"
    },
    {
        "metric": "Potential Saving Percentage",
        "value": saving_percent,
        "unit": "%"
    }
])


metrics.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "iitd_complete_validation_metrics.csv"
    ),
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("==========================================")
print(" COMPLETE IIT-D PIPELINE VALIDATION")
print("==========================================")

print()
print("OCCUPANCY MODEL")
print("----------------")
print(
    f"MAE  : {occupancy_mae:.4f} people"
)

print(
    f"RMSE : {occupancy_rmse:.4f} people"
)

print(
    f"R2   : {occupancy_r2:.4f}"
)


print()
print("ENERGY MODEL - ACTUAL OCCUPANCY")
print("--------------------------------")
print(
    f"MAE  : {oracle_mae:.4f} W"
)

print(
    f"RMSE : {oracle_rmse:.4f} W"
)

print(
    f"R2   : {oracle_r2:.4f}"
)


print()
print("ENERGY MODEL - PREDICTED OCCUPANCY")
print("-----------------------------------")
print(
    f"MAE  : {pipeline_mae:.4f} W"
)

print(
    f"RMSE : {pipeline_rmse:.4f} W"
)

print(
    f"R2   : {pipeline_r2:.4f}"
)


print()
print("OPTIMIZATION")
print("------------")

print(
    f"Predicted energy : {total_predicted:.2f}"
)

print(
    f"Optimized energy : {total_optimized:.2f}"
)

print(
    f"Potential saving : {total_saving:.2f}"
)

print(
    f"Saving percentage: {saving_percent:.2f}%"
)

print()
print("PEAK DEMAND")
print("-----------")

print(
    f"Actual peak     : {actual_peak:.2f} W"
)

print(
    f"Predicted peak  : {predicted_peak:.2f} W"
)

print(
    f"Optimized peak  : {optimized_peak:.2f} W"
)

print(
    f"Peak reduction  : {peak_reduction:.2f}%"
)


print()
print("Building summary:")
print(
    building_summary.to_string(
        index=False
    )
)


print()
print("Validation files saved successfully.")