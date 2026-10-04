import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


INPUT_FILE = "iitd_processed/iitd_energy_occupancy.csv"
OUTPUT_FOLDER = "iitd_processed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    ["building", "timestamp"]
).reset_index(drop=True)


# -----------------------------
# Time features
# -----------------------------

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


# -----------------------------
# Energy lag features
# -----------------------------

energy_group = df.groupby(
    "building"
)["power"]

df["energy_lag_1"] = energy_group.shift(1)

df["energy_lag_6"] = energy_group.shift(6)

df["energy_lag_144"] = energy_group.shift(144)

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


# -----------------------------
# Occupancy lag features
# -----------------------------

occupancy_group = df.groupby(
    "building"
)["occupancy_count"]

df["occupancy_lag_1"] = occupancy_group.shift(1)

df["occupancy_lag_6"] = occupancy_group.shift(6)

df["occupancy_lag_144"] = occupancy_group.shift(144)


df = df.dropna().reset_index(drop=True)


# -----------------------------
# Building encoding
# -----------------------------

df = pd.get_dummies(
    df,
    columns=["building"],
    dtype=float
)


building_columns = [
    col
    for col in df.columns
    if col.startswith("building_")
]


feature_columns = [
    "hour",
    "minute",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "hour_decimal",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "energy_lag_1",
    "energy_lag_6",
    "energy_lag_144",
    "energy_rolling_6",
    "energy_rolling_144",
    "occupancy_count",
    "occupancy_lag_1",
    "occupancy_lag_6",
    "occupancy_lag_144"
] + building_columns


X = df[feature_columns]

y = df["power"]


# -----------------------------
# Chronological split
# -----------------------------

split_time = df["timestamp"].quantile(0.80)

train_mask = (
    df["timestamp"] <= split_time
)

test_mask = (
    df["timestamp"] > split_time
)

X_train = X.loc[train_mask]

X_test = X.loc[test_mask]

y_train = y.loc[train_mask]

y_test = y.loc[test_mask]


print("\nEnergy model training")
print("---------------------")

print(
    f"Training records: {len(X_train)}"
)

print(
    f"Testing records: {len(X_test)}"
)

print(
    f"Training end: "
    f"{df.loc[train_mask, 'timestamp'].max()}"
)

print(
    f"Testing start: "
    f"{df.loc[test_mask, 'timestamp'].min()}"
)


# -----------------------------
# Model
# -----------------------------

model = HistGradientBoostingRegressor(
    max_iter=250,
    learning_rate=0.08,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# -----------------------------
# Evaluation
# -----------------------------

predictions = model.predict(
    X_test
)

predictions = np.maximum(
    predictions,
    0
)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\nEnergy model performance")
print("------------------------")

print(
    f"MAE:  {mae:.4f} W"
)

print(
    f"RMSE: {rmse:.4f} W"
)

print(
    f"R2:   {r2:.4f}"
)


# -----------------------------
# Save model
# -----------------------------

joblib.dump(
    {
        "model": model,
        "features": feature_columns
    },
    os.path.join(
        OUTPUT_FOLDER,
        "iitd_energy_model.pkl"
    )
)


# -----------------------------
# Save predictions
# -----------------------------

prediction_output = df.loc[
    test_mask,
    [
        "timestamp",
        "power"
    ]
].copy()

prediction_output[
    "predicted_power"
] = predictions

prediction_output.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "energy_predictions.csv"
    ),
    index=False
)


# -----------------------------
# Save metrics
# -----------------------------

metrics = pd.DataFrame([
    {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "train_records": len(X_train),
        "test_records": len(X_test)
    }
])

metrics.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "energy_model_metrics.csv"
    ),
    index=False
)


print("\nEnergy model saved successfully.")