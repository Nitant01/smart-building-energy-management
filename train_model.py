import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# ---------------------------------------------------------
# Load real building dataset
# ---------------------------------------------------------

print("Loading real building dataset...")

df = pd.read_csv("real_building_data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Start:", df["timestamp"].min())
print("End:", df["timestamp"].max())


# ---------------------------------------------------------
# Fill missing weather values
# ---------------------------------------------------------

weather_features = [
    "airTemperature",
    "cloudCoverage",
    "dewTemperature",
    "windSpeed"
]

for column in weather_features:
    if column in df.columns:
        df[column] = df[column].fillna(df[column].median())


# ---------------------------------------------------------
# Create cyclical time features
# ---------------------------------------------------------

df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

df["day_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["day_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)


# ---------------------------------------------------------
# Create historical energy features
# ---------------------------------------------------------

df["energy_lag_1"] = df["energy"].shift(1)

df["energy_lag_24"] = df["energy"].shift(24)

df["energy_lag_168"] = df["energy"].shift(168)


# ---------------------------------------------------------
# Create rolling energy features
# ---------------------------------------------------------

df["energy_rolling_24"] = (
    df["energy"]
    .shift(1)
    .rolling(window=24)
    .mean()
)

df["energy_rolling_168"] = (
    df["energy"]
    .shift(1)
    .rolling(window=168)
    .mean()
)


# ---------------------------------------------------------
# Features used by the model
# ---------------------------------------------------------

features = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",

    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",

    "airTemperature",
    "cloudCoverage",
    "dewTemperature",
    "windSpeed",

    "energy_lag_1",
    "energy_lag_24",
    "energy_lag_168",

    "energy_rolling_24",
    "energy_rolling_168"
]


target = "energy"


# ---------------------------------------------------------
# Remove rows where lag/rolling features are unavailable
# ---------------------------------------------------------

df_model = df.dropna(
    subset=features + [target]
).copy()

df_model = df_model.reset_index(drop=True)

print()
print("Rows available for modelling:", len(df_model))


# ---------------------------------------------------------
# Chronological train/test split
# ---------------------------------------------------------

split_index = int(
    len(df_model) * 0.80
)

train_data = df_model.iloc[:split_index].copy()

test_data = df_model.iloc[split_index:].copy()


X_train = train_data[features]

y_train = train_data[target]

X_test = test_data[features]

y_test = test_data[target]


print()
print("Training rows:", len(train_data))
print("Testing rows:", len(test_data))

print(
    "Training period:",
    train_data["timestamp"].min(),
    "to",
    train_data["timestamp"].max()
)

print(
    "Testing period:",
    test_data["timestamp"].min(),
    "to",
    test_data["timestamp"].max()
)


# ---------------------------------------------------------
# Train Random Forest model
# ---------------------------------------------------------

print()
print("Training Random Forest model...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ---------------------------------------------------------
# Make predictions
# ---------------------------------------------------------

y_pred = model.predict(
    X_test
)


# ---------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print()
print("========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(
    "MAE:",
    round(mae, 4)
)

print(
    "RMSE:",
    round(rmse, 4)
)

print(
    "R2 Score:",
    round(r2, 4)
)

print("========================================")


# ---------------------------------------------------------
# Save predictions for the test period
# ---------------------------------------------------------

test_predictions = test_data[
    [
        "timestamp",
        "energy"
    ]
].copy()

test_predictions = test_predictions.rename(
    columns={
        "energy": "actual_energy"
    }
)

test_predictions["predicted_energy"] = y_pred

test_predictions.to_csv(
    "test_predictions.csv",
    index=False
)


# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

feature_importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)


# ---------------------------------------------------------
# Save model and feature information
# ---------------------------------------------------------

model_bundle = {
    "model": model,
    "features": features
}

joblib.dump(
    model_bundle,
    "energy_model.pkl"
)


# ---------------------------------------------------------
# Save model summary
# ---------------------------------------------------------

model_summary = pd.DataFrame({
    "metric": [
        "MAE",
        "RMSE",
        "R2",
        "Training Rows",
        "Testing Rows",
        "Training Start",
        "Training End",
        "Testing Start",
        "Testing End"
    ],

    "value": [
        mae,
        rmse,
        r2,
        len(train_data),
        len(test_data),
        str(train_data["timestamp"].min()),
        str(train_data["timestamp"].max()),
        str(test_data["timestamp"].min()),
        str(test_data["timestamp"].max())
    ]
})

model_summary.to_csv(
    "model_summary.csv",
    index=False
)


# ---------------------------------------------------------
# Print important results
# ---------------------------------------------------------

print()
print("Files created successfully:")
print("- energy_model.pkl")
print("- test_predictions.csv")
print("- feature_importance.csv")
print("- model_summary.csv")

print()
print("Top features:")

print(
    feature_importance.head(10).to_string(
        index=False
    )
)

print()
print("Training completed successfully.")