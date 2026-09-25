import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("building_data.csv")

# -------------------------------
# Occupancy Prediction Model
# -------------------------------

X_occupancy = df[
    [
        "day",
        "hour",
        "temperature",
        "humidity",
        "light_level"
    ]
]

y_occupancy = df["occupancy"]

X_train, X_test, y_train, y_test = train_test_split(
    X_occupancy,
    y_occupancy,
    test_size=0.2,
    random_state=42
)

occupancy_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

occupancy_model.fit(X_train, y_train)

occupancy_prediction = occupancy_model.predict(X_test)

print("Occupancy MAE:",
      mean_absolute_error(y_test, occupancy_prediction))

print("Occupancy R2:",
      r2_score(y_test, occupancy_prediction))

joblib.dump(
    occupancy_model,
    "occupancy_model.pkl"
)


# -------------------------------
# Energy Prediction Model
# -------------------------------

X_energy = df[
    [
        "hour",
        "occupancy",
        "temperature",
        "humidity",
        "light_level",
        "ac",
        "lights"
    ]
]

y_energy = df["energy"]

X_train, X_test, y_train, y_test = train_test_split(
    X_energy,
    y_energy,
    test_size=0.2,
    random_state=42
)

energy_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

energy_model.fit(X_train, y_train)

energy_prediction = energy_model.predict(X_test)

print("Energy MAE:",
      mean_absolute_error(y_test, energy_prediction))

print("Energy R2:",
      r2_score(y_test, energy_prediction))

joblib.dump(
    energy_model,
    "energy_model.pkl"
)

print("Models saved successfully!")