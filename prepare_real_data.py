import pandas as pd
import numpy as np


print("Reading files...")

electricity = pd.read_csv("electricity.csv")
metadata = pd.read_csv("metadata.csv")
weather = pd.read_csv("weather.csv")

print("Files loaded successfully.")

electricity["timestamp"] = pd.to_datetime(
    electricity["timestamp"],
    errors="coerce"
)

weather["timestamp"] = pd.to_datetime(
    weather["timestamp"],
    errors="coerce"
)

metadata["building_id"] = metadata["building_id"].astype(str)

electricity = electricity.dropna(
    subset=["timestamp"]
)

weather = weather.dropna(
    subset=["timestamp"]
)


# ---------------------------------------------------------
# Find available building columns
# ---------------------------------------------------------

building_columns = [
    col for col in electricity.columns
    if col != "timestamp"
]

print("Number of buildings found:", len(building_columns))


# ---------------------------------------------------------
# Select building
# ---------------------------------------------------------

preferred_building = "Hog_education_Janell"

if preferred_building in building_columns:
    selected_building = preferred_building

else:

    print("Preferred building not found.")
    print("Finding another suitable building...")

    best_building = None
    best_score = -np.inf

    for building in building_columns:

        energy = electricity[building].dropna()

        if len(energy) < 5000:
            continue

        energy_mean = energy.mean()
        energy_std = energy.std()

        if energy_mean <= 10:
            continue

        if energy_std <= 1:
            continue

        building_lower = building.lower()

        score = energy_mean * energy_std

        if "education" in building_lower:
            score *= 2.0

        elif "office" in building_lower:
            score *= 1.5

        elif "school" in building_lower:
            score *= 1.8

        elif "university" in building_lower:
            score *= 1.8

        elif "parking" in building_lower:
            score *= 0.3

        if score > best_score:
            best_score = score
            best_building = building

    if best_building is None:
        raise ValueError(
            "No suitable building could be selected."
        )

    selected_building = best_building


print("Selected building:", selected_building)


# ---------------------------------------------------------
# Create building energy dataframe
# ---------------------------------------------------------

building_data = electricity[
    ["timestamp", selected_building]
].copy()

building_data = building_data.rename(
    columns={
        selected_building: "energy"
    }
)

building_data["building_id"] = selected_building

building_data = building_data.dropna(
    subset=["energy"]
)


# ---------------------------------------------------------
# Get metadata
# ---------------------------------------------------------

building_metadata = metadata[
    metadata["building_id"] == selected_building
].copy()

if len(building_metadata) == 0:
    raise ValueError(
        "Metadata not found for selected building."
    )

site_id = building_metadata.iloc[0]["site_id"]

building_data["site_id"] = site_id

print("Site:", site_id)


# ---------------------------------------------------------
# Get weather data
# ---------------------------------------------------------

site_weather = weather[
    weather["site_id"] == site_id
].copy()

site_weather = site_weather.sort_values(
    "timestamp"
)

site_weather = site_weather.drop_duplicates(
    subset=["timestamp", "site_id"],
    keep="first"
)


# ---------------------------------------------------------
# Merge energy and weather
# ---------------------------------------------------------

building_data = pd.merge(
    building_data,
    site_weather,
    on=["timestamp", "site_id"],
    how="left"
)


# ---------------------------------------------------------
# Time features
# ---------------------------------------------------------

building_data["hour"] = (
    building_data["timestamp"].dt.hour
)

building_data["day_of_week"] = (
    building_data["timestamp"].dt.dayofweek
)

building_data["month"] = (
    building_data["timestamp"].dt.month
)

building_data["is_weekend"] = (
    building_data["day_of_week"] >= 5
).astype(int)


# ---------------------------------------------------------
# Keep required columns
# ---------------------------------------------------------

columns_to_keep = [
    "timestamp",
    "building_id",
    "site_id",
    "energy",
    "airTemperature",
    "cloudCoverage",
    "dewTemperature",
    "windSpeed",
    "hour",
    "day_of_week",
    "month",
    "is_weekend"
]

available_columns = [
    col
    for col in columns_to_keep
    if col in building_data.columns
]

building_data = building_data[
    available_columns
].copy()


# ---------------------------------------------------------
# Remove duplicate timestamps
# ---------------------------------------------------------

building_data = (
    building_data
    .sort_values("timestamp")
    .drop_duplicates(
        subset=["timestamp"],
        keep="first"
    )
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------

start_time = building_data["timestamp"].min()
end_time = building_data["timestamp"].max()

expected_hours = (
    int(
        (
            end_time - start_time
        ).total_seconds() / 3600
    )
    + 1
)

actual_hours = len(
    building_data
)

missing_hours = max(
    expected_hours - actual_hours,
    0
)

duplicate_timestamps = (
    building_data["timestamp"]
    .duplicated()
    .sum()
)

zero_energy_records = (
    building_data["energy"] == 0
).sum()

weather_columns = [
    "airTemperature",
    "cloudCoverage",
    "dewTemperature",
    "windSpeed"
]

available_weather_columns = [
    col
    for col in weather_columns
    if col in building_data.columns
]

missing_weather_values = (
    building_data[
        available_weather_columns
    ]
    .isna()
    .sum()
    .sum()
)


summary = pd.DataFrame({
    "building_id": [
        selected_building
    ],
    "site_id": [
        site_id
    ],
    "rows": [
        len(building_data)
    ],
    "start": [
        start_time
    ],
    "end": [
        end_time
    ],
    "energy_min": [
        building_data["energy"].min()
    ],
    "energy_max": [
        building_data["energy"].max()
    ],
    "energy_mean": [
        building_data["energy"].mean()
    ],
    "energy_std": [
        building_data["energy"].std()
    ],
    "duplicate_timestamps": [
        duplicate_timestamps
    ],
    "missing_hours": [
        missing_hours
    ],
    "zero_energy_records": [
        zero_energy_records
    ],
    "missing_weather_values": [
        missing_weather_values
    ]
})

summary.to_csv(
    "dataset_summary.csv",
    index=False
)


# ---------------------------------------------------------
# Save final dataset
# ---------------------------------------------------------

building_data.to_csv(
    "real_building_data.csv",
    index=False
)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print()
print("Real dataset created successfully.")
print()
print("Selected building:", selected_building)
print("Site:", site_id)
print("Rows:", len(building_data))
print("Columns:", len(building_data.columns))
print("Start:", start_time)
print("End:", end_time)
print("Energy minimum:", building_data["energy"].min())
print("Energy maximum:", building_data["energy"].max())
print("Energy mean:", building_data["energy"].mean())
print("Energy standard deviation:", building_data["energy"].std())
print("Missing hours:", missing_hours)
print("Zero-energy records:", zero_energy_records)
print("Missing weather values:", missing_weather_values)
print()
print("Saved as: real_building_data.csv")
print("Saved as: dataset_summary.csv")
print()
print(building_data.head())