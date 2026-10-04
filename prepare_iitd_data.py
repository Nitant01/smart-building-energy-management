import os
import pandas as pd

ENERGY_FOLDER = "energy_dataset"
OCCUPANCY_FOLDER = "IIITD_occupancy_dataset"
OUTPUT_FOLDER = "iitd_processed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

BUILDING_MAPPING = {
    "Academic": ("ACB.csv", "Academic"),
    "Boys": ("BH.csv", "Boys_main"),
    "Dining": ("DB.csv", "Mess"),
    "Girls": ("GH.csv", "Girls_main"),
    "Library": ("LB.csv", "Library"),
    "Lecture": ("LCB.csv", "Lecture"),
    "Facilities": ("SRB.csv", "Facilities")
}


def convert_timestamp(series):
    return (
        pd.to_datetime(series, unit="s", utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
    )


all_data = []
summary = []

energy_file = os.path.join(ENERGY_FOLDER, "all_buildings_power.csv")

energy = pd.read_csv(energy_file)

energy["timestamp"] = convert_timestamp(energy["timestamp"])

energy = energy.set_index("timestamp")

energy_10min = energy.resample("10min").mean()

for building, (occupancy_file, energy_column) in BUILDING_MAPPING.items():

    print(f"\nProcessing {building}...")

    occupancy_path = os.path.join(
        OCCUPANCY_FOLDER,
        occupancy_file
    )

    occupancy = pd.read_csv(occupancy_path)

    occupancy["timestamp"] = convert_timestamp(
        occupancy["timestamp"]
    )

    occupancy = occupancy[
        ["timestamp", "occupancy_count"]
    ].copy()

    occupancy = occupancy.sort_values("timestamp")

    building_energy = energy_10min[
        [energy_column]
    ].copy()

    building_energy = building_energy.rename(
        columns={energy_column: "power"}
    )

    building_energy = building_energy.reset_index()

    merged = pd.merge_asof(
        occupancy.sort_values("timestamp"),
        building_energy.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("5min")
    )

    merged["building"] = building

    merged = merged.dropna(
        subset=["occupancy_count", "power"]
    )

    merged = merged[
        [
            "timestamp",
            "building",
            "power",
            "occupancy_count"
        ]
    ]

    all_data.append(merged)

    summary.append({
        "building": building,
        "records": len(merged),
        "start": merged["timestamp"].min(),
        "end": merged["timestamp"].max(),
        "average_power_w": merged["power"].mean(),
        "average_occupancy": merged["occupancy_count"].mean(),
        "maximum_occupancy": merged["occupancy_count"].max()
    })

    print(
        f"Occupancy records: {len(occupancy)}"
    )

    print(
        f"Merged records: {len(merged)}"
    )

    print(
        f"Time range: {merged['timestamp'].min()} "
        f"to {merged['timestamp'].max()}"
    )


combined = pd.concat(
    all_data,
    ignore_index=True
)

combined = combined.sort_values(
    ["building", "timestamp"]
)

combined.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "iitd_energy_occupancy.csv"
    ),
    index=False
)

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "iitd_summary.csv"
    ),
    index=False
)

print("\n--------------------------------")
print("IITD DATA PREPARATION COMPLETED")
print("--------------------------------")

print(
    f"Combined shape: {combined.shape}"
)

print("\nBuilding summary:")
print(summary_df.to_string(index=False))

print(
    "\nSaved:"
)

print(
    os.path.join(
        OUTPUT_FOLDER,
        "iitd_energy_occupancy.csv"
    )
)

print(
    os.path.join(
        OUTPUT_FOLDER,
        "iitd_summary.csv"
    )
)