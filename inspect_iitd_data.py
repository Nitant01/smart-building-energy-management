import pandas as pd
import os

ENERGY_FOLDER = "energy_dataset"
OCCUPANCY_FOLDER = "IIITD_occupancy_dataset"


print("=" * 70)
print("MAIN ENERGY DATA")
print("=" * 70)

energy_files = [
    "all_buildings_power.csv",
    "acad_build_mains.csv",
    "boys_hostel_mains.csv",
    "facilities_build_mains.csv",
    "girls_hostel_mains.csv",
    "lecture_build_mains.csv",
    "library_build_mains.csv",
    "mess_build_mains.csv"
]

for filename in energy_files:

    path = os.path.join(ENERGY_FOLDER, filename)

    if not os.path.exists(path):
        print(f"\n{filename} -> FILE NOT FOUND")
        continue

    df = pd.read_csv(path)

    print(f"\n--- {filename} ---")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))


print("\n")
print("=" * 70)
print("OCCUPANCY DATA")
print("=" * 70)


occupancy_files = [
    "ACB.csv",
    "BH.csv",
    "DB.csv",
    "GH.csv",
    "LB.csv",
    "LCB.csv",
    "SRB.csv"
]

for filename in occupancy_files:

    path = os.path.join(OCCUPANCY_FOLDER, filename)

    if not os.path.exists(path):
        print(f"\n{filename} -> FILE NOT FOUND")
        continue

    df = pd.read_csv(path)

    print(f"\n--- {filename} ---")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))