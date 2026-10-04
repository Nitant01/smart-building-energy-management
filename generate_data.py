import pandas as pd
import numpy as np


np.random.seed(42)


rows = []


for day in range(1, 31):

    for hour in range(8, 19):

        # -------------------------------------------------
        # Simulated occupancy
        # -------------------------------------------------

        if hour in [10, 11, 12]:

            occupancy = np.random.randint(
                25,
                41
            )

        elif hour in [13, 14]:

            occupancy = np.random.randint(
                10,
                25
            )

        elif hour in [15, 16]:

            occupancy = np.random.randint(
                20,
                36
            )

        else:

            occupancy = np.random.randint(
                2,
                15
            )


        # -------------------------------------------------
        # Simulated environmental conditions
        # -------------------------------------------------

        temperature = np.random.normal(
            29 + (hour - 12) * 0.3,
            1.5
        )

        humidity = np.random.normal(
            60,
            5
        )

        light_level = np.random.randint(
            250,
            600
        )


        # -------------------------------------------------
        # Equipment status
        # -------------------------------------------------

        ac = int(
            temperature > 27
            and occupancy > 5
        )

        lights = int(
            occupancy > 0
        )


        # -------------------------------------------------
        # Energy consumption
        # -------------------------------------------------

        base_energy = 0.8

        ac_energy = (
            ac
            * (
                1.5
                + occupancy * 0.04
            )
        )

        lighting_energy = (
            lights
            * (
                0.5
                + occupancy * 0.01
            )
        )

        energy = (
            base_energy
            + ac_energy
            + lighting_energy
        )

        energy += np.random.normal(
            0,
            0.15
        )

        energy = max(
            energy,
            0
        )


        rows.append([

            day,

            hour,

            occupancy,

            temperature,

            humidity,

            light_level,

            ac,

            lights,

            energy

        ])


# ---------------------------------------------------------
# Create dataframe
# ---------------------------------------------------------

columns = [

    "day",

    "hour",

    "occupancy",

    "temperature",

    "humidity",

    "light_level",

    "ac",

    "lights",

    "energy"

]


df = pd.DataFrame(
    rows,
    columns=columns
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    "building_data.csv",
    index=False
)


print(
    "Synthetic dataset created successfully."
)

print()

print(
    "Rows:",
    len(df)
)

print()

print(
    df.head()
)