import pandas as pd

from optimizer import optimize_energy


# ---------------------------------------------------------
# Load model predictions
# ---------------------------------------------------------

print("Loading test predictions...")

df = pd.read_csv(
    "test_predictions.csv"
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)


# ---------------------------------------------------------
# Fixed evaluation scenario
# ---------------------------------------------------------

OCCUPANCY = 40

TEMPERATURE = 24.0

HVAC_ON = True

LIGHTS_ON = True


# ---------------------------------------------------------
# Apply optimization
# ---------------------------------------------------------

results = []

for _, row in df.iterrows():

    predicted_energy = row[
        "predicted_energy"
    ]

    (
        optimized_energy,
        saving,
        saving_percent,
        recommendation
    ) = optimize_energy(
        predicted_energy=predicted_energy,
        occupancy=OCCUPANCY,
        temperature=TEMPERATURE,
        hvac_on=HVAC_ON,
        lights_on=LIGHTS_ON
    )

    results.append({

        "timestamp": row["timestamp"],

        "actual_energy":
            row["actual_energy"],

        "predicted_energy":
            predicted_energy,

        "optimized_energy":
            optimized_energy,

        "energy_saved":
            saving,

        "saving_percent":
            saving_percent,

        "occupancy":
            OCCUPANCY,

        "temperature":
            TEMPERATURE,

        "hvac_on":
            HVAC_ON,

        "lights_on":
            LIGHTS_ON,

        "recommendation":
            recommendation
    })


# ---------------------------------------------------------
# Create results dataframe
# ---------------------------------------------------------

results_df = pd.DataFrame(
    results
)


# ---------------------------------------------------------
# Overall metrics
# ---------------------------------------------------------

actual_energy = (
    results_df["actual_energy"]
    .sum()
)

predicted_energy = (
    results_df["predicted_energy"]
    .sum()
)

optimized_energy = (
    results_df["optimized_energy"]
    .sum()
)

total_saving = (
    results_df["energy_saved"]
    .sum()
)


if predicted_energy > 0:

    saving_percent = (
        total_saving
        / predicted_energy
    ) * 100

else:

    saving_percent = 0.0


# ---------------------------------------------------------
# Peak demand
# ---------------------------------------------------------

baseline_peak = (
    results_df["predicted_energy"]
    .max()
)

optimized_peak = (
    results_df["optimized_energy"]
    .max()
)


if baseline_peak > 0:

    peak_reduction_percent = (
        (baseline_peak - optimized_peak)
        / baseline_peak
    ) * 100

else:

    peak_reduction_percent = 0.0


# ---------------------------------------------------------
# Save detailed results
# ---------------------------------------------------------

results_df.to_csv(
    "optimization_results.csv",
    index=False
)


# ---------------------------------------------------------
# Optimization summary
# ---------------------------------------------------------

summary = pd.DataFrame({

    "metric": [

        "Test Start",

        "Test End",

        "Records",

        "Actual Historical Energy",

        "Predicted Baseline Energy",

        "Optimized Energy",

        "Potential Energy Saved",

        "Energy Saving Percent",

        "Baseline Peak Energy",

        "Optimized Peak Energy",

        "Peak Reduction Percent",

        "Occupancy Assumption",

        "Temperature Assumption",

        "HVAC State",

        "Lighting State"

    ],

    "value": [

        str(results_df["timestamp"].min()),

        str(results_df["timestamp"].max()),

        len(results_df),

        actual_energy,

        predicted_energy,

        optimized_energy,

        total_saving,

        saving_percent,

        baseline_peak,

        optimized_peak,

        peak_reduction_percent,

        OCCUPANCY,

        TEMPERATURE,

        HVAC_ON,

        LIGHTS_ON

    ]

})


summary.to_csv(
    "optimization_summary.csv",
    index=False
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print()
print("========================================")
print("OPTIMIZATION EVALUATION")
print("========================================")

print(
    "Test period:",
    results_df["timestamp"].min(),
    "to",
    results_df["timestamp"].max()
)

print(
    "Records:",
    len(results_df)
)

print(
    "Actual historical energy:",
    round(actual_energy, 2)
)

print(
    "Predicted baseline energy:",
    round(predicted_energy, 2)
)

print(
    "Optimized energy:",
    round(optimized_energy, 2)
)

print(
    "Potential energy saved:",
    round(total_saving, 2)
)

print(
    "Energy saving %:",
    round(saving_percent, 2)
)

print(
    "Baseline peak:",
    round(baseline_peak, 2)
)

print(
    "Optimized peak:",
    round(optimized_peak, 2)
)

print(
    "Peak reduction %:",
    round(peak_reduction_percent, 2)
)

print()
print("Scenario assumptions:")
print("Occupancy:", OCCUPANCY, "%")
print("Temperature:", TEMPERATURE, "°C")
print("HVAC ON:", HVAC_ON)
print("Lights ON:", LIGHTS_ON)

print()
print("Files created:")
print("- optimization_results.csv")
print("- optimization_summary.csv")

print()
print("Optimization evaluation completed.")