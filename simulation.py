import numpy as np


CLIMATE_PROFILES = {
    "Hot & Dry": {
        "base_temperature": 31.0,
        "temperature_amplitude": 7.0,
        "humidity": 45.0
    },
    "Warm & Humid": {
        "base_temperature": 29.0,
        "temperature_amplitude": 5.0,
        "humidity": 75.0
    },
    "Composite": {
        "base_temperature": 27.0,
        "temperature_amplitude": 8.0,
        "humidity": 60.0
    },
    "Temperate": {
        "base_temperature": 24.0,
        "temperature_amplitude": 6.0,
        "humidity": 55.0
    },
    "Cold": {
        "base_temperature": 18.0,
        "temperature_amplitude": 5.0,
        "humidity": 50.0
    }
}


def simulate_building_conditions(
    occupancy,
    hour,
    climate_zone="Composite"
):
    profile = CLIMATE_PROFILES[climate_zone]

    temperature = (
        profile["base_temperature"]
        + profile["temperature_amplitude"]
        * np.sin((hour - 6) * np.pi / 12)
    )

    daylight = max(
        0.0,
        np.sin((hour - 6) * np.pi / 12)
    ) * 100.0

    humidity = profile["humidity"]

    if occupancy <= 20:
        hvac_status = "Low"
        lighting_status = "Low"

    elif occupancy <= 50:
        hvac_status = "Medium"
        lighting_status = "Medium"

    else:
        hvac_status = "High"
        lighting_status = "High"

    if daylight > 60:
        lighting_status = "Low"

    return {
        "temperature": round(float(temperature), 2),
        "humidity": round(float(humidity), 2),
        "daylight": round(float(daylight), 2),
        "hvac_status": hvac_status,
        "lighting_status": lighting_status
    }


def check_comfort(
    temperature,
    humidity,
    min_temperature=22.0,
    max_temperature=26.0
):
    temperature_ok = (
        min_temperature
        <= temperature
        <= max_temperature
    )

    humidity_ok = (
        30.0
        <= humidity
        <= 70.0
    )

    return temperature_ok and humidity_ok