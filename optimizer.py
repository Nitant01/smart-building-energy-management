def optimize_energy(
    predicted_energy,
    occupancy,
    hvac_mode="Medium",
    lighting_mode="Medium"
):
    if predicted_energy <= 0:
        return {
            "optimized_energy": 0.0,
            "total_saving": 0.0,
            "saving_percent": 0.0,
            "recommendation": "No energy demand predicted.",
            "recommended_hvac": "Off",
            "recommended_lighting": "Off"
        }

    HVAC_SHARE = 0.50
    LIGHTING_SHARE = 0.15

    if hvac_mode == "High":
        hvac_reduction = 0.00

    elif hvac_mode == "Medium":
        hvac_reduction = 0.10

    else:
        hvac_reduction = 0.20

    if lighting_mode == "High":
        lighting_reduction = 0.00

    elif lighting_mode == "Medium":
        lighting_reduction = 0.15

    else:
        lighting_reduction = 0.30

    if occupancy <= 20:
        recommended_hvac = "Low"
        recommended_lighting = "Low"

        recommendation = (
            "Low occupancy detected. Use Low HVAC and Low lighting."
        )

    elif occupancy <= 50:
        recommended_hvac = "Medium"
        recommended_lighting = "Medium"

        recommendation = (
            "Moderate occupancy detected. Use Medium HVAC and Medium lighting."
        )

    else:
        recommended_hvac = "High"
        recommended_lighting = "High"

        recommendation = (
            "High occupancy detected. Use High HVAC and High lighting "
            "to maintain building comfort."
        )

    hvac_saving = (
        predicted_energy
        * HVAC_SHARE
        * hvac_reduction
    )

    lighting_saving = (
        predicted_energy
        * LIGHTING_SHARE
        * lighting_reduction
    )

    total_saving = (
        hvac_saving
        + lighting_saving
    )

    optimized_energy = (
        predicted_energy
        - total_saving
    )

    saving_percent = (
        total_saving
        / predicted_energy
    ) * 100

    return {
        "optimized_energy": optimized_energy,
        "total_saving": total_saving,
        "saving_percent": saving_percent,
        "recommendation": recommendation,
        "recommended_hvac": recommended_hvac,
        "recommended_lighting": recommended_lighting
    }


def comfort_aware_optimization(
    predicted_energy,
    occupancy,
    temperature,
    humidity,
    hvac_mode="Medium",
    lighting_mode="Medium",
    min_temperature=22.0,
    max_temperature=26.0
):
    comfort_ok = (
        min_temperature
        <= temperature
        <= max_temperature
        and
        30.0
        <= humidity
        <= 70.0
    )

    if comfort_ok:

        result = optimize_energy(
            predicted_energy,
            occupancy,
            hvac_mode,
            lighting_mode
        )

        result["comfort_status"] = (
            "Comfort condition is within the simulated target range."
        )

        return result

    if temperature > max_temperature:

        recommended_hvac = "High"

        recommendation = (
            "Simulated temperature is above the comfort range. "
            "Prioritize higher HVAC operation while optimizing lighting."
        )

    elif temperature < min_temperature:

        recommended_hvac = "Low"

        recommendation = (
            "Simulated temperature is below the comfort range. "
            "Avoid unnecessary cooling and optimize lighting."
        )

    else:

        recommended_hvac = hvac_mode

        recommendation = (
            "Simulated humidity is outside the preferred range. "
            "Maintain the selected HVAC level and avoid aggressive energy reduction."
        )

    result = optimize_energy(
        predicted_energy,
        occupancy,
        recommended_hvac,
        lighting_mode
    )

    result["recommended_hvac"] = recommended_hvac

    result["recommendation"] = recommendation

    result["comfort_status"] = (
        "Simulated comfort condition is outside the target range. "
        "HVAC operation has been adjusted to prioritize comfort."
    )

    return result


def optimize_building(
    predicted_energy,
    occupancy,
    temperature,
    current_ac,
    current_lights,
    comfort_min=22.0,
    comfort_max=26.0
):
    if not current_ac and not current_lights:

        return (
            float(predicted_energy),
            0.0,
            0.0,
            "HVAC and lighting are currently OFF."
        )

    if occupancy <= 20:

        hvac_mode = "Low"

        lighting_mode = "Low"

    elif occupancy <= 50:

        hvac_mode = "Medium"

        lighting_mode = "Medium"

    else:

        hvac_mode = "High"

        lighting_mode = "High"

    if not current_ac:
        hvac_mode = "High"

    if not current_lights:
        lighting_mode = "High"

    result = comfort_aware_optimization(
        predicted_energy=predicted_energy,
        occupancy=occupancy,
        temperature=temperature,
        humidity=50.0,
        hvac_mode=hvac_mode,
        lighting_mode=lighting_mode,
        min_temperature=comfort_min,
        max_temperature=comfort_max
    )

    return (
        result["optimized_energy"],
        result["total_saving"],
        result["saving_percent"],
        result["recommendation"]
    )