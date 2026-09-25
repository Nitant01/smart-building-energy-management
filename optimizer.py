def optimize_building(
    occupancy,
    temperature,
    humidity,
    current_ac,
    current_lights
):

    recommendation = []

    optimized_ac = current_ac
    optimized_lights = current_lights

    # Low occupancy
    if occupancy <= 5:

        optimized_lights = 0

        if temperature < 28:
            optimized_ac = 0

        recommendation.append(
            "Low occupancy detected. Reduce unnecessary HVAC and lighting."
        )

    # Medium occupancy
    elif occupancy <= 20:

        if temperature <= 28:
            optimized_ac = 0

        recommendation.append(
            "Moderate occupancy. Use adaptive cooling and lighting."
        )

    # High occupancy
    else:

        optimized_ac = 1
        optimized_lights = 1

        recommendation.append(
            "High occupancy detected. Maintain cooling and lighting for comfort."
        )

    # High temperature safety condition
    if temperature > 32:

        optimized_ac = 1

        recommendation.append(
            "High temperature detected. Cooling should remain active."
        )

    return {
        "optimized_ac": optimized_ac,
        "optimized_lights": optimized_lights,
        "recommendation": " ".join(recommendation)
    }