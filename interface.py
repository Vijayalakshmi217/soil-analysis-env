"""
inference.py - OpenEnv compatible inference script for Soil Analysis Environment
"""

import requests

BASE_URL = "http://localhost:7860"


def reset(task="easy", seed=42, session_id="default"):
    """Reset the environment and get initial observation."""
    payload = {
        "task": task,
        "seed": seed,
        "session_id": session_id
    }
    response = requests.post(f"{BASE_URL}/reset", json=payload)
    response.raise_for_status()
    return response.json()


def step(soil_type, fertilizer=None, crop=None, session_id="default"):
    """Take a step in the environment."""
    payload = {
        "session_id": session_id,
        "soil_type": soil_type
    }
    if fertilizer:
        payload["fertilizer"] = fertilizer
    if crop:
        payload["crop"] = crop

    response = requests.post(f"{BASE_URL}/step", json=payload)
    response.raise_for_status()
    return response.json()


def predict_soil(ph, moisture, nitrogen, organic):
    """
    Rule-based soil type prediction.

    Soil characteristics:
    - sandy:  low moisture below 25, low nitrogen below 0.15, low organic below 1.8
    - clay:   high moisture above 40, high organic above 2.0
    - loamy:  medium moisture 25 to 50, medium nitrogen above 0.15
    - silty:  medium moisture 25 to 55, moderate organic above 1.3
    - peaty:  very high moisture above 50, high nitrogen above 0.20, low ph below 5.5
    """

    # Peaty: very acidic, very wet, high nitrogen
    if ph < 5.5 and moisture > 50 and nitrogen >= 0.20:
        return "peaty"

    # Sandy: low moisture is the strongest signal
    if moisture < 25 and nitrogen < 0.15 and organic < 1.8:
        return "sandy"

    # Clay: high moisture, high organic
    if moisture >= 40 and organic >= 2.0:
        return "clay"

    # Loamy: balanced medium moisture, medium nitrogen, good organic
    if 25 <= moisture <= 50 and nitrogen >= 0.15 and organic >= 1.5 and ph <= 7.2:
        return "loamy"

    # Silty: medium moisture, moderate organic
    if 25 <= moisture <= 55 and organic >= 1.3:
        return "silty"

    # Sandy fallback: if moisture is low
    if moisture < 30:
        return "sandy"

    # Default fallback
    return "loamy"


def get_fertilizer(soil_type):
    """Get best fertilizer for soil type."""
    fertilizer_map = {
        "peaty": "lime",
        "sandy": "npk",
        "clay": "compost",
        "loamy": "balanced",
        "silty": "phosphorus"
    }
    return fertilizer_map.get(soil_type, "balanced")


def get_crop(soil_type):
    """Get best crop for soil type."""
    crop_map = {
        "peaty": "blueberry",
        "sandy": "carrot",
        "clay": "wheat",
        "loamy": "maize",
        "silty": "rice"
    }
    return crop_map.get(soil_type, "maize")


def run_inference(task="easy", session_id="default"):
    """Run one episode of inference."""
    result = reset(task=task, session_id=session_id)
    obs = result["observation"]

    soil_readings = obs.get("soil_readings", {})
    ph = soil_readings.get("ph", 6.5)
    moisture = soil_readings.get("moisture_pct", 40)
    nitrogen = soil_readings.get("nitrogen_pct", 0.1)
    organic = soil_readings.get("organic_matter_pct", 2.0)

    print("  Readings -> ph:" + str(ph) + ", moisture:" + str(moisture) + ", nitrogen:" + str(nitrogen) + ", organic:" + str(organic))

    soil_type = predict_soil(ph, moisture, nitrogen, organic)
    print("  Predicted soil type: " + soil_type)

    fertilizer = None
    crop = None

    if task in ("medium", "hard"):
        fertilizer = get_fertilizer(soil_type)
        print("  Fertilizer: " + str(fertilizer))

    if task == "hard":
        crop = get_crop(soil_type)
        print("  Crop: " + str(crop))

    step_result = step(
        soil_type=soil_type,
        fertilizer=fertilizer,
        crop=crop,
        session_id=session_id
    )
    return step_result


if __name__ == "__main__":
    print("Running Soil Analysis Inference...")
    print("=" * 50)
    for task in ("easy", "medium", "hard"):
        print("Task: " + task)
        result = run_inference(task=task, session_id=task)
        reward = result.get("reward", 0)
        done = result.get("done", False)
        true_soil = result.get("info", {}).get("true_soil_type", "unknown")
        print("  True soil: " + str(true_soil))
        print("  Reward: " + str(reward) + " | Done: " + str(done))
    print("=" * 50)
    print("Inference complete!")
