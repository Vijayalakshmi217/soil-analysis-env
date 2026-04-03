"""
inference.py - OpenEnv compatible inference script for Soil Analysis Environment
This file is required by the OpenEnv hackathon checker.
"""

import requests

BASE_URL = "http://localhost:7860"


def reset(task: str = "easy", seed: int = 42, session_id: str = "default"):
    """Reset the environment and get initial observation."""
    payload = {
        "task": task,
        "seed": seed,
        "session_id": session_id
    }
    response = requests.post(f"{BASE_URL}/reset", json=payload)
    response.raise_for_status()
    return response.json()


def step(soil_type: str, fertilizer: str = None, crop: str = None, session_id: str = "default"):
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


def run_inference(task: str = "easy", session_id: str = "default"):
    """Run one episode of inference."""
    # Reset environment
    result = reset(task=task, session_id=session_id)
    obs = result["observation"]

    # Simple rule-based prediction
    soil_readings = obs.get("soil_readings", {})
    ph = soil_readings.get("ph", 6.5)
    moisture = soil_readings.get("moisture_pct", 40)
    nitrogen = soil_readings.get("nitrogen_pct", 0.1)
    organic = soil_readings.get("organic_matter_pct", 2.0)

    # Predict soil type based on readings
    soil_type = predict_soil(ph, moisture, nitrogen, organic)

    # Choose fertilizer and crop based on soil type
    fertilizer = None
    crop = None

    if task in ("medium", "hard"):
        fertilizer = get_fertilizer(soil_type)
    if task == "hard":
        crop = get_crop(soil_type)

    # Take step
    step_result = step(soil_type=soil_type, fertilizer=fertilizer, crop=crop, session_id=session_id)
    return step_result


def predict_soil(ph, moisture, nitrogen, organic):
    """Rule-based soil type prediction."""
    if ph < 5.5 and moisture > 55 and nitrogen >= 0.25:
        return "peaty"
    elif ph < 6.5 and moisture < 30 and nitrogen < 0.05:
        return "sandy"
    elif ph <= 7.5 and moisture >= 35 and organic >= 1.8:
        return "clay"
    elif ph <= 7.0 and moisture >= 25 and nitrogen >= 0.18:
        return "loamy"
    elif ph <= 7.0 and moisture >= 30:
        return "silty"
    else:
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


if __name__ == "__main__":
    print("Running Soil Analysis Inference...")
    for task in ("easy", "medium", "hard"):
        result = run_inference(task=task, session_id=task)
        print(f"Task: {task} | Reward: {result.get('reward')} | Done: {result.get('done')}")