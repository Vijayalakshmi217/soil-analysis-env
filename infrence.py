import os
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://viji217-soil-analysis-env.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)

def reset(task="easy", seed=42, session_id="default"):
    payload = {"task": task, "seed": seed, "session_id": session_id}
    response = requests.post(f"{API_BASE_URL}/reset", json=payload)
    response.raise_for_status()
    return response.json()

def step(soil_type, fertilizer=None, crop=None, session_id="default"):
    payload = {"session_id": session_id, "soil_type": soil_type}
    if fertilizer:
        payload["fertilizer"] = fertilizer
    if crop:
        payload["crop"] = crop
    response = requests.post(f"{API_BASE_URL}/step", json=payload)
    response.raise_for_status()
    return response.json()

def predict_soil(ph, moisture, nitrogen, organic):
    if ph < 5.5 and moisture > 50 and nitrogen >= 0.20:
        return "peaty"
    if moisture < 25 and nitrogen < 0.15 and organic < 1.8:
        return "sandy"
    if moisture >= 40 and organic >= 2.0:
        return "clay"
    if 25 <= moisture <= 50 and nitrogen >= 0.15 and organic >= 1.5:
        return "loamy"
    if 25 <= moisture <= 55 and organic >= 1.3:
        return "silty"
    if moisture < 30:
        return "sandy"
    return "loamy"

def get_fertilizer(soil_type):
    return {"peaty": "lime", "sandy": "npk", "clay": "compost", "loamy": "balanced", "silty": "phosphorus"}.get(soil_type, "balanced")

def get_crop(soil_type):
    return {"peaty": "blueberry", "sandy": "carrot", "clay": "wheat", "loamy": "maize", "silty": "rice"}.get(soil_type, "maize")

def run_inference(task="easy", session_id="default"):
    print("START")
    result = reset(task=task, session_id=session_id)
    obs = result["observation"]
    soil_readings = obs.get("soil_readings", {})
    ph = soil_readings.get("ph", 6.5)
    moisture = soil_readings.get("moisture_pct", 40)
    nitrogen = soil_readings.get("nitrogen_pct", 0.1)
    organic = soil_readings.get("organic_matter_pct", 2.0)

    soil_type = predict_soil(ph, moisture, nitrogen, organic)
    fertilizer = get_fertilizer(soil_type) if task in ("medium", "hard") else None
    crop = get_crop(soil_type) if task == "hard" else None

    print("STEP soil_type=" + soil_type)
    step_result = step(soil_type=soil_type, fertilizer=fertilizer, crop=crop, session_id=session_id)

    reward = step_result.get("reward", 0)
    print("END reward=" + str(reward))
    return step_result

if __name__ == "__main__":
    for task in ("easy", "medium", "hard"):
        run_inference(task=task, session_id=task)
