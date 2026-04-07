import os
import sys

# Safe import of requests
try:
    import requests
except ImportError:
    print("ERROR: 'requests' module not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

API_BASE_URL = os.getenv("API_BASE_URL", "https://viji217-soil-analysis-env.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN", "")

TIMEOUT = 30  # seconds for all requests


def reset(task="easy", seed=42, session_id="default"):
    payload = {"task": task, "seed": seed, "session_id": session_id}
    try:
        response = requests.post(
            f"{API_BASE_URL}/reset",
            json=payload,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as e:
        print(f"ERROR: Cannot connect to server at {API_BASE_URL}. Is it running? {e}")
        raise
    except requests.exceptions.Timeout:
        print(f"ERROR: /reset request timed out after {TIMEOUT}s.")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: /reset returned HTTP error: {e}")
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error in reset(): {e}")
        raise


def step(soil_type, fertilizer=None, crop=None, session_id="default"):
    payload = {"session_id": session_id, "soil_type": soil_type}
    if fertilizer:
        payload["fertilizer"] = fertilizer
    if crop:
        payload["crop"] = crop
    try:
        response = requests.post(
            f"{API_BASE_URL}/step",
            json=payload,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as e:
        print(f"ERROR: Cannot connect to server at {API_BASE_URL}. Is it running? {e}")
        raise
    except requests.exceptions.Timeout:
        print(f"ERROR: /step request timed out after {TIMEOUT}s.")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: /step returned HTTP error: {e}")
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error in step(): {e}")
        raise


def predict_soil(ph, moisture, nitrogen, organic):
    try:
        ph       = float(ph)
        moisture = float(moisture)
        nitrogen = float(nitrogen)
        organic  = float(organic)

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
    except (TypeError, ValueError) as e:
        print(f"WARNING: Could not parse soil readings ({e}), defaulting to loamy.")
        return "loamy"


def get_fertilizer(soil_type):
    mapping = {
        "peaty":  "lime",
        "sandy":  "npk",
        "clay":   "compost",
        "loamy":  "balanced",
        "silty":  "phosphorus"
    }
    return mapping.get(soil_type, "balanced")


def get_crop(soil_type):
    mapping = {
        "peaty":  "blueberry",
        "sandy":  "carrot",
        "clay":   "wheat",
        "loamy":  "maize",
        "silty":  "rice"
    }
    return mapping.get(soil_type, "maize")


def run_inference(task="easy", session_id="default"):
    print(f"\n=== Task: {task.upper()} | Session: {session_id} ===")
    print("START")

    try:
        result = reset(task=task, session_id=session_id)
    except Exception:
        print(f"FAILED: Could not reset for task={task}. Skipping.")
        return None

    try:
        obs = result.get("observation", {})
        soil_readings = obs.get("soil_readings", {})
        ph       = soil_readings.get("ph", 6.5)
        moisture = soil_readings.get("moisture_pct", 40)
        nitrogen = soil_readings.get("nitrogen_pct", 0.1)
        organic  = soil_readings.get("organic_matter_pct", 2.0)
    except (AttributeError, KeyError) as e:
        print(f"WARNING: Could not parse observation ({e}), using defaults.")
        ph, moisture, nitrogen, organic = 6.5, 40, 0.1, 2.0

    soil_type  = predict_soil(ph, moisture, nitrogen, organic)
    fertilizer = get_fertilizer(soil_type) if task in ("medium", "hard") else None
    crop       = get_crop(soil_type)       if task == "hard"             else None

    print(f"STEP soil_type={soil_type}, fertilizer={fertilizer}, crop={crop}")

    try:
        step_result = step(
            soil_type=soil_type,
            fertilizer=fertilizer,
            crop=crop,
            session_id=session_id
        )
    except Exception:
        print(f"FAILED: Could not complete step for task={task}.")
        return None

    reward = step_result.get("reward", 0)
    print(f"END reward={reward}")
    return step_result


if __name__ == "__main__":
    for task in ("easy", "medium", "hard"):
        run_inference(task=task, session_id=task)
