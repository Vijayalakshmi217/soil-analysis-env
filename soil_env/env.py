import random
from typing import Any, Dict, Optional, Tuple

SOIL_PROFILES = {
    "sandy": {
        "ph_range": (5.5, 6.5),
        "nitrogen_range": (0.05, 0.15),
        "moisture_range": (10, 25),
        "organic_matter_range": (0.5, 1.5),
        "best_fertilizer": "NPK 10-20-20",
        "best_crop": "groundnut",
        "alt_crops": ["cassava", "watermelon", "sweet potato"],
        "alt_fertilizers": ["urea", "superphosphate"],
    },
    "clay": {
        "ph_range": (6.0, 7.5),
        "nitrogen_range": (0.15, 0.35),
        "moisture_range": (35, 55),
        "organic_matter_range": (2.0, 4.0),
        "best_fertilizer": "NPK 20-10-10",
        "best_crop": "rice",
        "alt_crops": ["wheat", "cotton", "soybean"],
        "alt_fertilizers": ["ammonium nitrate", "DAP"],
    },
    "loamy": {
        "ph_range": (6.0, 7.0),
        "nitrogen_range": (0.20, 0.40),
        "moisture_range": (25, 45),
        "organic_matter_range": (2.5, 5.0),
        "best_fertilizer": "NPK 15-15-15",
        "best_crop": "maize",
        "alt_crops": ["tomato", "wheat", "sunflower"],
        "alt_fertilizers": ["compost", "vermicompost"],
    },
    "silty": {
        "ph_range": (6.0, 7.0),
        "nitrogen_range": (0.10, 0.25),
        "moisture_range": (30, 50),
        "organic_matter_range": (1.5, 3.5),
        "best_fertilizer": "NPK 12-12-17",
        "best_crop": "sorghum",
        "alt_crops": ["barley", "oats", "flax"],
        "alt_fertilizers": ["potassium sulfate", "calcium nitrate"],
    },
    "peaty": {
        "ph_range": (3.5, 5.5),
        "nitrogen_range": (0.30, 0.60),
        "moisture_range": (55, 80),
        "organic_matter_range": (20.0, 60.0),
        "best_fertilizer": "lime + NPK 5-10-10",
        "best_crop": "blueberry",
        "alt_crops": ["cranberry", "potato", "cabbage"],
        "alt_fertilizers": ["garden lime", "dolomite"],
    },
}

ALL_CROPS = list({c for p in SOIL_PROFILES.values() for c in [p["best_crop"]] + p["alt_crops"]})
ALL_FERTILIZERS = list({f for p in SOIL_PROFILES.values() for f in [p["best_fertilizer"]] + p["alt_fertilizers"]})


def _sample_soil(soil_type, seed=None):
    rng = random.Random(seed)
    p = SOIL_PROFILES[soil_type]
    def rng_range(lo, hi):
        return round(rng.uniform(lo, hi), 3)
    return {
        "ph": rng_range(*p["ph_range"]),
        "nitrogen_pct": rng_range(*p["nitrogen_range"]),
        "moisture_pct": rng_range(*p["moisture_range"]),
        "organic_matter_pct": rng_range(*p["organic_matter_range"]),
    }


def grade_soil_type(predicted, true_type):
    return 1.0 if predicted.strip().lower() == true_type.lower() else 0.0


def grade_fertilizer(predicted, soil_type):
    p = SOIL_PROFILES[soil_type]
    pred = predicted.strip().lower()
    if pred == p["best_fertilizer"].lower():
        return 1.0
    for alt in p["alt_fertilizers"]:
        if pred == alt.lower():
            return 0.5
    return 0.0


def grade_crop(predicted, soil_type):
    p = SOIL_PROFILES[soil_type]
    pred = predicted.strip().lower()
    if pred == p["best_crop"].lower():
        return 1.0
    for alt in p["alt_crops"]:
        if pred == alt.lower():
            return 0.5
    return 0.0


class SoilAnalysisEnv:

    metadata = {"version": "1.0.0", "name": "SoilAnalysisEnv"}

    def __init__(self, task="easy", seed=None):
        assert task in ("easy", "medium", "hard")
        self.task = task
        self.seed = seed
        self._rng = random.Random(seed)
        self._true_soil_type = None
        self._done = True

    def reset(self):
        self._done = False
        self._true_soil_type = self._rng.choice(list(SOIL_PROFILES.keys()))
        episode_seed = self._rng.randint(0, 10000)
        readings = _sample_soil(self._true_soil_type, seed=episode_seed)
        obs = {
            "task_level": self.task,
            "soil_readings": readings,
            "valid_soil_types": sorted(SOIL_PROFILES.keys()),
        }
        if self.task in ("medium", "hard"):
            obs["valid_fertilizers"] = sorted(ALL_FERTILIZERS)
        if self.task == "hard":
            obs["valid_crops"] = sorted(ALL_CROPS)
        return obs

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset() before step().")
        self._done = True
        true = self._true_soil_type
        info = {"true_soil_type": true}
        scores = []

        soil_score = grade_soil_type(action.get("soil_type", ""), true)
        scores.append(soil_score)
        info["soil_type_score"] = soil_score

        if self.task in ("medium", "hard"):
            fert_score = grade_fertilizer(action.get("fertilizer", ""), true)
            scores.append(fert_score)
            info["fertilizer_score"] = fert_score

        if self.task == "hard":
            crop_score = grade_crop(action.get("crop", ""), true)
            scores.append(crop_score)
            info["crop_score"] = crop_score

        reward = round(sum(scores) / len(scores), 4)
        info["reward"] = reward
        obs = {"task_level": self.task, "episode_done": True}
        return obs, reward, True, info

    def render(self):
        if self._true_soil_type is None:
            return "Call reset() first."
        p = SOIL_PROFILES[self._true_soil_type]
        return (
            f"Task: {self.task.upper()}\n"
            f"True soil type: {self._true_soil_type}\n"
            f"Best fertilizer: {p['best_fertilizer']}\n"
            f"Best crop: {p['best_crop']}\n"
        )