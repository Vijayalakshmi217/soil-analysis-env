"""
soil_env/env.py  —  Soil Analysis RL Environment
Works with action passed as a dict OR an object with attributes.
"""

import random
from typing import Optional

# ─────────────────────────────────────────────
# Domain data
# ─────────────────────────────────────────────

SOIL_PROFILES = {
    "sandy": {
        "ph_range":        (5.5, 6.5),
        "moisture_range":  (10,  30),
        "nitrogen_range":  (0.00, 0.12),
        "organic_range":   (0.0,  1.0),
        "best_fertilizer": "NPK 10-10-10",
        "best_crop":       "carrot",
    },
    "clay": {
        "ph_range":        (6.0, 7.5),
        "moisture_range":  (35,  60),
        "nitrogen_range":  (0.12, 0.25),
        "organic_range":   (1.8,  3.5),
        "best_fertilizer": "Gypsum",
        "best_crop":       "wheat",
    },
    "loamy": {
        "ph_range":        (6.0, 7.0),
        "moisture_range":  (25,  50),
        "nitrogen_range":  (0.18, 0.30),
        "organic_range":   (2.0,  4.0),
        "best_fertilizer": "Compost",
        "best_crop":       "maize",
    },
    "silty": {
        "ph_range":        (6.0, 7.0),
        "moisture_range":  (30,  52),
        "nitrogen_range":  (0.08, 0.20),
        "organic_range":   (1.3,  2.8),
        "best_fertilizer": "Phosphorus",
        "best_crop":       "rice",
    },
    "peaty": {
        "ph_range":        (3.5, 5.5),
        "moisture_range":  (55,  100),
        "nitrogen_range":  (0.25, 0.50),
        "organic_range":   (15.0, 40.0),
        "best_fertilizer": "Lime",
        "best_crop":       "potato",
    },
}

ALL_SOIL_TYPES  = sorted(SOIL_PROFILES.keys())
ALL_FERTILIZERS = sorted({p["best_fertilizer"] for p in SOIL_PROFILES.values()})
ALL_CROPS       = sorted({p["best_crop"]        for p in SOIL_PROFILES.values()})


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def _sample_readings(soil_type: str, rng: random.Random) -> dict:
    p = SOIL_PROFILES[soil_type]
    return {
        "ph":                 round(rng.uniform(*p["ph_range"]),       2),
        "moisture_pct":       round(rng.uniform(*p["moisture_range"]), 2),
        "nitrogen_pct":       round(rng.uniform(*p["nitrogen_range"]), 4),
        "organic_matter_pct": round(rng.uniform(*p["organic_range"]),  2),
    }


def _get(action, key, default=None):
    """Read from action whether it's a dict or an object."""
    if isinstance(action, dict):
        return action.get(key, default)
    return getattr(action, key, default)


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

class SoilAnalysisEnv:
    """
    Soil Analysis RL Environment.

    reset() → dict observation
    step(action)  → dict observation   (action can be dict or object)
    """

    def __init__(self, task: str = "easy", seed: Optional[int] = None):
        if task not in ("easy", "medium", "hard"):
            raise ValueError("task must be 'easy', 'medium', or 'hard'")
        self._default_task = task
        self._seed         = seed

        self._rng        = random.Random(seed)
        self._true_type  = "loamy"
        self._task       = task
        self._step_count = 0
        self._done       = False

    # ── reset ──────────────────────────────────

    def reset(self, seed: Optional[int] = None, **kwargs) -> dict:
        task = kwargs.get("task", self._default_task)
        if task not in ("easy", "medium", "hard"):
            task = self._default_task

        actual_seed  = seed if seed is not None else self._seed
        self._rng        = random.Random(actual_seed)
        self._true_type  = self._rng.choice(ALL_SOIL_TYPES)
        self._task       = task
        self._step_count = 0
        self._done       = False

        return {
            "done":       False,
            "reward":     None,
            "metadata":   {},
            "soil_readings": _sample_readings(self._true_type, self._rng),
            "task_level":    self._task,
            "step_number":   self._step_count,
            "feedback":      "Analyse the readings and submit your answer.",
        }

    # ── step ───────────────────────────────────

    def step(self, action) -> tuple:
        """
        action: dict  e.g. {"soil_type": "loamy", "fertilizer": "Compost", "crop": "maize"}
                or object with .soil_type / .fertilizer / .crop attributes

        Returns (obs_dict, reward, done, info)
        """
        if self._done:
            obs = {
                "done": True, "reward": 0.0, "metadata": {},
                "soil_readings": _sample_readings(self._true_type, self._rng),
                "task_level": self._task, "step_number": self._step_count,
                "feedback": "Episode already finished. Call /reset first.",
            }
            return obs, 0.0, True, {}

        self._step_count += 1
        profile = SOIL_PROFILES[self._true_type]

        # ── read action safely (dict OR object) ──
        soil_type  = str(_get(action, "soil_type",  "")).lower().strip()
        fertilizer = _get(action, "fertilizer", None)
        crop       = _get(action, "crop",       None)
        if fertilizer:
            fertilizer = str(fertilizer).strip()
        if crop:
            crop = str(crop).strip()

        reward         = 0.0
        feedback_parts = []

        # soil type  (+0.40)
        if soil_type == self._true_type:
            reward += 0.40
            feedback_parts.append(f"✓ Soil type correct ({self._true_type})")
        else:
            feedback_parts.append(
                f"✗ Soil type wrong (you='{soil_type}', true='{self._true_type}')"
            )

        # fertilizer (+0.30) — medium / hard only
        if self._task in ("medium", "hard"):
            correct_fert = profile["best_fertilizer"]
            if fertilizer == correct_fert:
                reward += 0.30
                feedback_parts.append(f"✓ Fertilizer correct ({correct_fert})")
            else:
                feedback_parts.append(
                    f"✗ Fertilizer wrong (you='{fertilizer}', best='{correct_fert}')"
                )

        # crop (+0.30) — hard only
        if self._task == "hard":
            correct_crop = profile["best_crop"]
            if crop == correct_crop:
                reward += 0.30
                feedback_parts.append(f"✓ Crop correct ({correct_crop})")
            else:
                feedback_parts.append(
                    f"✗ Crop wrong (you='{crop}', best='{correct_crop}')"
                )

        self._done = True
        reward     = round(reward, 4)

        obs = {
            "done":     True,
            "reward":   reward,
            "metadata": {
                "true_soil_type":  self._true_type,
                "best_fertilizer": profile["best_fertilizer"],
                "best_crop":       profile["best_crop"],
            },
            "soil_readings": _sample_readings(self._true_type, self._rng),
            "task_level":    self._task,
            "step_number":   self._step_count,
            "feedback":      " | ".join(feedback_parts),
        }

        info = {
            "true_soil_type":  self._true_type,
            "best_fertilizer": profile["best_fertilizer"],
            "best_crop":       profile["best_crop"],
        }

        return obs, reward, True, info
