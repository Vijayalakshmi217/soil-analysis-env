"""
soil_env/env.py
Soil Analysis RL Environment — OpenEnv compatible.

The agent receives sensor readings and must correctly identify:
  easy   → soil type only
  medium → soil type + best fertilizer
  hard   → soil type + best fertilizer + best crop
"""

import random
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from soil_env.models import SoilAction, SoilObservation, SoilState


# ---------------------------------------------------------------------------
# Domain data
# ---------------------------------------------------------------------------

SOIL_PROFILES = {
    "sandy": {
        "ph_range":        (5.5, 6.5),
        "moisture_range":  (10, 30),
        "nitrogen_range":  (0.00, 0.12),
        "organic_range":   (0.0, 1.0),
        "best_fertilizer": "NPK 10-10-10",
        "best_crop":       "carrot",
    },
    "clay": {
        "ph_range":        (6.0, 7.5),
        "moisture_range":  (35, 60),
        "nitrogen_range":  (0.12, 0.25),
        "organic_range":   (1.8, 3.5),
        "best_fertilizer": "Gypsum",
        "best_crop":       "wheat",
    },
    "loamy": {
        "ph_range":        (6.0, 7.0),
        "moisture_range":  (25, 50),
        "nitrogen_range":  (0.18, 0.30),
        "organic_range":   (2.0, 4.0),
        "best_fertilizer": "Compost",
        "best_crop":       "maize",
    },
    "silty": {
        "ph_range":        (6.0, 7.0),
        "moisture_range":  (30, 52),
        "nitrogen_range":  (0.08, 0.20),
        "organic_range":   (1.3, 2.8),
        "best_fertilizer": "Phosphorus",
        "best_crop":       "rice",
    },
    "peaty": {
        "ph_range":        (3.5, 5.5),
        "moisture_range":  (55, 100),
        "nitrogen_range":  (0.25, 0.50),
        "organic_range":   (15.0, 40.0),
        "best_fertilizer": "Lime",
        "best_crop":       "potato",
    },
}

ALL_SOIL_TYPES  = sorted(SOIL_PROFILES.keys())
ALL_FERTILIZERS = sorted({p["best_fertilizer"] for p in SOIL_PROFILES.values()})
ALL_CROPS       = sorted({p["best_crop"]        for p in SOIL_PROFILES.values()})


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sample_readings(soil_type: str, rng: random.Random) -> dict:
    """Generate realistic noisy sensor readings for a given soil type."""
    p = SOIL_PROFILES[soil_type]
    return {
        "ph":               round(rng.uniform(*p["ph_range"]),       2),
        "moisture_pct":     round(rng.uniform(*p["moisture_range"]), 2),
        "nitrogen_pct":     round(rng.uniform(*p["nitrogen_range"]), 4),
        "organic_matter_pct": round(rng.uniform(*p["organic_range"]), 2),
    }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SoilAnalysisEnv(Environment):
    """
    OpenEnv-compatible Soil Analysis environment.

    Inherits from openenv.core Environment so the framework can automatically
    expose /health, /schema, /metadata, and /ws endpoints.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True   # each instance is fully isolated

    # task passed at construction or via reset() kwargs
    def __init__(self, task: str = "easy", seed: Optional[int] = None):
        super().__init__()
        if task not in ("easy", "medium", "hard"):
            raise ValueError("task must be 'easy', 'medium', or 'hard'")
        self._default_task = task
        self._seed         = seed

        # internal state (initialised properly in reset())
        self._rng:        random.Random = random.Random(seed)
        self._true_type:  str  = "loamy"
        self._task:       str  = task
        self._step_count: int  = 0
        self._done:       bool = False

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> SoilObservation:
        """Start a new episode. Accepts 'task' as an extra kwarg."""
        task = kwargs.get("task", self._default_task)
        if task not in ("easy", "medium", "hard"):
            task = self._default_task

        actual_seed = seed if seed is not None else self._seed
        self._rng        = random.Random(actual_seed)
        self._true_type  = self._rng.choice(ALL_SOIL_TYPES)
        self._task       = task
        self._step_count = 0
        self._done       = False

        readings = _sample_readings(self._true_type, self._rng)

        return SoilObservation(
            soil_readings=readings,
            task_level=self._task,
            step_number=self._step_count,
            feedback="Analyse the readings and submit your answer.",
            done=False,
            reward=None,
        )

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(
        self,
        action: SoilAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> SoilObservation:
        """
        Evaluate the agent's answer and return graded reward.

        Reward breakdown (all tasks cumulative):
          +0.40  correct soil type
          +0.30  correct fertilizer  (medium / hard only)
          +0.30  correct crop        (hard only)
        """
        if self._done:
            # Episode already finished — return terminal obs with zero reward
            readings = _sample_readings(self._true_type, self._rng)
            return SoilObservation(
                soil_readings=readings,
                task_level=self._task,
                step_number=self._step_count,
                feedback="Episode already finished. Call reset() to start a new one.",
                done=True,
                reward=0.0,
            )

        self._step_count += 1
        profile = SOIL_PROFILES[self._true_type]

        reward      = 0.0
        feedback_parts = []

        # --- soil type ---
        if action.soil_type.lower() == self._true_type:
            reward += 0.40
            feedback_parts.append(f"✓ Soil type correct ({self._true_type})")
        else:
            feedback_parts.append(
                f"✗ Soil type wrong (you said '{action.soil_type}', true='{self._true_type}')"
            )

        # --- fertilizer ---
        if self._task in ("medium", "hard"):
            correct_fert = profile["best_fertilizer"]
            if action.fertilizer and action.fertilizer == correct_fert:
                reward += 0.30
                feedback_parts.append(f"✓ Fertilizer correct ({correct_fert})")
            else:
                feedback_parts.append(
                    f"✗ Fertilizer wrong (you said '{action.fertilizer}', best='{correct_fert}')"
                )

        # --- crop ---
        if self._task == "hard":
            correct_crop = profile["best_crop"]
            if action.crop and action.crop == correct_crop:
                reward += 0.30
                feedback_parts.append(f"✓ Crop correct ({correct_crop})")
            else:
                feedback_parts.append(
                    f"✗ Crop wrong (you said '{action.crop}', best='{correct_crop}')"
                )

        self._done = True   # one-step episodes
        readings   = _sample_readings(self._true_type, self._rng)

        return SoilObservation(
            soil_readings=readings,
            task_level=self._task,
            step_number=self._step_count,
            feedback=" | ".join(feedback_parts),
            done=True,
            reward=round(reward, 4),
            metadata={
                "true_soil_type":    self._true_type,
                "best_fertilizer":   profile["best_fertilizer"],
                "best_crop":         profile["best_crop"],
            },
        )

    # ------------------------------------------------------------------
    # state  (abstract property required by OpenEnv)
    # ------------------------------------------------------------------

    @property
    def state(self) -> SoilState:
        return SoilState(
            true_soil_type=self._true_type,
            task_level=self._task,
            step_count=self._step_count,
            episode_done=self._done,
        )
