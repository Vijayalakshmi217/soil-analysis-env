"""
soil_env/models.py
Pydantic models for the Soil Analysis OpenEnv environment.
Action, Observation and State all inherit from the openenv-core base classes.
"""

from typing import Any, Dict, Optional
from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class SoilAction(Action):
    """
    Action sent by the agent each step.

    easy  → only soil_type is required
    medium → soil_type + fertilizer
    hard   → soil_type + fertilizer + crop
    """
    soil_type: str
    fertilizer: Optional[str] = None
    crop: Optional[str] = None


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class SoilObservation(Observation):
    """
    Observation returned after every reset() and step().

    'done' and 'reward' are inherited from the base Observation class.
    """
    # Sensor readings the agent must analyse
    soil_readings: Dict[str, Any]

    # Contextual information
    task_level: str          # "easy" | "medium" | "hard"
    step_number: int

    # Feedback after a step (empty string before any step)
    feedback: str = ""


# ---------------------------------------------------------------------------
# State  (internal env state — not sent to the agent)
# ---------------------------------------------------------------------------

class SoilState(State):
    """Internal state of the environment."""
    true_soil_type: str
    task_level: str
    step_count: int
    episode_done: bool
