"""
app.py  —  Soil Analysis OpenEnv Server

Uses openenv-core's create_app() which automatically registers:
  GET  /health      → {"status": "healthy"}
  GET  /schema      → action + observation + state JSON schemas
  GET  /metadata    → environment name, description, version
  POST /reset       → start new episode
  POST /step        → submit an action
  GET  /state       → current internal state
  WS   /ws          → WebSocket protocol (primary evaluation path)
"""

import uvicorn
from openenv.core.env_server.http_server import create_app

from soil_env.env import SoilAnalysisEnv
from soil_env.models import SoilAction, SoilObservation

# create_app() wires everything together and returns a FastAPI instance
# with all required OpenEnv endpoints already registered.
app = create_app(
    env=SoilAnalysisEnv,          # factory — called once per session
    action_cls=SoilAction,
    observation_cls=SoilObservation,
    env_name="soil-analysis-env",
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
