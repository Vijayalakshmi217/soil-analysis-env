"""
app.py  —  Soil Analysis Environment Server
Fully OpenEnv-compatible: /health, /schema, /metadata, /state, /reset, /step, /ws
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import json

from soil_env.env import SoilAnalysisEnv, SOIL_PROFILES, ALL_CROPS, ALL_FERTILIZERS

app = FastAPI(
    title="Soil Analysis AI Environment",
    description="OpenEnv-compatible environment for soil type identification, fertilizer, and crop recommendation.",
    version="1.0.0",
)

_sessions: dict = {}

# ─────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "easy"
    seed: Optional[int] = 42
    session_id: str = "default"

class StepRequest(BaseModel):
    session_id: str = "default"
    soil_type: str
    fertilizer: Optional[str] = None
    crop: Optional[str] = None

# ─────────────────────────────────────────────
# Required OpenEnv endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "name": "SoilAnalysisEnv",
        "description": "RL environment for soil type identification, fertilizer and crop recommendation.",
        "version": "1.0.0",
        "author": "Vijayalakshmi",
        "tasks": ["easy", "medium", "hard"],
    }

@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "soil_type":  {"type": "string", "enum": sorted(SOIL_PROFILES.keys())},
                "fertilizer": {"type": "string", "enum": sorted(ALL_FERTILIZERS), "nullable": True},
                "crop":       {"type": "string", "enum": sorted(ALL_CROPS), "nullable": True},
            },
            "required": ["soil_type"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "soil_readings": {"type": "object"},
                "task_level":    {"type": "string"},
                "step_number":   {"type": "integer"},
                "feedback":      {"type": "string"},
                "done":          {"type": "boolean"},
                "reward":        {"type": "number", "nullable": True},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "true_soil_type": {"type": "string"},
                "task_level":     {"type": "string"},
                "step_count":     {"type": "integer"},
                "episode_done":   {"type": "boolean"},
            },
        },
    }

@app.get("/state")
def state_endpoint(session_id: str = "default"):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
    return {
        "true_soil_type": env._true_type,
        "task_level":     env._task,
        "step_count":     env._step_count,
        "episode_done":   env._done,
    }

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    return """
<html><body style="font-family:sans-serif;max-width:700px;margin:40px auto;padding:0 20px">
<h1>🌱 Soil Analysis AI Environment</h1>
<p>An OpenEnv-compatible environment for training AI agents to analyse soil data.</p>
<h2>Endpoints</h2>
<ul>
  <li><b>GET  /health</b>   — health check</li>
  <li><b>GET  /metadata</b> — environment metadata</li>
  <li><b>GET  /schema</b>   — action / observation schemas</li>
  <li><b>POST /reset</b>    — start a new episode</li>
  <li><b>POST /step</b>     — submit an action</li>
  <li><b>GET  /state</b>    — current internal state</li>
  <li><b>WS   /ws</b>       — WebSocket protocol</li>
</ul>
<h2>Tasks</h2>
<ul>
  <li><b>easy</b>   — identify soil type only (+0.40)</li>
  <li><b>medium</b> — soil type + fertilizer (+0.70)</li>
  <li><b>hard</b>   — soil type + fertilizer + crop (+1.00)</li>
</ul>
<p><a href="/docs">Open API docs</a></p>
</body></html>
"""

@app.get("/info")
def info():
    return {
        "name": "SoilAnalysisEnv",
        "version": "1.0.0",
        "tasks": ["easy", "medium", "hard"],
        "soil_types": sorted(SOIL_PROFILES.keys()),
        "fertilizers": sorted(ALL_FERTILIZERS),
        "crops": sorted(ALL_CROPS),
    }

# ─────────────────────────────────────────────
# Core RL endpoints
# ─────────────────────────────────────────────

@app.post("/reset")
def reset(req: ResetRequest):
    if req.task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="task must be 'easy', 'medium', or 'hard'")
    env = SoilAnalysisEnv(task=req.task, seed=req.seed)
    obs = env.reset()
    _sessions[req.session_id] = env
    return {"session_id": req.session_id, "observation": obs}

@app.post("/step")
def step(req: StepRequest):
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    action = {"soil_type": req.soil_type}
    if req.fertilizer:
        action["fertilizer"] = req.fertilizer
    if req.crop:
        action["crop"] = req.crop

    try:
        result = env.step(action)
        # Support both old tuple return and new dict/object return
        if isinstance(result, tuple):
            obs, reward, done, info = result
        elif isinstance(result, dict):
            obs    = result.get("observation", result)
            reward = result.get("reward", 0.0)
            done   = result.get("done", True)
            info   = result.get("info", {})
        else:
            # Pydantic model
            d      = result.model_dump() if hasattr(result, "model_dump") else vars(result)
            reward = d.pop("reward", 0.0)
            done   = d.pop("done", True)
            info   = d.pop("metadata", {})
            obs    = d
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"observation": obs, "reward": reward, "done": done, "info": info}

# ─────────────────────────────────────────────
# WebSocket  (/ws) — primary evaluation path
# ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    env: Optional[SoilAnalysisEnv] = None

    try:
        while True:
            raw   = await websocket.receive_text()
            msg   = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "reset":
                data = msg.get("data", {})
                env  = SoilAnalysisEnv(task=data.get("task", "easy"), seed=data.get("seed"))
                obs  = env.reset()
                await websocket.send_text(json.dumps({
                    "type": "observation",
                    "data": {"observation": obs, "reward": None, "done": False},
                }))

            elif mtype == "step":
                if env is None:
                    await websocket.send_text(json.dumps({"type": "error", "data": {"message": "Call reset first"}}))
                    continue
                data   = msg.get("data", {})
                action = {k: data[k] for k in ("soil_type", "fertilizer", "crop") if data.get(k)}
                if "soil_type" not in action:
                    action["soil_type"] = "loamy"
                try:
                    result = env.step(action)
                    if isinstance(result, tuple):
                        obs, reward, done, info = result
                    elif isinstance(result, dict):
                        obs    = result.get("observation", result)
                        reward = result.get("reward", 0.0)
                        done   = result.get("done", True)
                        info   = result.get("info", {})
                    else:
                        d      = result.model_dump() if hasattr(result, "model_dump") else vars(result)
                        reward = d.pop("reward", 0.0)
                        done   = d.pop("done", True)
                        info   = d.pop("metadata", {})
                        obs    = d
                    await websocket.send_text(json.dumps({
                        "type": "observation",
                        "data": {"observation": obs, "reward": reward, "done": done, "info": info},
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "data": {"message": str(e)}}))

            elif mtype == "state":
                if env is None:
                    await websocket.send_text(json.dumps({"type": "error", "data": {"message": "Call reset first"}}))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "state",
                        "data": {
                            "true_soil_type": env._true_type,
                            "task_level":     env._task,
                            "step_count":     env._step_count,
                            "episode_done":   env._done,
                        },
                    }))
            else:
                await websocket.send_text(json.dumps({"type": "error", "data": {"message": f"Unknown type: {mtype}"}}))

    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
