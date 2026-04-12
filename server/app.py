
server/app.py  —  Soil Analysis Environment Server
Required by openenv validate: must have main() function callable with if __name__ == "__main__"
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from typing import Optional, Any, Dict
import uvicorn
import json
import sys
import os

# Allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soil_env.env import SoilAnalysisEnv, SOIL_PROFILES, ALL_CROPS, ALL_FERTILIZERS

app = FastAPI(
    title="Soil Analysis AI Environment",
    description="OpenEnv-compatible environment for soil type identification, fertilizer, and crop recommendation.",
    version="1.0.0",
)

_sessions: dict = {}

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

@app.get("/", response_class=HTMLResponse)
def root():
    return """
<html><body style="font-family:sans-serif;max-width:700px;margin:40px auto;padding:0 20px">
<h1>🌱 Soil Analysis AI Environment</h1>
<p>An OpenEnv-compatible environment for training AI agents to analyse soil data.</p>
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

@app.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if body is None:
        body = {}

    task       = body.get("task", "easy") or "easy"
    seed       = body.get("seed", None)
    session_id = body.get("session_id", "default") or "default"
    if task not in ("easy", "medium", "hard"):
        task = "easy"

    env = SoilAnalysisEnv(task=task, seed=seed)
    obs = env.reset(seed=seed)
    _sessions[session_id] = env
    return {"session_id": session_id, "observation": obs}

@app.post("/step")
async def step(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if body is None:
        body = {}

    session_id = body.get("session_id", "default") or "default"
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    if "action" in body and isinstance(body["action"], dict):
        action_data = body["action"]
    else:
        action_data = body

    action = {
        "soil_type":  action_data.get("soil_type", "loamy"),
        "fertilizer": action_data.get("fertilizer", None),
        "crop":       action_data.get("crop", None),
    }

    try:
        result = env.step(action)
        if isinstance(result, tuple):
            obs, reward, done, info = result
        else:
            obs, reward, done, info = result, 0.0, True, {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"observation": obs, "reward": reward, "done": done, "info": info}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    env: Optional[SoilAnalysisEnv] = None
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await websocket.send_text(json.dumps({"type": "error", "data": {"message": "Invalid JSON"}}))
                continue

            mtype = msg.get("type")
            data  = msg.get("data", {}) or {}

            if mtype == "reset":
                task = data.get("task", "easy") or "easy"
                seed = data.get("seed", None)
                env  = SoilAnalysisEnv(task=task, seed=seed)
                obs  = env.reset(seed=seed)
                await websocket.send_text(json.dumps({
                    "type": "observation",
                    "data": {"observation": obs, "reward": None, "done": False},
                }))

            elif mtype == "step":
                if env is None:
                    await websocket.send_text(json.dumps({"type": "error", "data": {"message": "Call reset first"}}))
                    continue
                if "action" in data and isinstance(data["action"], dict):
                    action_data = data["action"]
                else:
                    action_data = data
                action = {
                    "soil_type":  action_data.get("soil_type", "loamy"),
                    "fertilizer": action_data.get("fertilizer", None),
                    "crop":       action_data.get("crop", None),
                }
                try:
                    result = env.step(action)
                    if isinstance(result, tuple):
                        obs, reward, done, info = result
                    else:
                        obs, reward, done, info = result, 0.0, True, {}
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
                await websocket.send_text(json.dumps({"type": "error", "data": {"message": f"Unknown: {mtype}"}}))

    except WebSocketDisconnect:
        pass


# ─────────────────────────────────────────────
# Required by openenv validate
# ─────────────────────────────────────────────

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
