from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from soil_env.env import SoilAnalysisEnv, SOIL_PROFILES, ALL_CROPS, ALL_FERTILIZERS

app = FastAPI(
    title="Soil Analysis AI Environment",
    description="OpenEnv-compatible environment for soil type identification, fertilizer, and crop recommendation.",
    version="1.0.0",
)

_sessions: dict = {}

class ResetRequest(BaseModel):
    task: str = "easy"
    seed: Optional[int] = 42
    session_id: str = "default"

class StepRequest(BaseModel):
    session_id: str = "default"
    soil_type: str
    fertilizer: Optional[str] = None
    crop: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html><body style="font-family:sans-serif;max-width:700px;margin:40px auto;padding:0 20px">
    <h1>Soil Analysis AI Environment</h1>
    <p>An OpenEnv-compatible environment for training AI agents to analyze soil data.</p>
    <h2>Tasks</h2>
    <ul>
      <li><b>easy</b> - identify soil type only</li>
      <li><b>medium</b> - identify soil type + recommend fertilizer</li>
      <li><b>hard</b> - identify soil type + fertilizer + best crop</li>
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

@app.post("/reset")
async def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    if req.task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="task must be 'easy', 'medium', or 'hard'")
    env = SoilAnalysisEnv(task=req.task, seed=req.seed)
    obs = env.reset()
    _sessions[req.session_id] = env
    return {"session_id": req.session_id, "observation": obs}

@app.post("/step")
async def step(req: Optional[StepRequest] = None):
    if req is None:
        raise HTTPException(status_code=400, detail="Request body required")
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
    action = {"soil_type": req.soil_type}
    if req.fertilizer:
        action["fertilizer"] = req.fertilizer
    if req.crop:
        action["crop"] = req.crop
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"observation": obs, "reward": reward, "done": done, "info": info}

# ✅ Required for [project.scripts] entry point
def start():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    start()
