"""
inference.py  —  AI Agent that interacts with the Soil Analysis Environment.

Required environment variables:
  API_BASE_URL  → LLM endpoint  (e.g. https://api.openai.com/v1)
  MODEL_NAME    → model id      (e.g. gpt-4o-mini)
  HF_TOKEN      → HuggingFace / API key

Usage:
  python inference.py
"""

import json
import os

import requests
from openenv.core.generic_client import GenericEnvClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",       "https://viji217-soil-analysis-env.hf.space")

SOIL_TYPES   = ["sandy", "clay", "loamy", "silty", "peaty"]
FERTILIZERS  = ["Compost", "Gypsum", "Lime", "NPK 10-10-10", "Phosphorus"]
CROPS        = ["carrot", "maize", "potato", "rice", "wheat"]


# ---------------------------------------------------------------------------
# LLM call (OpenAI-compatible)
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> str:
    """Call the LLM and return the raw text response."""
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type":  "application/json",
    }
    body = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert soil scientist. "
                    "Given soil sensor readings, identify the soil type, "
                    "best fertilizer, and best crop. "
                    "Always respond with a single valid JSON object and nothing else."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens":  256,
    }
    resp = requests.post(f"{API_BASE_URL}/chat/completions", headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Build agent prompt
# ---------------------------------------------------------------------------

def build_prompt(obs: dict, task: str) -> str:
    r = obs.get("soil_readings", {})
    lines = [
        f"Soil sensor readings:",
        f"  pH              = {r.get('ph')}",
        f"  Moisture %      = {r.get('moisture_pct')}",
        f"  Nitrogen %      = {r.get('nitrogen_pct')}",
        f"  Organic matter  = {r.get('organic_matter_pct')}",
        f"",
        f"Task level: {task}",
        f"",
        f"Valid soil types : {SOIL_TYPES}",
    ]
    if task in ("medium", "hard"):
        lines.append(f"Valid fertilizers: {FERTILIZERS}")
    if task == "hard":
        lines.append(f"Valid crops       : {CROPS}")

    lines += [
        "",
        "Respond ONLY with a JSON object like:",
    ]
    if task == "easy":
        lines.append('  {"soil_type": "loamy"}')
    elif task == "medium":
        lines.append('  {"soil_type": "loamy", "fertilizer": "Compost"}')
    else:
        lines.append('  {"soil_type": "loamy", "fertilizer": "Compost", "crop": "maize"}')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parse LLM response
# ---------------------------------------------------------------------------

def parse_action(text: str, task: str) -> dict:
    """Extract JSON dict from LLM text; fall back to random guesses on error."""
    try:
        # strip markdown fences if present
        clean = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data  = json.loads(clean)
        action = {"soil_type": str(data.get("soil_type", SOIL_TYPES[0]))}
        if task in ("medium", "hard") and "fertilizer" in data:
            action["fertilizer"] = str(data["fertilizer"])
        if task == "hard" and "crop" in data:
            action["crop"] = str(data["crop"])
        return action
    except Exception:
        # Fallback — never crash the agent loop
        return {"soil_type": SOIL_TYPES[0], "fertilizer": FERTILIZERS[0], "crop": CROPS[0]}


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent(task: str = "hard", episodes: int = 5):
    print(f"\n{'='*60}")
    print(f"  Soil Analysis Agent  |  task={task}  |  episodes={episodes}")
    print(f"{'='*60}\n")

    total_reward = 0.0

    with GenericEnvClient(base_url=ENV_URL).sync() as env:
        for ep in range(1, episodes + 1):
            # Reset — pass task as extra kwarg
            result = env.reset(task=task, seed=ep)
            obs    = result.observation

            # Build prompt and call LLM
            prompt = build_prompt(obs, task)
            llm_text = call_llm(prompt)
            action   = parse_action(llm_text, task)

            print(f"Episode {ep:02d} | action={action}")

            # Step
            result = env.step(action)
            reward = result.reward or 0.0
            total_reward += reward

            print(f"          | reward={reward:.2f} | feedback={result.observation.get('feedback','')}\n")

    avg = total_reward / episodes
    print(f"\nAverage reward over {episodes} episodes: {avg:.4f}")
    return avg


if __name__ == "__main__":
    for level in ("easy", "medium", "hard"):
        run_agent(task=level, episodes=5)
