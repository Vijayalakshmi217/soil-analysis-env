"""
inference.py  —  Soil Analysis AI Agent
Prints [START]/[STEP]/[END] blocks required by the evaluator.
"""

import json
import os
import sys
import requests

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL", "https://viji217-soil-analysis-env.hf.space").rstrip("/")

SOIL_TYPES  = ["sandy", "clay", "loamy", "silty", "peaty"]
FERTILIZERS = ["Compost", "Gypsum", "Lime", "NPK 10-10-10", "Phosphorus"]
CROPS       = ["carrot", "maize", "potato", "rice", "wheat"]

SOIL_PROFILES = {
    "sandy": {"best_fertilizer": "NPK 10-10-10", "best_crop": "carrot"},
    "clay":  {"best_fertilizer": "Gypsum",        "best_crop": "wheat"},
    "loamy": {"best_fertilizer": "Compost",        "best_crop": "maize"},
    "silty": {"best_fertilizer": "Phosphorus",     "best_crop": "rice"},
    "peaty": {"best_fertilizer": "Lime",           "best_crop": "potato"},
}

RULES = [
    (3.5, 5.5,  55, 100, 0.25, 15.0, "peaty"),
    (5.5, 6.5,  10,  30, 0.00,  0.0, "sandy"),
    (6.0, 7.5,  35,  60, 0.12,  1.8, "clay"),
    (6.0, 7.0,  25,  50, 0.18,  2.0, "loamy"),
    (6.0, 7.0,  30,  52, 0.08,  1.3, "silty"),
]

# ─────────────────────────────────────────────
# Rule-based agent (never fails)
# ─────────────────────────────────────────────

def rule_based_predict(obs):
    try:
        r    = obs.get("soil_readings", {})
        ph   = float(r.get("ph",   6.5))
        mois = float(r.get("moisture_pct", 30))
        nit  = float(r.get("nitrogen_pct", 0.1))
        org  = float(r.get("organic_matter_pct", 2.0))
        task = obs.get("task_level", "easy")

        predicted = "loamy"
        for ph_lo, ph_hi, m_lo, m_hi, n_lo, o_lo, stype in RULES:
            if ph_lo <= ph <= ph_hi and m_lo <= mois <= m_hi and nit >= n_lo and org >= o_lo:
                predicted = stype
                break

        action = {"soil_type": predicted}
        if task in ("medium", "hard"):
            action["fertilizer"] = SOIL_PROFILES[predicted]["best_fertilizer"]
        if task == "hard":
            action["crop"] = SOIL_PROFILES[predicted]["best_crop"]
        return action
    except Exception:
        return {"soil_type": "loamy", "fertilizer": "Compost", "crop": "maize"}

# ─────────────────────────────────────────────
# LLM call (optional — falls back if it fails)
# ─────────────────────────────────────────────

def call_llm(prompt, task):
    if not API_BASE_URL or not HF_TOKEN:
        return None
    try:
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type":  "application/json",
        }
        body = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a soil scientist. Respond ONLY with a JSON object."},
                {"role": "user",   "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens":  128,
        }
        resp = requests.post(
            f"{API_BASE_URL}/chat/completions",
            headers=headers, json=body, timeout=15
        )
        resp.raise_for_status()
        text  = resp.json()["choices"][0]["message"]["content"]
        clean = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data  = json.loads(clean)
        action = {"soil_type": str(data.get("soil_type", "loamy"))}
        if task in ("medium", "hard") and "fertilizer" in data:
            action["fertilizer"] = str(data["fertilizer"])
        if task == "hard" and "crop" in data:
            action["crop"] = str(data["crop"])
        return action
    except Exception:
        return None

# ─────────────────────────────────────────────
# Environment HTTP helpers
# ─────────────────────────────────────────────

def env_reset(task="easy", seed=42, session_id="default"):
    try:
        r = requests.post(
            f"{ENV_URL}/reset",
            json={"task": task, "seed": seed, "session_id": session_id},
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("observation", {})
    except Exception as e:
        print(f"  [reset error] {e}", flush=True)
        return {}

def env_step(action, session_id="default"):
    try:
        payload = {"session_id": session_id, **action}
        r = requests.post(f"{ENV_URL}/step", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [step error] {e}", flush=True)
        return {"reward": 0.0, "done": True, "observation": {}, "info": {}}

def build_prompt(obs, task):
    r = obs.get("soil_readings", {})
    lines = [
        f"Soil readings: pH={r.get('ph')}, moisture={r.get('moisture_pct')}%,"
        f" nitrogen={r.get('nitrogen_pct')}, organic={r.get('organic_matter_pct')}%",
        f"Task: {task}  |  Valid soil types: {SOIL_TYPES}",
    ]
    if task in ("medium", "hard"):
        lines.append(f"Valid fertilizers: {FERTILIZERS}")
    if task == "hard":
        lines.append(f"Valid crops: {CROPS}")
    lines.append('Reply ONLY with JSON e.g. {"soil_type":"loamy","fertilizer":"Compost","crop":"maize"}')
    return "\n".join(lines)

# ─────────────────────────────────────────────
# Agent loop — prints required [START]/[STEP]/[END]
# ─────────────────────────────────────────────

def run_agent(task="hard", episodes=5):
    # ── REQUIRED: print [START] block ──────────
    print(f"[START] task={task}", flush=True)

    total_reward = 0.0

    for ep in range(1, episodes + 1):
        try:
            session_id = f"{task}_{ep}"
            obs = env_reset(task=task, seed=ep, session_id=session_id)

            if not obs:
                reward = 0.0
                # ── REQUIRED: print [STEP] block ──
                print(f"[STEP] step={ep} reward={reward:.4f}", flush=True)
                total_reward += reward
                continue

            # Try LLM, fall back to rule-based
            prompt = build_prompt(obs, task)
            action = call_llm(prompt, task)
            if action is None:
                action = rule_based_predict(obs)

            result  = env_step(action, session_id=session_id)
            reward  = float(result.get("reward") or 0.0)
            total_reward += reward

            # ── REQUIRED: print [STEP] block ──────
            print(f"[STEP] step={ep} reward={reward:.4f}", flush=True)

        except Exception as e:
            print(f"[STEP] step={ep} reward=0.0", flush=True)
            continue

    avg = total_reward / episodes if episodes > 0 else 0.0

    # ── REQUIRED: print [END] block ────────────
    print(f"[END] task={task} score={avg:.4f} steps={episodes}", flush=True)

    return avg


def main():
    print("Soil Analysis Inference Agent", flush=True)

    results = {}
    try:
        for level in ("easy", "medium", "hard"):
            results[level] = run_agent(task=level, episodes=5)

        overall = sum(results.values()) / len(results)
        print(f"[END] task=all score={overall:.4f} steps={5*3}", flush=True)

    except Exception as e:
        print(f"[END] task=all score=0.0 steps=0", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
