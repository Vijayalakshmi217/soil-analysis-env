import os
from openai import OpenAI

# Use the injected proxy - THIS IS THE KEY FIX
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ.get("API_KEY", "dummy-key"),
)

def llm_predict_soil(obs):
    r = obs["soil_readings"]
    task_level = obs["task_level"]

    prompt = f"""You are a soil analysis expert.
Given these soil readings:
- pH: {r['ph']}
- Moisture: {r['moisture_pct']}%
- Nitrogen: {r['nitrogen_pct']}%
- Organic Matter: {r['organic_matter_pct']}%

Task level: {task_level}

Respond ONLY in this exact format with no extra text:
soil_type: <one of: sandy, clay, loamy, silty, peaty>
fertilizer: <fertilizer name or none>
crop: <crop name or none>"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0,
    )

    text = response.choices[0].message.content.strip().lower()

    # Parse response
    soil_type = "loamy"
    fertilizer = None
    crop = None

    for line in text.split("\n"):
        if line.startswith("soil_type:"):
            soil_type = line.split(":", 1)[1].strip()
        elif line.startswith("fertilizer:") and task_level in ("medium", "hard"):
            val = line.split(":", 1)[1].strip()
            if val != "none":
                fertilizer = val
        elif line.startswith("crop:") and task_level == "hard":
            val = line.split(":", 1)[1].strip()
            if val != "none":
                crop = val

    action = {"soil_type": soil_type}
    if fertilizer:
        action["fertilizer"] = fertilizer
    if crop:
        action["crop"] = crop
    return action


def run_task(task, episodes=10, seed=42):
    try:
        from soil_env.env import SoilAnalysisEnv
    except ImportError:
        print(f"[START] task={task}", flush=True)
        print(f"[STEP] step=1 reward=0.0", flush=True)
        print(f"[END] task={task} score=0.0 steps=1", flush=True)
        return 0.0

    env = SoilAnalysisEnv(task=task, seed=seed)
    print(f"[START] task={task}", flush=True)

    total_reward = 0.0
    for ep in range(episodes):
        obs = env.reset()
        try:
            action = llm_predict_soil(obs)
        except Exception as e:
            print(f"[STEP] step={ep+1} reward=0.0", flush=True)
            continue
        _, reward, _, info = env.step(action)
        total_reward += reward
        print(f"[STEP] step={ep+1} reward={round(reward, 4)}", flush=True)

    score = round(total_reward / episodes, 4)
    print(f"[END] task={task} score={score} steps={episodes}", flush=True)
    return score


if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    scores = {}
    for task in tasks:
        scores[task] = run_task(task, episodes=10, seed=42)

    overall = round(sum(scores.values()) / len(scores), 4)
    print(f"[START] task=overall", flush=True)
    print(f"[STEP] step=1 reward={overall}", flush=True)
    print(f"[END] task=overall score={overall} steps=1", flush=True)
