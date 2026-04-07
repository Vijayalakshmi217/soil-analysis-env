import sys

def run_task(task, episodes=10, seed=42):
    try:
        from soil_env.env import SoilAnalysisEnv, SOIL_PROFILES
    except ImportError:
        print(f"[START] task={task}", flush=True)
        print(f"[STEP] step=1 reward=0.0", flush=True)
        print(f"[END] task={task} score=0.0 steps=1", flush=True)
        return 0.0

    RULES = [
        (3.5, 5.5, 55, 100, 0.25, 15.0, "peaty"),
        (5.5, 6.5, 10, 30, 0.00, 0.0,  "sandy"),
        (6.0, 7.5, 35, 60, 0.12, 1.8,  "clay"),
        (6.0, 7.0, 25, 50, 0.18, 2.0,  "loamy"),
        (6.0, 7.0, 30, 52, 0.08, 1.3,  "silty"),
    ]

    def rule_based_agent(obs):
        r = obs["soil_readings"]
        ph   = r["ph"]
        mois = r["moisture_pct"]
        nit  = r["nitrogen_pct"]
        org  = r["organic_matter_pct"]
        predicted_soil = "loamy"
        for ph_lo, ph_hi, m_lo, m_hi, n_lo, o_lo, stype in RULES:
            if ph_lo <= ph <= ph_hi and m_lo <= mois <= m_hi and nit >= n_lo and org >= o_lo:
                predicted_soil = stype
                break
        action = {"soil_type": predicted_soil}
        if obs["task_level"] in ("medium", "hard"):
            action["fertilizer"] = SOIL_PROFILES[predicted_soil]["best_fertilizer"]
        if obs["task_level"] == "hard":
            action["crop"] = SOIL_PROFILES[predicted_soil]["best_crop"]
        return action

    env = SoilAnalysisEnv(task=task, seed=seed)
    print(f"[START] task={task}", flush=True)
    total_reward = 0.0
    for ep in range(episodes):
        obs = env.reset()
        action = rule_based_agent(obs)
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
