from soil_env.env import SoilAnalysisEnv, SOIL_PROFILES

RULES = [
    (3.5, 5.5,  55, 100, 0.25, 15.0, "peaty"),
    (5.5, 6.5,  10,  30, 0.00,  0.0, "sandy"),
    (6.0, 7.5,  35,  60, 0.12,  1.8, "clay"),
    (6.0, 7.0,  25,  50, 0.18,  2.0, "loamy"),
    (6.0, 7.0,  30,  52, 0.08,  1.3, "silty"),
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

def run_baseline(task, episodes=10, seed=42):
    env = SoilAnalysisEnv(task=task, seed=seed)
    total_reward = 0.0
    print(f"\n{'='*50}")
    print(f"  Task: {task.upper()}  |  Episodes: {episodes}")
    print(f"{'='*50}")
    for ep in range(episodes):
        obs = env.reset()
        action = rule_based_agent(obs)
        _, reward, _, info = env.step(action)
        true_type = info.get("true_soil_type", "?")
        predicted = action["soil_type"]
        match = "CORRECT" if predicted == true_type else "WRONG"
        print(f"  Ep {ep+1:02d} | true={true_type:<8} pred={predicted:<8} {match} | reward={reward:.2f}")
        total_reward += reward
    avg = round(total_reward / episodes, 4)
    print(f"  Average reward: {avg:.4f}")
    return avg

if __name__ == "__main__":
    print("Soil Analysis Environment - Baseline Agent")
    results = {}
    for level in ("easy", "medium", "hard"):
        results[level] = run_baseline(level, episodes=10, seed=42)
    print("\n" + "="*50)
    print("  SUMMARY")
    print("="*50)
    for level, score in results.items():
        print(f"  {level:<8} {score:.4f}")
    overall = round(sum(results.values()) / 3, 4)
    print(f"\n  Overall avg: {overall:.4f}")