"""
baseline.py  —  Rule-based baseline agent (no LLM needed).

Tests the environment locally without any API calls.
"""

import random
from soil_env.env import SoilAnalysisEnv, SOIL_PROFILES, ALL_SOIL_TYPES
from soil_env.models import SoilAction


# ---------------------------------------------------------------------------
# Simple rule-based agent
# ---------------------------------------------------------------------------

RULES = [
    # (ph_lo, ph_hi, mois_lo, mois_hi, nit_lo, org_lo, soil_type)
    (3.5, 5.5,  55, 100, 0.25, 15.0, "peaty"),
    (5.5, 6.5,  10,  30, 0.00,  0.0, "sandy"),
    (6.0, 7.5,  35,  60, 0.12,  1.8, "clay"),
    (6.0, 7.0,  25,  50, 0.18,  2.0, "loamy"),
    (6.0, 7.0,  30,  52, 0.08,  1.3, "silty"),
]


def rule_based_agent(obs: dict) -> SoilAction:
    r    = obs["soil_readings"]
    ph   = r["ph"]
    mois = r["moisture_pct"]
    nit  = r["nitrogen_pct"]
    org  = r["organic_matter_pct"]
    task = obs["task_level"]

    predicted = "loamy"  # default
    for ph_lo, ph_hi, m_lo, m_hi, n_lo, o_lo, stype in RULES:
        if ph_lo <= ph <= ph_hi and m_lo <= mois <= m_hi and nit >= n_lo and org >= o_lo:
            predicted = stype
            break

    fert = SOIL_PROFILES[predicted]["best_fertilizer"] if task in ("medium", "hard") else None
    crop = SOIL_PROFILES[predicted]["best_crop"]        if task == "hard"            else None

    return SoilAction(soil_type=predicted, fertilizer=fert, crop=crop)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_baseline(task: str, episodes: int = 10, seed: int = 42):
    env          = SoilAnalysisEnv(task=task, seed=seed)
    total_reward = 0.0

    print(f"\n{'='*55}")
    print(f"  Task: {task.upper()}  |  Episodes: {episodes}")
    print(f"{'='*55}")

    for ep in range(episodes):
        obs    = env.reset(seed=seed + ep)
        action = rule_based_agent(obs.model_dump())

        result     = env.step(action)
        reward     = result.reward or 0.0
        true_type  = result.metadata.get("true_soil_type", "?")
        match      = "✓" if action.soil_type == true_type else "✗"

        print(
            f"  Ep {ep+1:02d} | true={true_type:<8} pred={action.soil_type:<8} "
            f"{match} | reward={reward:.2f}"
        )
        total_reward += reward

    avg = round(total_reward / episodes, 4)
    print(f"  Average reward: {avg:.4f}")
    return avg


if __name__ == "__main__":
    print("Soil Analysis Environment — Baseline Agent")
    results = {}
    for level in ("easy", "medium", "hard"):
        results[level] = run_baseline(level, episodes=10, seed=42)

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    for level, score in results.items():
        print(f"  {level:<8} {score:.4f}")
    overall = round(sum(results.values()) / 3, 4)
    print(f"\n  Overall avg: {overall:.4f}")
