"""
Baseline (rule-based) controller for Smart Energy Management.

Logic:
  - HVAC ON during business hours (08:00–18:00) for all zones.
  - Lighting ON during business hours + evening lobby (18:00–22:00).
  - Everything OFF at night.

This is the "Fixed-Timer" equivalent for our building domain.
Used as the comparison baseline against the RL policy.
"""

import numpy as np
from sim.building_env import BuildingEnv, NUM_ZONES


def baseline_action(hour: int) -> int:
    """Return a fixed rule-based action index for the given hour."""
    hvac_on = np.zeros(NUM_ZONES, dtype=bool)
    light_on = np.zeros(NUM_ZONES, dtype=bool)

    if 8 <= hour < 18:
        hvac_on[:] = True
        light_on[:] = True
    elif 18 <= hour < 22:
        light_on[3] = True   # lobby light only

    # Encode back to action integer
    action = 0
    for z in range(NUM_ZONES):
        zone_action = int(hvac_on[z]) | (int(light_on[z]) << 1)
        action += zone_action * (4 ** z)
    return action


def run_baseline(n_episodes: int = 100, seed: int = 0) -> dict:
    """Run baseline controller for n_episodes, return aggregated metrics."""
    env = BuildingEnv(seed=seed)
    total_rewards = []
    total_energies = []
    total_comfort_violations = []

    for ep in range(n_episodes):
        env.reset()
        ep_reward = 0.0
        ep_energy = 0.0
        ep_comfort = 0.0

        for h in range(24):
            action = baseline_action(h)
            _, reward, done, info = env.step(action)
            ep_reward += reward
            ep_energy += info["energy"]
            ep_comfort += info["comfort_penalty"]
            if done:
                break

        total_rewards.append(ep_reward)
        total_energies.append(ep_energy)
        total_comfort_violations.append(ep_comfort)

    return {
        "avg_reward": float(np.mean(total_rewards)),
        "avg_energy": float(np.mean(total_energies)),
        "avg_comfort_violation": float(np.mean(total_comfort_violations)),
        "all_rewards": total_rewards,
    }


if __name__ == "__main__":
    results = run_baseline(n_episodes=50)
    print(f"Baseline avg reward:           {results['avg_reward']:.2f}")
    print(f"Baseline avg energy cost:      {results['avg_energy']:.2f}")
    print(f"Baseline avg comfort violation:{results['avg_comfort_violation']:.2f}")
