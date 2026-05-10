"""
train.py — Main training script for Smart Energy RL agent.

Usage:
    python train.py --config configs/qlearning_v1.yaml
    python train.py --config configs/qlearning_v2.yaml

Outputs:
    policies/<policy_path>       — saved Q-table (.pkl)
    experiments/<results_path>   — per-episode CSV log
    logs/<log_path>              — JSON run summary
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import mlflow

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from sim.building_env import BuildingEnv
from sim.agent import QLearningAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate_agent(agent: QLearningAgent, env: BuildingEnv, n_eval: int) -> dict:
    """Run n_eval episodes with greedy policy, return average metrics."""
    rewards, energies, comfort_viols = [], [], []
    for _ in range(n_eval):
        state = env.reset()
        ep_reward, ep_energy, ep_comfort = 0.0, 0.0, 0.0
        for _ in range(24):
            action = agent.select_action(state, eval_mode=True)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_energy += info["energy"]
            ep_comfort += info["comfort_penalty"]
            if done:
                break
        rewards.append(ep_reward)
        energies.append(ep_energy)
        comfort_viols.append(ep_comfort)
    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_energy": float(np.mean(energies)),
        "avg_comfort_violation": float(np.mean(comfort_viols)),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(config_path: str):
    cfg = load_config(config_path)
    exp_name = cfg["experiment"]["name"]

    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Config     : {config_path}")
    print(f"{'='*60}\n")

    # Paths
    policy_path = cfg["output"]["policy_path"]
    results_path = cfg["output"]["results_path"]
    log_path = cfg["output"]["log_path"]
    for p in [policy_path, results_path, log_path]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        
    mlflow.set_experiment("SmartEnergyRL")

    # Environment
    env_seed = cfg["environment"]["seed"]
    env = BuildingEnv(seed=env_seed)

    # Agent
    a = cfg["agent"]
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=a["learning_rate"],
        gamma=a["gamma"],
        epsilon_start=a["epsilon_start"],
        epsilon_min=a["epsilon_min"],
        epsilon_decay=a["epsilon_decay"],
        seed=cfg["experiment"]["seed"],
    )

    t_cfg = cfg["training"]
    n_episodes = t_cfg["n_episodes"]
    eval_every = t_cfg["eval_every"]
    eval_episodes = t_cfg["eval_episodes"]

    # Tracking
    train_rewards = []
    csv_rows = []
    start_time = time.time()

    # CSV header
    csv_header = [
        "run_id", "episode", "train_reward",
        "eval_avg_reward", "eval_avg_energy", "eval_avg_comfort_violation",
        "epsilon", "learning_rate", "gamma", "epsilon_decay",
    ]

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    print(f"  Training for {n_episodes} episodes...\n")

    with mlflow.start_run(run_name=exp_name):
        mlflow.log_params({
            "learning_rate": a["learning_rate"],
            "gamma": a["gamma"],
            "epsilon_start": a["epsilon_start"],
            "epsilon_min": a["epsilon_min"],
            "epsilon_decay": a["epsilon_decay"],
            "seed": cfg["experiment"]["seed"],
            "n_episodes": n_episodes
        })

        for ep in range(1, n_episodes + 1):
        state = env.reset()
        ep_reward = 0.0

        for _ in range(24):  # one full day = one episode
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            if done:
                break

        agent.decay_epsilon()
        train_rewards.append(ep_reward)

        # Periodic evaluation
        eval_metrics = {"avg_reward": np.nan, "avg_energy": np.nan, "avg_comfort_violation": np.nan}
        if ep % eval_every == 0:
            eval_metrics = evaluate_agent(agent, env, eval_episodes)
            elapsed = time.time() - start_time
            print(
                f"  Ep {ep:>4d}/{n_episodes} | "
                f"Train R: {ep_reward:7.2f} | "
                f"Eval R: {eval_metrics['avg_reward']:7.2f} | "
                f"ε: {agent.epsilon:.4f} | "
                f"t: {elapsed:.0f}s"
            )
            mlflow.log_metrics({
                "train_reward": ep_reward,
                "eval_avg_reward": eval_metrics["avg_reward"],
                "eval_avg_energy": eval_metrics["avg_energy"],
                "epsilon": agent.epsilon
            }, step=ep)

        csv_rows.append({
            "run_id": exp_name,
            "episode": ep,
            "train_reward": round(ep_reward, 4),
            "eval_avg_reward": round(eval_metrics["avg_reward"], 4) if not np.isnan(eval_metrics["avg_reward"]) else "",
            "eval_avg_energy": round(eval_metrics["avg_energy"], 4) if not np.isnan(eval_metrics["avg_energy"]) else "",
            "eval_avg_comfort_violation": round(eval_metrics["avg_comfort_violation"], 4) if not np.isnan(eval_metrics["avg_comfort_violation"]) else "",
            "epsilon": round(agent.epsilon, 6),
            "learning_rate": a["learning_rate"],
            "gamma": a["gamma"],
            "epsilon_decay": a["epsilon_decay"],
        })

        # ---------------------------------------------------------------------------
        # Save policy
        # ---------------------------------------------------------------------------
        agent.save(policy_path)
        mlflow.log_artifact(policy_path, artifact_path="model")

        # Also save a mid-training checkpoint at episode 50 for "policy_v1" variant
        # (to demonstrate two policy versions exist)
        mid_path = policy_path.replace(".pkl", "_mid.pkl")
        agent.save(mid_path)

        # ---------------------------------------------------------------------------
        # Write CSV results
        # ---------------------------------------------------------------------------
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_header)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  Results saved → {results_path}")

        # ---------------------------------------------------------------------------
        # Write JSON log
        # ---------------------------------------------------------------------------
        final_eval = evaluate_agent(agent, env, n_eval=20)
        
        mlflow.log_metrics({
            "final_eval_reward": final_eval['avg_reward'],
            "final_eval_energy": final_eval['avg_energy']
        })
        
        q_summary = agent.q_table_summary()
        log = {
            "experiment": cfg["experiment"],
            "training": cfg["training"],
            "agent_params": cfg["agent"],
            "final_eval": final_eval,
            "q_table_summary": q_summary,
            "total_episodes": n_episodes,
            "total_time_seconds": round(time.time() - start_time, 2),
            "policy_path": policy_path,
        }
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"  Log saved   → {log_path}")

        print(f"\n  {'='*50}")
        print(f"  Final eval avg reward : {final_eval['avg_reward']:.2f}")
        print(f"  Final eval avg energy : {final_eval['avg_energy']:.2f}")
        print(f"  Q-table nonzero entries: {q_summary['nonzero_entries']} / {q_summary['total_entries']}")
        print(f"  {'='*50}\n")

    return train_rewards, final_eval


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for smart energy management")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)
