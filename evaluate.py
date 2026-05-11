"""
evaluate.py — Compare rule-based baseline vs trained RL policy.
Generates comparison table + plots for the final report.

Usage:
    python evaluate.py --policy policies/policy_v1.pkl --results experiments/results_qlearning_v1.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).parent))

from sim.building_env import BuildingEnv
from sim.agent import QLearningAgent
from sim.baseline import run_baseline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_rl_policy(policy_path: str, n_episodes: int = 100, seed: int = 0) -> dict:
    env = BuildingEnv(seed=seed)
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
    agent.load(policy_path)

    total_rewards, total_energies, total_comfort = [], [], []
    for _ in range(n_episodes):
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
        total_rewards.append(ep_reward)
        total_energies.append(ep_energy)
        total_comfort.append(ep_comfort)

    return {
        "avg_reward": float(np.mean(total_rewards)),
        "avg_energy": float(np.mean(total_energies)),
        "avg_comfort_violation": float(np.mean(total_comfort)),
        "all_rewards": total_rewards,
    }


def print_comparison_table(baseline: dict, rl: dict):
    improvement_energy = (baseline["avg_energy"] - rl["avg_energy"]) / baseline["avg_energy"] * 100
    improvement_reward = (rl["avg_reward"] - baseline["avg_reward"]) / abs(baseline["avg_reward"]) * 100
    improvement_comfort = (baseline["avg_comfort_violation"] - rl["avg_comfort_violation"]) / max(baseline["avg_comfort_violation"], 1e-6) * 100

    print("\n" + "="*65)
    print(f"  {'Metric':<35} {'Baseline':>10} {'RL Policy':>10} {'Change':>7}")
    print("  " + "-"*61)
    print(f"  {'Avg episode reward':<35} {baseline['avg_reward']:>10.2f} {rl['avg_reward']:>10.2f} {improvement_reward:>+6.1f}%")
    print(f"  {'Avg energy cost (per episode)':<35} {baseline['avg_energy']:>10.2f} {rl['avg_energy']:>10.2f} {-improvement_energy:>+6.1f}%")
    print(f"  {'Avg comfort violation (per ep)':<35} {baseline['avg_comfort_violation']:>10.2f} {rl['avg_comfort_violation']:>10.2f} {improvement_comfort:>+6.1f}%")
    print("="*65)
    print(f"\n  Energy saved vs baseline: {improvement_energy:.1f}%")
    print(f"  SDG 11/12/13 impact: Reduced energy consumption by ~{improvement_energy:.0f}%,")
    print(f"  contributing to lower carbon emissions and more sustainable building operations.\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_training_curve(results_csv: str, out_dir: str = "reports"):
    Path(out_dir).mkdir(exist_ok=True)
    df = pd.read_csv(results_csv)

    # Smooth training reward
    window = 20
    df["smooth_reward"] = df["train_reward"].rolling(window=window, min_periods=1).mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Q-Learning Training — Smart Energy Management", fontsize=13, fontweight="bold")

    # Panel 1: Reward over episodes
    ax = axes[0]
    ax.plot(df["episode"], df["train_reward"], color="#d0d0d0", linewidth=0.6, label="Per-episode reward")
    ax.plot(df["episode"], df["smooth_reward"], color="#1D9E75", linewidth=2, label=f"{window}-ep moving avg")

    # Plot eval points if present
    eval_df = df.dropna(subset=["eval_avg_reward"])
    if len(eval_df) > 0:
        ax.scatter(eval_df["episode"], eval_df["eval_avg_reward"],
                   color="#E8593C", zorder=5, s=40, label="Greedy eval reward")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Reward over training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Epsilon decay
    ax2 = axes[1]
    ax2.plot(df["episode"], df["epsilon"], color="#534AB7", linewidth=1.8)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("ε (exploration rate)")
    ax2.set_title("ε-greedy exploration decay")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    out_path = f"{out_dir}/training_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {out_path}")


def plot_comparison(baseline: dict, rl: dict, out_dir: str = "reports"):
    Path(out_dir).mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Baseline vs RL Policy — Smart Energy Management", fontsize=13, fontweight="bold")

    # Panel 1: Reward distribution
    ax = axes[0]
    ax.hist(baseline["all_rewards"], bins=20, alpha=0.6, color="#888780", label="Rule-based baseline", edgecolor="white")
    ax.hist(rl["all_rewards"], bins=20, alpha=0.7, color="#1D9E75", label="RL policy", edgecolor="white")
    ax.axvline(baseline["avg_reward"], color="#444441", linestyle="--", linewidth=1.5, label=f"Baseline mean: {baseline['avg_reward']:.1f}")
    ax.axvline(rl["avg_reward"], color="#0F6E56", linestyle="--", linewidth=1.5, label=f"RL mean: {rl['avg_reward']:.1f}")
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward distribution (100 episodes)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Bar chart metrics
    ax2 = axes[1]
    metrics = ["Avg reward", "Avg energy cost", "Avg comfort violation"]
    b_vals = [abs(baseline["avg_reward"]), baseline["avg_energy"], baseline["avg_comfort_violation"]]
    r_vals = [abs(rl["avg_reward"]), rl["avg_energy"], rl["avg_comfort_violation"]]

    x = np.arange(len(metrics))
    w = 0.35
    bars1 = ax2.bar(x - w/2, b_vals, w, label="Rule-based baseline", color="#888780", edgecolor="white")
    bars2 = ax2.bar(x + w/2, r_vals, w, label="RL policy", color="#1D9E75", edgecolor="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=8)
    ax2.set_title("Key metric comparison")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = f"{out_dir}/baseline_vs_rl.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="policies/policy_v1.pkl")
    parser.add_argument("--results", type=str, default="experiments/results_qlearning_v1.csv")
    parser.add_argument("--n_eval", type=int, default=100)
    args = parser.parse_args()

    print("\n  Running baseline controller...")
    baseline = run_baseline(n_episodes=args.n_eval, seed=0)

    print(f"  Running RL policy: {args.policy}")
    rl = run_rl_policy(args.policy, n_episodes=args.n_eval, seed=0)

    print_comparison_table(baseline, rl)

    print("  Generating plots...")
    plot_training_curve(args.results)
    plot_comparison(baseline, rl)

    # Save comparison JSON
    comparison = {
        "baseline": {k: v for k, v in baseline.items() if k != "all_rewards"},
        "rl_policy": {k: v for k, v in rl.items() if k != "all_rewards"},
    }
    with open("reports/comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("  Comparison saved -> reports/comparison_results.json")
