"""
dashboard.py — Streamlit dashboard for Smart Energy RL project.

Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sim.building_env import BuildingEnv
from sim.agent import QLearningAgent
from sim.baseline import baseline_action

# ---------------------------------------------------------------------------
st.set_page_config(page_title="Smart Energy RL", layout="wide")

st.title("Smart Energy Management — RL Dashboard")
st.caption("SDG 11 · SDG 12 · SDG 13 | Algorithm: Q-Learning | Zones: 4")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")
policy_choice = st.sidebar.radio(
    "Policy to evaluate",
    ["Rule-based baseline", "RL Policy v1", "RL Policy v2 (explored)"]
)
n_eval_episodes = st.sidebar.slider("Evaluation episodes", 10, 200, 50, step=10)
show_live_sim = st.sidebar.checkbox("Show live simulation", value=True)

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Training curves",
    "Baseline vs RL",
    "Live simulation",
    "Experiment logs"
])

# ---------------------------------------------------------------------------
# TAB 1 — Training curves
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Training curves")

    col1, col2 = st.columns(2)

    for i, (label, csv_path) in enumerate([
        ("v1 — moderate exploration (α=0.1, decay=0.995)", "experiments/results_qlearning_v1.csv"),
        ("v2 — extended exploration (α=0.2, decay=0.990)", "experiments/results_qlearning_v2.csv"),
    ]):
        col = col1 if i == 0 else col2
        with col:
            st.markdown(f"**{label}**")
            try:
                df = pd.read_csv(csv_path)
                df["smooth"] = df["train_reward"].rolling(20, min_periods=1).mean()

                fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

                axes[0].plot(df["episode"], df["train_reward"], color="#d0d0d0", lw=0.5, label="Per-episode")
                axes[0].plot(df["episode"], df["smooth"], color="#1D9E75", lw=2, label="20-ep avg")
                eval_df = df.dropna(subset=["eval_avg_reward"])
                if len(eval_df):
                    axes[0].scatter(eval_df["episode"], eval_df["eval_avg_reward"],
                                    color="#E8593C", s=30, zorder=5, label="Greedy eval")
                axes[0].set_ylabel("Reward")
                axes[0].legend(fontsize=7)
                axes[0].grid(alpha=0.3)
                axes[0].set_title("Episode reward")

                axes[1].plot(df["episode"], df["epsilon"], color="#534AB7", lw=1.5)
                axes[1].set_ylabel("ε")
                axes[1].set_xlabel("Episode")
                axes[1].grid(alpha=0.3)
                axes[1].set_title("Exploration decay")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            except FileNotFoundError:
                st.warning(f"Run `python train.py --config configs/qlearning_v{'1' if i==0 else '2'}.yaml` first.")

# ---------------------------------------------------------------------------
# TAB 2 — Baseline vs RL
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Baseline vs RL policy comparison")

    policy_map = {
        "Rule-based baseline": None,
        "RL Policy v1": "policies/policy_v1.pkl",
        "RL Policy v2 (explored)": "policies/policy_v2_explored.pkl",
    }

    @st.cache_data
    def run_comparison(policy_path, n_ep, seed=0):
        env = BuildingEnv(seed=seed)

        def run_policy(use_rl, pkl_path=None):
            if use_rl:
                agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
                agent.load(pkl_path)
            rewards, energies, comforts = [], [], []
            for _ in range(n_ep):
                state = env.reset()
                r, e, c = 0.0, 0.0, 0.0
                for h in range(24):
                    if use_rl:
                        action = agent.select_action(state, eval_mode=True)
                    else:
                        action = baseline_action(h)
                    state, reward, done, info = env.step(action)
                    r += reward; e += info["energy"]; c += info["comfort_penalty"]
                    if done: break
                rewards.append(r); energies.append(e); comforts.append(c)
            return rewards, energies, comforts

        b_r, b_e, b_c = run_policy(False)

        if policy_path and Path(policy_path).exists():
            rl_r, rl_e, rl_c = run_policy(True, policy_path)
        else:
            rl_r, rl_e, rl_c = b_r, b_e, b_c  # fallback

        return (b_r, b_e, b_c), (rl_r, rl_e, rl_c)

    selected_policy = policy_map[policy_choice]
    (b_r, b_e, b_c), (rl_r, rl_e, rl_c) = run_comparison(selected_policy, n_eval_episodes)

    # Metrics row
    energy_improvement = (np.mean(b_e) - np.mean(rl_e)) / np.mean(b_e) * 100
    reward_improvement = (np.mean(rl_r) - np.mean(b_r)) / abs(np.mean(b_r)) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baseline avg energy", f"{np.mean(b_e):.1f}")
    m2.metric("RL avg energy", f"{np.mean(rl_e):.1f}", delta=f"{-energy_improvement:.1f}%")
    m3.metric("Baseline avg reward", f"{np.mean(b_r):.1f}")
    m4.metric("RL avg reward", f"{np.mean(rl_r):.1f}", delta=f"{reward_improvement:.1f}%")

    # Comparison table
    st.markdown("### Summary table")
    comp_df = pd.DataFrame({
        "Metric": ["Avg episode reward", "Avg energy cost", "Avg comfort violation"],
        "Rule-based baseline": [f"{np.mean(b_r):.2f}", f"{np.mean(b_e):.2f}", f"{np.mean(b_c):.2f}"],
        f"{policy_choice}": [f"{np.mean(rl_r):.2f}", f"{np.mean(rl_e):.2f}", f"{np.mean(rl_c):.2f}"],
        "Change": [
            f"{reward_improvement:+.1f}%",
            f"{-energy_improvement:+.1f}%",
            f"{(np.mean(b_c)-np.mean(rl_c))/max(np.mean(b_c),1)*100:+.1f}%"
        ]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Distribution plot
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.hist(b_r, bins=20, alpha=0.6, color="#888780", label="Rule-based baseline", edgecolor="white")
    ax.hist(rl_r, bins=20, alpha=0.7, color="#1D9E75", label=policy_choice, edgecolor="white")
    ax.axvline(np.mean(b_r), color="#444441", linestyle="--", lw=1.5, label=f"Baseline mean: {np.mean(b_r):.1f}")
    ax.axvline(np.mean(rl_r), color="#0F6E56", linestyle="--", lw=1.5, label=f"RL mean: {np.mean(rl_r):.1f}")
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info(f"**SDG Impact:** Energy reduction of ~{energy_improvement:.0f}% supports SDG 11, 12, 13 — "
            f"less wasted energy, lower carbon emissions, smarter cities.")

# ---------------------------------------------------------------------------
# TAB 3 — Live simulation
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Live simulation — one full day")

    if not show_live_sim:
        st.info("Enable 'Show live simulation' in the sidebar to run this.")
    else:
        sim_mode = st.radio("Controller", ["Rule-based baseline", "RL Policy v1", "RL Policy v2 (explored)"],
                            horizontal=True)
        run_btn = st.button("▶ Run one day simulation")

        if run_btn:
            env = BuildingEnv(seed=np.random.randint(0, 1000))
            use_rl = sim_mode != "Rule-based baseline"

            if use_rl:
                pkl = "policies/policy_v1.pkl" if "v1" in sim_mode else "policies/policy_v2_explored.pkl"
                agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
                if Path(pkl).exists():
                    agent.load(pkl)
                else:
                    st.error(f"Policy file not found: {pkl}")
                    st.stop()

            env.reset()
            zone_names = ["Office A", "Office B", "Meeting Rm", "Lobby"]
            records = []

            for h in range(24):
                state = env._get_state_index()
                if use_rl:
                    action = agent.select_action(state, eval_mode=True)
                else:
                    action = baseline_action(h)
                _, reward, done, info = env.step(action)

                for z in range(4):
                    records.append({
                        "Hour": h,
                        "Zone": zone_names[z],
                        "Temp (°C)": round(info["zone_temps"][z], 1),
                        "Occupancy": info["zone_occ"][z],
                        "HVAC": "ON" if info["hvac_on"][z] else "OFF",
                        "Lighting": "ON" if info["light_on"][z] else "OFF",
                        "Energy": round(info["energy"] / 4, 2),
                        "Reward": round(reward / 4, 2),
                    })

            df_sim = pd.DataFrame(records)

            # Temperature over time
            fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
            colors = ["#1D9E75", "#534AB7", "#E8593C", "#BA7517"]
            for i, z in enumerate(zone_names):
                zdf = df_sim[df_sim["Zone"] == z]
                axes[0].plot(zdf["Hour"], zdf["Temp (°C)"], label=z, color=colors[i], lw=1.8)
            axes[0].axhline(20, color="gray", linestyle="--", lw=1, label="Comfort min (20°C)")
            axes[0].axhline(26, color="gray", linestyle=":", lw=1, label="Comfort max (26°C)")
            axes[0].set_xlabel("Hour")
            axes[0].set_ylabel("Temperature (°C)")
            axes[0].set_title(f"Zone temperatures — {sim_mode}")
            axes[0].legend(fontsize=7)
            axes[0].grid(alpha=0.3)

            # Energy per hour
            hourly_e = df_sim.groupby("Hour")["Energy"].sum()
            axes[1].bar(hourly_e.index, hourly_e.values, color="#1D9E75", alpha=0.8, edgecolor="white")
            axes[1].set_xlabel("Hour")
            axes[1].set_ylabel("Energy (units)")
            axes[1].set_title("Energy consumption per hour")
            axes[1].grid(alpha=0.3, axis="y")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("### Hour-by-hour detail")
            st.dataframe(
                df_sim.pivot_table(index="Hour", columns="Zone",
                                   values=["Temp (°C)", "HVAC", "Lighting"],
                                   aggfunc="first"),
                use_container_width=True
            )

# ---------------------------------------------------------------------------
# TAB 4 — Experiment logs
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Experiment logs")

    col1, col2 = st.columns(2)
    for col, log_path, label in [
        (col1, "logs/log_qlearning_v1.json", "v1 — moderate exploration"),
        (col2, "logs/log_qlearning_v2.json", "v2 — extended exploration"),
    ]:
        with col:
            st.markdown(f"**{label}**")
            try:
                with open(log_path) as f:
                    log = json.load(f)
                st.json(log)
            except FileNotFoundError:
                st.warning("Log not found — run training first.")

    st.markdown("### Raw CSV — experiment v1")
    try:
        df_csv = pd.read_csv("experiments/results_qlearning_v1.csv")
        st.dataframe(df_csv.tail(50), use_container_width=True)
    except FileNotFoundError:
        st.warning("CSV not found.")