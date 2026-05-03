# Part B — MLOps Implementation Report
## Smart Energy Management in Buildings

**Student:** [Your Name]  
**Date:** May 2026

---

## 1. Overview

This document describes the MLOps practices applied to the Smart Energy RL project,
covering experiment versioning, tracking, reproducibility, and a monitoring plan
for hypothetical real-world deployment.

---

## 2. Versioning

### Git commit structure

The project uses Git with meaningful commit messages and tags per experiment run:

```bash
# Initial setup
git init
git add .
git commit -m "Initial project structure: sim, configs, experiments, policies"

# After first training run
git add experiments/results_qlearning_v1.csv policies/policy_v1.pkl logs/log_qlearning_v1.json
git commit -m "exp-qlearning-1: baseline run, alpha=0.1, eps_decay=0.995, 500 episodes"
git tag exp-qlearning-1

# After second training run
git add experiments/results_qlearning_v2.csv policies/policy_v2_explored.pkl logs/log_qlearning_v2.json
git commit -m "exp-qlearning-2: extended exploration, alpha=0.2, eps_decay=0.990, 500 episodes"
git tag exp-qlearning-2

# After evaluation
git add reports/
git commit -m "evaluation: baseline vs RL comparison, plots generated"
git tag final-eval
```

**Tags created:** `exp-qlearning-1`, `exp-qlearning-2`, `final-eval`

To switch back to a previous experiment:
```bash
git checkout exp-qlearning-1
```

---

## 3. Experiment Tracking

### 3.1 Per-run CSV logs

Every training run produces a `results_x.csv` file in `experiments/`. Each row records:

| Column | Description |
|---|---|
| `run_id` | Experiment name (e.g., `qlearning_v1`) |
| `episode` | Episode number |
| `train_reward` | Total reward during that training episode |
| `eval_avg_reward` | Greedy policy average reward (logged every 50 episodes) |
| `eval_avg_energy` | Average energy cost during greedy evaluation |
| `eval_avg_comfort_violation` | Average comfort penalty during greedy evaluation |
| `epsilon` | Exploration rate at this episode |
| `learning_rate` | α used for this run |
| `gamma` | Discount factor |
| `epsilon_decay` | Decay rate used |

**Files produced:**
- `experiments/results_qlearning_v1.csv` (500 rows)
- `experiments/results_qlearning_v2.csv` (500 rows)

### 3.2 JSON run summary

Each run also produces a `logs/log_x.json` with a full summary:

```json
{
  "experiment": { "name": "qlearning_v1", "seed": 42 },
  "agent_params": { "learning_rate": 0.1, "gamma": 0.95, ... },
  "final_eval": {
    "avg_reward": -769.09,
    "avg_energy": 91.40,
    "avg_comfort_violation": 636.85
  },
  "q_table_summary": {
    "nonzero_entries": 5950,
    "total_entries": 6718464
  },
  "total_time_seconds": 0.8
}
```

---

## 4. Reproducibility

### 4.1 How to reproduce a run

```bash
# 1. Clone the repository
git clone <repo-url>
cd smart_energy_rl

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run experiment v1
python train.py --config configs/qlearning_v1.yaml

# 4. Run experiment v2
python train.py --config configs/qlearning_v2.yaml

# 5. Evaluate and generate plots
python evaluate.py --policy policies/policy_v2_explored.pkl \
                   --results experiments/results_qlearning_v2.csv
```

All random seeds are fixed in the YAML config (`experiment.seed` and `environment.seed`),
ensuring identical results on every run on any machine.

### 4.2 Reproducibility guarantees

| Factor | How it is controlled |
|---|---|
| Random seed | NumPy `default_rng(seed)` in both env and agent |
| Hyperparameters | All defined in YAML config files, not hardcoded |
| Environment version | `requirements.txt` pins package versions |
| Policy snapshots | `policies/policy_v1.pkl` committed with the experiment tag |

Anyone can clone the repository, run the two commands above, and obtain the
same CSV results, the same Q-table, and the same evaluation metrics.

---

## 5. Monitoring Plan (Design Only)

If this RL agent were deployed in a real commercial building, the following metrics
and signals would be monitored continuously:

### 5.1 Energy metrics
- **Total daily energy consumption** (kWh) per zone — alert if >10% above rolling 7-day average.
- **Peak demand** (kW) — flag if the agent creates demand spikes during peak tariff windows.
- **Wasted energy** — energy delivered to zones with zero occupancy.

### 5.2 Comfort metrics
- **Zone temperature** (°C) every 15 minutes — alert if outside 20–26 °C for >30 minutes.
- **Occupant comfort complaints** — manual override count per day as a proxy.

### 5.3 Policy health metrics
- **Average Q-value drift** — monitor if the deployed Q-table produces actions with Q-values
  that differ significantly from training-time values (indicates distribution shift).
- **Action entropy** — if the agent consistently picks the same action regardless of state,
  it may have collapsed to a suboptimal policy.
- **Override frequency** — how often building operators manually override the RL decision.

### 5.4 Safety constraints (hard rules)
- **No red-zone HVAC** — if outdoor temperature is below 10 °C, at least one zone must have
  HVAC on to prevent pipe freezing, regardless of RL decision.
- **Emergency lighting** — lighting in escape routes (lobby) must never be turned off.
- **Maximum off-time** — HVAC cannot be off for more than 4 consecutive hours in occupied zones.

### 5.5 Retraining triggers
- Comfort violation rate > 5% of steps over a rolling 7-day window.
- Energy use rises >15% above the pre-deployment baseline for 3 consecutive days.
- Building occupancy pattern changes significantly (e.g., new shift schedule).

---

## 6. Folder Structure

```
smart_energy_rl/
├── sim/
│   ├── __init__.py
│   ├── building_env.py      # Simulator
│   ├── agent.py             # Q-learning agent
│   └── baseline.py          # Rule-based baseline
├── configs/
│   ├── qlearning_v1.yaml    # Experiment 1 config
│   └── qlearning_v2.yaml    # Experiment 2 config
├── experiments/
│   ├── results_qlearning_v1.csv
│   └── results_qlearning_v2.csv
├── policies/
│   ├── policy_v1.pkl
│   ├── policy_v1_mid.pkl
│   ├── policy_v2_explored.pkl
│   └── policy_v2_explored_mid.pkl
├── logs/
│   ├── log_qlearning_v1.json
│   └── log_qlearning_v2.json
├── reports/
│   ├── RL_Report_PartA.md
│   ├── MLOps_Report_PartB.md
│   ├── Final_Report.md
│   ├── training_curve.png
│   └── baseline_vs_rl.png
├── train.py                 # Training entry point
├── evaluate.py              # Evaluation + plots
├── requirements.txt
└── README.md
```
