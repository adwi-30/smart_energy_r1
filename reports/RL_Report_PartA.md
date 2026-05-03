# Part A — RL Methodology Report
## Smart Energy Management in Buildings

**Student:** [Your Name]  
**SDGs Addressed:** SDG 11 (Sustainable Cities), SDG 12 (Responsible Consumption), SDG 13 (Climate Action)  
**Date:** May 2026

---

## 1. Problem Statement

Modern commercial buildings consume 30–40% of global energy, much of it wasted through fixed schedules
that run HVAC and lighting regardless of actual occupancy or outdoor conditions. The goal of this project
is to train a Reinforcement Learning agent to control HVAC and lighting systems across four building zones
— Office A, Office B, Meeting Room, and Lobby — in order to minimise energy consumption while maintaining
occupant comfort (temperature within 20–26 °C).

**SDG alignment:**
- **SDG 11** — Sustainable Cities and Communities: smart building management reduces urban energy demand.
- **SDG 12** — Responsible Consumption and Production: avoiding wasteful energy use in unoccupied zones.
- **SDG 13** — Climate Action: lower electricity consumption directly reduces CO₂ emissions.

---

## 2. Algorithm Choice

**Algorithm selected: Q-learning (tabular)**

**One-line justification:**  
Q-learning was chosen because the state space (hour bucket × occupancy bins × temperature bins per zone)
is discrete and small (~324 states for 4 zones), and Q-learning is guaranteed to converge to the optimal
policy for finite MDPs without requiring a neural network or GPU.

Alternative algorithms considered:
- **DQN** — unnecessary complexity for this state space size; tabular Q-learning converges faster here.
- **SARSA** — on-policy; would learn a more conservative policy, but off-policy Q-learning is preferable
  since we want to separate exploration from the target policy.

---

## 3. State, Action, and Reward Design

### 3.1 State Space

The state is a discrete integer encoding of:

| Component | Discretisation | Values |
|---|---|---|
| Hour of day | 4 buckets (0–5, 6–11, 12–17, 18–23) | 0–3 |
| Occupancy per zone | 3 bins: empty / partial / full | 0–2 (×4 zones) |
| Indoor temperature per zone | 3 bins: cold / comfort / hot | 0–2 (×4 zones) |

Total states ≈ 4 × 3⁴ × 3⁴ = **26,244 states** (actual reachable states: ~5,950 as seen in Q-table).

### 3.2 Action Space

Each zone has two binary controls:
- **HVAC** — on (1) or off (0)
- **Lighting** — on (1) or off (0)

Each zone's 4 combinations are encoded into a single integer:  
`action = Σ zone_action × 4^z` for z in 0..3 → **256 total actions**.

### 3.3 Reward Function

```
reward = −(energy_cost + comfort_penalty + waste_penalty)
```

| Component | Formula | Rationale |
|---|---|---|
| Energy cost | `2.0 × HVAC_on + 0.5 × Light_on + 0.1 × idle` per zone | Penalises total power draw |
| Comfort penalty | `5.0 × |temp − comfort_range|` per zone | Heavy penalty for thermal discomfort |
| Waste penalty | `1.5 × energy_spent` on unoccupied zone | Discourages running equipment when nobody is present |

The agent receives a negative reward each step; maximising reward means minimising cost.

---

## 4. Exploration Strategy

**Strategy: ε-greedy with exponential decay**

```
ε_t = max(ε_min, ε_0 × decay^t)
```

| Parameter | v1 value | v2 value |
|---|---|---|
| ε start | 1.0 | 1.0 |
| ε min | 0.05 | 0.05 |
| ε decay | 0.995 | 0.990 |
| Learning rate (α) | 0.1 | 0.2 |

In v1, ε reaches 0.05 around episode 598; in v2 it reaches floor earlier (~episode 299),
giving more exploitation time at the cost of less exploration.

---

## 5. Training Results and Convergence

Training was run for **500 episodes** (each episode = one 24-hour day) for both configurations.

### Convergence observation

| Checkpoint | v1 Eval Reward | v2 Eval Reward |
|---|---|---|
| Episode 50 | −1,385 | −915 |
| Episode 100 | −936 | −755 |
| Episode 150 | −1,218 | −560 |
| Episode 200 | −667 | −1,079 |
| Episode 300 | −670 | −694 |
| Episode 350 | −634 | −557 |
| Episode 500 | −799 | −889 |

**Discussion:**  
Average reward improves substantially over the first 200 episodes as the agent learns to turn off
HVAC and lighting in unoccupied zones at night. Rewards partially stabilise after episode 300,
though variance remains because the episode reward depends on stochastic occupancy and outdoor
temperature. v2 (slower ε decay, higher α) explores more broadly and achieves a stronger early
reduction in episode 150 (−560), but converges more noisily.

The Q-table has **5,950 non-zero entries** out of 6,718,464 total, confirming that only a small
portion of the state-action space is visited — as expected for a building that follows regular
daily patterns.

---

## 6. Saved Policies

| File | Description |
|---|---|
| `policies/policy_v1.pkl` | Q-learning v1 — moderate exploration, α=0.1 |
| `policies/policy_v1_mid.pkl` | Mid-training checkpoint (v1) |
| `policies/policy_v2_explored.pkl` | Q-learning v2 — extended exploration, α=0.2 |
| `policies/policy_v2_explored_mid.pkl` | Mid-training checkpoint (v2) |

Policies contain: Q-table (NumPy array), ε at save time, episode count, hyperparameters.

---

## 7. Limitations and Future Work

- The state space bins temperature and occupancy coarsely — a finer discretisation or a DQN
  with continuous state inputs would capture more nuance.
- The thermal model is simplified (first-order ODE); a real building would use EnergyPlus or
  a calibrated RC-network model.
- Multi-zone interactions (shared air handling units) are not modelled.
- Future work: add electricity price as a state feature to enable demand-response scheduling.
