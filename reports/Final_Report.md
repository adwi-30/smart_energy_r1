# Final Report — Smart Energy Management in Buildings
### Reinforcement Learning + MLOps Project

**Student:** [Your Name]  
**SDGs:** 11 · 12 · 13  
**Date:** May 2026

---

## 1. Problem Statement

Commercial buildings account for 30–40% of global energy use. Fixed-schedule controllers
(the industry norm) run HVAC and lighting on rigid timetables regardless of actual occupancy
or weather, wasting significant energy in empty rooms and off-hours periods.

**Goal:** Train a Reinforcement Learning agent to control HVAC and lighting across four zones
of an office building — minimising energy consumption while keeping occupant temperatures
within a comfortable range (20–26 °C).

---

## 2. SDG Alignment

| SDG | How this project contributes |
|---|---|
| **SDG 11** — Sustainable Cities | Smart building control reduces urban energy demand and supports liveable, resource-efficient cities. |
| **SDG 12** — Responsible Consumption | Avoiding wasted energy in unoccupied zones directly reduces resource consumption per productive work-hour. |
| **SDG 13** — Climate Action | Lower electricity demand reduces grid-level CO₂ emissions; every 20% energy reduction in buildings matters at city scale. |

---

## 3. Simulator

The building simulator (`sim/building_env.py`) models a four-zone office building:

| Zone | Type | Peak occupancy |
|---|---|---|
| Office A | Open plan | 50 people |
| Office B | Open plan | 50 people |
| Meeting Room | Conference | 25 people |
| Lobby | Reception | 10 people |

**Occupancy profile:** Business hours (08:00–18:00) are fully occupied; evenings see only
lobby traffic; nights are empty. Occupancy is sampled stochastically each episode.

**Thermal model:** Indoor temperature drifts toward outdoor temperature (sinusoidal daily cycle,
mean 30 °C, ±5 °C peak) with HVAC correction toward a 23 °C setpoint when active.

**State discretisation:** Hour bucket (×4) × occupancy bins (×3 per zone) × temperature bins (×3 per zone).

---

## 4. RL Methodology

### 4.1 Algorithm

**Q-learning** (tabular) — chosen because the state space is discrete and small (~26K states),
making tabular convergence feasible without neural networks.

### 4.2 State · Action · Reward

```
State  : (hour_bin, occ_bin×4, temp_bin×4) → single integer index
Action : HVAC on/off × Lighting on/off per zone → 256 integer actions
Reward : −(energy_cost + comfort_penalty + waste_penalty)
```

### 4.3 Exploration

ε-greedy with exponential decay: `ε_t = max(0.05, 1.0 × decay^t)`  
- v1: decay = 0.995  
- v2: decay = 0.990 (slower, more exploration)

### 4.4 Training convergence

Average reward improves from approximately −1,385 (episode 50) to −634 (episode 350),
demonstrating that the agent learns to reduce unnecessary energy use in unoccupied zones.
Reward variance remains due to stochastic occupancy and weather.

---

## 5. MLOps

| Practice | Implementation |
|---|---|
| Versioning | Git tags: `exp-qlearning-1`, `exp-qlearning-2` |
| Experiment tracking | `experiments/results_x.csv` + `logs/log_x.json` per run |
| Hyperparameter management | YAML configs in `configs/` — no hardcoded values |
| Reproducibility | Fixed seeds in config; `pip install -r requirements.txt` |
| Policy snapshots | `.pkl` files committed at each experiment tag |

**To reproduce:**
```bash
python train.py --config configs/qlearning_v1.yaml
python evaluate.py --policy policies/policy_v1.pkl \
                   --results experiments/results_qlearning_v1.csv
```

---

## 6. Baseline vs RL Comparison

The rule-based baseline runs HVAC and lighting for all zones during business hours (08–18)
and lobby lighting in the evening (18–22), regardless of occupancy or temperature.

| Metric | Rule-based baseline | RL policy (v2) | Change |
|---|---|---|---|
| Avg episode reward | −490.5 | −897.6 | — |
| Avg energy cost / episode | 111.6 units | 89.7 units | **−19.6%** |
| Avg comfort violation / episode | 378.6 | 761.3 | +101% |

### Interpretation

**When RL performs better:**  
The RL agent correctly identifies that night-time and empty zones need minimal or no HVAC,
reducing energy cost by ~20% compared to the baseline.

**When RL performs worse:**  
Comfort violations are significantly higher for the RL agent in this 500-episode run.
The reward function penalises comfort at weight 5.0, but the agent has not yet fully
learnt to balance comfort vs. energy in the meeting room zone (lower occupancy frequency
means fewer learning samples for that zone's states). With more training episodes (1,000–2,000),
comfort violations would be expected to decrease as those states are visited more.

**Sensitivity analysis:**  
When traffic is shifted (e.g., all four zones fully occupied in evenings), the RL agent
initially acts suboptimally — it has learnt that evenings have low occupancy and turns off HVAC.
This represents a distribution shift failure; the monitoring plan (Section 7) addresses this
with a retraining trigger.

---

## 7. Monitoring Plan

If deployed in a real building:

- **Energy metrics:** Daily kWh per zone; alert if >10% above 7-day rolling average.
- **Comfort metrics:** Temperature every 15 min; alert if outside 20–26 °C for >30 min.
- **Policy health:** Q-value drift; action entropy; operator override frequency.
- **Safety hard rules:** Lobby lighting always on; HVAC must not be off >4h in occupied zones.
- **Retraining triggers:** Comfort violation rate >5% over 7 days; energy up >15% for 3 consecutive days.

---

## 8. SDG Impact

Reducing average energy cost by **~20%** across a commercial building has concrete SDG implications:

- **SDG 11:** A city where 10,000 buildings adopt smart control saves the equivalent of several
  medium-sized power plants' daily output — directly supporting sustainable urban infrastructure.
- **SDG 12:** Eliminating wasted HVAC in empty meeting rooms means energy is only consumed when
  it produces value (occupied, comfortable space).
- **SDG 13:** At India's grid emission factor of ~0.71 kg CO₂/kWh, a 20% reduction in a 100-zone
  building running 8 hours/day saves approximately **2–4 tonnes CO₂/year per building**.

---

## 9. Limitations

- Thermal model is simplified; real deployment would use EnergyPlus or a calibrated RC model.
- Comfort violations are higher than baseline in this run — needs more training or reward re-weighting.
- Multi-zone thermal coupling (shared air handling unit) is not modelled.
- The Q-table explores only 0.09% of the full state-action space — a DQN could generalise better.

---

## 10. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train
python train.py --config configs/qlearning_v1.yaml
python train.py --config configs/qlearning_v2.yaml

# Evaluate and generate plots
python evaluate.py --policy policies/policy_v2_explored.pkl \
                   --results experiments/results_qlearning_v2.csv
```

All outputs (policies, CSVs, logs, plots) are saved automatically.

---

## Appendix A — File List

| File | Purpose |
|---|---|
| `sim/building_env.py` | 4-zone building simulator |
| `sim/agent.py` | Q-learning agent |
| `sim/baseline.py` | Rule-based baseline controller |
| `configs/qlearning_v1.yaml` | Experiment 1 config |
| `configs/qlearning_v2.yaml` | Experiment 2 config |
| `train.py` | Training entry point |
| `evaluate.py` | Evaluation + plots |
| `policies/policy_v1.pkl` | Saved Q-table (v1) |
| `policies/policy_v2_explored.pkl` | Saved Q-table (v2) |
| `experiments/results_qlearning_v*.csv` | Per-episode tracking |
| `logs/log_qlearning_v*.json` | Run summaries |
| `reports/training_curve.png` | Reward + epsilon plots |
| `reports/baseline_vs_rl.png` | Comparison plots |
