# Smart Energy Management in Buildings ⚡️
### Reinforcement Learning + MLOps Project for Sustainable Cities

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project implements a **Reinforcement Learning (RL)** solution to optimize HVAC and lighting control in a 4-zone office building. By balancing energy consumption with occupant comfort, the agent supports global sustainability goals (**SDGs 11, 12, and 13**).

---

## 🚀 Key Features

- **RL-Based Control**: Tabular Q-Learning agent trained to manage energy across multiple building zones.
- **Custom Simulation**: A discrete-time building environment (`BuildingEnv`) modeling occupancy, thermal dynamics, and energy costs.
- **Interactive Dashboard**: Streamlit-based UI for live simulations, training visualization, and baseline comparisons.
- **REST API**: Production-ready FastAPI service for real-time action predictions.
- **Experiment Tracking**: Integrated MLflow for logging parameters, rewards, and model artifacts.
- **Production Infrastructure**: Fully containerized with Docker/Docker Compose and Kubernetes readiness.
- **Reproducible Pipeline**: Configuration-driven experiments via YAML for consistent results across environments.

---

## 🛠 Tech Stack

- **Core**: Python 3.10+
- **Reinforcement Learning**: Tabular Q-Learning (Custom implementation)
- **Data & Math**: NumPy, Pandas, Matplotlib
- **API & Web**: FastAPI, Streamlit, Uvicorn
- **MLOps & DevOps**: MLflow, DVC (Data Tracking), Docker, Kubernetes
- **Configuration**: YAML
- **Testing**: Pytest, HTTPX

---

## 📦 Project Structure

```text
smart_energy_r1/
├── sim/                     # Simulator core logic
│   ├── building_env.py      # 4-zone building environment
│   ├── agent.py             # Q-learning agent implementation
│   └── baseline.py          # Rule-based control logic
├── configs/                 # Experiment YAML configurations
├── experiments/             # Training logs (CSV)
├── policies/                # Serialized RL models (.pkl)
├── logs/                    # JSON run summaries
├── reports/                 # Analysis and MLOps reports
├── k8s/                     # Kubernetes deployment manifests
├── scripts/                 # Utility scripts (e.g., rollback)
├── api.py                   # FastAPI prediction service
├── dashboard.py             # Streamlit visualization UI
├── train.py                 # Main training pipeline
├── evaluate.py              # Performance evaluation script
├── Dockerfile               # Container build instructions
├── docker-compose.yml       # Multi-container orchestration
└── requirements.txt         # Project dependencies
```

---

## 🚦 Getting Started

### 1. Local Setup
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Using Docker
Run the entire stack (API + Dashboard) using Docker Compose:
```bash
docker-compose up --build
```
- **API**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`

---

## 🏃‍♂️ Usage

### Training
Train new agents using different exploration strategies defined in the configs:
```bash
python train.py --config configs/qlearning_v1.yaml
python train.py --config configs/qlearning_v2.yaml
```

### Interactive Dashboard
Launch the dashboard to visualize results and run live simulations:
```bash
streamlit run dashboard.py
```

### Evaluation
Compare a trained policy against the rule-based baseline:
```bash
python evaluate.py --policy policies/policy_v2_explored.pkl \
                   --results experiments/results_qlearning_v2.csv
```

### Running Tests
```bash
pytest
```

---

## 📊 Results Summary

| Metric | Rule-based Baseline | RL Policy (v2) | Change |
|---|---|---|---|
| Avg Energy Cost / Episode | 111.60 | 90.45 | **−18.9%** |
| Avg Episode Reward | −490.47 | −320.12 | **+34.7%** |
| Comfort Violations | 378.60 | 124.20 | **−67.2%** |

*Note: Results may vary slightly based on environment stochasticity.*

---

## 🌍 SDG Impact
- **SDG 11 (Sustainable Cities)**: Reduces urban energy load and carbon footprint.
- **SDG 12 (Responsible Consumption)**: Optimizes resource use through intelligent automation.
- **SDG 13 (Climate Action)**: Direct reduction in CO₂ emissions (~2 tonnes/year for a typical office building).

---

## 📄 Documentation
- [RL Methodology](reports/RL_Report_PartA.md)
- [MLOps Architecture](reports/MLOps_Report_PartB.md)
- [Final Impact Report](reports/Final_Report.md)
