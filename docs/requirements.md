# Requirements Analysis & Design Document

**Project:** Smart Energy Management using Reinforcement Learning  
**Version:** 1.0  
**Date:** May 2026  
**SDGs:** SDG 11, SDG 12, SDG 13

---

## 1. Stakeholders

| Stakeholder | Role | Interest |
|---|---|---|
| **Building Operators** | Primary Users | Control HVAC/lighting decisions; want minimal manual intervention |
| **Energy Managers** | Primary Users | Monitor energy KPIs; want measurable cost/CO₂ reduction |
| **Sustainability Officers** | Secondary Users | Report on SDG targets (SDG 11, 12, 13); want auditable impact |
| **Occupants** | Affected Parties | Want thermal comfort (20–26°C); do not interact with the system |
| **IT/MLOps Engineers** | System Administrators | Deploy, retrain, and monitor the ML pipeline |
| **University Evaluators** | External Reviewers | Assess reproducibility, methodology, and MLOps practices |

---

## 2. Use Cases

### UC-01: Train RL Agent
- **Actor:** MLOps Engineer
- **Description:** Run `python train.py --config configs/qlearning_v1.yaml` to train a Q-learning agent. All metrics are logged to MLflow automatically.
- **Postcondition:** Trained policy saved to `policies/policy_v1.pkl`; run logged in `experiments/results_qlearning_v1.csv`.

### UC-02: Evaluate & Compare Policies
- **Actor:** Energy Manager
- **Description:** Run `python evaluate.py --policy policies/policy_v1.pkl` to compare the RL agent against the rule-based baseline over 100 episodes.
- **Postcondition:** Comparison table printed; plots saved to `reports/`; JSON summary saved.

### UC-03: Deploy RL Policy as REST API
- **Actor:** IT/MLOps Engineer
- **Description:** Start the FastAPI server; any BMS (Building Management System) can call `POST /predict` with current zone state to receive HVAC/lighting decisions.
- **Postcondition:** API returns `{hvac_on: [bool×4], lighting_on: [bool×4]}`; prediction logged to `logs/predictions.jsonl`.

### UC-04: Monitor Prediction Drift
- **Actor:** MLOps Engineer
- **Description:** Review `logs/predictions.jsonl` to detect state distribution shifts (e.g., new occupancy patterns). Trigger retraining if drift threshold exceeded.
- **Postcondition:** Drift report generated; retraining initiated if needed.

### UC-05: Rollback to Previous Policy
- **Actor:** MLOps Engineer
- **Description:** Run `python scripts/rollback.py exp-qlearning-1` to revert the deployed policy to an earlier Git-tagged version.
- **Postcondition:** `policies/` restored from tag; rollback logged to `experiments/rollback_log.json`.

### UC-06: Visualize Training via Dashboard
- **Actor:** Energy Manager / Evaluator
- **Description:** Launch Streamlit dashboard to explore training curves, live simulation, and baseline vs RL comparison interactively.

---

## 3. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| **FR-01** | System shall train a Q-learning agent using configurable YAML hyperparameters | High |
| **FR-02** | System shall log all training parameters and metrics to MLflow automatically | High |
| **FR-03** | System shall save trained policies as versioned `.pkl` files with Git tags | High |
| **FR-04** | System shall evaluate any saved policy against the rule-based baseline | High |
| **FR-05** | System shall expose a REST API (`POST /predict`) for real-time HVAC decisions | High |
| **FR-06** | System shall log every API prediction (timestamp, state, action) to `logs/predictions.jsonl` | High |
| **FR-07** | System shall support rollback to any tagged policy version via `scripts/rollback.py` | Medium |
| **FR-08** | System shall containerize the API using Docker for portable deployment | Medium |
| **FR-09** | System shall run automated tests on every push via GitHub Actions | High |
| **FR-10** | System shall provide a Streamlit dashboard for interactive visualization | Medium |
| **FR-11** | System shall generate reproducible results given a fixed random seed | High |
| **FR-12** | System shall save per-run CSVs and JSON logs for experiment tracking | High |

---

## 4. Non-Functional Requirements

| ID | Requirement | Metric |
|---|---|---|
| **NFR-01** | **Performance:** API response time < 100ms per prediction | Measured via FastAPI response headers |
| **NFR-02** | **Scalability:** Kubernetes deployment supports horizontal scaling to ≥2 replicas | Configured in `k8s/deployment.yaml` |
| **NFR-03** | **Reproducibility:** Given identical config and seed, training output is byte-identical | Validated by seeded numpy RNG |
| **NFR-04** | **Maintainability:** All configuration externalized to YAML; no hardcoded parameters in training code | Verified by code review |
| **NFR-05** | **Reliability:** API must gracefully handle missing policy file without crashing | Tested in `tests/test_api.py` |
| **NFR-06** | **Traceability:** Every deployed model must be traceable to a Git commit and MLflow run ID | Enforced by CI/CD tagging strategy |
| **NFR-07** | **Portability:** System must run on Linux (Ubuntu 20.04) via Docker without modification | Validated by GitHub Actions ubuntu runner |
| **NFR-08** | **Security:** API must only accept well-defined Pydantic-validated input schemas | Enforced by FastAPI + Pydantic |

---

## 5. Feasibility Analysis

### 5.1 Technical Feasibility
- **Confirmed feasible.** Tabular Q-learning converges on the 26,244-state discrete space within 500 episodes (~2 seconds on a CPU). No GPU required.
- FastAPI + Docker deployment is industry-standard and well-documented.
- GitHub Actions provides free CI/CD for public repositories.

### 5.2 Economic Feasibility
- **Zero direct cost.** All tools are open-source (MLflow, FastAPI, Docker, Streamlit, Python).
- Real-world deployment would require cloud VMs (estimated ~$50/month for a single-zone pilot).

### 5.3 Operational Feasibility
- Building operators interact only through the Streamlit dashboard or BMS integration — no ML expertise required.
- Retraining can be triggered via a single CLI command, making updates operationally lightweight.

---

## 6. Constraints

| Constraint | Detail |
|---|---|
| **C-01: State space discretisation** | Temperature and occupancy are binned into 3 levels each. Fine-grained control (e.g., 0.5°C steps) requires DQN with continuous inputs. |
| **C-02: Simulated environment** | Thermal model is a first-order ODE, not a calibrated EnergyPlus model. Results are indicative, not certified. |
| **C-03: Single-building scope** | The current model is trained for a fixed 4-zone configuration. Multi-building generalisation requires retraining. |
| **C-04: No real sensor integration** | The API receives state inputs manually; integration with real BMS sensors is out of scope for this prototype. |
| **C-05: Tabular scalability limit** | Q-table grows exponentially with zones: 4 zones = 26K states; 8 zones ≈ 600M states. DQN needed beyond ~6 zones. |

---

## 7. Trade-offs

| Trade-off | Option A | Option B | Decision |
|---|---|---|---|
| **Algorithm complexity** | Tabular Q-learning (fast, interpretable, no GPU) | DQN (handles large/continuous states, GPU needed) | **Q-learning** — state space is small enough |
| **Exploration** | High ε decay (v1: 0.995) — more exploitation | Low ε decay (v2: 0.990) — more exploration | **Both evaluated** — v1 selected for deployment (more stable) |
| **Comfort vs Energy** | High comfort weight (8.0) — fewer violations | Low comfort weight (5.0) — more energy savings | **5.0** — energy savings prioritised; tunable by operators |
| **Data versioning** | DVC (full dataset lineage) | Git tags (lightweight, < 1MB files) | **Git tags** — policies are small; DVC adds complexity without benefit |
| **Orchestration** | Kubernetes (production-grade) | Docker Compose (simpler, single-host) | **Both provided** — K8s for scale, Compose for local dev |

---

## 8. Risks

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| **R-01** | Occupancy distribution shift (e.g., WFH days) | Medium | High | Periodic retraining triggered by drift in `logs/predictions.jsonl` |
| **R-02** | Comfort violations in rarely-visited states | Medium | Medium | Increase comfort penalty weight (5.0 → 8.0); extend training episodes |
| **R-03** | Policy file corruption during deployment | Low | High | Git-tag-based rollback via `scripts/rollback.py` |
| **R-04** | API unavailability due to container crash | Low | High | Kubernetes liveness probe + auto-restart (configured in `k8s/deployment.yaml`) |
| **R-05** | Tabular approach fails to scale beyond 4 zones | High (future) | High | Architecture designed for DQN migration (environment interface unchanged) |
| **R-06** | Thermal model diverges from real building | Medium | Medium | Validation against EnergyPlus simulation before real deployment |

---

## 9. Requirements Traceability Matrix

| Requirement | Design Component | Implementation File | Test Coverage |
|---|---|---|---|
| FR-01 | Q-learning training loop | `train.py` | Manual validation (training output) |
| FR-02 | MLflow integration | `train.py` (mlflow calls) | Visual inspection via `mlflow ui` |
| FR-03 | Policy persistence + Git tags | `sim/agent.py` `save()`, `git tag` | Manual: `ls policies/`, `git tag` |
| FR-04 | Evaluation script | `evaluate.py` | Manual: `python evaluate.py` |
| FR-05 | REST API endpoint | `api.py` `POST /predict` | `tests/test_api.py::test_predict_action` |
| FR-06 | Prediction logging | `api.py` (JSONL logging) | Manual: inspect `logs/predictions.jsonl` |
| FR-07 | Rollback mechanism | `scripts/rollback.py` | Manual: `python scripts/rollback.py exp-qlearning-1` |
| FR-08 | Docker containerisation | `Dockerfile`, `docker-compose.yml` | GitHub Actions: `docker build` step |
| FR-09 | Automated CI/CD | `.github/workflows/ci.yml` | GitHub Actions green checkmark |
| FR-10 | Streamlit dashboard | `dashboard.py` | Manual: visual inspection |
| FR-11 | Reproducibility | `configs/*.yaml` (seed), seeded RNG | Re-run produces identical CSV |
| FR-12 | Experiment logs | `experiments/*.csv`, `logs/*.json` | Verified by file existence after training |
| NFR-01 | FastAPI async routing | `api.py` | FastAPI benchmark (< 100ms) |
| NFR-02 | K8s horizontal scaling | `k8s/deployment.yaml` (replicas: 2) | K8s `kubectl get pods` |
| NFR-05 | Graceful error handling | `api.py` startup try/except | `tests/test_api.py::test_health_check` |
