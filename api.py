"""
api.py — FastAPI REST API for Smart Energy RL agent.

Endpoints:
  GET  /          — health check
  POST /predict   — HVAC + lighting decision for current zone state
  GET  /metrics   — prediction log summary (monitoring)

Prediction logs are written to logs/predictions.jsonl for drift monitoring.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from sim.agent import QLearningAgent
from sim.building_env import BuildingEnv

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("smart_energy_api")

PREDICTION_LOG = Path("logs/predictions.jsonl")
PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
agent = None
env = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global agent, env
    env = BuildingEnv(seed=42)
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
    policy_path = "policies/policy_v2_explored.pkl"
    try:
        agent.load(policy_path)
        logger.info(f"Loaded RL policy from {policy_path}")
    except Exception as e:
        logger.warning(f"Could not load policy. Starting with untrained agent. Error: {e}")
    yield
    logger.info("API shutting down.")

app = FastAPI(
    title="Smart Energy RL API",
    version="1.0.0",
    description="REST API for the Smart Energy Q-Learning agent. Provides real-time HVAC and lighting decisions.",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class StateInput(BaseModel):
    hour: int                  # 0–23
    occupancies: list[int]     # 4 ints: 0=empty, 1=partial, 2=full
    temperatures: list[int]    # 4 ints: 0=cold, 1=comfort, 2=hot

    model_config = {
        "json_schema_extra": {
            "example": {
                "hour": 14,
                "occupancies": [2, 1, 0, 0],
                "temperatures": [1, 1, 1, 1]
            }
        }
    }

class ActionOutput(BaseModel):
    hvac_on: list[bool]
    lighting_on: list[bool]
    state_index: int
    action_index: int
    timestamp: str

# ---------------------------------------------------------------------------
# Helper: encode state
# ---------------------------------------------------------------------------
def encode_state(hour: int, occupancies: list, temperatures: list) -> int:
    """Map continuous state -> discrete integer index (matches BuildingEnv.py)."""
    hour_bin = min(hour // 6, 3)
    
    occ_idx = 0
    for z, o in enumerate(occupancies):
        occ_idx += o * (3 ** z)
        
    temp_idx = 0
    for z, t in enumerate(temperatures):
        temp_idx += t * (3 ** z)
        
    # State = hour_bin * 3^4 * 3^4 + occ_idx * 3^4 + temp_idx
    return (hour_bin * 6561) + (occ_idx * 81) + temp_idx

# ---------------------------------------------------------------------------
# Helper: structured prediction log (for drift monitoring)
# ---------------------------------------------------------------------------
def log_prediction(state_input: StateInput, state_idx: int, action_idx: int,
                   hvac: list, light: list):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": {
            "hour": state_input.hour,
            "occupancies": state_input.occupancies,
            "temperatures": state_input.temperatures,
        },
        "state_index": state_idx,
        "action_index": action_idx,
        "output": {
            "hvac_on": hvac,
            "lighting_on": light,
        }
    }
    with open(PREDICTION_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Smart Energy RL API is running"}


@app.post("/predict", response_model=ActionOutput, tags=["Inference"])
def predict_action(state_input: StateInput):
    """
    Return HVAC and lighting decisions for 4 building zones.

    - **hour**: Current hour (0–23)
    - **occupancies**: Occupancy level per zone (0=empty, 1=partial, 2=full)
    - **temperatures**: Temperature bin per zone (0=cold <20°C, 1=comfort 20–26°C, 2=hot >26°C)
    """
    if len(state_input.occupancies) != 4 or len(state_input.temperatures) != 4:
        raise HTTPException(
            status_code=400,
            detail="Must provide exactly 4 occupancies and 4 temperatures."
        )

    if agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Calculate state index to match BuildingEnv.py
    state_idx = encode_state(
        state_input.hour,
        state_input.occupancies,
        state_input.temperatures
    )
        
    # Get action
    action_idx = agent.select_action(state_idx, eval_mode=True)
    
    # Decode action
    hvac = []
    light = []
    temp_action = action_idx
    for _ in range(4):
        zone_act = temp_action % 4
        hvac.append(bool(zone_act & 1))
        light.append(bool((zone_act >> 1) & 1))
        temp_action //= 4
        
    logger.info(f"Prediction requested (v2). State: {state_idx}, Action: {action_idx}")
    logger.info(f"Predict | hour={state_input.hour} | state={state_idx} | action={action_idx}")
    log_prediction(state_input, state_idx, action_idx, hvac, light)

    ts = datetime.now(timezone.utc).isoformat()
    return ActionOutput(
        hvac_on=hvac,
        lighting_on=light,
        state_index=state_idx,
        action_index=action_idx,
        timestamp=ts,
    )


@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    """
    Return a summary of all predictions logged so far.
    Used for drift monitoring — check if state distribution has shifted.
    """
    if not PREDICTION_LOG.exists():
        return {"total_predictions": 0, "log_file": str(PREDICTION_LOG)}

    records = []
    with open(PREDICTION_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {"total_predictions": 0}

    state_indices = [r["state_index"] for r in records]
    action_indices = [r["action_index"] for r in records]
    hours = [r["input"]["hour"] for r in records]

    # Drift detection: compare current state std vs training baseline
    # Training baseline: state_std ~ 1800 (empirically from 500-episode run)
    BASELINE_STATE_STD = 1800.0
    DRIFT_THRESHOLD = 0.30  # alert if std deviates >30% from baseline
    current_std = float(np.std(state_indices))
    drift_ratio = abs(current_std - BASELINE_STATE_STD) / BASELINE_STATE_STD
    drift_alert = drift_ratio > DRIFT_THRESHOLD

    return {
        "total_predictions": len(records),
        "first_prediction": records[0]["timestamp"],
        "last_prediction": records[-1]["timestamp"],
        "state_distribution": {
            "mean": round(float(np.mean(state_indices)), 2),
            "std": round(current_std, 2),
            "min": int(np.min(state_indices)),
            "max": int(np.max(state_indices)),
        },
        "action_distribution": {
            "mean": round(float(np.mean(action_indices)), 2),
            "std": round(float(np.std(action_indices)), 2),
            "unique_actions": len(set(action_indices)),
        },
        "hour_distribution": {
            "mean_hour": round(float(np.mean(hours)), 1),
        },
        "drift_monitoring": {
            "baseline_state_std": BASELINE_STATE_STD,
            "current_state_std": round(current_std, 2),
            "drift_ratio": round(drift_ratio, 3),
            "drift_alert": drift_alert,
            "recommendation": "Consider retraining — state distribution has shifted." if drift_alert else "Distribution stable. No retraining needed.",
        },
        "log_file": str(PREDICTION_LOG),
    }
