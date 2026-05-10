from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from sim.agent import QLearningAgent
from sim.building_env import BuildingEnv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("smart_energy_api")

app = FastAPI(title="Smart Energy RL API", version="1.0.0")

# Global variables to hold model and env
agent = None
env = None

class StateInput(BaseModel):
    hour: int
    occupancies: list[int]  # 4 integers (0, 1, or 2)
    temperatures: list[int] # 4 integers (0, 1, or 2)

class ActionOutput(BaseModel):
    hvac_on: list[bool]
    lighting_on: list[bool]

@app.on_event("startup")
def load_model():
    global agent, env
    env = BuildingEnv(seed=42)
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
    try:
        agent.load("policies/policy_v2_explored.pkl")
        logger.info("Successfully loaded RL policy.")
    except Exception as e:
        logger.warning(f"Could not load policy. Starting with untrained agent. Error: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Smart Energy RL API is running"}

@app.post("/predict", response_model=ActionOutput)
def predict_action(state_input: StateInput):
    if len(state_input.occupancies) != 4 or len(state_input.temperatures) != 4:
        raise HTTPException(status_code=400, detail="Must provide exactly 4 occupancies and 4 temperatures.")
    
    # Calculate state index manually
    hour_bin = state_input.hour // 6
    if hour_bin > 3: hour_bin = 3
    
    state_idx = hour_bin
    
    # Add occupancy
    multiplier = 4
    for o in state_input.occupancies:
        state_idx += o * multiplier
        multiplier *= 3
        
    # Add temperature
    for t in state_input.temperatures:
        state_idx += t * multiplier
        multiplier *= 3
        
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
        
    logger.info(f"Prediction requested. State: {state_idx}, Action: {action_idx}")
    
    return ActionOutput(hvac_on=hvac, lighting_on=light)
