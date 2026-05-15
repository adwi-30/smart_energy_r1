"""
Q-Learning Agent for Smart Energy Management.

Algorithm choice: Q-learning
Reason: The state space (hour bucket × occupancy bins × temperature bins per zone)
        is discrete and small (~324 states), and the action space (HVAC+lighting
        on/off per zone) is also discrete (256 actions). Q-learning converges
        reliably on tabular MDPs of this size without needing a neural network.

State  : Discrete index encoding (hour_bin, occ_bin × zones, temp_bin × zones).
Action : Integer encoding HVAC on/off × Lighting on/off for each zone.
Reward : Negative of (energy cost + comfort penalty + wasted energy penalty).

Exploration: ε-greedy with exponential decay — start greedy-random, anneal
             toward greedy-optimal to balance exploration and convergence.
"""

import numpy as np
import pickle
from pathlib import Path


class QLearningAgent:
    """
    Tabular Q-Learning agent.

    Parameters
    ----------
    n_states      : number of discrete states in the environment
    n_actions     : number of discrete actions
    learning_rate : α — step size for Bellman update
    gamma         : discount factor
    epsilon_start : initial exploration rate
    epsilon_min   : minimum exploration rate (floor)
    epsilon_decay : multiplicative decay applied each episode
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # Q-table initialised to zero (optimistic init also works)
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

        # Tracking
        self.episode_count = 0
        self.epsilon_history = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, state: int, eval_mode: bool = False) -> int:
        """ε-greedy action selection. Uses greedy policy in eval_mode."""
        if not eval_mode and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state]))

    # ------------------------------------------------------------------
    # Q-table update
    # ------------------------------------------------------------------
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Bellman update:
            Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
        """
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state])
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------
    def decay_epsilon(self):
        """Call once per episode after training."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
        self.episode_count += 1

    # ------------------------------------------------------------------
    # Save / load policy
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Persist Q-table and hyperparameters to a .pkl file."""
        payload = {
            "Q": self.Q,
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "hyperparams": {
                "learning_rate": self.lr,
                "gamma": self.gamma,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min,
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"  [Agent] Policy saved -> {path}")

    def load(self, path: str):
        """Load Q-table and hyperparameters from a .pkl file."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.Q = payload["Q"]
        self.epsilon = payload["epsilon"]
        self.episode_count = payload["episode_count"]
        print(f"  [Agent] Policy loaded <- {path}  (episode {self.episode_count})")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def q_table_summary(self) -> dict:
        return {
            "mean_q": float(np.mean(self.Q)),
            "max_q": float(np.max(self.Q)),
            "min_q": float(np.min(self.Q)),
            "nonzero_entries": int(np.count_nonzero(self.Q)),
            "total_entries": int(self.Q.size),
        }
