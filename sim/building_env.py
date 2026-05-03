"""
Smart Energy Management Simulator
Building environment with HVAC, lighting, and occupancy zones.

SDG Alignment:
  SDG 11 - Sustainable Cities and Communities
  SDG 12 - Responsible Consumption and Production
  SDG 13 - Climate Action
"""

import numpy as np
import random


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_ZONES = 4          # e.g., Office A, Office B, Meeting Room, Lobby
HOURS_PER_DAY = 24
STEPS_PER_HOUR = 1     # 1 step = 1 hour (keep state space small)
MAX_OCCUPANCY = 50     # max people per zone

# Comfort temperature range (°C)
TEMP_MIN = 20.0
TEMP_MAX = 26.0
OUTDOOR_TEMP_MEAN = 30.0   # hot city climate (Bengaluru-like)
OUTDOOR_TEMP_STD = 4.0

# Energy costs (arbitrary units, proportional to kWh)
HVAC_COST_PER_STEP = 2.0      # per zone when HVAC is ON
LIGHTING_COST_PER_STEP = 0.5  # per zone when lights are ON
IDLE_COST = 0.1               # base standby draw per zone

# Penalty weights for comfort violation
COMFORT_PENALTY = 5.0         # per zone per degree outside comfort range
WASTED_ENERGY_PENALTY = 1.5   # energy spent on unoccupied zone


class BuildingEnv:
    """
    Discrete building energy environment.

    State  : (hour_bin, occupancy_bin per zone, indoor_temp_bin per zone)
    Action : 2^NUM_ZONES * 2^NUM_ZONES  (HVAC on/off × Lighting on/off per zone)
             Encoded as a single integer index.
    Reward : negative energy cost + comfort satisfaction bonus
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.num_zones = NUM_ZONES

        # Discretisation buckets
        self.hour_bins = 4          # 0-5, 6-11, 12-17, 18-23
        self.occ_bins = 3           # empty, partial, full
        self.temp_bins = 3          # cold, comfort, hot

        # State/action sizes
        self.n_states = (
            self.hour_bins
            * (self.occ_bins ** self.num_zones)
            * (self.temp_bins ** self.num_zones)
        )
        # Each zone has 2 binary controls: HVAC and Lighting → 4 choices/zone
        self.n_actions = 4 ** self.num_zones

        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        self.hour = 0
        self.day_step = 0

        # Zone indoor temperatures (continuous, °C)
        self.zone_temps = self.rng.uniform(22, 25, size=self.num_zones)

        # Zone occupancy (number of people)
        self.zone_occ = self._sample_occupancy(self.hour)

        # Outdoor temp
        self.outdoor_temp = self._outdoor_temp(self.hour)

        self.total_energy = 0.0
        self.total_comfort_violation = 0.0
        self.episode_rewards = []

        return self._get_state_index()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        Execute action, advance one hour, return (next_state, reward, done, info).
        Action is an integer in [0, n_actions).
        """
        hvac_on, light_on = self._decode_action(action)

        # Thermal dynamics: each zone drifts toward outdoor temp
        # HVAC pulls toward comfort midpoint (23°C)
        TARGET_HVAC = 23.0
        for z in range(self.num_zones):
            drift = 0.3 * (self.outdoor_temp - self.zone_temps[z])
            if hvac_on[z]:
                correction = 0.6 * (TARGET_HVAC - self.zone_temps[z])
            else:
                correction = 0.0
            noise = self.rng.normal(0, 0.2)
            self.zone_temps[z] += drift + correction + noise
            self.zone_temps[z] = float(np.clip(self.zone_temps[z], 15, 40))

        # Energy consumption this step
        energy = 0.0
        for z in range(self.num_zones):
            energy += IDLE_COST
            if hvac_on[z]:
                energy += HVAC_COST_PER_STEP
            if light_on[z]:
                energy += LIGHTING_COST_PER_STEP

        # Comfort and waste penalties
        comfort_penalty = 0.0
        waste_penalty = 0.0
        for z in range(self.num_zones):
            # Temperature comfort
            if self.zone_temps[z] < TEMP_MIN:
                comfort_penalty += COMFORT_PENALTY * (TEMP_MIN - self.zone_temps[z])
            elif self.zone_temps[z] > TEMP_MAX:
                comfort_penalty += COMFORT_PENALTY * (self.zone_temps[z] - TEMP_MAX)

            # Wasted energy on empty zones
            if self.zone_occ[z] == 0:
                if hvac_on[z]:
                    waste_penalty += WASTED_ENERGY_PENALTY * HVAC_COST_PER_STEP
                if light_on[z]:
                    waste_penalty += WASTED_ENERGY_PENALTY * LIGHTING_COST_PER_STEP

        reward = -(energy + comfort_penalty + waste_penalty)

        # Advance time
        self.hour = (self.hour + 1) % HOURS_PER_DAY
        self.day_step += 1
        self.zone_occ = self._sample_occupancy(self.hour)
        self.outdoor_temp = self._outdoor_temp(self.hour)

        self.total_energy += energy
        self.total_comfort_violation += comfort_penalty
        self.episode_rewards.append(reward)

        done = self.day_step >= HOURS_PER_DAY

        info = {
            "hour": self.hour,
            "energy": energy,
            "comfort_penalty": comfort_penalty,
            "waste_penalty": waste_penalty,
            "zone_temps": self.zone_temps.copy(),
            "zone_occ": self.zone_occ.copy(),
            "hvac_on": hvac_on,
            "light_on": light_on,
        }

        return self._get_state_index(), reward, done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _decode_action(self, action: int):
        """Decode integer action → hvac_on[z], light_on[z] arrays."""
        hvac_on = np.zeros(self.num_zones, dtype=bool)
        light_on = np.zeros(self.num_zones, dtype=bool)
        for z in range(self.num_zones):
            zone_action = (action // (4 ** z)) % 4
            hvac_on[z] = bool(zone_action & 1)
            light_on[z] = bool(zone_action & 2)
        return hvac_on, light_on

    def _get_state_index(self) -> int:
        """Map continuous state → discrete integer index."""
        hour_bin = min(int(self.hour // 6), self.hour_bins - 1)

        occ_idx = 0
        for z in range(self.num_zones):
            occ_ratio = self.zone_occ[z] / MAX_OCCUPANCY
            if occ_ratio == 0:
                ob = 0
            elif occ_ratio < 0.5:
                ob = 1
            else:
                ob = 2
            occ_idx += ob * (self.occ_bins ** z)

        temp_idx = 0
        for z in range(self.num_zones):
            t = self.zone_temps[z]
            if t < TEMP_MIN:
                tb = 0
            elif t <= TEMP_MAX:
                tb = 1
            else:
                tb = 2
            temp_idx += tb * (self.temp_bins ** z)

        state = (
            hour_bin * (self.occ_bins ** self.num_zones) * (self.temp_bins ** self.num_zones)
            + occ_idx * (self.temp_bins ** self.num_zones)
            + temp_idx
        )
        return int(state % self.n_states)

    def _sample_occupancy(self, hour: int) -> np.ndarray:
        """Realistic occupancy profile for an office building."""
        occ = np.zeros(self.num_zones, dtype=int)
        if 8 <= hour < 18:
            # Business hours
            for z in range(self.num_zones):
                peak = MAX_OCCUPANCY if z < 2 else MAX_OCCUPANCY // 2
                occ[z] = int(self.rng.integers(peak // 2, peak + 1))
        elif 18 <= hour < 22:
            # Evening — only lobby and one office
            occ[3] = int(self.rng.integers(0, 10))
        # Night: all zeros
        return occ

    def _outdoor_temp(self, hour: int) -> float:
        """Sinusoidal outdoor temperature (peaks at 14:00)."""
        base = OUTDOOR_TEMP_MEAN + 5 * np.sin(np.pi * (hour - 6) / 12)
        noise = self.rng.normal(0, OUTDOOR_TEMP_STD * 0.3)
        return float(base + noise)

    # ------------------------------------------------------------------
    # Human-readable rendering
    # ------------------------------------------------------------------
    def render(self, info: dict):
        print(f"\n  Hour {info['hour']:02d}:00 | Outdoor: {self.outdoor_temp:.1f}°C")
        for z in range(self.num_zones):
            zone_names = ["Office A", "Office B", "Meeting Rm", "Lobby"]
            print(
                f"  {zone_names[z]:12s} | "
                f"Occ: {info['zone_occ'][z]:3d} | "
                f"Temp: {info['zone_temps'][z]:.1f}°C | "
                f"HVAC: {'ON ' if info['hvac_on'][z] else 'OFF'} | "
                f"Light: {'ON ' if info['light_on'][z] else 'OFF'}"
            )
        print(
            f"  Energy: {info['energy']:.2f}  "
            f"Comfort penalty: {info['comfort_penalty']:.2f}  "
            f"Waste penalty: {info['waste_penalty']:.2f}"
        )
