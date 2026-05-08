"""CityLearn v2 environment factory, reward functions, and state utilities.

All notebooks and scripts import from here — no env logic lives in cells.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT  = _PROJECT_ROOT / "data" / "citylearn_datasets" / "citylearn_challenge_2022_phase_all"
SCHEMA_FILE   = DATASET_ROOT / "schema.json"

# ── Reproducibility ───────────────────────────────────────────────────────
SEED: int = 42

# ── Buildings ─────────────────────────────────────────────────────────────
BUILDINGS:        list[int] = [0, 1, 2, 3, 4, 5]
UNSEEN_BUILDINGS: list[int] = [6, 7, 8, 9, 10, 11]

# ── Episode bounds ────────────────────────────────────────────────────────
SIM_START: int = 0
SIM_END:   int = 8759   # full year (8 760 hourly steps, indices 0–8759)

# ── Observation sets ──────────────────────────────────────────────────────
# OBSERVATIONS_SAC: 13 variables — 9 real-time + 4 short-horizon forecasts.
#   Used for SAC training. Forecasts follow Nweye et al. (2024, MERLIN).
OBSERVATIONS_SAC: list[str] = [
    "month", "hour", "day_type",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "non_shiftable_load",
    "solar_generation",
    "electricity_pricing",
    "carbon_intensity",
    "electricity_pricing_predicted_1",       # price  +6 h
    "electricity_pricing_predicted_2",       # price  +12 h
    "diffuse_solar_irradiance_predicted_1",  # solar  +6 h (diffuse)
    "direct_solar_irradiance_predicted_1",   # solar  +6 h (direct)
]

# OBSERVATIONS_LLM: 9 real-time variables — no forecasts.
#   Used for LLM-as-policy. The LLM receives state via snapshot_state(),
#   not the obs vector, so forecast columns in the vector are unused noise.
OBSERVATIONS_LLM: list[str] = [
    "month", "hour", "day_type",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "non_shiftable_load",
    "solar_generation",
    "electricity_pricing",
    "carbon_intensity",
]

ACTIVE_ACTIONS: list[str] = ["electrical_storage"]

# ── Reward weights and normalisation constants ────────────────────────────
# Measured from the full 2022 dataset, all 17 buildings, full year.
W_COST:       float = 0.4
W_CARBON:     float = 0.4
W_PEAK:       float = 0.2
MAX_PRICE:    float = 0.54   # EUR/kWh  — exact max of 5 discrete tariff levels
MAX_CARBON:   float = 0.30   # kgCO₂/kWh — observed max 0.282, +6 % headroom
MAX_NET_LOAD: float = 10.0   # kWh/step  — per-building; observed max 8.51, +18 % headroom


# ── Reward functions ──────────────────────────────────────────────────────

class MERLINReward(RewardFunction):
    """SoC-aware net-consumption reward (Nweye et al., 2024).

    reward = −(1 + sign(net) · SoC) · |net_consumption|

    Uses raw kWh values — no dataset-specific normalisation required.
    Grid-searched optimal parameters (Table 3): w1=1, w2=0, e1=1, e2=1.

    Reference: Nweye et al. (2024). Applied Energy, 358, 121958.
    https://doi.org/10.1016/j.apenergy.2023.121958
    """

    def __init__(self, env_metadata: dict, w1: float = 1.0, w2: float = 0.0,
                 e1: float = 1.0, e2: float = 1.0):
        super().__init__(env_metadata)
        self.w1, self.w2, self.e1, self.e2 = w1, w2, e1, e2

    def calculate(self, observations: list[dict]) -> list[float]:
        rewards = []
        for o in observations:
            net    = o.get("net_electricity_consumption", 0.0)
            carbon = o.get("carbon_intensity", 0.0) * abs(net)
            soc    = o.get("electrical_storage_soc", 0.0)
            signal = self.w1 * (abs(net) ** self.e1) + self.w2 * (abs(carbon) ** self.e2)
            rewards.append(float(-(1.0 + np.sign(net) * soc) * signal))
        return [float(sum(rewards))] if self.central_agent else rewards


class EcoPeakBatteryReward(RewardFunction):
    """Multi-objective reward: cost + carbon + peak shaving, all normalised.

    eco    = w_cost · (price / MAX_PRICE) + w_carbon · (carbon / MAX_CARBON)
    base   = −(1 + sign(net·eco) · SoC) · |net·eco / MAX_NET_LOAD|
    peak   = −w_peak · (net / MAX_NET_LOAD)²
    reward = base + peak

    Activate via: make_env(reward_fn="eco")
    """

    def __init__(self, env_metadata: dict,
                 w_cost: float = W_COST, w_carbon: float = W_CARBON, w_peak: float = W_PEAK,
                 max_price: float = MAX_PRICE, max_carbon: float = MAX_CARBON,
                 max_net_load: float = MAX_NET_LOAD):
        super().__init__(env_metadata)
        self.w_cost, self.w_carbon, self.w_peak = w_cost, w_carbon, w_peak
        self.max_price, self.max_carbon, self.max_net_load = max_price, max_carbon, max_net_load

    def calculate(self, observations: list[dict]) -> list[float]:
        rewards = []
        for o in observations:
            norm_price  = o.get("electricity_pricing",         0.0) / self.max_price
            norm_carbon = o.get("carbon_intensity",            0.0) / self.max_carbon
            norm_net    = o.get("net_electricity_consumption", 0.0) / self.max_net_load
            soc         = o.get("electrical_storage_soc",      0.0)
            eco    = self.w_cost * norm_price + self.w_carbon * norm_carbon
            cost   = norm_net * eco
            base   = -(1.0 + np.sign(cost) * soc) * abs(cost)
            peak   = -(max(norm_net, 0.0) ** 2) * self.w_peak
            rewards.append(float(base + peak))
        return [float(sum(rewards))] if self.central_agent else rewards


# ── Schema loader ─────────────────────────────────────────────────────────

def load_schema() -> dict:
    """Load schema.json and patch root_directory to an absolute path."""
    with open(SCHEMA_FILE) as f:
        schema = json.load(f)
    schema["root_directory"] = str(DATASET_ROOT.resolve())
    return schema


# ── Environment factory ───────────────────────────────────────────────────

def make_env(
    buildings: list[int] | None = None,
    start: int = SIM_START,
    end: int = SIM_END,
    reward_fn: str = "merlin",
    obs_set: str = "sac",
    session_name: str | None = None,
    render_mode: str = "end",
    artifacts_dir: Path | None = None,
) -> CityLearnEnv:
    """Build a CityLearnEnv with the canonical thesis configuration.

    Args:
        buildings:     Building indices to include. Defaults to BUILDINGS (0–5).
        start:         Simulation start timestep. Default 0 (full year).
        end:           Simulation end timestep. Default 8758 (full year).
        reward_fn:     'merlin' (default, dataset-agnostic) or 'eco' (multi-objective).
        obs_set:       'sac' → 13 variables (with forecasts); 'llm' → 9 real-time only.
        session_name:  Enables render output when provided.
        render_mode:   'end' (buffer per episode) or 'during' (live writes).
        artifacts_dir: Base directory for render output. Defaults to notebooks/artifacts/.
    """
    observations = OBSERVATIONS_SAC if obs_set == "sac" else OBSERVATIONS_LLM

    kwargs: dict[str, Any] = dict(
        schema=load_schema(),
        buildings=buildings if buildings is not None else BUILDINGS,
        central_agent=False,
        active_actions=ACTIVE_ACTIONS,
        active_observations=observations,
        random_seed=SEED,
        simulation_start_time_step=start,
        simulation_end_time_step=end,
    )

    if session_name:
        render_dir = (
            artifacts_dir or (_PROJECT_ROOT / "notebooks" / "artifacts")
        ) / "SimulationData"
        render_dir.mkdir(parents=True, exist_ok=True)
        kwargs.update(
            render_mode=render_mode,
            render_directory=str(render_dir),
            render_session_name=session_name,
        )

    env = CityLearnEnv(**kwargs)
    env.reward_function = (
        EcoPeakBatteryReward(env.get_metadata())
        if reward_fn == "eco"
        else MERLINReward(env.get_metadata())
    )
    return env


# ── State snapshot ────────────────────────────────────────────────────────

def snapshot_state(env: CityLearnEnv) -> list[dict]:
    """Read current district state directly from building objects.

    Bypasses the CityLearn v2.5 obs-vector bug where electrical_storage_soc
    and net_electricity_consumption report stale (next-step initialisation)
    values. Always safe to call right after env.reset() or env.step().

    Forecast fields included
    ────────────────────────
    electricity_pricing_predicted_1  : expected price +6 h ($/kWh) — same scale as current price
    electricity_pricing_predicted_2  : expected price +12 h ($/kWh)
    solar_irradiance_predicted_1     : diffuse + direct irradiance +6 h (W/m²)

    All three are read from building sub-objects directly and fall back to
    None if the attributes are unavailable (e.g. when using a custom dataset
    that does not include forecast columns).
    """
    out = []

    def _at(arr, idx: int):
        """Safe index — clamps to the last valid entry of `arr`."""
        return arr[min(idx, len(arr) - 1)]

    for b in env.buildings:
        # Clamp to last valid index — env.time_step can sit one past the end
        # after termination (CityLearn 2.6 behaviour). Some arrays are
        # shorter than energy_simulation.* (e.g. non_shiftable_load is
        # populated up to t-1), so we clamp per-array via _at().
        t = env.time_step
        # ── Forecast helpers ──────────────────────────────────────────────
        def _price_fc(attr: str) -> float | None:
            try:
                return float(_at(getattr(b.pricing, attr), t))
            except Exception:
                return None

        def _irr_sum_fc(attr_diffuse: str, attr_direct: str) -> float | None:
            try:
                diffuse = float(_at(getattr(b.weather, attr_diffuse), t))
                direct  = float(_at(getattr(b.weather, attr_direct), t))
                return diffuse + direct
            except Exception:
                return None

        out.append({
            # ── Real-time signals ─────────────────────────────────────────
            "month":                            int(_at(b.energy_simulation.month, t)),
            "day_type":                         int(_at(b.energy_simulation.day_type, t)),
            "hour":                             int(_at(b.energy_simulation.hour, t)),
            "electricity_pricing":              float(_at(b.pricing.electricity_pricing, t)),
            "carbon_intensity":                 float(_at(b.carbon_intensity.carbon_intensity, t)),
            "solar_generation":                 float(_at(b.energy_simulation.solar_generation, t)),
            "non_shiftable_load":               float(_at(b.non_shiftable_load, t)),
            "electrical_storage_soc":           float(_at(b.electrical_storage.soc, t - 1)) if t > 0 else float(b.electrical_storage.soc[0]),
            "net_electricity_consumption_last": float(_at(b.net_electricity_consumption, t - 1)) if t > 0 else 0.0,
            # ── Short-horizon forecasts ───────────────────────────────────
            "electricity_pricing_predicted_1":  _price_fc("electricity_pricing_predicted_1"),
            "electricity_pricing_predicted_2":  _price_fc("electricity_pricing_predicted_2"),
            "solar_irradiance_predicted_1":     _irr_sum_fc(
                "_diffuse_solar_irradiance_predicted_1",
                "_direct_solar_irradiance_predicted_1",
            ),
        })
    return out
