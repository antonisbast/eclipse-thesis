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
# Phases 1–3 train a SINGLE group-centralized agent over 3 buildings (one SLM
# call per step, not two). Phase 4 splits the canonical 6-building district
# across two agents (α=TRAINING_BUILDINGS, β=HELDOUT_BUILDINGS) for the
# partial-observability + no-comms multi-agent experiment.
#
#   TRAINING_BUILDINGS — single-agent training/eval through Phase 3
#   HELDOUT_BUILDINGS  — in-distribution generalization test (unseen buildings,
#                        same dataset) AND Phase 4 agent β's slice
#   BUILDINGS          — full district (Phase 4 deployment, dual-agent rollout)
#   UNSEEN_BUILDINGS   — out-of-distribution generalization test (different
#                        buildings from the same 2022 dataset)
TRAINING_BUILDINGS: list[int] = [0, 1, 2]
HELDOUT_BUILDINGS:  list[int] = [3, 4, 5]
BUILDINGS:          list[int] = [0, 1, 2, 3, 4, 5]
UNSEEN_BUILDINGS:   list[int] = [6, 7, 8, 9, 10, 11]

# ── Episode bounds ────────────────────────────────────────────────────────
SIM_START: int = 0
SIM_END:   int = 8759   # full year (8 760 hourly steps, indices 0–8759)

# ── Observation sets ──────────────────────────────────────────────────────
# NOTE: CityLearn exposes oracle short-horizon forecasts (price/solar +6 h, +12 h)
# in the raw dataset. We deliberately DO NOT use them — they are perfect look-ahead
# values from the simulation tape, not signals an agent could realistically obtain
# in deployment. Including them would let the policy "cheat" against the test
# distribution and inflate KPIs. The agent must reason about future conditions
# from real-time state alone (time of day, calendar, current price/solar trend).
#
# Canonical 9 real-time variables — used by SAC (vector input) and the LLM
# (which actually reads state via snapshot_state(), but the env still needs an
# active_observations list to construct the obs vector). Forecast fields are
# deliberately excluded; see the note above.
OBSERVATIONS: list[str] = [
    "month", "hour", "day_type",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "non_shiftable_load",
    "solar_generation",
    "electricity_pricing",
    "carbon_intensity",
]

# Legacy aliases — older notebook code referenced obs_set="sac" / "llm".
# Both resolve to the same list now.
OBSERVATIONS_SAC: list[str] = OBSERVATIONS
OBSERVATIONS_LLM: list[str] = OBSERVATIONS

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

    Per-building cost/carbon term (still per-building because each building has
    its own SoC and net):
        eco_i  = w_cost · (price / MAX_PRICE) + w_carbon · (carbon / MAX_CARBON)
        base_i = −(1 + sign(net_i·eco_i) · SoC_i) · |net_i·eco_i / MAX_NET_LOAD|

    DISTRICT-LEVEL peak term — the actual `daily_peak_average` KPI is computed
    on the district sum, not per-building. Squaring per-building rewarded a
    building that *exported* (negative net) just as much as one that imported;
    summing first matches the KPI. The district peak is then distributed
    equally across buildings so the per-building reward list still sums to the
    intended district peak.

        net_district = Σ_i (net_i / MAX_NET_LOAD)
        peak_total   = −w_peak · max(net_district, 0.0)²
        peak_i       = peak_total / n_buildings

        reward_i = base_i + peak_i

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
        bases: list[float] = []
        norm_nets: list[float] = []
        for o in observations:
            norm_price  = o.get("electricity_pricing",         0.0) / self.max_price
            norm_carbon = o.get("carbon_intensity",            0.0) / self.max_carbon
            norm_net    = o.get("net_electricity_consumption", 0.0) / self.max_net_load
            soc         = o.get("electrical_storage_soc",      0.0)
            eco    = self.w_cost * norm_price + self.w_carbon * norm_carbon
            cost   = norm_net * eco
            base   = -(1.0 + np.sign(cost) * soc) * abs(cost)
            bases.append(float(base))
            norm_nets.append(float(norm_net))

        # District-level peak (matches the daily_peak_average KPI).
        net_district = sum(norm_nets)
        peak_total   = -(max(net_district, 0.0) ** 2) * self.w_peak
        n            = max(len(bases), 1)
        peak_share   = peak_total / n

        rewards = [b + peak_share for b in bases]
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
        buildings:     Building indices to include. Defaults to BUILDINGS (0–5)
                       for backward compatibility with Phase 1/2 notebooks. For
                       Phase 3 single-agent SLM work, pass
                       `buildings=TRAINING_BUILDINGS` ([0,1,2]) explicitly.
        start:         Simulation start timestep. Default 0 (full year).
        end:           Simulation end timestep. Default 8759 (inclusive — full
                       year of 8760 hourly steps, indices 0..8759).
        reward_fn:     'merlin' (default, dataset-agnostic) or 'eco' (multi-objective).
        obs_set:       'sac' or 'llm' — both are the same 9 real-time variables
                       (oracle forecast fields are intentionally excluded; see
                       the note on OBSERVATIONS_SAC).
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

    NOTE: We deliberately exclude the oracle price/solar forecast fields that
    CityLearn exposes — see the comment on OBSERVATIONS_SAC. The agent must
    plan from real-time state only.
    """
    out = []

    def _at(arr, idx: int):
        """Safe index — clamps to [0, len(arr)-1] so t-1 at t=0 is fine."""
        return arr[max(0, min(idx, len(arr) - 1))]

    for b in env.buildings:
        # env.time_step can sit one past the end after termination
        # (CityLearn 2.6 behaviour). Some arrays are populated only up to
        # t-1 (electrical_storage.soc, net_electricity_consumption — see
        # the SoC obs-vector bug in docs/CITYLEARN_INSIGHTS.md), so we
        # read those with index t-1 and clamp per-array via _at().
        t = env.time_step

        out.append({
            "month":                            int(_at(b.energy_simulation.month, t)),
            "day_type":                         int(_at(b.energy_simulation.day_type, t)),
            "hour":                             int(_at(b.energy_simulation.hour, t)),
            "electricity_pricing":              float(_at(b.pricing.electricity_pricing, t)),
            "carbon_intensity":                 float(_at(b.carbon_intensity.carbon_intensity, t)),
            "solar_generation":                 float(_at(b.energy_simulation.solar_generation, t)),
            "non_shiftable_load":               float(_at(b.non_shiftable_load, t)),
            "electrical_storage_soc":           float(_at(b.electrical_storage.soc, t - 1)),
            "net_electricity_consumption_last": float(_at(b.net_electricity_consumption, t - 1)) if t > 0 else 0.0,
        })
    return out
