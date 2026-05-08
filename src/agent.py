"""LLM-as-policy primitives shared by notebooks 02 (remote APIs) and 03 (local SLM).

Contents:
- Threshold constants and bucket functions (price / carbon / solar / irradiance).
- `render_state` — snapshot → human-readable prompt string.
- `parse_actions` — extract `<action building=i>...</action>` tags into floats.
- `make_minimal_prompt` — system prompt with discrete CHARGE/IDLE/DISCHARGE bins.
- `make_policy_llm` — bind any `.step()`-providing provider into a rollout policy.
- Reference policies: `policy_noop`, `policy_random`, `policy_rbc`.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np

from src.env import SEED


# ── Thresholds ────────────────────────────────────────────────────────────────
PRICE_PEAK_THRESHOLD: float      = 0.30   # $/kWh — above this = PEAK price
IRRADIANCE_LOW_THRESHOLD: float  = 50     # W/m²  — below this = NONE
IRRADIANCE_HIGH_THRESHOLD: float = 600    # W/m²  — above this = HIGH


def price_bucket(v: float | None) -> str:
    if v is None:
        return "?"
    return "PEAK" if v >= PRICE_PEAK_THRESHOLD else "LOW"


def carbon_bucket(v: float | None) -> str:
    if v is None:
        return "?"
    if v < 0.12:
        return "LOW"
    if v < 0.25:
        return "MID"
    return "HIGH"


def solar_bucket(v: float | None) -> str:
    if v is None:
        return "?"
    if v <= 0.0:
        return "NONE"
    if v < 0.5:
        return "LOW"
    return "HIGH"


def irradiance_bucket(v: float | None) -> str:
    if v is None:
        return "?"
    if v < IRRADIANCE_LOW_THRESHOLD:
        return "NONE"
    if v < IRRADIANCE_HIGH_THRESHOLD:
        return "LOW"
    return "HIGH"


# ── State renderer ────────────────────────────────────────────────────────────
def render_state(snap: list[dict]) -> str:
    """Convert a snapshot (list of building dicts) into an LLM prompt string.

    Buildings are renumbered locally from B0 — both agents see identical structure
    regardless of which slice of the district they observe.
    """
    if not snap:
        return "(empty snapshot)"
    d0   = snap[0]
    hour = int(d0.get("hour", 1)) - 1
    day  = int(d0.get("day_type", 1)) - 1
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    prc = d0.get("electricity_pricing", None)
    crb = d0.get("carbon_intensity", None)

    header = (
        f"Month {d0.get('month', '?')}, {day_names[day]} {hour:02d}:00  |  "
        f"price={prc:.3f} ({price_bucket(prc)})  |  "
        f"carbon={crb:.3f} ({carbon_bucket(crb)})"
    )

    fp1 = d0.get("electricity_pricing_predicted_1", None)
    fp2 = d0.get("electricity_pricing_predicted_2", None)
    fi1 = d0.get("solar_irradiance_predicted_1", None)
    forecast = (
        f"Forecast:  price+6h={price_bucket(fp1)}  "
        f"price+12h={price_bucket(fp2)}  "
        f"solar+6h={irradiance_bucket(fi1)}"
    )

    lines = [header, forecast, "Buildings:"]
    for i, d in enumerate(snap):
        soc  = d.get("electrical_storage_soc", 0.0)
        sol  = d.get("solar_generation", 0.0)
        load = d.get("non_shiftable_load", 0.0)
        net  = d.get("net_electricity_consumption_last", 0.0)
        lines.append(
            f"  B{i}: SoC={soc*100:5.1f}%  "
            f"load={load:.2f} kWh  "
            f"last_net={net:+.2f} kWh  "
            f"solar={solar_bucket(sol)}"
        )
    return "\n".join(lines)


# ── Action parser — discrete CHARGE/DISCHARGE/IDLE bins ───────────────────────
ACTION_RE = re.compile(
    r"<action\s+building\s*=\s*(\d+)\s*>\s*(CHARGE|DISCHARGE|IDLE)_?(\d+)?\s*</action>",
    re.IGNORECASE,
)


def parse_actions(text: str, n_buildings: int) -> list[float]:
    """Extract per-building discrete actions and map to [-1.0, 1.0] floats.

    CHARGE_<n>    →  +n/100
    DISCHARGE_<n> →  -n/100
    IDLE          →   0.0
    Missing buildings default to 0.0.
    """
    acts = [0.0] * n_buildings
    for m in ACTION_RE.finditer(text):
        idx       = int(m.group(1))
        direction = m.group(2).upper()
        amt_str   = m.group(3)

        val = 0.0
        if direction == "CHARGE" and amt_str:
            val = float(amt_str) / 100.0
        elif direction == "DISCHARGE" and amt_str:
            val = -float(amt_str) / 100.0
        elif direction == "IDLE":
            val = 0.0

        if 0 <= idx < n_buildings:
            acts[idx] = float(np.clip(val, -1.0, 1.0))
    return acts


# ── Prompt ────────────────────────────────────────────────────────────────────
def make_minimal_prompt(n_buildings: int = 6) -> str:
    """Prompt with variable semantics, indirect instructions, and brief CoT."""
    action_fmt = "\n".join(
        f"<action building={i}>YOUR_CHOICE</action>" for i in range(n_buildings)
    )
    return f"""\
You are an energy management agent for {n_buildings} buildings. Goal: minimize grid dependency and energy costs over time.

[Actions] — choose exactly one per building:
CHARGE_100, CHARGE_80, CHARGE_60, CHARGE_40, CHARGE_20, IDLE, DISCHARGE_20, DISCHARGE_40, DISCHARGE_60, DISCHARGE_80, DISCHARGE_100

[State Variables & Environment]
- 'price': Current cost of grid electricity. PEAK indicates high cost.
- 'solar': Renewable energy generated locally.
- 'load': Energy demanded by the building's operations. High load means the building needs a lot of power.
- 'SoC': Battery State of Charge (0% = empty, 100% = full).
- Charging stores energy. Doing so when solar is HIGH or price is LOW is efficient, but charging from the grid increases district demand.
- Discharging uses stored energy to serve the 'load', directly reducing grid dependency. This is highly beneficial when 'price' is PEAK or 'load' is high and SoC is sufficient.
- Forecast fields show anticipated conditions 6 or 12 hours ahead, helping you plan when to store or release energy.
-Avoid aggresive actions, prefer CHARGE_20,CHARGE_40,DISCHARGE_20 OR DISCHARGE_40.
-Never charge when SOC is higher than 90% and never discharge when SOC is lower than 10%.

[Reasoning]
Before choosing actions, briefly analyze the state in a <thought> block.
CRITICAL: Keep your thought extremely brief (UNDER 15 WORDS) to save computation time.

[Output Format]
<thought>
Ultra-short analysis here...
</thought>
{action_fmt}
"""


# ── Policy adapter ────────────────────────────────────────────────────────────
def make_policy_llm(
    provider,
    n_buildings: int = 6,
    agent_label: str = "",
    system: str | None = None,
    verbose: bool = True,
    **step_kwargs,
) -> Callable:
    """Bind a provider into a rollout-compatible policy function.

    Provider must expose `.step(state_text, system, n_buildings, **kwargs)`
    returning `(actions, raw_response, fallback_flag)`. Works with both
    `APIProvider` (remote) and `LocalHFProvider` (local SLM).

    Verbose print format (one line per call):
      t=  5 [α] B0:42%→+0.40  B1:61%→+0.00  B2:28%→-0.80  | '<action ...'
    """
    _system = system or make_minimal_prompt(n_buildings)
    _label  = f"[{agent_label}] " if agent_label else ""

    def policy(snap: list[dict], t: int):
        state_text = render_state(snap)
        acts, raw, fallback = provider.step(
            state_text,
            system=_system,
            n_buildings=n_buildings,
            **step_kwargs,
        )
        if verbose:
            fb_tag   = " [FALLBACK]" if fallback else ""
            bldg_str = "  ".join(
                f"B{i}:{snap[i]['electrical_storage_soc']*100:.0f}%→{acts[i]:+.2f}"
                for i in range(len(acts))
            )
            print(
                f"  t={t:3d} {_label}{bldg_str}"
                f"  |  {raw.replace(chr(10), ' ')[:55].strip()!r}{fb_tag}"
            )
        return acts, raw, fallback

    return policy


# ── Reference policies ────────────────────────────────────────────────────────
def policy_noop(snap: list[dict], t: int) -> list[float]:
    return [0.0] * len(snap)


_rng = np.random.default_rng(SEED)


def policy_random(snap: list[dict], t: int) -> list[float]:
    return _rng.uniform(-1.0, 1.0, size=len(snap)).tolist()


def policy_rbc(snap: list[dict], t: int) -> list[float]:
    """Price + solar aware rule-based controller.

    Exploits battery asymmetry: charge small (avoids demand spikes),
    discharge full (-1.0 is hardware-capped and safe).
    """
    acts = []
    for d in snap:
        soc = d["electrical_storage_soc"]
        prc = d["electricity_pricing"]
        sol = d["solar_generation"]
        if solar_bucket(sol) == "HIGH" and soc < 0.85:
            acts.append(0.2)
        elif price_bucket(prc) == "PEAK" and soc > 0.10:
            acts.append(-1.0)
        elif price_bucket(prc) == "LOW" and soc < 0.90:
            acts.append(0.25)
        else:
            acts.append(0.0)
    return acts
