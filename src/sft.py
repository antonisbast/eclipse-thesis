"""Helpers for SAC→SLM behavior-cloning distillation.

Pipeline
--------
1. Train SAC on CityLearn (notebook 04, local).
2. Roll the trained SAC out for one full year, dumping per-step
   (state_text, action_tokens) pairs to JSONL via `dump_sac_trajectory_jsonl`.
3. Fine-tune a small LM with LoRA on that JSONL (notebook 05, Colab).
4. Evaluate the fine-tuned SLM in CityLearn using the same prompt format.

Design choices
--------------
* `render_state` is identical to the one in notebook 03 — same text format
  the SLM has already proven it can parse.
* `action_to_token` discretises continuous SAC actions into the same 11-bucket
  vocabulary the prompt uses (CHARGE_20…100, IDLE, DISCHARGE_20…100).
  20 % steps match the prompt; SAC outputs in [-1, 1] are bucketed by
  rounding |a|·100 to the nearest 20, then clamped to {20, 40, 60, 80, 100}.
* The SFT prompt (`make_sft_prompt`) drops the <thought> block from the
  inference prompt — distilling without rationales is simpler and avoids
  having to fabricate one for each SAC action.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# ── Bucket thresholds (kept in sync with notebook 03) ─────────────────────

PRICE_PEAK_THRESHOLD      = 0.30
IRRADIANCE_LOW_THRESHOLD  = 50
IRRADIANCE_HIGH_THRESHOLD = 600


def price_bucket(v: float | None) -> str:
    if v is None: return "?"
    return "PEAK" if v >= PRICE_PEAK_THRESHOLD else "LOW"

def carbon_bucket(v: float | None) -> str:
    if v is None: return "?"
    if v < 0.12: return "LOW"
    if v < 0.25: return "MID"
    return "HIGH"

def solar_bucket(v: float | None) -> str:
    if v is None: return "?"
    if v <= 0.0:  return "NONE"
    if v < 0.5:   return "LOW"
    return "HIGH"

def irradiance_bucket(v: float | None) -> str:
    if v is None: return "?"
    if v < IRRADIANCE_LOW_THRESHOLD:  return "NONE"
    if v < IRRADIANCE_HIGH_THRESHOLD: return "LOW"
    return "HIGH"


# ── State text rendering ──────────────────────────────────────────────────

def render_state(snap: list[dict]) -> str:
    """Format a snapshot (list of building dicts from `snapshot_state`) as
    the LLM-facing prompt body. Mirrors notebook 03's `render_state`.
    """
    if not snap:
        return "(empty snapshot)"
    d0 = snap[0]
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


# ── Action discretisation (SAC float ↔ prompt token) ──────────────────────

ACTION_BUCKETS_PCT = (20, 40, 60, 80, 100)
IDLE_THRESHOLD = 0.10  # |a| < 0.10 → IDLE


def action_to_token(a: float, idle_threshold: float = IDLE_THRESHOLD) -> str:
    """Map a SAC action ∈ [-1, 1] to a discrete prompt token.

    |a| < idle_threshold        → 'IDLE'
    a > 0  → 'CHARGE_{20|40|60|80|100}'    (rounded to nearest 20%)
    a < 0  → 'DISCHARGE_{20|40|60|80|100}'
    """
    a = float(np.clip(a, -1.0, 1.0))
    if abs(a) < idle_threshold:
        return "IDLE"
    direction = "CHARGE" if a > 0 else "DISCHARGE"
    pct = int(round(abs(a) * 100 / 20) * 20)
    pct = max(ACTION_BUCKETS_PCT[0], min(ACTION_BUCKETS_PCT[-1], pct))
    return f"{direction}_{pct}"


_ACTION_RE = re.compile(
    r"<action\s+building\s*=\s*(\d+)\s*>\s*(CHARGE|DISCHARGE|IDLE)_?(\d+)?\s*</action>",
    re.IGNORECASE,
)


def parse_actions(text: str, n_buildings: int) -> list[float]:
    """Inverse of `action_to_token` — extracts per-building actions from
    LLM output. Missing buildings default to 0.0 (no-op)."""
    acts = [0.0] * n_buildings
    for m in _ACTION_RE.finditer(text):
        idx       = int(m.group(1))
        direction = m.group(2).upper()
        amt_str   = m.group(3)
        val = 0.0
        if direction == "CHARGE" and amt_str:
            val = float(amt_str) / 100.0
        elif direction == "DISCHARGE" and amt_str:
            val = -float(amt_str) / 100.0
        if 0 <= idx < n_buildings:
            acts[idx] = float(np.clip(val, -1.0, 1.0))
    return acts


def format_action_block(actions: Iterable[float], n_buildings: int) -> str:
    """Format a list of float actions as the assistant response body."""
    tokens = [action_to_token(a) for a in list(actions)[:n_buildings]]
    while len(tokens) < n_buildings:
        tokens.append("IDLE")
    return "\n".join(
        f"<action building={i}>{tok}</action>"
        for i, tok in enumerate(tokens)
    )


# ── Prompts ───────────────────────────────────────────────────────────────

def make_sft_prompt(n_buildings: int = 6) -> str:
    """SFT/eval prompt. Mirrors nb03's `make_minimal_prompt` but WITHOUT the
    [Reasoning] / <thought> block, since the SAC teacher provides no rationale.

    The SAME text is used at training and at inference — any drift between the
    two creates an OOD evaluation and destroys KPIs (see the CoT eval blowup
    in nb05 § 19).
    """
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
- Avoid aggressive actions, prefer CHARGE_20, CHARGE_40, DISCHARGE_20 or DISCHARGE_40.
- Never charge when SoC is higher than 90% and never discharge when SoC is lower than 10%.

[Output Format]
Output exactly {n_buildings} action lines, one per building, and nothing else:
{action_fmt}
"""


# ── Dataset filtering ─────────────────────────────────────────────────────

_SOC_RE = re.compile(r"SoC=\s*([\d.]+)%")


def filter_uninformative_rows(
    rows: list[dict],
    soc_eps: float = 0.02,
    act_eps: float = 0.05,
) -> list[dict]:
    """Drop rows where EVERY building's action is physically a no-op.

    A per-building (SoC, action) pair is uninformative when:
      • SoC ≤ soc_eps        AND  action < -act_eps   (discharge from empty)
      • SoC ≥ 1 - soc_eps    AND  action > +act_eps   (charge into full)
      • |action| < act_eps                             (near-IDLE)

    These (state, action) pairs carry no learnable signal — the action token
    has no effect on the next state — and dilute the gradient toward the
    marginal "DISCHARGE_20" mode. A row is dropped only when ALL buildings
    are uninformative simultaneously.

    SoC is parsed from the `prompt` text (no schema change needed). If parsing
    fails (mismatched length), the row is kept.
    """
    kept: list[dict] = []
    for row in rows:
        socs_pct = _SOC_RE.findall(row.get("prompt", ""))
        acts     = row.get("actions_float", [])
        if not socs_pct or len(socs_pct) != len(acts):
            kept.append(row)
            continue

        socs = [float(s) / 100.0 for s in socs_pct]
        noop = 0
        for soc, a in zip(socs, acts):
            a = float(a)
            if abs(a) < act_eps:
                noop += 1
            elif soc <= soc_eps and a < 0:
                noop += 1
            elif soc >= 1.0 - soc_eps and a > 0:
                noop += 1
        if noop < len(socs):
            kept.append(row)
    return kept


# ── Trajectory dumper ─────────────────────────────────────────────────────

def dump_sac_trajectory_jsonl(
    env,
    agent,
    out_path: str | Path,
    snapshot_fn,
    n_buildings: int | None = None,
    include_meta: bool = True,
) -> dict[str, Any]:
    """Run SAC deterministically for one full episode and write a JSONL
    SFT dataset.

    Each line is:
        {"prompt": "STATE:\\n...", "response": "<action ...>\\n...",
         "t": int, "actions_float": [..], "reward": [..]}

    Args:
        env:          A fresh CityLearnEnv (will be reset).
        agent:        Trained SAC agent with .predict(obs, deterministic=True).
        out_path:     Destination .jsonl path.
        snapshot_fn:  Callable env -> list[dict] (typically `snapshot_state`).
        n_buildings:  Override; defaults to len(env.buildings).
        include_meta: Include t, actions_float, reward fields per row.

    Returns:
        Stats dict: {"n_steps", "path", "fallbacks", "n_buildings"}.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_b = n_buildings if n_buildings is not None else len(env.buildings)
    obs, _ = env.reset()
    done, t = False, 0
    n_written = 0

    with open(out_path, "w") as f:
        while not done:
            snap        = snapshot_fn(env)
            state_text  = render_state(snap)
            actions     = agent.predict(obs, deterministic=True)  # list-of-list
            # Flatten: SAC returns one [a] per building (active_actions=1)
            acts_flat   = [float(a[0]) if hasattr(a, "__len__") else float(a)
                           for a in actions]
            response    = format_action_block(acts_flat, n_b)
            row         = {
                "prompt":   f"STATE:\n{state_text}",
                "response": response,
            }
            if include_meta:
                row["t"]              = t
                row["actions_float"]  = acts_flat
            obs, reward, terminated, truncated, _ = env.step(actions)
            if include_meta:
                row["reward"] = [float(r) for r in reward]
            f.write(json.dumps(row) + "\n")
            n_written += 1
            done = bool(terminated or truncated)
            t   += 1

    return {
        "n_steps":     n_written,
        "path":        str(out_path),
        "n_buildings": n_b,
    }
