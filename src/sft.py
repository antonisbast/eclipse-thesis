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

# ── Single source of truth lives in src.agent ─────────────────────────────
# State rendering, buckets, thresholds, and the inference-time action regex
# are defined ONCE in src/agent.py. We re-export them here so nb 04/05
# can do `from src.sft import render_state, parse_actions, ...` without
# pulling src.agent explicitly. _ACTION_RE alias kept for older callers.
from src.agent import (
    PRICE_PEAK_THRESHOLD,
    IRRADIANCE_LOW_THRESHOLD,
    IRRADIANCE_HIGH_THRESHOLD,
    price_bucket,
    carbon_bucket,
    solar_bucket,
    irradiance_bucket,
    render_state,
    parse_actions,
    ACTION_RE,
)

_ACTION_RE = ACTION_RE   # legacy alias — keep for any external import


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

def make_sft_prompt(n_buildings: int = 3) -> str:
    """SFT-only prompt — the [Reasoning] / <thought> block of the canonical
    CoT prompt (`src.agent.make_minimal_prompt`) is intentionally STRIPPED
    here, because the SAC teacher trajectories in the distillation JSONL
    don't include rationales.

    For everything else (zero-shot LLM-as-policy in nb 02/03, eval of the
    fine-tuned SLM, Phase 4 deployment) use `src.agent.make_minimal_prompt`
    which keeps CoT — that is the canonical prompt.

    CRITICAL: at eval time of the fine-tuned SLM, use the SAME prompt that
    was used at SFT — i.e. THIS function — to avoid an OOD eval (see the
    CoT eval blowup in nb 05 § 19). If you want a CoT-capable fine-tuned
    SLM you must re-distill with synthesised <thought> blocks.
    """
    action_fmt = "\n".join(
        f"<action building={i}>YOUR_CHOICE</action>" for i in range(n_buildings)
    )
    return f"""\
You manage batteries in {n_buildings} buildings that share one grid meter. Each step, pick one action per building.

[Actions]
CHARGE_100, CHARGE_80, CHARGE_60, CHARGE_40, CHARGE_20, IDLE, DISCHARGE_20, DISCHARGE_40, DISCHARGE_60, DISCHARGE_80, DISCHARGE_100

[State]
- 'price' (LOW / MID / PEAK): how expensive grid electricity is now.
- 'carbon' (LOW / MID / HIGH): how dirty grid electricity is now.
- 'solar' (NONE / LOW / MID / HIGH): the building's solar generation now.
- 'load' (kWh): the building's electricity demand now.
- 'SoC' (%): how full the battery is. 0% empty, 100% full.
- 'last_net' (kWh): grid draw last step — your feedback signal.
- Time: month, weekday, hour. No forecasts.

[Physics]
A building's grid draw is its load, minus its solar, plus any charging, minus any discharging. A negative result means the building exports to the grid for almost no reward. The {n_buildings} buildings share one meter, so the district's draw is the sum across them. Battery charge stays between 0% and 100%.

[Hints]
- To keep cost down: discharge when grid electricity is expensive; charge when it is cheap or when solar can cover it.
- To keep carbon low: avoid buying from the grid when it is dirty (HIGH carbon) — IDLE is better than charging in those moments.
- To keep ramping low: prefer small actions (CHARGE_20/40, DISCHARGE_20/40), and avoid switching the same battery from charging to discharging on the very next step.
- To keep peak low: discharge to help serve the load when district demand is high; do not charge from the grid then.

[Output]
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
    building_slices: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Run SAC deterministically for one full episode and write a JSONL
    SFT dataset.

    Each JSONL line is:
        {"prompt": "STATE:\\n...", "response": "<action ...>\\n...",
         "t": int, "actions_float": [..], "reward": [..], "slice": [...]}

    Args:
        env:             A fresh CityLearnEnv (will be reset). The SAC teacher
                         is rolled out on ALL its buildings (full 6-building
                         env for the canonical thesis setup) — see
                         `building_slices` for emitting only a subset per row.
        agent:           Trained SAC agent with .predict(obs, deterministic=True).
        out_path:        Destination .jsonl path.
        snapshot_fn:     Callable env -> list[dict] (typically `snapshot_state`).
        n_buildings:     Number of buildings PER OUTPUT ROW. Defaults to the
                         slice width if `building_slices` is given, else
                         `len(env.buildings)`.
        include_meta:    Include t, actions_float, reward, slice fields per row.
        building_slices: Optional list of index-lists. For each env step, one
                         row is emitted per slice — the state_text and action
                         block are restricted to that subset of buildings.
                         The SAC teacher still acts on the full env; only the
                         OUTPUT is sliced. Example for the Phase-3 single-agent
                         setup on a 6-building SAC:
                             building_slices=[[0,1,2], [3,4,5]]
                         doubles the dataset and makes the SLM building-agnostic
                         within the 3-building shape, so the same LoRA can drop
                         into Phase 4 agent α (B0–2) or β (B3–5).
                         If None, emits one row per step over all env buildings
                         (legacy behaviour).

    Returns:
        Stats dict: {"n_steps", "n_rows", "path", "n_buildings"}.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_env = len(env.buildings)
    if building_slices is None:
        slices = [list(range(n_env))]
    else:
        slices = [list(s) for s in building_slices]
        widths = {len(s) for s in slices}
        if len(widths) != 1:
            raise ValueError(
                f"All building_slices must have the same width; got widths={widths}"
            )

    slice_width = len(slices[0])
    n_b = n_buildings if n_buildings is not None else slice_width
    if n_b != slice_width:
        raise ValueError(
            f"n_buildings={n_b} must equal slice width ({slice_width}); "
            f"pass building_slices with the desired per-row width instead."
        )

    obs, _ = env.reset()
    done, t = False, 0
    n_steps = 0
    n_rows  = 0

    with open(out_path, "w") as f:
        while not done:
            snap        = snapshot_fn(env)
            actions     = agent.predict(obs, deterministic=True)  # list-of-list
            # Flatten: SAC returns one [a] per building (active_actions=1)
            acts_flat   = [float(a[0]) if hasattr(a, "__len__") else float(a)
                           for a in actions]
            obs, reward, terminated, truncated, _ = env.step(actions)
            reward_flat = [float(r) for r in reward]

            for sl in slices:
                snap_sl  = [snap[i] for i in sl]
                acts_sl  = [acts_flat[i] for i in sl]
                state_text = render_state(snap_sl)
                response   = format_action_block(acts_sl, n_b)
                row = {
                    "prompt":   f"STATE:\n{state_text}",
                    "response": response,
                }
                if include_meta:
                    row["t"]             = t
                    row["slice"]         = sl
                    row["actions_float"] = acts_sl
                    row["reward"]        = [reward_flat[i] for i in sl]
                f.write(json.dumps(row) + "\n")
                n_rows += 1

            n_steps += 1
            done = bool(terminated or truncated)
            t   += 1

    return {
        "n_steps":     n_steps,
        "n_rows":      n_rows,
        "path":        str(out_path),
        "n_buildings": n_b,
        "n_slices":    len(slices),
    }
