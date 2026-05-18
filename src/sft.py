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
  When the building SoC is supplied, physical no-ops (discharge from an empty
  battery, charge into a full one) are relabelled IDLE — they are clipped to
  zero by CityLearn, so cloning them as DISCHARGE_*/CHARGE_* teaches a token
  that is harmful elsewhere.
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
# pulling src.agent explicitly.
from src.agent import (
    PRICE_PEAK_THRESHOLD,
    price_bucket,
    carbon_bucket,
    solar_bucket,
    render_state,
    parse_actions,
    ACTION_RE,
)


# ── Action discretisation (SAC float ↔ prompt token) ──────────────────────

ACTION_BUCKETS_PCT = (20, 40, 60, 80, 100)
IDLE_THRESHOLD = 0.10  # |a| < 0.10 → IDLE

# Physical no-op thresholds (SoC as a fraction in [0, 1], matching
# snapshot_state's `electrical_storage_soc`). Discharging at/below EMPTY_SOC,
# or charging at/above FULL_SOC, is clipped to zero by CityLearn — the
# realised effect is identical to IDLE.
EMPTY_SOC = 0.03
FULL_SOC  = 0.97


def action_to_token(
    a: float,
    soc: float | None = None,
    idle_threshold: float = IDLE_THRESHOLD,
    empty_soc: float = EMPTY_SOC,
    full_soc: float = FULL_SOC,
) -> str:
    """Map a SAC action ∈ [-1, 1] to a discrete prompt token.

    |a| < idle_threshold        → 'IDLE'
    a > 0  → 'CHARGE_{20|40|60|80|100}'    (round-half-up to nearest 20%)
    a < 0  → 'DISCHARGE_{20|40|60|80|100}'

    Bucket boundaries land at 0.30, 0.50, 0.70, 0.90 (uniform 0.20-wide
    buckets, except IDLE at [0, 0.10) and *_100 at [0.90, 1.00]).

    NOTE: do NOT use ``round(...)`` here — Python 3 uses banker's rounding,
    which makes 0.50 → CHARGE_40 (instead of 60) and 0.70 → CHARGE_80
    (instead of 60-or-80), squeezing the 60 and 100 buckets. Integer
    round-half-up is symmetric and matches the prompt's stated 20%-step
    layout.

    Physical no-op relabel: if ``soc`` is given, a discharge at an empty
    battery (soc ≤ empty_soc) or a charge into a full one (soc ≥ full_soc)
    is clipped to zero by CityLearn, so its realised effect IS IDLE. We emit
    'IDLE' for those pairs — cloning them verbatim as DISCHARGE_*/CHARGE_*
    teaches the student a token that is harmful in states where the battery
    is NOT empty/full (the dominant failure mode of the SAC-distilled SFT:
    ~77 % of the teacher's discharge labels were discharges from an empty
    battery). Pass ``soc=None`` to disable.
    """
    a = float(np.clip(a, -1.0, 1.0))
    if abs(a) < idle_threshold:
        return "IDLE"
    direction = "CHARGE" if a > 0 else "DISCHARGE"
    if soc is not None:
        if direction == "DISCHARGE" and soc <= empty_soc:
            return "IDLE"
        if direction == "CHARGE" and soc >= full_soc:
            return "IDLE"
    units = int(round(abs(a) * 100))            # 0..100
    pct   = ((units + 10) // 20) * 20           # round half up to nearest 20
    pct = max(ACTION_BUCKETS_PCT[0], min(ACTION_BUCKETS_PCT[-1], pct))
    return f"{direction}_{pct}"


def format_action_block(
    actions: Iterable[float],
    n_buildings: int,
    socs: Iterable[float] | None = None,
) -> str:
    """Format a list of float actions as the assistant response body.

    If ``socs`` is given (per-building SoC fractions, same order as
    ``actions``), physical no-op actions are relabelled to IDLE — see
    `action_to_token`.
    """
    actions = list(actions)[:n_buildings]
    if socs is None:
        tokens = [action_to_token(a) for a in actions]
    else:
        socs = list(socs)[:n_buildings]
        tokens = [action_to_token(a, soc=s) for a, s in zip(actions, socs)]
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
- 'price' (LOW / PEAK): how expensive grid electricity is now.
- 'carbon' (LOW / MID / HIGH): how dirty grid electricity is now.
- 'solar' (NONE / LOW / MID / HIGH): the building's solar generation now.
- 'load' (kWh): the building's electricity demand now.
- 'SoC' (%): how full the battery is. 0% empty, 100% full.
- 'last_net' (kWh): grid draw last step — your feedback signal.
- Time: month, weekday, hour. No forecasts.

[Physics]
A building meets its load from its own solar first and draws the rest from the grid. Charging a battery adds to that grid draw, while discharging it covers part of the load and lowers the draw. If solar and discharging together produce more than the load needs, the surplus is exported to the grid for almost no reward. The {n_buildings} buildings share one meter, so the district's draw is the sum across them. Battery charge stays between 0% and 100%.

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


# ── Per-cell supervision masking (physical no-ops) ────────────────────────

_PROMPT_SOC_RE = re.compile(r"SoC=\s*([\d.]+)%")


def supervision_mask(row: dict) -> list[bool]:
    """Per-building supervision flags for one distillation row.

    Returns one bool per building: ``True`` = supervise this building's
    action token in the SFT loss, ``False`` = mask it (loss label -100).

    A cell is masked when the SAC teacher's continuous action is a *physical
    no-op* — a discharge from a (near-)empty battery or a charge into a
    (near-)full one. CityLearn clips those to zero, so the realised action is
    nothing, and the row's ``response`` already shows IDLE there (the
    `action_to_token` relabel). That IDLE is an artifact of clipping, NOT a
    teacher decision: supervising it teaches "battery near empty → IDLE" and
    the student then never charges out of a cold start (the dominant failure
    mode of the first SAC-distilled SFT run). Masking removes the artifact
    label without fabricating anything — the student is simply given no
    lesson where SAC had no opinion.

    A masked cell is determined exactly the way `action_to_token`'s relabel
    is: the SoC-aware token differs from the raw token. SoC is read from the
    row ``prompt`` text (percent). If it cannot be parsed, or its length
    mismatches ``actions_float``, every building is supervised (fail-safe —
    never silently drop signal).
    """
    socs_pct = _PROMPT_SOC_RE.findall(row.get("prompt", ""))
    acts     = row.get("actions_float", [])
    if not acts or len(socs_pct) != len(acts):
        return [True] * len(acts)
    mask: list[bool] = []
    for s_pct, a in zip(socs_pct, acts):
        soc = float(s_pct) / 100.0
        a   = float(a)
        is_noop = action_to_token(a) != action_to_token(a, soc=soc)
        mask.append(not is_noop)
    return mask


def attach_supervision_masks(rows: list[dict]) -> list[dict]:
    """Attach a per-building ``supervise`` flag list to every row and drop
    rows where NO building is supervised.

    Run AFTER `filter_uninformative_rows`. Each returned row gains a
    ``supervise`` key (``list[bool]``, see `supervision_mask`). The handful
    of rows where every building's action is a physical no-op are dropped —
    all their loss labels would be -100, so they contribute zero gradient
    and only waste a forward pass.

    Returns new row dicts; the input rows are not mutated.
    """
    kept: list[dict] = []
    for row in rows:
        m = supervision_mask(row)
        if any(m):
            kept.append({**row, "supervise": m})
    return kept


# ── Class rebalancing (imbalanced distillation dataset) ───────────────────

_RESPONSE_TOKEN_RE = re.compile(
    r"<action\s+building\s*=\s*\d+\s*>\s*([A-Z]+(?:_\d+)?)\s*</action>",
    re.IGNORECASE,
)


def token_counts(rows: list[dict]) -> "dict[str, int]":
    """Count action tokens across the `response` field of every row.

    Parses the verbatim assistant response (post-relabel, post-filter), so the
    counts are exactly what cross-entropy will see at SFT time.
    """
    from collections import Counter

    c: Counter = Counter()
    for row in rows:
        c.update(t.upper() for t in _RESPONSE_TOKEN_RE.findall(row.get("response", "")))
    return dict(c)


def rebalance_rows(
    rows: list[dict],
    *,
    beta: float = 2.0,
    floor: float = 0.25,
    target_size: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Resample an imbalanced distillation dataset so the dominant action
    token (IDLE) no longer drowns out the actual control actions.

    The SAC-distill JSONL is severely skewed — IDLE alone is ~48 % of all
    action tokens after `filter_uninformative_rows`. Behaviour cloning with
    greedy decoding on such a marginal collapses: the student minimises loss
    by emitting IDLE for (almost) every state (measured: 98.6 % IDLE at
    eval). Rebalancing flattens that marginal so no single token is left for
    greedy decoding to collapse onto.

    The action *vocabulary* is left fully intact — every token, including
    ones the SAC teacher never used (CHARGE_100, …), stays available to the
    SLM for zero-shot and for the Phase-3 RL stage. We change only how often
    each row is *seen* during SFT.

    Each row is weighted by how many *informative* (non-IDLE) action tokens
    it carries, raised to `beta`::

        n_info(row) = #{ action tokens in row.response that are not IDLE }
        W(row)      = (floor + n_info(row)) ** beta

    Rows are drawn WITH REPLACEMENT in proportion to ``W(row)``. IDLE-only
    rows are heavily down-sampled; rows where the teacher actively cycles
    several batteries are up-sampled. On the SAC-distill JSONL the default
    (beta=2) pulls IDLE from ~48 % of all tokens to ~24 %, with every other
    token in the 10–25 % band.

    Why not per-token inverse-frequency weighting? Each row carries
    `n_buildings` *coupled* tokens — an IDLE-heavy row that happens to hold
    one rare token still drags the other IDLE labels along. Per-token
    inverse-frequency barely moves the marginal (measured: 48 % → 41 %).
    Counting informative tokens per row and weighting super-linearly
    (`beta` > 1) is what actually flattens the distribution.

    Args:
        rows:        Distillation rows (already filtered). Each must have a
                     `response` field with `<action building=...>` tags.
        beta:        Rebalancing strength. 1.0 = linear in n_info; 2.0
                     (default) pulls IDLE to ~24 %; higher values concentrate
                     harder on multi-action rows at the cost of row diversity
                     (more duplication → mild overfitting risk).
        floor:       Small positive offset so all-IDLE rows keep a tiny but
                     non-zero chance of being sampled — the model must still
                     see when idling is correct.
        target_size: Number of rows to draw. Defaults to ``len(rows)`` — the
                     dataset keeps its size, only its composition changes.
        seed:        RNG seed for reproducible resampling.

    Returns:
        A new list of rows (references reused; rows may repeat). Resample
        the TRAIN split ONLY — never the held-out eval split, or duplicated
        rows leak across the split and contaminate ``eval_loss``.
    """
    if not rows:
        return rows
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    if floor <= 0:
        raise ValueError(f"floor must be positive, got {floor}")

    row_w = np.empty(len(rows), dtype=float)
    for i, row in enumerate(rows):
        toks   = [t.upper() for t in _RESPONSE_TOKEN_RE.findall(row.get("response", ""))]
        n_info = sum(t != "IDLE" for t in toks)
        row_w[i] = (floor + n_info) ** beta
    row_w /= row_w.sum()

    rng  = np.random.default_rng(seed)
    n    = target_size if target_size is not None else len(rows)
    idx  = rng.choice(len(rows), size=n, replace=True, p=row_w)
    return [rows[i] for i in idx]


# ── Trajectory dumper ─────────────────────────────────────────────────────

def dump_sac_trajectory_jsonl(
    env,
    agent,
    out_path: str | Path,
    snapshot_fn,
    n_buildings: int | None = None,
    include_meta: bool = True,
    building_slices: list[list[int]] | None = None,
    seed: int | None = None,
    relabel_noops: bool = True,
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
        seed:            Optional seed passed to env.reset() — required for
                         deterministic JSONL across re-runs if the schema
                         configures stochastic initial battery SoC. If None,
                         env.reset() uses whatever seed CityLearn was built with.
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
        relabel_noops:   If True (default), a per-building action that is a
                         physical no-op (discharge from an empty battery /
                         charge into a full one) is written as IDLE rather
                         than DISCHARGE_*/CHARGE_*. See `action_to_token`.
                         The raw SAC float is still kept in `actions_float`.

    Returns:
        Stats dict: {"n_steps", "n_rows", "path", "n_buildings", "n_slices",
        "n_relabeled"}.
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

    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
    done, t = False, 0
    n_steps = 0
    n_rows  = 0
    n_relabeled = 0

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
                socs_sl  = [float(d.get("electrical_storage_soc", 0.0))
                            for d in snap_sl]
                state_text = render_state(snap_sl)
                if relabel_noops:
                    n_relabeled += sum(
                        action_to_token(a) != action_to_token(a, soc=s)
                        for a, s in zip(acts_sl, socs_sl)
                    )
                response = format_action_block(
                    acts_sl, n_b, socs=socs_sl if relabel_noops else None
                )
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
        "n_relabeled": n_relabeled,
    }
