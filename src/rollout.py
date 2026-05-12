"""Rollout drivers and result-summary helpers shared by notebooks 02 and 03.

`run_policy`            — single-agent rollout. THIS IS THE DEFAULT through
                          Phase 3. The policy sees all buildings of the env
                          (TRAINING_BUILDINGS=[0,1,2] for SLM training/eval;
                          BUILDINGS=[0..5] when used on the full district).
`run_policy_dual_agent` — PHASE 4 ONLY. Dual-agent rollout with partial
                          observability: two policy calls per step (α + β),
                          actions merged in global building-index order.
                          Do NOT use for Phases 1–3 — single-agent
                          group-centralized over 3 buildings is the design.

Summary helpers return DataFrames / Series suitable for direct `display()`:
`summarize_district`, `district_kpis`, `per_agent_summary`.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
from citylearn.citylearn import CityLearnEnv

from src.agent import render_state
from src.env import make_env, snapshot_state


N_BUILDINGS_DEFAULT = 6

HEADLINE_KPIS = [
    "electricity_consumption_total",
    "cost_total",
    "carbon_emissions_total",
    "daily_peak_average",
    "ramping_average",
    "daily_one_minus_load_factor_average",
]


# ──────────────────────────────────────────────────────────────────────────────
#  Rollouts
# ──────────────────────────────────────────────────────────────────────────────

def run_policy(
    name: str,
    policy_fn: Callable,
    start: int,
    length: int,
    reward_fn: str = "merlin",
    obs_set: str = "llm",
    env_factory: Callable | None = None,
) -> tuple[pd.DataFrame, CityLearnEnv, list[dict]]:
    """Single-agent rollout — policy_fn sees all buildings.

    `policy_fn(snap, t)` may return either `list[float]` (simple policies) or
    `(actions, raw, fallback)` (LLM policies). The latter is logged in `raw_log`.

    `env_factory(start, end, obs_set, reward_fn)` overrides `make_env` — useful
    for the Colab variant that downloads the schema by name.
    """
    factory = env_factory or _default_env_factory
    env = factory(start=start, end=start + length - 1, obs_set=obs_set, reward_fn=reward_fn)
    env.reset()

    rows: list[dict] = []
    raw_log: list[dict] = []
    done, t, t0 = False, 0, time.time()

    while not done:
        snap   = snapshot_state(env)
        result = policy_fn(snap, t)

        if isinstance(result, tuple):
            acts, raw, fb = result
            raw_log.append({
                "t": t, "state_text": render_state(snap),
                "raw": raw, "fallback": bool(fb),
            })
        else:
            acts = result

        n = len(acts)
        _obs, reward, terminated, truncated, _ = env.step([[float(a)] for a in acts])
        done = bool(terminated or truncated)
        post = snapshot_state(env)

        rows.append({
            "policy": name, "t": t, "price": snap[0]["electricity_pricing"],
            "reward_sum": float(np.sum(reward)),
            **{f"a{i}":   acts[i]                                     for i in range(n)},
            **{f"r{i}":   float(reward[i])                            for i in range(n)},
            **{f"soc{i}": post[i]["electrical_storage_soc"]           for i in range(n)},
            **{f"net{i}": post[i]["net_electricity_consumption_last"] for i in range(n)},
        })
        t += 1

    df    = pd.DataFrame(rows)
    n_fb  = sum(1 for r in raw_log if r["fallback"])
    fb_msg = f" | fallbacks={n_fb}/{len(raw_log)}" if raw_log else ""
    print(f"[{name}] {t} steps in {time.time()-t0:.1f}s | "
          f"reward={df['reward_sum'].sum():.4f}{fb_msg}")
    return df, env, raw_log


def run_policy_dual_agent(
    name: str,
    policy_a: Callable,
    policy_b: Callable,
    agent_a_bldgs: list[int],
    agent_b_bldgs: list[int],
    start: int,
    length: int,
    reward_fn: str = "merlin",
    obs_set: str = "llm",
    summary_every: int = 24,
    env_factory: Callable | None = None,
) -> dict:
    """Dual-agent rollout — partial observability, no inter-agent communication.

    **PHASE 4 ONLY.** Through Phase 3 we train a single group-centralized agent
    over TRAINING_BUILDINGS=[0,1,2] (one policy call per step). At Phase 4
    deployment, the same fine-tuned LoRA is loaded into two agent instances
    and this function is used to roll them out on the full 6-building env
    with partial observability enforced by the snap slicing below.

    Agent α receives `snap[agent_a_bldgs]`, Agent β receives `snap[agent_b_bldgs]`.
    Their actions are combined in global building-index order before `env.step`.

    `summary_every=24` prints a compact daily progress line; pass 0 to disable.

    Returns dict with keys: df, env, raw_log_a, raw_log_b.
    """
    factory = env_factory or _default_env_factory
    n_a = len(agent_a_bldgs)
    n_b = len(agent_b_bldgs)
    n_total = n_a + n_b
    all_bldgs = agent_a_bldgs + agent_b_bldgs

    env = factory(start=start, end=start + length - 1, obs_set=obs_set, reward_fn=reward_fn)
    env.reset()

    rows: list[dict] = []
    raw_log_a: list[dict] = []
    raw_log_b: list[dict] = []
    done, t, t0 = False, 0, time.time()
    day_reward = 0.0

    while not done:
        snap   = snapshot_state(env)
        snap_a = [snap[i] for i in agent_a_bldgs]
        snap_b = [snap[i] for i in agent_b_bldgs]

        result_a = policy_a(snap_a, t)
        result_b = policy_b(snap_b, t)

        if isinstance(result_a, tuple):
            acts_a, raw_a, fb_a = result_a
            raw_log_a.append({"t": t, "state_text": render_state(snap_a),
                              "raw": raw_a, "fallback": bool(fb_a)})
        else:
            acts_a, fb_a = result_a, False

        if isinstance(result_b, tuple):
            acts_b, raw_b, fb_b = result_b
            raw_log_b.append({"t": t, "state_text": render_state(snap_b),
                              "raw": raw_b, "fallback": bool(fb_b)})
        else:
            acts_b, fb_b = result_b, False

        # Merge α + β into a single action vector in global building-index order
        acts_combined = [0.0] * n_total
        for local_i, global_i in enumerate(agent_a_bldgs):
            acts_combined[global_i] = acts_a[local_i]
        for local_i, global_i in enumerate(agent_b_bldgs):
            acts_combined[global_i] = acts_b[local_i]

        _obs, reward, terminated, truncated, _ = env.step(
            [[float(a)] for a in acts_combined]
        )
        done = bool(terminated or truncated)
        post = snapshot_state(env)

        step_reward = float(np.sum(reward))
        day_reward += step_reward

        rows.append({
            "policy": name, "t": t, "price": snap[0]["electricity_pricing"],
            "reward_sum": step_reward,
            "reward_a":   float(sum(reward[i] for i in agent_a_bldgs)),
            "reward_b":   float(sum(reward[i] for i in agent_b_bldgs)),
            "fallback_a": fb_a,
            "fallback_b": fb_b,
            **{f"a{i}":   acts_combined[i]                            for i in range(n_total)},
            **{f"r{i}":   float(reward[i])                            for i in range(n_total)},
            **{f"soc{i}": post[i]["electrical_storage_soc"]           for i in range(n_total)},
            **{f"net{i}": post[i]["net_electricity_consumption_last"] for i in range(n_total)},
        })

        if summary_every > 0 and (t + 1) % summary_every == 0:
            day_num   = (t + 1) // summary_every
            soc_a_str = "/".join(f"{post[i]['electrical_storage_soc']*100:.0f}" for i in agent_a_bldgs)
            soc_b_str = "/".join(f"{post[i]['electrical_storage_soc']*100:.0f}" for i in agent_b_bldgs)
            dist_net  = sum(post[i]["net_electricity_consumption_last"] for i in all_bldgs)
            elapsed   = time.time() - t0
            print(
                f"  ── Day {day_num:2d} | "
                f"SoC α:[{soc_a_str}]%  β:[{soc_b_str}]%  | "
                f"dist_net={dist_net:+.1f} kWh  reward={day_reward:.1f}  | "
                f"{elapsed:.0f}s elapsed"
            )
            day_reward = 0.0

        t += 1

    df     = pd.DataFrame(rows)
    n_fb_a = sum(1 for r in raw_log_a if r["fallback"])
    n_fb_b = sum(1 for r in raw_log_b if r["fallback"])
    print(
        f"[{name}] {t} steps in {time.time()-t0:.1f}s | "
        f"reward={df['reward_sum'].sum():.4f} "
        f"(α={df['reward_a'].sum():.4f}  β={df['reward_b'].sum():.4f}) | "
        f"fallbacks α={n_fb_a} β={n_fb_b}"
    )
    return {"df": df, "env": env, "raw_log_a": raw_log_a, "raw_log_b": raw_log_b}


def _default_env_factory(start: int, end: int, obs_set: str, reward_fn: str) -> CityLearnEnv:
    return make_env(start=start, end=end, obs_set=obs_set, reward_fn=reward_fn)


# ──────────────────────────────────────────────────────────────────────────────
#  Summaries / KPIs
# ──────────────────────────────────────────────────────────────────────────────

def summarize_district(df: pd.DataFrame, label: str, n_buildings: int = N_BUILDINGS_DEFAULT) -> dict:
    """One-row district summary: total reward, est. cost, peak load, total kWh."""
    net_cols = [f"net{i}" for i in range(n_buildings)]
    dist_net = df[net_cols].sum(axis=1)
    return {
        "policy":         label,
        "total_reward":   float(df["reward_sum"].sum()),
        "total_cost_est": float((dist_net * df["price"]).sum()),
        "peak_net_kW":    float(dist_net.max()),
        "total_net_kWh":  float(dist_net.sum()),
    }


# district_kpis was here — removed. Use src.eval.district_kpis (evaluate_v2,
# CityLearn 2.6+) for the single canonical KPI extractor.


def per_agent_summary(df: pd.DataFrame, agent_name: str, bldg_indices: list[int]) -> dict:
    """Per-agent metrics from a full multi-building rollout dataframe."""
    net_cols = [f"net{i}" for i in bldg_indices]
    r_cols   = [f"r{i}"   for i in bldg_indices]
    soc_cols = [f"soc{i}" for i in bldg_indices]
    a_cols   = [f"a{i}"   for i in bldg_indices]
    dist_net = df[net_cols].sum(axis=1)
    return {
        "agent":         agent_name,
        "buildings":     str(bldg_indices),
        "total_reward":  float(df[r_cols].sum().sum()),
        "mean_soc_pct":  float(df[soc_cols].mean().mean() * 100),
        "peak_net_kW":   float(dist_net.max()),
        "total_net_kWh": float(dist_net.sum()),
        "mean_action":   float(df[a_cols].values.mean()),
        "std_action":    float(df[a_cols].values.std()),
    }
