"""Standardised evaluation utilities for the ECLIPSE thesis.

All notebooks and scripts import from here — no KPI logic lives in cells.

Challenge scoring follows the 2022 CityLearn Challenge specification
(Appendix A).  All ratios are normalised against the no-battery-control
baseline: 1.0 = no improvement, < 1.0 = better.

Quick-start
-----------
    from src.eval import evaluate, generalisation_gap

    result = evaluate(env_after_episode, label="SAC")
    print(result.challenge)   # DataFrame row
    print(result.zne)         # DataFrame row
    gap = generalisation_gap(sac_train_result, sac_unseen_result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
from citylearn.citylearn import CityLearnEnv

logger = logging.getLogger(__name__)

# ── KPI column names (CityLearn v2 evaluate_v2 output) ────────────────────

CHALLENGE_KPIS: dict[str, str] = {
    "C  — cost":        "district_cost_ratio_to_baseline_total_ratio",
    "G  — carbon":      "district_emissions_ratio_to_baseline_total_ratio",
    "R  — ramping":     "district_energy_grid_shape_quality_ramping_average_to_baseline_ratio",
    "1L — load factor": "district_energy_grid_shape_quality_load_factor_penalty_daily_average_to_baseline_ratio",
}

_ZNE_SOLAR_COL = "district_solar_self_consumption_total_generation_kwh"
_ZNE_IMPORT_COL = "district_energy_grid_total_import_control_kwh"
_ZNE_SC_COL = "district_solar_self_consumption_ratio_self_consumption_ratio"

# ── Result container ───────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Holds all KPIs for one agent / environment pair.

    Attributes:
        label:     Human-readable agent name (used as DataFrame index).
        challenge: Series with C, G, R, 1L, Phase I, and Combined scores.
        zne:       Series with solar generation, grid import, ZNE ratio,
                   ZNE achieved flag, and self-consumption ratio.
    """
    label:     str
    challenge: pd.Series = field(default_factory=pd.Series)
    zne:       pd.Series = field(default_factory=pd.Series)

    # ── Convenience accessors ──────────────────────────────────────────────

    @property
    def phase1(self) -> float:
        """Phase I score: (C + G) / 2."""
        return float(self.challenge["Phase I (C+G)/2"])

    @property
    def combined(self) -> float:
        """Combined score: (C + G + D) / 3, D = (R + 1-L) / 2."""
        return float(self.challenge["Combined (C+G+D)/3"])



# ── Low-level helpers ──────────────────────────────────────────────────────

def district_kpis(env: CityLearnEnv) -> pd.Series:
    """Extract district-level KPIs from env.evaluate_v2() as a Series.

    Args:
        env: A CityLearnEnv after at least one completed episode.

    Returns:
        Series indexed by cost_function name, values as floats.
    """
    df = env.evaluate_v2()
    mask = df["level"].astype(str).str.lower() == "district"
    return df[mask].set_index("cost_function")["value"].astype(float)


# ── Primary evaluation functions ───────────────────────────────────────────

def challenge_score(env: CityLearnEnv, label: str) -> pd.Series:
    """Compute the 2022 Challenge scores from env.evaluate_v2().

    Scores:
        Phase I  = (C + G) / 2        — primary thesis metric
        D        = (R + (1-L)) / 2    — grid quality composite
        Combined = (C + G + D) / 3    — full challenge score

    Args:
        env:   A CityLearnEnv after at least one completed episode.
        label: Agent name (used as Series name).

    Returns:
        Series with all KPIs plus Phase I and Combined composites.
    """
    kpis = district_kpis(env)
    C   = float(kpis[CHALLENGE_KPIS["C  — cost"]])
    G   = float(kpis[CHALLENGE_KPIS["G  — carbon"]])
    R   = float(kpis[CHALLENGE_KPIS["R  — ramping"]])
    oml = float(kpis[CHALLENGE_KPIS["1L — load factor"]])
    D   = (R + oml) / 2
    return pd.Series(
        {
            "C  — cost":            round(C,              4),
            "G  — carbon":          round(G,              4),
            "R  — ramping":         round(R,              4),
            "1L — load factor":     round(oml,            4),
            "Phase I (C+G)/2":      round((C + G) / 2,   4),
            "Combined (C+G+D)/3":   round((C + G + D) / 3, 4),
        },
        name=label,
    )


def zne_metric(env: CityLearnEnv, label: str) -> pd.Series:
    """Compute Zero Net Energy and self-consumption metrics.

    ZNE ratio = total solar generation / total grid import.
    ≥ 1.0 means the district generated at least as much solar as it imported.

    Args:
        env:   A CityLearnEnv after at least one completed episode.
        label: Agent name (used as Series name).

    Returns:
        Series with solar generation, grid import, ZNE ratio, ZNE flag, and
        self-consumption ratio.
    """
    d     = district_kpis(env)
    solar = float(d.get(_ZNE_SOLAR_COL,  0.0))
    imp   = float(d.get(_ZNE_IMPORT_COL, 1.0))
    sc    = float(d.get(_ZNE_SC_COL,     float("nan")))
    zne   = solar / max(imp, 1e-6)
    return pd.Series(
        {
            "solar generation (kWh)":     round(solar, 1),
            "grid import (kWh)":          round(imp,   1),
            "ZNE ratio (solar / import)": round(zne,   4),
            "ZNE achieved (≥ 1.0)":       solar >= imp,
            "self-consumption ratio":     round(sc,    4),
        },
        name=label,
    )


def evaluate(env: CityLearnEnv, label: str) -> EvalResult:
    """Run both challenge_score and zne_metric in one call.

    Args:
        env:   A CityLearnEnv after at least one completed episode.
        label: Human-readable agent name.

    Returns:
        EvalResult with .challenge and .zne Series.
    """
    return EvalResult(
        label=label,
        challenge=challenge_score(env, label),
        zne=zne_metric(env, label),
    )


# ── Multi-agent comparison ─────────────────────────────────────────────────

def comparison_table(results: list[EvalResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build challenge-score and ZNE DataFrames from a list of EvalResults.

    Args:
        results: One EvalResult per agent / condition.

    Returns:
        Tuple of (challenge_df, zne_df), both indexed by agent label.
    """
    challenge_df = pd.DataFrame([r.challenge for r in results])
    zne_df       = pd.DataFrame([r.zne       for r in results])
    challenge_df.index = [r.label for r in results]
    zne_df.index       = [r.label for r in results]
    return challenge_df, zne_df


# ── Generalisation gap ─────────────────────────────────────────────────────

def generalisation_gap(
    train_result:  EvalResult,
    unseen_result: EvalResult,
) -> dict[str, float]:
    """Compute the generalisation gap between train and unseen-building results.

    A positive gap means the agent performs worse on unseen buildings.
    The RBC gap estimates environment difficulty; the SAC gap estimates
    policy transfer degradation beyond environment difficulty.

    Args:
        train_result:  EvalResult on training buildings.
        unseen_result: EvalResult on held-out buildings.

    Returns:
        Dict with phase1_gap and combined_gap (unseen − train).
    """
    phase1_gap   = unseen_result.phase1   - train_result.phase1
    combined_gap = unseen_result.combined - train_result.combined
    logger.info(
        "%s → %s  Phase I gap: %+.4f  Combined gap: %+.4f",
        train_result.label, unseen_result.label, phase1_gap, combined_gap,
    )
    return {
        "agent":        train_result.label,
        "phase1_gap":   round(phase1_gap,   4),
        "combined_gap": round(combined_gap, 4),
    }
