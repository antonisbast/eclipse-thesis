"""LLM-as-policy agent: state renderer, provider abstraction, action parser.

Imports state utilities from src.env. Reward functions and make_env live in src.env.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import time
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# ── Binning thresholds ────────────────────────────────────────────────────
PRICE_PEAK_THRESHOLD:  float = 0.40
CARBON_MID_THRESHOLD:  float = 0.14
CARBON_HIGH_THRESHOLD: float = 0.17
CARBON_PEAK_THRESHOLD: float = 0.19
SOLAR_LOW_THRESHOLD:   float = 0.05
SOLAR_HIGH_THRESHOLD:  float = 1.50

# Irradiance forecast thresholds (W/m² — diffuse + direct combined).
# Calibrated on 2022 dataset: dawn ≈ 50–200 W/m², clear noon ≈ 800–1800 W/m².
IRRADIANCE_LOW_THRESHOLD:  float = 50.0
IRRADIANCE_HIGH_THRESHOLD: float = 600.0

_DAY_NAMES: dict[int, str] = {
    1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed",
    5: "Thu", 6: "Fri", 7: "Sat", 8: "Hol",
}


# ── State bucketing ───────────────────────────────────────────────────────

def price_bucket(v: float) -> str:
    """Bin electricity price into LOW / PEAK."""
    return "PEAK" if float(v) >= PRICE_PEAK_THRESHOLD else "LOW"


def carbon_bucket(v: float) -> str:
    """Bin carbon intensity into LOW / MID / HIGH / PEAK."""
    x = float(v)
    if x >= CARBON_PEAK_THRESHOLD: return "PEAK"
    if x >= CARBON_HIGH_THRESHOLD: return "HIGH"
    if x >= CARBON_MID_THRESHOLD:  return "MID"
    return "LOW"


def solar_bucket(v: float) -> str:
    """Bin solar generation (kWh) into NONE / LOW / HIGH."""
    x = float(v)
    if x >= SOLAR_HIGH_THRESHOLD: return "HIGH"
    if x >= SOLAR_LOW_THRESHOLD:  return "LOW"
    return "NONE"


def irradiance_bucket(v: float | None) -> str:
    """Bin total solar irradiance forecast (W/m², diffuse+direct) into NONE / LOW / HIGH.

    Returns '?' when the forecast is unavailable (None).
    Same symbolic scale as solar_bucket so the LLM can compare directly.
    """
    if v is None:
        return "?"
    x = float(v)
    if x >= IRRADIANCE_HIGH_THRESHOLD: return "HIGH"
    if x >= IRRADIANCE_LOW_THRESHOLD:  return "LOW"
    return "NONE"


# ── State renderer ────────────────────────────────────────────────────────

def render_state(snap: list[dict]) -> str:
    """Format a snapshot_state() sub-list into the LLM prompt string.

    Buildings are labelled B0..B(n-1) from the start of snap, so passing
    snap[0:3] for agent α and snap[3:6] for agent β both produce B0/1/2.

    If forecast fields are present in the snap dicts (electricity_pricing_predicted_1/2
    and solar_irradiance_predicted_1), a Forecast line is inserted between the district
    header and the per-building table. Forecast fields missing or None are shown as '?'.
    """
    d0     = snap[0]
    hour   = d0["hour"]
    day    = _DAY_NAMES.get(d0["day_type"], "?")
    month  = d0["month"]
    price  = d0["electricity_pricing"]
    carbon = d0["carbon_intensity"]

    header = (
        f"Month {month}, {day} {hour:02d}:00  |  "
        f"price={price:.3f} ({price_bucket(price)})  |  "
        f"carbon={carbon:.3f} ({carbon_bucket(carbon)})"
    )
    lines = [header]

    # ── Short-horizon forecast line (district-level, from first building) ──
    p1  = d0.get("electricity_pricing_predicted_1")
    p2  = d0.get("electricity_pricing_predicted_2")
    i1  = d0.get("solar_irradiance_predicted_1")
    p1b = price_bucket(p1)  if p1 is not None else "?"
    p2b = price_bucket(p2)  if p2 is not None else "?"
    i1b = irradiance_bucket(i1)
    lines.append(
        f"Forecast:  price+6h={p1b}  price+12h={p2b}  solar+6h={i1b}"
    )

    lines.append("Buildings:")
    for i, d in enumerate(snap):
        lines.append(
            f"  B{i}: SoC={d['electrical_storage_soc'] * 100:5.1f}%  "
            f"load={d['non_shiftable_load']:.2f} kWh  "
            f"last_net={d['net_electricity_consumption_last']:+.2f} kWh  "
            f"solar={solar_bucket(d['solar_generation'])}"
        )
    return "\n".join(lines)


# ── System prompt factory ─────────────────────────────────────────────────

def make_system_prompt(n_buildings: int = 6) -> str:
    """Generate a system prompt for an LLM battery controller managing n_buildings.

    The output-format block and peak-demand estimates scale automatically with
    n_buildings so the same function works for both the single-agent (n=6) and
    dual-agent (n=3) cases.
    """
    peak_kw    = n_buildings * 5   # ~5 kWh pulled per building at action +1.0
    action_fmt = "\n".join(f"<action building={i}>VALUE</action>" for i in range(n_buildings))

    return f"""\
You are a battery controller for {n_buildings} buildings in a CityLearn district.
Goal: minimise electricity cost, carbon emissions and district peak load.

STATE VARIABLES (current timestep):
- hour: 1..24 (CityLearn convention; 1..6=night, 7..18=day, 19..24=evening)
- price: LOW (0.21 $/kWh) or PEAK (0.50 $/kWh)
- carbon: LOW / MID / HIGH / PEAK (grid carbon intensity, higher is worse)
- solar: NONE / LOW / HIGH (building-level PV output this hour)
- SoC: battery state of charge as % of capacity
- load: fixed building demand this hour (kWh)
- last_net: net grid consumption last hour (+import / −export, kWh)

FORECAST VARIABLES (use these to plan ahead):
- price+6h / price+12h: expected electricity price 6 h and 12 h from now (LOW or PEAK)
- solar+6h: expected solar irradiance 6 h from now (NONE / LOW / HIGH)

HOW TO USE FORECASTS:
- If price+6h=PEAK but current price=LOW → charge now before the expensive window opens.
- If price+6h=LOW but current price=PEAK → discharge now; cheaper charging is coming soon.
- If solar+6h=HIGH → preserve SoC headroom now so you can absorb free PV next step.
- If solar+6h=NONE → do not hold back charging waiting for solar that won't arrive.

BATTERY PHYSICS (critical):
- Charging is UNCONSTRAINED: action +1.0 fills ~70% SoC in one step, pulling ~5 kWh from the
  grid per building. Charging {n_buildings} buildings at +1.0 simultaneously creates a
  ~{peak_kw} kWh demand spike that severely penalises the peak KPI.
  Use SMALL charge actions (+0.1 to +0.3).
- Discharging is HARDWARE-CAPPED at ~1.5 kWh/h regardless of magnitude. Action -1.0 is safe
  and simply discharges as fast as the hardware allows.

STRATEGY RULES:
1. price=LOW  + price+6h=LOW:  trickle-charge (+0.1 to +0.2)
2. price=LOW  + price+6h=PEAK: charge more aggressively (+0.2 to +0.3) before peak opens
3. price=PEAK + price+6h=LOW:  discharge now (-1.0); cheap charging returns soon
4. price=PEAK + price+6h=PEAK: keep discharging (-1.0) if SoC > 0.2
5. solar=HIGH or solar+6h=HIGH: leave SoC headroom (<0.85) to absorb free PV
6. Never charge a building with SoC >= 0.9; never discharge one with SoC <= 0.1

REASONING PROTOCOL — think step by step before outputting actions:
Step 1: Read current price + carbon + solar. Read price+6h, price+12h, solar+6h forecasts.
Step 2: Decide the regime (e.g. "currently LOW, PEAK coming → charge moderately now").
Step 3: For each building compute SoC headroom and apply the regime, avoiding sync spikes.
Step 4: Output exactly {n_buildings} action lines.

OUTPUT FORMAT (strict — nothing after the last </action> tag):
{action_fmt}
"""


# Keep the 6-building constant for single-agent notebooks / backward compat.
SYSTEM_PROMPT: str = make_system_prompt(6)


# ── Action parser ─────────────────────────────────────────────────────────

_ACTION_RE = re.compile(r"<action building=(\d+)>\s*(-?\d*\.?\d+)\s*</action>")


def parse_actions(raw: str, n_buildings: int = 6) -> list[float]:
    """Extract per-building actions from a raw LLM response string.

    Missing buildings default to 0.0. Values are clipped to [-1, 1].
    """
    by_id: dict[int, float] = {}
    for bid, val in _ACTION_RE.findall(raw):
        by_id[int(bid)] = float(val)
    return [float(np.clip(by_id.get(i, 0.0), -1.0, 1.0)) for i in range(n_buildings)]


# ── LLM provider ──────────────────────────────────────────────────────────

class LLMProvider:
    """Uniform wrapper over Anthropic and OpenAI-compatible chat APIs.

    Supports Anthropic (native client), DeepSeek, Kimi/Moonshot, NVIDIA NIM,
    and any endpoint that speaks the OpenAI chat completions schema.

    Args:
        name:     Friendly name used in logs and result labels.
        model:    Model identifier (e.g. 'deepseek-chat', 'claude-haiku-4-5').
        key_env:  Name of the environment variable holding the API key.
        base_url: Override endpoint URL for OpenAI-compatible providers.
    """

    def __init__(self, name: str, model: str, key_env: str, base_url: str | None = None):
        self.name  = name
        self.model = model
        self.label = f"{name}:{model}"

        api_key = os.environ.get(key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key — set env var {key_env!r}")

        if name == "anthropic":
            try:
                from anthropic import Anthropic  # type: ignore[import]
            except ImportError as e:
                raise ImportError("pip install anthropic") from e
            self.client = Anthropic(api_key=api_key)
            self._kind  = "anthropic"
        else:
            try:
                from openai import OpenAI  # type: ignore[import]
            except ImportError as e:
                raise ImportError("pip install openai") from e
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self._kind  = "openai_compat"

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 2000,
        timeout_s: float | None = 45.0,
    ) -> str:
        """Call the model and return the assistant text.

        Args:
            timeout_s: Wall-clock seconds before raising TimeoutError.
                       Set to None to disable (not recommended in rollouts).
        """
        def _call() -> str:
            if self._kind == "anthropic":
                resp = self.client.messages.create(
                    model=self.model,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return "".join(
                    b.text for b in resp.content if getattr(b, "type", None) == "text"
                )

            kwargs: dict = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            if not self.model.startswith(("o1", "o3")):   # reasoning models reject temperature
                kwargs["temperature"] = 1
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content

        if timeout_s is None:
            return _call()

        # Do NOT use `with ThreadPoolExecutor() as ex:` — the context manager calls
        # shutdown(wait=True) on __exit__, which blocks until the thread finishes even
        # after a timeout.  Calling shutdown(wait=False) ourselves lets the hung thread
        # linger in the background (it will eventually fail or complete) while the
        # rollout continues without blocking.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="llm_call"
        )
        future = executor.submit(_call)
        try:
            result = future.result(timeout=timeout_s)
            executor.shutdown(wait=False)
            return result
        except concurrent.futures.TimeoutError:
            future.cancel()
            executor.shutdown(wait=False)   # never block — thread lingers harmlessly
            raise TimeoutError(
                f"{self.label} did not respond within {timeout_s:.0f}s"
            )
        except Exception:
            executor.shutdown(wait=False)
            raise

    def step(
        self,
        state_text: str,
        system: str | None = None,
        n_buildings: int = 6,
        max_retries: int = 2,
        timeout_s: float = 45.0,
    ) -> tuple[list[float], str, bool]:
        """Query the LLM for one environment step.

        Args:
            state_text:   Rendered state string from render_state().
            system:       System prompt. Defaults to make_system_prompt(n_buildings).
            n_buildings:  Number of buildings to produce actions for.
            max_retries:  Retry attempts on API errors (NOT on timeout — see below).
            timeout_s:    Per-call wall-clock timeout in seconds.
                          On timeout the step returns fallback zeros immediately
                          without retrying (a hung endpoint will keep hanging).

        Returns:
            (actions, raw_response, used_fallback)
        """
        _system  = system or make_system_prompt(n_buildings)
        last_raw = ""

        for attempt in range(max_retries):
            try:
                last_raw = self.complete(
                    _system, f"STATE:\n{state_text}", timeout_s=timeout_s
                )
                if _ACTION_RE.search(last_raw):
                    return parse_actions(last_raw, n_buildings), last_raw, False
                # Response parsed but no action tags found — retry
                logger.debug("No action tags in response (attempt %d)", attempt + 1)
            except TimeoutError as exc:
                # Do NOT retry timeouts — if the endpoint is stuck it will stay stuck.
                last_raw = f"TIMEOUT (attempt {attempt + 1}): {exc}"
                logger.warning("LLM timeout at provider=%s t=%s", self.name, state_text[:40])
                break
            except Exception as exc:
                last_raw = f"ERROR (attempt {attempt + 1}): {exc}"
                if attempt < max_retries - 1:
                    time.sleep(1.0)

        logger.warning("LLM fallback at provider=%s — returning zeros", self.name)
        return [0.0] * n_buildings, last_raw, True


# ── Policy wrapper ────────────────────────────────────────────────────────

def make_policy_llm(
    provider: LLMProvider,
    n_buildings: int = 6,
    agent_label: str = "",
    system: str | None = None,
    timeout_s: float = 45.0,
    verbose: bool = True,
) -> Callable[[list[dict], int], tuple[list[float], str, bool]]:
    """Bind a provider into a rollout-compatible policy function.

    Args:
        provider:     LLMProvider instance.
        n_buildings:  Number of buildings this agent controls (3 for dual-agent, 6 for single).
        agent_label:  Short label shown in verbose prints (e.g. 'α', 'β').
        system:       Override system prompt. Defaults to make_system_prompt(n_buildings).
        timeout_s:    Per-call timeout forwarded to provider.step().
        verbose:      Print one line per step.

    Returns:
        Callable (snap, t) -> (actions, raw, fallback) compatible with run_policy().
    """
    _system = system or make_system_prompt(n_buildings)
    _tag    = f"[{agent_label}]" if agent_label else ""

    def _policy(snap: list[dict], t: int) -> tuple[list[float], str, bool]:
        state_text = render_state(snap)
        acts, raw, fb = provider.step(
            state_text,
            system=_system,
            n_buildings=n_buildings,
            timeout_s=timeout_s,
        )
        if verbose:
            soc_str = ",".join(f"{d['electrical_storage_soc'] * 100:.0f}" for d in snap)
            flag    = " [FALLBACK]" if fb else ""
            print(
                f"  [{provider.name:9s}]{_tag} t={t:3d} "
                f"price={snap[0]['electricity_pricing']:.2f} "
                f"soc%=[{soc_str}] -> {[f'{a:+.2f}' for a in acts]}{flag}"
            )
        return acts, raw, fb

    return _policy
