# Progress Log

> Update this file after EVERY work session. Claude Code reads this first.

## Current status

**Phase:** Phase 1, Month 2 — `src/env.py` + `src/agent.py` populated; `02_llm_policy.ipynb` created
**Working on:** LLM-as-policy integration (Phase 2 groundwork)
**Blockers:** None
**Compute:** MacBook Air (local dev), Google Colab Pro (GPU training)
**Next step:** Run `02_llm_policy.ipynb` locally to verify imports work; then train SAC on Colab

---

## Log

### 2026-05-05 — 03_slm_colab.ipynb: local SLM inference on Colab GPU [LOCAL]
- Created `notebooks/03_slm_colab.ipynb` — fully self-contained Colab notebook
- **LocalHFProvider** class defined inline: same `.complete()` / `.step()` interface as
  `LLMProvider` so `make_policy_llm()` and all rollout functions work unchanged
  - Greedy decoding (`do_sample=False`) — deterministic, reproducible
  - No timeout needed (local GPU calls finish in < 2 s)
  - Qwen3 thinking mode disabled (`enable_thinking=False`) — 2× faster
  - Retry logic: up to `max_retries=2` on missing action tags, then zeros
- **make_colab_env()** passes dataset name as string to `CityLearnEnv` → auto-download
- Dual-agent setup: α controls B0-2, β controls B3-5, same as `02_llm_policy`
- Model presets in config cell (Qwen2.5-1.5B default, Qwen3-4B, Phi-3.5-mini, Qwen3-8B)
- 4-bit quantization support (bitsandbytes) for ≥7B models on T4
- Drive mount option — set MOUNT_DRIVE=True to persist results across sessions
- § 13.4 timing analysis: tokens/call, tokens/s, fallback rate
- Estimated rollout time: ~2 min (1.5B on T4) vs 1+ hour for remote API calls

### 2026-05-05 — Smoke-test split + forecast labels in notebook [LOCAL]
- `src/agent.py`: fixed `ThreadPoolExecutor` shutdown bug in `complete()` — replaced
  `with ThreadPoolExecutor() as ex:` (calls `shutdown(wait=True)` on any exception, blocking
  indefinitely) with explicit executor + `executor.shutdown(wait=False)` in all code paths
  (success, TimeoutError, other exception). Hung NVIDIA/slow calls now return in ≤ timeout_s.
- `notebooks/02_llm_policy.ipynb` — § 2, § 3, § 5 updated:
  - `env-check` (§ 2): expanded to show all 12 snapshot fields (9 real-time + 3 forecasts),
    prints forecast availability at t=0, shows price+6h and irr+6h per building
  - `s3-header` (§ 3): updated to mention 12-field snapshot, added example of the
    `Forecast: price+6h=X  price+12h=Y  solar+6h=Z` line now shown in rendered state
  - `s5-header` (§ 5): rewritten to explain per-provider cell structure
  - `provider-setup` (§ 5, cell 1): now only initialises `PROVIDER_OBJS = {}`
  - Added 5 individual smoke-test cells (one per provider: anthropic, deepseek, kimi,
    nvidia, gemma) — each independently interruptible; NVIDIA gets 30 s (cold-start headroom)

### 2026-05-05 — Google Gemma + forecast variables in state/prompt [LOCAL]
- `src/env.py` — `snapshot_state()` extended with 3 forecast fields:
  - `electricity_pricing_predicted_1` — price +6 h ($/kWh), via `b.pricing.electricity_pricing_predicted_1[t]`
  - `electricity_pricing_predicted_2` — price +12 h ($/kWh)
  - `solar_irradiance_predicted_1` — diffuse+direct irradiance +6 h (W/m²), via private
    `b.weather._diffuse/direct_solar_irradiance_predicted_1[t]`; reads wrapped in try/except
    so missing forecast columns degrade gracefully to None
- `src/agent.py`:
  - Added `IRRADIANCE_LOW_THRESHOLD=50`, `IRRADIANCE_HIGH_THRESHOLD=600` (W/m²)
  - Added `irradiance_bucket(v)` → NONE/LOW/HIGH (returns '?' on None)
  - `render_state()` now inserts `Forecast: price+6h=X  price+12h=Y  solar+6h=Z` between
    header and buildings; uses `price_bucket()` for price forecasts (same $/kWh scale),
    `irradiance_bucket()` for solar irradiance (W/m²)
  - `make_system_prompt(n)` rewritten: FORECAST VARIABLES section, expanded STRATEGY RULES
    to 6 rules (4 price-regime combos + solar headroom + limits), updated REASONING PROTOCOL
    Step 1 to read forecasts before deciding actions
- `notebooks/02_llm_policy.ipynb`:
  - Added Google Gemma (`gemma-3-12b-it`, Google AI Studio OpenAI-compat, `GOOGLE_API_KEY`)
    as 5th provider; standalone § 13 cell; sections renumbered § 13→14 through § 16→17

### 2026-05-05 — Dual-agent notebook + timeout + NVIDIA NIM [LOCAL]
- `src/agent.py` updated:
  - `make_system_prompt(n_buildings)` — parametric prompt; action-format block and peak-demand
    estimates scale automatically (used for both 3-building agents and 6-building single-agent)
  - `SYSTEM_PROMPT` kept as `make_system_prompt(6)` for backward compatibility
  - `complete()` gains `timeout_s` param via `ThreadPoolExecutor.result(timeout=...)`;
    raises `TimeoutError` on expiry, cancels the in-flight thread best-effort
  - `step()` breaks immediately on `TimeoutError` (no retry — a hung endpoint stays hung);
    API errors still retry up to `max_retries` times with 1 s backoff
  - `make_policy_llm()` gains `n_buildings`, `agent_label`, `system`, `timeout_s` params;
    `agent_label` ("α"/"β") appears in every verbose print line
- `notebooks/02_llm_policy.ipynb` rewritten as dual-agent experiment:
  - Agent α controls B0-B2, Agent β controls B3-B5 (partial observability, mirrors Phase 4)
  - Two LLM calls per timestep; actions combined in global building-index order before env.step()
  - NVIDIA NIM added as 4th provider (`meta/llama-3.1-8b-instruct`, `https://integrate.api.nvidia.com/v1`)
  - `LLM_TIMEOUT_S = 45.0` config constant; each call hard-stops and returns zeros on expiry
  - **One cell per provider** (§ 9–12) — interrupt a hung cell without losing other results
  - `llm_runs` dict initialised in § 8b; provider cells append to it; results cells gracefully
    handle any subset (empty, partial, full)
  - § 14 per-agent breakdown: reward split α/β, fallback counts per agent, mean SoC, peak net
  - § 15 diagnostics: SoC coloured by agent group (blue=α, red=β), district net load, behaviour
    table with sync_rate/fallback/rule-violations per agent, raw response sample

### 2026-05-04 — src/ modules + 02_llm_policy notebook [LOCAL]
- Populated `src/env.py`:
  - `MERLINReward`, `EcoPeakBatteryReward` (extracted from 01 notebook)
  - `make_env()` — supports `start`/`end` windowing, `obs_set` (`sac`=13 vars, `llm`=9 vars), `reward_fn`
  - `snapshot_state()` — bypasses obs-vector SoC bug by reading building objects directly
  - `OBSERVATIONS_SAC` (13 vars, with forecasts) and `OBSERVATIONS_LLM` (9 real-time, no forecasts)
  - Absolute `DATASET_ROOT` via `Path(__file__).parent.parent` — importable from any notebook
- Populated `src/agent.py`:
  - `price_bucket`, `carbon_bucket`, `solar_bucket`, `render_state()` — state-to-text pipeline
  - `SYSTEM_PROMPT` — battery physics + strategy rules + strict XML output format
  - `LLMProvider` — uniform Anthropic / OpenAI-compat wrapper with `.complete()` and `.step()`
  - `parse_actions()` — regex parser with per-building fallback to 0.0 and [-1,1] clip
  - `make_policy_llm()` — binds a provider into a rollout-compatible policy function
- Created `notebooks/02_llm_policy.ipynb`:
  - Imports env/agent entirely from `src/` (no logic in cells)
  - 11 sections: config → imports → env → renderer → LLM interface → reference policies → baselines → LLM runs → results → diagnostics → save
  - Same 1-week window (t=3624, 168 steps) as `04_llm_policy_clean.ipynb` for direct comparison
  - Saves rollout CSV, KPI CSV, behaviour CSV, raw JSON logs per provider

### 2026-05-04 — Project cleanup + docs reorganization [LOCAL]
- Updated `docs/CONTEXT.md` with accurate thesis scope and four-phase plan
- Renamed `CLAUDE_CITYLEARN_INSTRUCTIONS.md` → `docs/CITYLEARN_API.md` (CityLearn v2 API reference)
- Renamed `citylearn_insights.md` → `docs/CITYLEARN_INSIGHTS.md` (observation quirks, battery dynamics, prompting tips)
- Rewrote `CLAUDE.md` to align with revised research questions and actual folder structure
- `src/` and `scripts/` are confirmed empty stubs — to be populated in Phase 1 (SAC) and Phase 2 (SLM)

### 2026-04-30 — Cleanup + archive of exploratory work [LOCAL]
- Built `notebooks/04_llm_policy_clean.ipynb` — clean rewrite of `04_llm_policyV3` with:
  - Single config cell, no hardcoded API keys (env var only)
  - Solar bucket added to state renderer (NONE/LOW/HIGH)
  - System prompt updated with battery asymmetry insight (charge small, discharge full = safe)
  - `make_env(start, length)` parametric (no global mutation)
  - RBC baseline added alongside no-op/random/LLM
  - Clean model routing — no broken `responses` API fallback
- Moved old work to `archive/`:
  - `archive/notebooks/` ← 8 versioned notebooks (`01_environment_and_experts` … `04_llm_policyV3`)
  - `archive/root_notebooks/` ← `ECLIPSE_Meeting1_v3`, `ECLIPSE_diagnostic`, `citylearn_ccai_tutorial`
  - `archive/notebook_generators/` ← 4 `_gen_*.py` scripts
  - `archive/CLAUDE_CODE_PROMPT.md` — one-off prompt file
- `notebooks/` now contains only `04_llm_policy_clean.ipynb` (LLM-as-policy baseline)
- See `archive/README.md` for index of archived files



### 2026-03-21 — CityLearn v2 environment exploration [LOCAL]
- Installed CityLearn v2.5.0 (latest) on Python 3.13 — works despite version pin mismatches
- Created `sandbox/` folder with 5 exploration scripts + shared `_env_helpers.py`
- **01_basic_env.py**: 4-building env, 28 obs dims per building, 1 action (battery charge/discharge [-1,1])
- **02_explore_observations.py**: Full year (8760 steps) with plots of solar, SoC, net load, pricing
- **03_simple_rules.py**: Do-nothing vs peak-shaving baselines. Peak shaving saves ~5% on cost
- **04_ppo_baseline.py**: PPO via SB3 with CityLearn wrapper. 50K steps (5 episodes) — not enough to beat rules
- **05_observation_to_text.py**: 3 prompt formats (terse/medium/verbose) + robust action parser with 12 test cases

Key findings:
- CityLearn v2.5 `electrical_storage_soc` in observations shows next-step initial value (always 0), NOT current SoC — must read from `building.electrical_storage.soc[t]` directly
- Observations are not raw values — some are normalized/transformed. Use building objects for decision-making
- Battery works correctly (6.4 kWh capacity, 5 kW nominal power, 90% efficiency)
- CityLearn data is cached locally after first download — use schema path to avoid GitHub API rate limits
- Dataset: `citylearn_challenge_2022_phase_all`, pricing has binary low/high ($0.22/$0.54)
- Episode terminates at step 8759 (not 8760) — handle terminated/truncated signals
- PPO needs significantly more than 50K steps to learn (only ~5 episodes with 8760-step episodes)

Plots saved in `sandbox/plots/`:
- solar_generation.png, battery_soc.png, net_electricity_load.png, electricity_pricing.png
- ppo_training_curve.png, strategy_comparison.png

Next steps:
- Build proper Gymnasium wrapper in `src/env.py` based on the CityLearnWrapper from script 04
- Start notebook 01 with environment setup narrative
- Scale up PPO training on Colab (500K+ timesteps)

<!-- Example entry:
### 2026-04-01 — CityLearn environment setup [LOCAL]
- Installed CityLearn v2 via pip, confirmed Gymnasium compatibility
- Configured 4-building scenario with default weather data
- Defined observation and action spaces in src/env.py
- Tested with debug scale (2 buildings, 10 episodes) — works
- Issue: reward function returns NaN when battery SoC hits bounds — needs clipping
- Next: fix reward clipping, start PPO training in notebook 02

### 2026-04-05 — PPO baseline training [COLAB]
- Pushed code to GitHub, pulled into Colab
- Ran scripts/train_baseline.py with PPO on A100
- Training: 500K timesteps, took ~45 min
- Results logged to wandb (run: ppo-baseline-v1)
- PPO achieves 18% peak demand reduction vs no-storage baseline
- Checkpoints saved to Google Drive
- Next: train SAC, compare performance
-->
