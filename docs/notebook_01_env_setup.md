# Notebook 01 — Environment Setup & Baseline Evaluation
## Complete Knowledge Document

**Notebook:** `notebooks/01_env_setup.ipynb`  
**Phase:** Phase 1, Month 1  
**Purpose:** Establish the CityLearn v2 environment, validate observation/reward design, and produce SAC + RBC baselines that define the 70% gate for SLM agents in Phase 2.

---

## Table of Contents

1. [Dataset & Environment](#1-dataset--environment)
2. [Observation Space](#2-observation-space)
3. [Action Space](#3-action-space)
4. [Reward Function](#4-reward-function)
5. [Baseline Agents](#5-baseline-agents)
6. [KPI System & Challenge Scoring](#6-kpi-system--challenge-scoring)
7. [70% Validation Gate](#7-70-validation-gate)
8. [ZNE Metric (MERLIN)](#8-zne-metric-merlin)
9. [Generalisation Section](#9-generalisation-section)
10. [Key API Patterns](#10-key-api-patterns)
11. [Gotchas & Hard-Won Lessons](#11-gotchas--hard-won-lessons)
12. [Numerical Results (reference)](#12-numerical-results-reference)
13. [References](#13-references)

---

## 1. Dataset & Environment

### Dataset
- **Name:** `citylearn_challenge_2022_phase_all`
- **Path:** `data/citylearn_datasets/citylearn_challenge_2022_phase_all/`
- **Buildings:** 17 residential/commercial buildings (indices 0–16)
- **Timestep:** 1 hour (`seconds_per_time_step: 3600`)
- **Full year:** 8 759 steps (indices 0–8758 inclusive; simulation ends at step 8758, not 8759)
- **Schema:** `schema.json` at dataset root — must patch `root_directory` to absolute path at runtime

### Training / held-out split
```
BUILDINGS       = [0, 1, 2, 3, 4, 5]   # training buildings
UNSEEN_BUILDINGS = [6, 7, 8, 9, 10, 11]  # held-out for generalisation testing
```
Buildings 12–16 are unused. The split mirrors the partial-observability thesis design: Agent α (buildings 0–2), Agent β (buildings 3–5).

### CityLearn version
The notebook requires **CityLearn v2.6.0b2** (Python 3.12 venv at `.venv312/`).  
CityLearn v2.5.0 (system Python) does **not** have `evaluate_v2()` — use the project venv for Jupyter.

### Environment construction
```python
env = CityLearnEnv(
    schema=load_schema(),        # patched with absolute root_directory
    buildings=[0, 1, 2, 3, 4, 5],
    central_agent=False,         # one policy per building (decentralised)
    active_actions=["electrical_storage"],
    active_observations=ACTIVE_OBSERVATIONS,
    random_seed=42,
    simulation_start_time_step=0,
    simulation_end_time_step=8758,
)
```

**`central_agent=False` is critical** — with `True`, the env concatenates all building observations into one vector and expects a single joint action. The thesis uses decentralised agents, so `False` is the only correct setting.

### Schema loading pattern
```python
def load_schema() -> dict:
    with open(SCHEMA_FILE) as f:
        schema = json.load(f)
    schema["root_directory"] = str(DATASET_ROOT.resolve())  # must be absolute
    return schema
```
Always call `load_schema()` fresh per `CityLearnEnv` instantiation — CityLearn caches the schema object internally and mutations from one instance bleed into the next if the same dict is reused.

---

## 2. Observation Space

### Base observation set (9 variables) — `ACTIVE_OBSERVATIONS`

Used for all training runs in this notebook.

| Variable | Type | Range | Why included |
|---|---|---|---|
| `month` | int | 1–12 | Seasonal pattern learning |
| `hour` | int | 0–23 | Diurnal pattern learning |
| `day_type` | int | 1–8 | Weekday/weekend distinction |
| `electrical_storage_soc` | float | 0–1 | Current battery charge level |
| `net_electricity_consumption` | float | ~-8 to +10 kWh | Grid exchange; **positive = import, negative = export** |
| `non_shiftable_load` | float | 0.1–7 kWh | Fixed building demand before solar/battery |
| `solar_generation` | float | ≥ 0 kWh | PV output this hour |
| `electricity_pricing` | float | {0.21, 0.22, 0.40, 0.50, 0.54} EUR/kWh | Tariff signal (5 discrete levels) |
| `carbon_intensity` | float | 0–0.282 kgCO₂/kWh | Grid carbon signal |

**Obs dim per building:** 9 → after CityLearn's internal encoding the network input is 17 dims (CityLearn adds sin/cos encodings and normalises).

### Extended observation set (13 variables) — `EXTENDED_OBSERVATIONS`

`EXTENDED_OBSERVATIONS = ACTIVE_OBSERVATIONS + FORECAST_OBSERVATIONS`

Adds 4 forecast variables inspired by Nweye et al. (2024, MERLIN):

| Variable | Horizon | Notes |
|---|---|---|
| `electricity_pricing_predicted_1` | +6 h | All three price predictions exist in schema (`_1/_2/_3`) |
| `electricity_pricing_predicted_2` | +12 h | Most useful for overnight charge planning |
| `diffuse_solar_irradiance_predicted_1` | +6 h | Cloud-cover signal; complements direct irradiance |
| `direct_solar_irradiance_predicted_1` | +6 h | Direct sunlight; use together with diffuse |

Forecast variables are **already in the schema and marked `active=True`** — no data file changes needed, just add them to `active_observations`.  
MERLIN used all three horizons (+6h/+12h/+18h) for both price and both irradiance components and achieved Phase I = **0.815**. The subset here (4 variables) is the highest-return slice.

**The SLM agents in Phase 2 will use `EXTENDED_OBSERVATIONS` by default** — language model reasoning is particularly well-suited to forecast context ("price will be high in 6 hours, so charge now from solar").

### Key observation gotchas
- `solar_generation` in the observation vector is already in **kWh** (converted by CityLearn from irradiance), not W/m² — can be used directly in reward calculations.
- `net_electricity_consumption` **can be negative** (building is a net exporter after solar surplus). The reward must handle negative values correctly — do not clip to zero.
- `electrical_storage_soc` in the observation is reliable in v2.6.0b2 (the bug from v2.5 where it always showed 0 appears fixed).

---

## 3. Action Space

Only `electrical_storage` is active. Action is a continuous scalar in **[-1, +1]** per building:

- **+1.0** = charge at full rate (up to hardware limit)
- **-1.0** = discharge at full rate (up to hardware limit)
- **0.0** = hold (no charge/discharge)

### Battery hardware (2022 dataset, typical building)
- Capacity: ~6.4 kWh
- Nominal power: ~5 kW (charges fast: +1.0 action ≈ 70% fill in one hour)
- Discharge rate: physically capped (≈1.5 kWh/h regardless of action magnitude)
- Efficiency: ~90%

### Charging asymmetry — critical for reward design
| Direction | Speed | Grid effect |
|---|---|---|
| Charge (+) | Very fast — +1.0 fills ~70% in one step | Pulls heavily from grid; causes demand spikes |
| Discharge (-) | Hardware-capped — max ~1.5 kWh/h regardless of action | Gradual; safe to use -1.0 at any time |

**Best practice:** Use small fractional charge actions (+0.1 to +0.3) to avoid peak demand spikes. Use -1.0 freely during peak price/carbon hours — the hardware cap prevents negative spikes.  
BasicBatteryRBC uses +0.11 (charge) and -0.067 (discharge) — the charging rate is already conservative.

### HVAC and other actuators
HVAC and DHW actuators are set to `inactive_actions` in the schema. The 2022 buildings have no active thermal loads — only `electrical_storage` can be controlled.

---

## 4. Reward Function

`MERLINReward` is the project's reward function — applied by `make_env()` and defined in the `env-factory` cell (kept byte-for-byte in sync with `src/env.py`).

---

### MERLINReward

From Nweye et al. (2024), formula (per building):
```
net    = net_electricity_consumption
carbon = carbon_intensity * |net|
signal = w1 * |net|^e1 + w2 * |carbon|^e2
p      = -(1 + sign(net) * SoC)
reward = p * signal
```

**Grid-searched optimal parameters (Nweye et al. Table 3):**
- `w1=1.0, w2=0.0, e1=1, e2=1` → **pure cost signal, no carbon term**

Collapses to: `reward = -(1 + sign(net) * SoC) * |net_consumption|`

Uses raw kWh values — no dataset-specific normalisation constants required. Selected because it is simple, published, and dataset-agnostic.

**SoC amplification logic:**
- `sign(net) * SoC` amplifies the penalty when importing with a full battery (agent should have discharged but didn't → strong negative signal).
- Reduces the penalty when forced to import at low SoC (unavoidable → weaker signal).
- This gives SAC a gradient to learn discharge timing that matches tariff windows.

**Design decision rationale:**  
CityLearn's default reward `−max(net_consumption, 0)` causes a lazy-discharge trap: the agent drains the battery in the first few hours and then sits idle because any discharge immediately removes the penalty regardless of timing. The SoC-amplified signal forces the agent to hold charge for high-price windows.

---

## 5. Baseline Agents

### 5.1 BasicBatteryRBC — zero-learning lower bound

**Import:** `from citylearn.agents.rbc import BasicBatteryRBC`  
**Usage:** `agent = BasicBatteryRBC(env=env); agent.learn(episodes=1)`

**Rule (solar-aware hour-of-use controller):**
- Hours 06:00–14:00: charge at **+0.11** (11% of capacity/hour, captures solar window)
- All other hours: discharge at **-0.067** (6.7% of capacity/hour)

No price or carbon awareness — purely time-based. This makes it interpretable: any C or G improvement by a learning agent is due to learning price/carbon timing, not the control structure.

**Why use BasicBatteryRBC:**
- Stateless and reproducible — no training required
- Eliminates experimenter degrees of freedom (no hand-tuning)
- **Generalises by construction** — same rule applies to any building
- Single full-year episode takes ~20s on MacBook CPU

**RBC as sanity check:** RBC ignores the reward signal during action selection, so its KPIs are a stable anchor — re-running it should give near-identical numbers (any difference is env-reset noise). Large differences indicate a bug.

---

### 5.2 SAC — learned upper bound

**Import:** `from citylearn.agents.sac import SAC`  
**Usage:** `agent = SAC(env=env, seed=42); agent.learn(episodes=SAC_EPISODES, deterministic_finish=True)`

**Architecture:**
- One independent SAC policy network per building (`central_agent=False`)
- Standard continuous-action SAC with replay buffer, soft policy updates
- CityLearn's built-in implementation — no SB3 wrapper needed

**Training parameters (from CityLearn defaults):**
- `batch_size=256` — replay buffer must accumulate ≥256 steps before training begins
- `end_exploration_time_step=8759` — exploration for exactly one full episode
- `standardize_start_time_step=8758` — observation normalisation starts after episode 0

**`deterministic_finish=True` — how it works:**
- All episodes except the last use stochastic exploration
- Final episode runs the greedy (deterministic) policy for clean KPI measurement
- **Requires SAC_EPISODES ≥ 2 AND full-year episodes (SIM_END=8758)** to work safely
  - Episode 0: 8759 steps of exploration fills the buffer (`8759 > 256`), `norm_mean` is populated
  - Episodes 1+: training + deterministic final
  - If episodes are short (e.g., 96 steps), `96 < batch_size=256` and training never starts → `norm_mean` stays `None` → crash

**Training time on MacBook CPU:**
- 5 episodes (full year) ≈ 24 min
- 10 episodes ≈ 48 min
- 30+ episodes → move to `scripts/train_baseline.py` on Colab GPU

**SAC generalisation (inference on unseen buildings):**
```python
obs, _ = env_gen.reset()
while not done:
    actions = agent_sac.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env_gen.step(actions)
    done = bool(terminated or truncated)
```
- Do **not** call `agent_sac.learn()` on unseen env — that would reset internal state and update weights
- `predict(obs, deterministic=True)` works on unseen buildings because same `ACTIVE_OBSERVATIONS` → same obs structure (9 dims) → same internal CityLearn encoder → same 17-dim network input

---

## 6. KPI System & Challenge Scoring

### 6.1 `evaluate_v2()` — primary KPI extraction

```python
df = env.evaluate_v2()
# Returns DataFrame with columns: [cost_function, value, name, level]
# level ∈ {"building", "district"}
```

Extract district-level KPIs:
```python
def district_kpis(env_obj: CityLearnEnv) -> pd.Series:
    df = env_obj.evaluate_v2()
    mask = df["level"].astype(str).str.lower() == "district"
    d = df[mask] if mask.any() else df
    return d.set_index("cost_function")["value"].astype(float)
```

**All KPI values are normalised to baseline** (no-battery-control). Value = 1.0 means identical to no-control. Value < 1.0 means improvement.

### 6.2 KPI name format (v2.6.0b2)

KPI names are verbose ratio strings — do **not** use v1-style names. Full list of commonly used ones:

| Symbol | v2 KPI name |
|---|---|
| C | `district_cost_ratio_to_baseline_total_ratio` |
| G | `district_emissions_ratio_to_baseline_total_ratio` |
| R | `district_energy_grid_shape_quality_ramping_average_to_baseline_ratio` |
| 1-L | `district_energy_grid_shape_quality_load_factor_penalty_daily_average_to_baseline_ratio` |
| Peak | `district_energy_grid_shape_quality_peak_daily_average_to_baseline_ratio` |
| Grid import | `district_energy_grid_ratio_to_baseline_import_total_ratio` |

**Solar / ZNE KPIs (not challenge metrics, used for MERLIN comparison):**

| Purpose | v2 KPI name |
|---|---|
| Total solar generated | `district_solar_self_consumption_total_generation_kwh` |
| Total grid import (control) | `district_energy_grid_total_import_control_kwh` |
| Total grid import (baseline) | `district_energy_grid_total_import_baseline_kwh` |
| Self-consumption ratio | `district_solar_self_consumption_ratio_self_consumption_ratio` |

### 6.3 Official 2022 Challenge scoring formulas

From the challenge paper (Appendix A). All ratios are control / no-battery-baseline:

| Symbol | Formula | KPI name in evaluate_v2() |
|---|---|---|
| **C** | cost_control / cost_baseline | `district_cost_ratio_to_baseline_total_ratio` |
| **G** | emissions_control / emissions_baseline | `district_emissions_ratio_to_baseline_total_ratio` |
| **R** | ramping_control / ramping_baseline | `...ramping_average_to_baseline_ratio` |
| **1−L** | (1−load_factor)_control / (1−load_factor)_baseline | `...load_factor_penalty...to_baseline_ratio` |
| **D** | (R + 1−L) / 2 | grid quality |
| **Phase I** | **(C + G) / 2** | **primary thesis metric** |
| **Combined** | **(C + G + D) / 3** | full evaluation |

**These map 1:1 to `evaluate_v2()` outputs** — the challenge organisers used CityLearn's own evaluation, so no reimplementation is needed.

```python
CHALLENGE_KPI_MAP = {
    "C":   "district_cost_ratio_to_baseline_total_ratio",
    "G":   "district_emissions_ratio_to_baseline_total_ratio",
    "R":   "district_energy_grid_shape_quality_ramping_average_to_baseline_ratio",
    "1-L": "district_energy_grid_shape_quality_load_factor_penalty_daily_average_to_baseline_ratio",
}

def challenge_score(env_obj, label):
    kpis = district_kpis(env_obj)
    C   = float(kpis[CHALLENGE_KPI_MAP["C"]])
    G   = float(kpis[CHALLENGE_KPI_MAP["G"]])
    R   = float(kpis[CHALLENGE_KPI_MAP["R"]])
    oml = float(kpis[CHALLENGE_KPI_MAP["1-L"]])
    D   = (R + oml) / 2
    return {
        "Phase I (C+G)/2":  round((C + G) / 2, 4),
        "Combined (C+G+D)/3": round((C + G + D) / 3, 4),
        ...
    }
```

---

## 7. 70% Validation Gate

**Definition:** Before SLM agents proceed to Phase 2 (multi-agent coordination), a single SLM agent must achieve a Phase I score ≤ the gate threshold.

```
sac_improvement = 1.0 - sac_phase1_score
gate_score      = 1.0 - 0.70 * sac_improvement
```

Interpretation: if SAC achieves Phase I = 0.83 (17% improvement over baseline), the gate is `1.0 - 0.70×0.17 = 0.881`. Any agent with Phase I ≤ 0.881 clears the gate.

**Gate threshold depends on SAC training quality** — more SAC episodes → lower SAC score → stricter gate. Always report the SAC score and episode count alongside the gate value so the comparison is reproducible.

---

## 8. ZNE Metric (MERLIN)

**Definition (Nweye et al. 2024, §2.4):** A district achieves Zero Net Energy if its total solar generation ≥ total grid imports over the year.

**Implementation:**
```python
def zne_metric(env_obj, label):
    d = district_kpis_raw(env_obj)   # or extract from evaluate_v2()
    solar_gen   = d["district_solar_self_consumption_total_generation_kwh"]
    grid_import = d["district_energy_grid_total_import_control_kwh"]
    zne_ratio   = solar_gen / max(grid_import, 1e-6)
    self_cons   = d["district_solar_self_consumption_ratio_self_consumption_ratio"]
    return {"ZNE ratio": zne_ratio, "ZNE achieved": zne_ratio >= 1.0, ...}
```

**ZNE ratio = total_solar_generation / total_grid_import**
- ≥ 1.0 → ZNE achieved (generates at least as much as it imports)
- < 1.0 → grid-dependent; value shows how close

**Self-consumption ratio** (separate metric): fraction of generated solar consumed on-site rather than exported to grid. Captures whether the battery is buffering local PV effectively.

**MERLIN's finding:** Both SAC and RBC **worsen** ZNE ratio vs. the no-battery baseline. Reason: battery overnight charging increases grid import in off-solar hours more than the battery's daytime discharge reduces it, because the district already exports surplus solar even without a battery.

**ZNE is not a primary thesis metric** — the 2022 challenge does not include it. It is reported for completeness and because MERLIN tracked it. It will become more relevant if the ECLIPSE project moves to buildings with larger PV capacity.

---

## 9. Generalisation Section

### Why it matters

The thesis operates in a transfer setting: SLM agents must work on buildings they have not seen during training. This section establishes the baseline transfer difficulty using classical agents.

### Method

- RBC runs fresh on `UNSEEN_BUILDINGS = [6, 7, 8, 9, 10, 11]` — stateless, so it always generalises by definition
- SAC does **inference-only** on unseen buildings (no weight updates):
  ```python
  obs, _ = env_gen_sac.reset()
  while not done:
      actions = agent_sac.predict(obs, deterministic=True)
      ...
  ```
- Both evaluated with the same `challenge_score()` function

### Interpreting the generalisation gap

```
generalisation_gap = Phase_I(SAC, unseen) - Phase_I(SAC, train)
```
- Positive gap → SAC performance degrades on unseen buildings (some overfitting to training distribution)
- Compare against RBC's gap: if SAC degrades much more than RBC, that is a clear overfitting signal
- The gap quantifies how hard the cross-building transfer problem is for the SLM agents

### Why SAC policy can run on different buildings

Same `ACTIVE_OBSERVATIONS` list → same raw observation structure (9 dims) → CityLearn's internal encoder produces same 17-dim network input. The policy network is agnostic to which physical building it controls — it only sees the normalised observation vector.

---

## 10. Key API Patterns

### `make_env()` — environment factory

```python
make_env(
    buildings=None,          # list of building indices; defaults to BUILDINGS
    render_mode="end",       # 'end' or 'during'
    session_name=None,       # enables render; creates sub-folder under RENDER_DIR
)
```

The env always uses `MERLINReward`. Defaults are backward-compatible — all existing call sites work unchanged.

### Render mode

```python
make_env(session_name="my_run", render_mode="end")
```
- `render_mode='end'` buffers all per-step data in memory, flushes to disk at episode end
- `render_session_name` keeps the folder stable across re-runs (no new folder per episode)
- Output goes to `notebooks/artifacts/SimulationData/<session_name>/`
- Required for CityLearn UI visualisation

### Accessing raw episode data after training

After `agent.learn()` completes, the env's simulation data is in the CityLearn render folder (if render enabled). For KPIs, use `env.evaluate_v2()` — it reads from internal episode buffers, not disk files.

### Checking episode termination

```python
obs, _, terminated, truncated, _ = env.step(actions)
done = bool(terminated or truncated)
```
For full-year episodes, `terminated=True` at step 8758. `truncated` can also be True in some edge cases — always combine both.

---

## 11. Gotchas & Hard-Won Lessons

### 1. `deterministic=True` crash on SAC
**Symptom:** `AssertionError` in `sac.py get_normalized_observations()` — `norm_mean is None`  
**Cause:** With `deterministic=True` on all episodes, SAC skips exploration and calls `get_post_exploration_prediction()` before training has started (requires buffer ≥ batch_size=256 samples)  
**Fix:** Always use `deterministic_finish=True` (not `deterministic=True`)  
**Extra condition:** Full-year episodes (8759 steps >> 256) are required. Short episodes (e.g., 96 steps) with SAC_EPISODES=2 still crash because episode 0's 96 steps never fill the replay buffer.

### 2. Wrong KPI names from v1 API (silent failure)
**Symptom:** `present` list is empty; `display()` shows all 77 rows; bar chart silently skips  
**Cause:** v1-style names like `district_cost_total` don't exist in `evaluate_v2()`. The v2 names are long ratio strings.  
**Fix:** Use exact v2 names — see §6.2 above. Always validate with `k in kpi_table.index`.

### 3. RENDER_DIR must be defined before make_env()
**Symptom:** `NameError: RENDER_DIR` inside `make_env()` even with try/except, because the except block also references `RENDER_DIR`  
**Fix:** Define `RENDER_DIR` in the env-factory cell, before `make_env()` is defined. Never split the definition into a later cell.

### 4. Schema caching across env instances
**Symptom:** Second `make_env()` call uses patched schema from first call, but then a third mutates it — inconsistent `root_directory`  
**Fix:** Always call `load_schema()` inside `make_env()` to get a fresh dict per instance. Never pass the same schema dict to two `CityLearnEnv()` calls.

### 5. `evaluate_v2()` only available in v2.6.0b2+
**Symptom:** `AttributeError: 'CityLearnEnv' object has no attribute 'evaluate_v2'`  
**Fix:** Use the project's `.venv312` (Python 3.12) which has CityLearn 2.6.0b2. The system Python has v2.5.0.

### 6. RBC's `learn(episodes=1)` vs manual rollout
`BasicBatteryRBC.learn(episodes=1)` internally runs the full episode and populates the env's internal episode buffer. After it returns, `env.evaluate_v2()` is valid. For SAC on unseen buildings, do **not** call `learn()` — use the manual `predict` + `step` loop to avoid resetting the trained policy's internal state.

---

## 12. Numerical Results (reference)

These are indicative results from 5-episode SAC runs on MacBook CPU. Numbers will improve with more episodes on GPU.

### Challenge scores (training buildings, 5-episode SAC)

| Agent | C — cost | G — carbon | R — ramping | 1-L | Phase I (C+G)/2 | Combined |
|---|---|---|---|---|---|---|
| RBC | ~0.917 | ~0.967 | ~1.074 | ~0.946 | ~0.942 | ~0.968 |
| SAC (5 ep) | ~0.799 | ~0.862 | ~0.800 | ~0.954 | ~0.830 | ~0.872 |

- SAC Phase I ≈ 0.83 → **17% improvement over no-control baseline**
- 70% gate ≈ 0.88 (SAC improvement × 0.70 = 11.9% required minimum)
- MERLIN's full setup (10 ep, all forecast obs): Phase I = 0.815 — our 5-ep run is in the right ballpark

### KPI interpretation
- C ≈ 0.80 → SAC reduces electricity cost by ~20% vs. no battery control
- G ≈ 0.86 → ~14% carbon reduction
- R ≈ 0.80 → 20% smoother grid ramping
- 1-L ≈ 0.95 → marginal load factor improvement (this KPI is hard for decentralised SAC)

### Generalisation gap (train → unseen buildings)
- RBC gap: typically small (±0.01) — rule generalises by construction
- SAC gap: typically +0.02 to +0.06 on Phase I — moderate overfitting to training buildings
- If SAC gap >> RBC gap → clear overfitting signal, policy relies on building-specific statistics

---

## 13. References

1. **2022 CityLearn Challenge paper:**  
   Vazquez-Canteli, J.R. et al. (2022). *CityLearn Challenge 2022*. Challenge paper and evaluation appendix (Appendix A defines C, G, R, 1-L, D, Phase I, Combined formulas).

2. **MERLIN:**  
   Nweye, K., Sankaranarayanan, S., & Nagy, G. (2024). MERLIN: Multi-agent offline and transfer learning for occupant-centric operation of grid-interactive communities. *Applied Energy*, 358, 121958.  
   https://doi.org/10.1016/j.apenergy.2023.121958  
   — Source of: `MERLINReward` formula, forecast observation design, ZNE metric definition, Phase I = 0.815 benchmark

3. **CityLearn v2:**  
   Vazquez-Canteli, J.R. et al. CityLearn — open-source simulation framework for demand response.  
   https://github.com/intelligent-environments-lab/CityLearn  
   — v2.6.0b2 required for `evaluate_v2()`, `BasicBatteryRBC`, and `central_agent=False` SAC

4. **Soft Actor-Critic:**  
   Haarnoja, T. et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. ICML 2018.
