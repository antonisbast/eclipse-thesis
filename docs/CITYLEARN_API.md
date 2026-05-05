# CLAUDE_CITYLEARN_INSTRUCTIONS.md

> **Purpose:** Technical reference for Claude Code when assisting with CityLearn-based thesis work.
> Read this before writing any CityLearn code.

---

## 1. Import Paths

```python
# Environment
from citylearn.citylearn import CityLearnEnv

# Agents
from citylearn.agents.base import Agent, BaselineAgent
from citylearn.agents.rbc import RBC, HourRBC, BasicRBC, OptimizedRBC, BasicBatteryRBC
from citylearn.agents.rlc import RLC
from citylearn.agents.sac import SAC, SACRBC
from citylearn.agents.marlisa import MARLISA, MARLISARBC
from citylearn.agents.q_learning import TabularQLearning

# Wrappers
from citylearn.wrappers import (
    NormalizedObservationWrapper,
    NormalizedActionWrapper,
    NormalizedSpaceWrapper,
    ClippedObservationWrapper,
    DiscreteObservationWrapper,
    DiscreteActionWrapper,
    DiscreteSpaceWrapper,
    TabularQLearningWrapper,
    StableBaselines3Wrapper,
    RLlibSingleAgentWrapper,
    RLlibMultiAgentEnv,
)

# Reward Functions
from citylearn.reward_function import (
    RewardFunction,
    MARL,
    IndependentSACReward,
    SolarPenaltyReward,
    ComfortReward,
    SolarPenaltyAndComfortReward,
    MultiBuildingRewardFunction,
)

# Cost Functions (KPIs)
from citylearn.cost_function import CostFunction
```

---

## 2. Environment API

### 2.1 Constructor

```python
CityLearnEnv(
    schema: Union[str, Path, Mapping[str, Any]],  # dataset name, JSON filepath, or dict
    root_directory: Union[str, Path] = None,
    buildings: Union[List[Building], List[str], List[int]] = None,
    simulation_start_time_step: int = None,
    simulation_end_time_step: int = None,
    episode_time_steps: Union[int, List[Tuple[int, int]]] = None,
    rolling_episode_split: bool = None,
    random_episode_split: bool = None,
    seconds_per_time_step: float = None,
    reward_function: Union[RewardFunction, str] = None,
    reward_function_kwargs: Mapping[str, Any] = None,
    central_agent: bool = None,
    shared_observations: List[str] = None,
    active_observations: Union[List[str], List[List[str]]] = None,
    inactive_observations: Union[List[str], List[List[str]]] = None,
    active_actions: Union[List[str], List[List[str]]] = None,
    inactive_actions: Union[List[str], List[List[str]]] = None,
    simulate_power_outage: bool = None,
    solar_generation: bool = None,
    random_seed: int = None,
    render_mode: str = 'none',
    **kwargs: Any,
)
```

**Schema accepts three formats:**
1. **Dataset name** (string): e.g. `'baeda_3dem'` — must exist in `DataSet.get_dataset_names()`
2. **File path** (str/Path): path to a JSON schema file
3. **Dict**: inline schema definition

### 2.2 `reset()`

```python
def reset(
    self,
    seed: int = None,
    options: Mapping[str, Any] = None,
) -> Tuple[List[List[float]], dict]
```

**Returns:** `(observations, info)`
- `observations`: `List[List[float]]` — if `central_agent=True`, length 1; otherwise length = number of buildings
- `info`: `dict` — auxiliary diagnostic info (empty by default)

**Internal flow:**
1. Calls `super().reset()`
2. Updates seed if provided
3. Advances episode tracker (`episode_tracker.next_episode()`)
4. Resets all buildings and electric vehicles
5. Resets reward function
6. Clears reward/consumption/cost/emission histories
7. Refreshes action cache
8. Calls `update_variables()`

### 2.3 `step()`

```python
def step(
    self,
    actions: List[List[float]],
) -> Tuple[List[List[float]], List[float], bool, bool, dict]
```

**Parameters:**
- `actions`: All values in `[-1.0, 1.0]`. Fraction of device capacity.
  - `central_agent=True`: `[[a0, a1, ..., aN]]` — single list, all buildings concatenated
  - `central_agent=False`: `[[b0_a0, ...], [b1_a0, ...], ...]` — one sublist per building

**Returns:** `(observations, rewards, terminated, truncated, info)`
- `observations`: `List[List[float]]`
- `rewards`: `List[float]` — length 1 if central_agent, else per-building
- `terminated`: `bool` — True when episode reaches end
- `truncated`: `bool` — True if time limit exceeded
- `info`: `dict`

### 2.4 `evaluate()`

```python
def evaluate(
    self,
    control_condition: EvaluationCondition = None,
    baseline_condition: EvaluationCondition = None,
    comfort_band: float = None,
) -> pd.DataFrame
```

**Returns:** `pd.DataFrame` with KPI rows. Delegates to `CityLearnKPIService`.

**EvaluationCondition enum values:**
- `WITH_STORAGE_AND_PV` (default)
- `WITHOUT_STORAGE_BUT_WITH_PV`
- `WITHOUT_STORAGE_AND_PV`
- `WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV`
- `WITHOUT_STORAGE_BUT_WITH_PARTIAL_LOAD_AND_PV`
- `WITHOUT_STORAGE_AND_PARTIAL_LOAD_BUT_WITH_PV`
- `WITHOUT_STORAGE_AND_PARTIAL_LOAD_AND_PV`

### 2.5 `load_agent()`

```python
def load_agent(
    self,
    agent: Union[str, 'Agent'] = None,
    **kwargs,
) -> Agent
```

Loads agent from schema definition or explicit class/string. String format: `'citylearn.agents.sac.SAC'`.

### 2.6 Key Properties

| Property | Type | Notes |
|----------|------|-------|
| `observation_space` | `List[spaces.Box]` | Length 1 if central_agent, else per-building |
| `action_space` | `List[spaces.Box]` | Same length rule |
| `observations` | `List[List[float]]` | Current timestep; cached |
| `observation_names` | `List[List[str]]` | Observation labels |
| `action_names` | `List[List[str]]` | Action labels |
| `buildings` | `List[Building]` | Building objects |
| `rewards` | `List[List[float]]` | Full reward history |
| `terminated` | `bool` | Episode done? |
| `truncated` | `bool` | Time limit hit? |
| `time_step` | `int` | Current step |
| `time_steps` | `int` | Total steps in episode |
| `central_agent` | `bool` | Control mode |

---

## 3. `central_agent` Constraint Rules

| Aspect | `central_agent=True` | `central_agent=False` |
|--------|----------------------|------------------------|
| `observation_space` | `[Box(...)]` (1 element) | `[Box(...), Box(...), ...]` (N elements) |
| `action_space` | `[Box(...)]` (1 element) | `[Box(...), Box(...), ...]` (N elements) |
| Observations | Concatenated; shared obs included once | Per-building; each gets full obs |
| Actions to `step()` | `[[all_building_actions]]` | `[[b0_actions], [b1_actions], ...]` |
| Rewards from `step()` | `[single_scalar]` | `[r0, r1, ..., rN]` |
| SB3 wrappers | **Required True** | Will break |
| RLlib single-agent | **Required True** | Will break |
| RLlib multi-agent | Will break | **Required False** |
| MARLISA agent | Not supported | **Required False** |

---

## 4. Actions Reference

All actions are floats in `[-1.0, 1.0]`. Positive = charge/increase, negative = discharge/decrease.

| Action Name | Controlled Device | Unit |
|-------------|-------------------|------|
| `cooling_storage` | `Building.cooling_storage` | kWh/kWh_capacity |
| `heating_storage` | `Building.heating_storage` | kWh/kWh_capacity |
| `dhw_storage` | `Building.dhw_storage` | kWh/kWh_capacity |
| `electrical_storage` | `Building.electrical_storage` | kWh/kWh_capacity |
| `cooling_device` | `Building.cooling_device` | kW/kW_nominal |
| `heating_device` | `Building.heating_device` | kW/kW_nominal |
| `dhw_device` | `Building.dhw_device` | kW/kW_nominal |

---

## 5. Reward Functions

| Class | Formula | Use Case |
|-------|---------|----------|
| `RewardFunction` | `-max(e, 0)^exponent` | Basic consumption penalty |
| `MARL` | `sign(-e) × 0.01(e²) × max(0, E)` | Multi-agent with district signal |
| `IndependentSACReward` | `min(-e³, 0)` | Independent SAC agents |
| `SolarPenaltyReward` | Penalized consumption + SOC terms | Net-zero incentive |
| `ComfortReward` | Temperature delta penalty | Thermal comfort |
| `SolarPenaltyAndComfortReward` | Weighted sum of above two | Combined objective |

Where `e` = building net electricity consumption, `E` = district net consumption.

**Custom reward function pattern:**

```python
from citylearn.reward_function import RewardFunction

class MyReward(RewardFunction):
    def __init__(self, env_metadata):
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        rewards = []
        for obs in observations:
            r = -abs(obs['net_electricity_consumption'])
            rewards.append(r)
        if self.central_agent:
            return [sum(rewards)]
        return rewards
```

---

## 6. Cost Functions (KPIs)

All are static methods on `CostFunction`:

| Method | What it measures |
|--------|-----------------|
| `ramping(net_elec)` | Rolling Σ|E_i − E_{i−1}| — grid flexibility |
| `peak(net_elec, window=24)` | Average daily peak consumption |
| `one_minus_load_factor(net_elec, window=730)` | 1 − (mean/peak) over window |
| `electricity_consumption(net_elec)` | Rolling sum of positive consumption |
| `zero_net_energy(net_elec)` | Rolling sum of net consumption |
| `carbon_emissions(emissions)` | Rolling sum of CO₂ |
| `cost(cost)` | Rolling sum of electricity cost |
| `quadratic(net_elec)` | Rolling sum of consumption² |
| `discomfort(...)` | Returns 9 lists: (total/cold/hot %, cold delta min/max/avg, hot delta min/max/avg) |
| `one_minus_thermal_resilience(...)` | Discomfort % during outage |
| `normalized_unserved_energy(...)` | Proportion of unmet demand |

**Extracting KPIs after simulation:**

```python
kpis = env.evaluate()
# kpis is a pd.DataFrame
# Columns include: cost_function, value, building (or 'District')
print(kpis.to_string())

# Filter specific KPIs
electricity_kpi = kpis[kpis['cost_function'] == 'electricity_consumption']
```

---

## 7. Wrappers — Transformation Pipeline

### 7.1 Stable Baselines3 Pipeline

**Requirement:** `central_agent=True`

```
CityLearnEnv
  → NormalizedObservationWrapper     # periodic (sin/cos) + min-max to [0,1]
  → ClippedObservationWrapper        # clip to space bounds (recommended)
  → StableBaselines3Wrapper          # converts List[List] → np.ndarray, List[float] → float
```

`StableBaselines3Wrapper` is a composite of:
- `StableBaselines3ActionWrapper`: 1D array → `List[List[float]]`
- `StableBaselines3RewardWrapper`: `List[float]` → `float`
- `StableBaselines3ObservationWrapper`: `List[List[float]]` → `np.ndarray`

### 7.2 RLlib Single-Agent Pipeline

**Requirement:** `central_agent=True`

```python
env_config = {
    'env_kwargs': {'schema': '...', 'central_agent': True, 'buildings': [0, 1]},
    'wrappers': [NormalizedObservationWrapper, ClippedObservationWrapper],
}
env = RLlibSingleAgentWrapper(env_config)
```

### 7.3 RLlib Multi-Agent Pipeline

**Requirement:** `central_agent=False`

```python
env_config = {
    'env_kwargs': {'schema': '...', 'central_agent': False},
    'wrappers': [ClippedObservationWrapper],
}
env = RLlibMultiAgentEnv(env_config)
# Agent IDs: 'agent_0', 'agent_1', ...
```

### 7.4 NormalizedObservationWrapper Details

- `hour`, `day_type`, `month` → transformed to sin/cos pairs (2 values each)
- All observations → min-max scaled to `[0, 1]`
- Updates `observation_space` and `observation_names` accordingly
- When `central_agent=True`, shared observations appear only once

---

## 8. Boilerplate: Stable Baselines3

```python
from stable_baselines3 import SAC as SB3_SAC
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import (
    NormalizedObservationWrapper,
    ClippedObservationWrapper,
    StableBaselines3Wrapper,
)

# 1. Create environment — central_agent MUST be True for SB3
env = CityLearnEnv(
    schema='baeda_3dem',
    central_agent=True,
    buildings=[0, 1],
    active_observations=[
        'month', 'hour', 'day_type',
        'outdoor_dry_bulb_temperature',
        'non_shiftable_load', 'solar_generation',
        'electrical_storage_soc', 'net_electricity_consumption',
    ],
    active_actions=['electrical_storage'],
    reward_function='citylearn.reward_function.IndependentSACReward',
    random_seed=42,
)

# 2. Wrap for SB3 compatibility
env = NormalizedObservationWrapper(env)
env = ClippedObservationWrapper(env)
env = StableBaselines3Wrapper(env)

# 3. Train
model = SB3_SAC(
    'MlpPolicy',
    env,
    verbose=1,
    learning_starts=env.unwrapped.time_steps,  # fill replay buffer first
    seed=42,
)
model.learn(total_timesteps=env.unwrapped.time_steps * 3)

# 4. Evaluate
obs, info = env.reset()
terminated = truncated = False
while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

# 5. Extract KPIs
kpis = env.unwrapped.evaluate()
print(kpis.pivot(index='cost_function', columns='name', values='value'))
```

---

## 9. Boilerplate: RLlib

### Single-Agent (Central)

```python
from ray.rllib.algorithms.sac import SACConfig
from citylearn.wrappers import (
    NormalizedObservationWrapper,
    ClippedObservationWrapper,
    RLlibSingleAgentWrapper,
)

env_config = {
    'env_kwargs': {
        'schema': 'baeda_3dem',
        'central_agent': True,
        'buildings': [0, 1],
        'random_seed': 42,
    },
    'wrappers': [NormalizedObservationWrapper, ClippedObservationWrapper],
}

config = (
    SACConfig()
    .environment(env=RLlibSingleAgentWrapper, env_config=env_config)
    .framework('torch')
    .training(lr=3e-4)
)

algo = config.build()
for i in range(10):
    result = algo.train()
    print(f"Episode {i}: reward_mean={result['episode_reward_mean']:.2f}")
```

### Multi-Agent (Decentralized)

```python
from ray.rllib.algorithms.ppo import PPOConfig
from citylearn.wrappers import (
    ClippedObservationWrapper,
    RLlibMultiAgentEnv,
)

num_buildings = 3
env_config = {
    'env_kwargs': {
        'schema': 'baeda_3dem',
        'central_agent': False,
        'buildings': list(range(num_buildings)),
    },
    'wrappers': [ClippedObservationWrapper],
}

# Build a temp env to read spaces
tmp_env = RLlibMultiAgentEnv(env_config)
policies = {
    f'agent_{i}': (None, tmp_env.observation_space[f'agent_{i}'],
                   tmp_env.action_space[f'agent_{i}'], {})
    for i in range(num_buildings)
}
tmp_env.close()

config = (
    PPOConfig()
    .environment(env=RLlibMultiAgentEnv, env_config=env_config)
    .multi_agent(policies=policies, policy_mapping_fn=lambda aid, **kw: aid)
    .framework('torch')
)

algo = config.build()
for i in range(10):
    result = algo.train()
```

---

## 10. Built-in Agent Training Pattern

```python
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.sac import SAC

env = CityLearnEnv(schema='baeda_3dem', central_agent=False)
agent = SAC(env=env, lr=3e-4, batch_size=256, discount=0.99)

# Train for 3 episodes; last episode uses deterministic policy
agent.learn(episodes=3, deterministic_finish=True)

# KPIs after training
kpis = env.evaluate()
```

---

## 11. Agent Hierarchy Quick Reference

| Agent | Parent | Learns? | Notes |
|-------|--------|---------|-------|
| `Agent` | — | No | Random actions (base class) |
| `BaselineAgent` | Agent | No | Deactivates all actions |
| `RBC` | Agent | No | Rule-based |
| `HourRBC` | RBC | No | Hour-of-day action map |
| `BasicRBC` | HourRBC | No | Preset thermal strategy |
| `OptimizedRBC` | BasicRBC | No | Tuned thermal strategy |
| `BasicBatteryRBC` | BasicRBC | No | Solar battery strategy |
| `TabularQLearning` | Agent | Yes | Q-table, discrete spaces only |
| `RLC` | Agent | Yes | Base RL controller (PyTorch) |
| `SAC` | RLC | Yes | Soft Actor-Critic |
| `SACRBC` | SAC | Yes | SAC + RBC exploration phase |
| `MARLISA` | SAC | Yes | Multi-agent coordinated RL |
| `MARLISARBC` | MARLISA | Yes | MARLISA + RBC exploration |

---

## 12. Schema Structure (JSON)

```json
{
    "root_directory": "path/to/data",
    "central_agent": false,
    "simulation_start_time_step": 0,
    "simulation_end_time_step": 8759,
    "episode_time_steps": 8760,
    "seconds_per_time_step": 3600,
    "observations": {
        "month": {"active": true, "shared_in_central_agent": true},
        "hour": {"active": true, "shared_in_central_agent": true},
        "outdoor_dry_bulb_temperature": {"active": true, "shared_in_central_agent": true},
        "electrical_storage_soc": {"active": true, "shared_in_central_agent": false},
        "net_electricity_consumption": {"active": true, "shared_in_central_agent": false}
    },
    "actions": {
        "electrical_storage": {"active": true},
        "cooling_storage": {"active": false}
    },
    "reward_function": {
        "type": "citylearn.reward_function.IndependentSACReward",
        "attributes": {}
    },
    "agent": {
        "type": "citylearn.agents.sac.SAC",
        "attributes": {"lr": 0.0003}
    },
    "buildings": {
        "Building_1": {
            "include": true,
            "energy_simulation": "Building_1.csv",
            "weather": "weather.csv",
            "carbon_intensity": "carbon_intensity.csv",
            "pricing": "pricing.csv",
            "type": "citylearn.building.Building",
            "cooling_device": {
                "type": "citylearn.energy_model.HeatPump",
                "autosize": true,
                "attributes": {"nominal_power": 10.0}
            },
            "electrical_storage": {
                "type": "citylearn.energy_model.Battery",
                "autosize": false,
                "attributes": {"capacity": 10.0}
            },
            "pv": {
                "type": "citylearn.energy_model.PV",
                "autosize": true,
                "attributes": {"nominal_power": 8.0}
            }
        }
    }
}
```

---

## 13. Dataset CSV File Formats

**Building CSV** (`Building_1.csv`): Per-timestep building data.
Key columns: `month`, `day_type`, `hour`, `indoor_dry_bulb_temperature_cooling_set_point`, `indoor_dry_bulb_temperature_heating_set_point`, `non_shiftable_load`, `solar_generation`, `power_outage`

**Weather CSV** (`weather.csv`): Per-timestep weather data.
Columns: `outdoor_dry_bulb_temperature`, `outdoor_relative_humidity`, `diffuse_solar_irradiance`, `direct_solar_irradiance` (plus `_predicted_6h/12h/24h` variants)

**Carbon Intensity CSV** (`carbon_intensity.csv`): Single column `carbon_intensity` (kgCO₂/kWh)

**Pricing CSV** (`pricing.csv`): Single column `electricity_pricing` ($/kWh, plus `_predicted_6h/12h/24h` variants)

---

## 14. Common Pitfalls

1. **SB3 without `central_agent=True`**: `StableBaselines3Wrapper` expects a single observation/action space. It will fail silently or error if central_agent is False.
2. **Forgetting wrappers**: Raw CityLearnEnv returns `List[List[float]]` which SB3/RLlib cannot handle. Always apply the appropriate composite wrapper.
3. **Action bounds**: Actions must be in `[-1, 1]`. Out-of-range actions are clipped internally but may produce unexpected behavior.
4. **`env.unwrapped`**: After wrapping, use `env.unwrapped` to access CityLearnEnv properties like `evaluate()`, `time_steps`, `buildings`.
5. **Episode length**: Default is 8760 timesteps (1 year, hourly). Set `learning_starts` in SB3 to at least 1 episode length to fill the replay buffer before learning.
6. **Observation normalization**: Always apply `NormalizedObservationWrapper` before training. Raw observations have vastly different scales.
7. **MARLISA requires `central_agent=False`**: It is a decentralized-coordinated algorithm.
8. **Reward function metadata**: Custom reward functions receive `env_metadata` in `__init__`, not the env itself. Access `self.central_agent` (bool) from metadata.
