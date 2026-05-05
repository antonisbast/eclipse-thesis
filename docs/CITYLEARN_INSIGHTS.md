# CityLearn Environment Insights & Best Practices

This document summarizes critical mechanics, bugs, and best practices for developing reinforcement learning or LLM-based controllers in CityLearn, based on recent experiments.

## 1. Observation State Bug & Workaround
- **The Bug**: The raw observation vector returned by `env.step()` can fail to correctly update `electrical_storage_soc` and `net_electricity_consumption` due to a known CityLearn bug.
- **The Fix**: Bypass the observation vector by reading directly from the building objects (e.g., `building.electrical_storage.soc[t]`, `building.net_electricity_consumption[t]`). A custom `snapshot_state(env)` function is highly recommended.
- **Formatting Note**: Be aware of NumPy 2.0+ scalar formatting (e.g., `np.int32(12)`) when parsing raw observations directly.

## 2. Variable Scales & Meanings
- **`non_shiftable_load`**: Fixed building demand in kWh per hour. Typical range: ~0.1 to 7.0 kWh.
- **`net_electricity_consumption`**: Grid exchange in kWh. Positive (+) means import (pull from grid), negative (-) means export (push to grid).
- **`electrical_storage_soc`**: State of Charge. Range: [0.0, 1.0] (0% to 100%).

## 3. Solar Generation (PV) Mechanics
- **Irradiance, not kWh**: The `solar_generation` observation provides raw solar irradiance (W/m²), not directly usable energy in kWh. High values (e.g., >100) mean strong sun.
- **Energy Balancing**: Solar energy first covers the building's `non_shiftable_load`. Any remaining energy is used to charge the battery (if a charge action is applied). Any final excess is automatically exported to the grid.
- **Strategy**: To prevent unnecessary grid exports, use small charge actions during high solar generation hours to capture the free energy.

## 4. Battery Dynamics & Action Space
Actions are defined in the range `[-1.0, 1.0]` (Positive = Charge, Negative = Discharge).

- **Charging (+): Highly Unconstrained**
  - Charging is extremely fast. An action of `+1.0` can fill ~70% of the battery capacity in a single hour.
  - It pulls heavily from the grid (e.g., ~5.2 kWh in one step).
  - **Best Practice**: Use small fractional actions (e.g., `+0.1` to `+0.3`) to charge slowly. A `+1.0` action will likely cause a massive district-wide demand spike, severely penalizing the peak load KPI.
  
- **Discharging (-): Physically Constrained**
  - Discharging is physically capped by the battery's nominal power (e.g., max ~1.5 kWh discharged per hour).
  - Discharged energy covers the building load first; excess is exported to the grid.
  - **Best Practice**: Since it is capped by the hardware, using `-1.0` during PEAK price/carbon hours is generally safe and maximizes savings without extreme negative spikes.

## 5. LLM Agent Prompting Strategy (Categorical Binning)
Instead of feeding raw continuous numbers to an LLM agent, discretizing key exogenous variables drastically improves reasoning:
- **Electricity Price**: `LOW` vs `PEAK`.
- **Carbon Intensity**: `LOW`, `MID`, `HIGH`, `PEAK`.
- **Solar Generation**: `NONE`, `LOW`, `HIGH`.
- **Rationale**: This allows the prompt to state simple categorical rules (e.g., "If price is PEAK, discharge (-1.0). If price is LOW, trickle charge (+0.2).") which LLMs follow much more reliably than continuous numerical thresholds.
