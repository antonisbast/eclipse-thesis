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
Actions are in `[-1.0, 1.0]` and represent **requested energy as a fraction of battery capacity** (positive = charge, negative = discharge). Per step the SoC change is:

```
ΔSoC ≈ action × capacity   (clipped by nominal power, available SoC, and round-trip efficiency)
```

Charge and discharge are **roughly symmetric**. Measured on `citylearn_challenge_2022_phase_all` in `notebooks/03_slm_colab.ipynb`:

| `|action|` | ΔSoC per step (charge) | ΔSoC per step (discharge) |
|------------|------------------------|---------------------------|
| 0.20       | +14 pp                 | −17 pp                    |
| 0.40       | +29 pp                 | −33 pp                    |
| 0.60       | —                      | −41 pp (sub-linear, efficiency loss) |
| 1.00       | +70 pp                 | (extrapolated ~−70 pp)    |

ΔSoC is **action-driven, not load-driven**. Buildings with very different `non_shiftable_load` show identical SoC drops under the same discharge action — load does not gate how much energy leaves the battery.

- **Charging (+):** Pulls from solar first, then the grid. A `+1.0` action pulls several kWh in a single step (~5 kWh on this schema) — if all 6 buildings do it concurrently, district demand spikes and `daily_peak_average` blows up. Use small actions (`+0.1` to `+0.3`) to absorb solar without grid spikes.

- **Discharging (−):** **Not** asymmetrically capped. A `−1.0` action from a full battery discharges roughly the same magnitude as `+1.0` charges. Where load matters is *downstream* of the battery, in the grid balance:

  ```
  net_electricity_consumption = non_shiftable_load − solar_generation
                                + battery_charge_power − battery_discharge_power
  ```

  If `discharge > load − solar`, surplus is exported as negative net consumption. On the 2022 phase tariff, exports do not earn back what imports cost, so over-discharging is wasteful but not physically blocked.

- **Best Practice — discharge sizing:** During PEAK, target `|action|` such that battery output roughly matches `load − solar`. A useful heuristic: discharge ~`-0.4` to `-0.6` for typical loads, escalating toward `-1.0` only when load is high *and* SoC is high. Avoid timid `-0.2`-only policies — they leave most of the stored energy unused at the moment it's most valuable.

- **Best Practice — SoC bounds:** Charging at SoC ≥ 0.95 wastes the request (clipped by ceiling); discharging at SoC ≤ 0.05 likewise. Hard-clip in the action parser rather than relying on the LLM to obey "never charge above 90%".

## 5. LLM Agent Prompting Strategy (Categorical Binning)
Instead of feeding raw continuous numbers to an LLM agent, discretizing key exogenous variables drastically improves reasoning:
- **Electricity Price**: `LOW` vs `PEAK`.
- **Carbon Intensity**: `LOW`, `MID`, `HIGH`, `PEAK`.
- **Solar Generation**: `NONE`, `LOW`, `HIGH`.
- **Rationale**: This allows the prompt to state simple categorical rules (e.g., "If price is PEAK and SoC is high, discharge aggressively (−0.6 to −1.0). If price is LOW, trickle charge (+0.2).") which LLMs follow much more reliably than continuous numerical thresholds.
