# CityLearn Environment Insights & Best Practices

This document summarizes critical mechanics, bugs, and best practices for developing reinforcement learning or LLM-based controllers in CityLearn, based on recent experiments.

## 1. Observation State Bug & Workaround
- **The Bug (CityLearn 2.5.x ONLY)**: The raw observation vector returned by `env.step()` reads `electrical_storage_soc` and `net_electricity_consumption` at index `self.time_step` of their respective arrays — but at that point the slot has not yet been written (it's the next-step initial value, typically 0). The agent therefore sees stale (≈0) values for these two fields. Verified directly in `citylearn==2.5.0`'s `building.py`:
  ```python
  'electrical_storage_soc': self.electrical_storage.soc[self.time_step],          # wrong: slot not yet written
  'net_electricity_consumption': self.net_electricity_consumption[self.time_step], # same
  ```
- **Fixed in CityLearn 2.6.0b2+**: the `BuildingOpsService.get_observations_data()` helper now uses
  ```python
  endogenous_t = t if include_all else max(t - 1, 0)
  ```
  so for the agent-facing call (`include_all=False`) it reads `soc[t-1]` / `net[t-1]` — the just-realised values. Project is pinned to 2.6.0b2 everywhere, so any agent that consumes the obs vector (including the bundled CityLearn SAC) sees correct SoC and net.
- **`snapshot_state` is still recommended** even on 2.6+: it always reads from building objects directly, so it's version-independent and makes the data path explicit when prompting an LLM. It is also slightly more flexible (you can choose which timestep to read).
- **Formatting Note**: Be aware of NumPy 2.0+ scalar formatting (e.g., `np.int32(12)`) when parsing raw observations directly.
- **Reference**: CityLearn issue [#37](https://github.com/intelligent-environments-lab/CityLearn/issues/37) is a different SoC bug (capacity-after-degradation divisor). The obs-vector indexing bug above isn't filed upstream — it was identified by inspecting installed source.

## 2. Variable Scales & Meanings
- **`non_shiftable_load`**: Fixed building demand in kWh per hour. Typical range: ~0.1 to 7.0 kWh.
- **`net_electricity_consumption`**: Grid exchange in kWh. Positive (+) means import (pull from grid), negative (-) means export (push to grid).
- **`electrical_storage_soc`**: State of Charge. Range: [0.0, 1.0] (0% to 100%).

## 3. Solar Generation (PV) Mechanics
- **Raw tape is a capacity factor, not kWh**: The raw `energy_simulation.solar_generation` tape is a W/kW capacity factor (range 0–976), **not** usable energy. `snapshot_state` emits the **capacity factor** = `raw / 1000` — generation as a fraction of nameplate (standard-test-condition) output, in [0, ~1]. The /1000 reference is universal, so the `solar_bucket` thresholds are absolute numbers needing no per-building calibration — they apply to any building or dataset, like the price/carbon thresholds. (Per-building panel orientation/shading still shifts the distribution ~38%; that is real physics, not noise.) See `notebooks/01_5_bin_design.ipynb` for the scale comparison and why own-peak normalisation was rejected.
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
Instead of feeding raw continuous numbers to an LLM agent, discretizing key exogenous variables drastically improves reasoning. Thresholds are data-driven from the deterministic full-year 2022 tape — see `notebooks/01_5_bin_design.ipynb` for the derivation and justification:
- **Electricity Price**: `LOW` vs `PEAK`, split at 0.30 $/kWh (the empty gap in the 5-level discrete tariff). ~79 / 21 %.
- **Carbon Intensity**: `LOW` / `MID` / `HIGH`, terciles at 0.14 / 0.17 kgCO₂/kWh. ~34 / 33 / 33 %.
- **Solar Generation**: `NONE` / `LOW` / `MID` / `HIGH`, on the capacity factor (`raw / 1000`), edges 0 / 0.17 / 0.50 (pooled daytime terciles). ~51 / 16 / 16 / 16 %.
- **Rendering**: price, carbon and solar appear in the prompt as the bucket **label only** (`price=LOW`, not `price=0.220 (LOW)`) — the raw number is the continuous value the discretisation deliberately abstracts away. `SoC`, `load` and `last_net` are instead shown as raw numbers: they are energy-state quantities the action is quantitatively sized against (see nb 01.5 § 7).
- **Rationale**: This allows the prompt to state simple categorical rules (e.g., "If price is PEAK and SoC is high, discharge aggressively (−0.6 to −1.0). If price is LOW, trickle charge (+0.2).") which LLMs follow much more reliably than continuous numerical thresholds. Equal-mass bins ensure every label is actually seen — a bin that absorbs ~all the mass carries no signal.
