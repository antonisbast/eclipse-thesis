# Thesis Status — ECLIPSE / SLM-Based Energy Management

**MSc Thesis · University of West Attica · AIDL program**
Student: Antonios Bastoulis · Supervisor: Dr. Panagiotis Kasnesis
Last updated: 2026-05-08

---

## 1. What this thesis is

A subset of the ECLIPSE project (Edge Coordination via Learning In Partially observable
Shared Environments). The investigation: can **Small Language Model (SLM)-based agents**
manage building energy in CityLearn, and can two such agents — one per edge node —
**cooperate under partial observability without explicit communication**?

Hardware target: 2× NVIDIA DGX Spark (academic grant) for the final multi-agent
deployment. Development on MacBook Air, training on Google Colab Pro.

### Research questions

**Primary:**
1. How does an SLM-based agent compare to rule-based and RL (SAC) baselines on energy KPIs?
2. How well does the SLM agent generalize to unseen buildings and weather conditions?
3. Can the SLM produce interpretable natural-language rationales for its control actions?

**Secondary:**
- Can two SLM agents develop implicit coordination through behavioral observation
  alone, with no explicit communication channel (Action-Only condition)?

### Four-phase plan

| Phase | Goal | Compute |
|-------|------|---------|
| 1 — Expert Baselines | RBC, no-op, LLM-as-policy, SAC | Local + Colab GPU |
| 2 — SLM Integration | Obs→text, action parsing, local SLM rollouts | Colab GPU |
| 3 — SLM Fine-Tuning | LoRA / GRPO online RL on CityLearn rollouts | Colab / DGX Spark |
| 4 — Multi-Agent Deployment | 2 agents, partial obs, joint reward, no comms | DGX Spark ×2 |

### Key design decisions (locked-in)

- **Online RL, not frozen deployment.** Agents continue learning during evaluation.
- **LoRA-only updates.** Base SLM weights stay frozen; only adapters are trained.
- **KL penalty against reference model.** Prevents catastrophic forgetting.
- **GRPO over PPO** for online RL — no value network, works on full generations.
- **Joint reward.** Both agents share the same reward at each step → cooperative incentive
  without an explicit channel.
- **Partial observability split.** Agent α sees buildings {0,1,2}, Agent β sees {3,4,5}.
  Never crossed.
- **Action-Only condition.** No language channel between agents — coordination emerges
  through environment state alone.
- **Validation gate before Phase 4:** single-agent SLM must reach ≥70% of SAC expert
  performance.

---

## 2. Environment & dataset

- **Simulator:** CityLearn v2 (Gymnasium-compatible). Currently using **2.6.0b2** in the
  active `.venv312`; `.venv` (Python 3.13) has 2.5.0 as a fallback.
- **Dataset:** `citylearn_challenge_2022_phase_all`, 6 buildings, 1-year hourly trace
  (8 760 steps).
- **Building model:** Solar PV, BESS (~6.4 kWh / 5 kW, 90% efficiency), dynamic loads,
  shared grid with time-of-use pricing.
- **Action:** continuous `[-1, 1]` per building (battery charge/discharge fraction).
- **Reward:** joint negative cost — peak demand + electricity cost + ramping penalty.

### CityLearn quirks we worked around (already documented in `docs/CITYLEARN_INSIGHTS.md`)

- `electrical_storage_soc` in the raw obs vector is bugged (next-step init, always 0).
  → We read SoC directly from `building.electrical_storage.soc[t]` via `snapshot_state()`.
- Battery is **asymmetric**: `+1.0` charge fills ~70% in one step and spikes district
  demand; `-1.0` discharge is hardware-capped at ~1.5 kWh/hr and is therefore safe at
  peak. Our RBC and prompts exploit this.
- In CityLearn 2.6 `env.time_step` advances one past the last valid index after
  termination, and `non_shiftable_load` is shorter than `energy_simulation.*`.
  → `snapshot_state()` clamps each per-array index defensively.

---

## 3. Repository layout

```
eclipse-thesis/
├── CLAUDE.md, README.md, requirements.txt
├── docs/
│   ├── CONTEXT.md                  ← thesis background
│   ├── PROGRESS.md                 ← session-by-session log
│   ├── CITYLEARN_API.md            ← v2 API reference
│   └── CITYLEARN_INSIGHTS.md       ← obs quirks, battery dynamics, prompting tips
├── notebooks/
│   ├── 01_env_setup.ipynb          ← env, RBC/SAC, KPIs (Phase 1)
│   ├── 02_llm_policy.ipynb         ← dual-agent LLM-as-policy, multi-provider
│   ├── 03_slm_colab.ipynb          ← local SLM on Colab GPU
│   └── 04_llm_policy_clean.ipynb   ← earlier reference baseline
├── src/
│   ├── env.py                      ← env factory, reward fns, snapshot_state
│   ├── eval.py                     ← KPIs, comparison_table, generalisation_gap
│   ├── agent.py                    ← buckets, render_state, parse_actions, prompt, policies
│   ├── providers.py                ← APIProvider (remote) + LocalHFProvider (GPU)
│   └── rollout.py                  ← run_policy, run_policy_dual_agent, summaries
├── scripts/                        ← (Phase 1 SAC, Phase 3 fine-tune — TBD)
└── configs/experiment.yaml         ← all hyperparameters
```

---

## 4. What's been done

### Phase 1 — Expert baselines (in progress)

**Done:**
- Sandbox exploration (`sandbox/01–05_*.py`) — env behaviour, obs quirks, PPO smoke test,
  obs-to-text experiments.
- `notebooks/01_env_setup.ipynb` — env factory, RBC/no-op/random baselines, KPI
  evaluation against the CityLearn 2022 challenge metrics.
- `src/env.py` — `make_env()` with `obs_set` (`sac` / `llm`) and `reward_fn` (`merlin` /
  `eco`) switches; custom rewards `MERLINReward`, `EcoPeakBatteryReward`;
  `snapshot_state()` exposing 12 fields per building (9 real-time + 3 forecasts).
- `src/eval.py` — `evaluate()`, `comparison_table()`, `generalisation_gap()`,
  `EvalResult` dataclass; both Phase-I `(C+G)/2` and Combined `(C+G+D)/3` scores.

**Still to do:**
- Train SAC at full scale on Colab (300 episodes × 5 seeds), log to wandb.
- Lock in headline KPIs vs all baselines.

### Phase 2 — SLM integration (groundwork done)

**Done:**
- `notebooks/02_llm_policy.ipynb` — dual-agent (α/β), 4 remote providers tested with
  zero-shot prompting:
  - Anthropic `claude-haiku-4-5`
  - DeepSeek `deepseek-chat`
  - Kimi `kimi-k2.5` (requires `temperature=1`)
  - OpenAI `gpt-5.4-nano` (reasoning family — uses `max_completion_tokens`,
    forced `temperature=1`)
  Each provider lives in its own cell so a hung call can be interrupted without
  losing the others. `LLM_TIMEOUT_S=45 s` enforced via background thread.
- `notebooks/03_slm_colab.ipynb` — local SLM inference on Colab T4. Self-contained, runs
  end-to-end with a single Run-All. Supports 4-bit quantization for ≥7B models.
  Tested with Qwen2.5-1.5B (~2 min/rollout), Qwen3-4B (~5 min), Llama-3-8B (4-bit).
- **Discrete action bins** stabilised: `CHARGE_{20,40,60,80,100}` / `IDLE` /
  `DISCHARGE_{20,40,60,80,100}`. Parsed via `<action building=i>...</action>` regex,
  clipped to `[-1, 1]`, fallback to 0.0 on parse failure.
- **Prompt:** `make_minimal_prompt(n)` — variable semantics, indirect strategy hints,
  brief CoT (`<thought>` block, capped at 15 words). No prescribed rules — SLM
  develops its own strategy.
- **State rendering:** header (month / day / hour / price + bucket / carbon + bucket)
  + forecast line (`price+6h`, `price+12h`, `solar+6h`) + per-building line
  (`SoC%, load, last_net, solar bucket`). Each agent sees buildings renumbered locally
  from B0 — identical structure regardless of which 3 buildings it controls.
- **Code consolidation (today, 2026-05-08):** the inline notebook code was extracted
  into reusable `src/` modules. `notebooks/02` and `notebooks/03` now share:
  - `src/agent.py` — buckets, `render_state`, `parse_actions`, `make_minimal_prompt`,
    `make_policy_llm`, reference policies (`noop`, `random`, `rbc`).
  - `src/providers.py` — `APIProvider` (Anthropic + OpenAI-compat with per-model
    quirks) and `LocalHFProvider` (HF causal LM, Qwen3 thinking-mode disabled, Gemma
    system-role workaround, fallback chat template).
  - `src/rollout.py` — `run_policy`, `run_policy_dual_agent` (with `env_factory` hook
    for the Colab auto-download variant), plus `summarize_district`, `district_kpis`,
    `per_agent_summary`.

**Still to do:**
- Refactor `02` and `03` to import from `src/` instead of inlining (keeps the
  notebooks thin per the project rule).
- Land single-agent + dual-agent SLM results vs SAC baseline.

### Phase 3 — SLM fine-tuning (not started)

Planned: LoRA + GRPO. KL penalty against frozen base. Multiple candidate generations
per observation. Reference policy = best zero-shot or imitation-warmed SLM.

### Phase 4 — Multi-agent deployment (not started)

2× DGX Spark, one SLM per node. NVIDIA TensorRT-LLM + Triton for serving. Joint
reward, partial obs split, Action-Only coordination.

---

## 5. Compute environments

| Environment | Hardware | Use |
|-------------|----------|-----|
| MacBook Air | Apple Silicon, CPU | Local dev, debugging, debug-scale tests, analysis |
| Google Colab Pro | A100 / V100 / T4 | SAC training, SLM fine-tuning, full experiments |
| DGX Spark ×2 | NVIDIA GPU (academic grant) | Phase 4 distributed multi-agent |

**Local Python venvs:**
- `.venv` (Python 3.13.7, citylearn 2.5.0) — older, has gymnasium/openstudio version
  drift.
- **`.venv312` (Python 3.12.13, citylearn 2.6.0b2) — preferred.** All required packages
  installed: `anthropic`, `openai`, `python-dotenv`, `transformers`, `accelerate`,
  `huggingface_hub`, `peft`, `trl`, `wandb`, `stable_baselines3`, `matplotlib`,
  `torch 2.11`. `bitsandbytes` skipped (Colab/Linux only — 4-bit quant).

---

## 6. Open items / next session

1. **Refactor notebooks 02 & 03** to import from `src/agent`, `src/providers`,
   `src/rollout` instead of inlining. Notebooks should stay thin.
2. **Run full SAC on Colab.** 300 episodes × 5 seeds, log to wandb. This is the
   Phase-1 expert benchmark.
3. **Land the headline comparison table:** No-Op / Random / RBC / SAC / LLM (×4
   providers) / SLM, on both the canonical 168- and 300-step windows.
4. **Generalisation gap experiment.** Train/eval on disjoint building subsets and on
   unseen weather slices.
5. **Begin Phase 3 design doc.** GRPO setup, KL coefficient, candidate count, reward
   shaping. Lock the Phase-3 → Phase-4 validation gate (≥70% of SAC).

---

## 7. References inside the repo

- `docs/CONTEXT.md` — fuller thesis background.
- `docs/PROGRESS.md` — session-level changelog (read first, write last).
- `docs/CITYLEARN_API.md` — CityLearn v2 API + boilerplate.
- `docs/CITYLEARN_INSIGHTS.md` — obs quirks, battery dynamics, prompting tips.
- `CLAUDE.md` — rules-of-engagement for AI assistance on this repo.
