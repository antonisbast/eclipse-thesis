# ECLIPSE Thesis — Status Snapshot for RL-Phase Research

**Date:** 2026-05-13  
**Purpose:** Hand-off document for an external Claude/LLM session to research and propose the **online RL phase (PPO or GRPO) on top of the SFT'd SLM**.  
**Author:** Antonios Bastoulis (MSc, AIDL @ UniWA) · Supervisor: Dr. Panagiotis Kasnesis

---

## 1. Thesis goal in one paragraph

Investigate whether **Small Language Model (SLM)-based agents** can effectively manage building energy in **CityLearn 2.6**, and whether **two such agents** can cooperate under partial observability with **no explicit communication channel** (action-only coordination).

**Primary research questions:**
1. How does an SLM agent compare to rule-based and RL baselines (SAC) on energy KPIs?
2. How well does it generalize to unseen buildings and seasons?
3. Can the SLM provide interpretable natural-language rationales (CoT) for its control actions?

**Secondary research question:** Can two SLM agents develop **implicit coordination** through behavioral observation alone, without explicit communication?

---

## 2. Four-phase plan

| Phase | Goal | Status |
|-------|------|--------|
| 1 — Expert Baselines | Train SAC; benchmark RBC vs no-op vs LLM-as-policy | ✅ Done |
| 2 — SLM Integration | Translate observations to text; parse SLM actions; zero-shot eval | ✅ Done |
| 3 — SLM Fine-Tuning | LoRA SFT on SAC distillation, then **online RL** (PPO / GRPO) | 🟡 SFT in progress; **RL is what this doc is asking about** |
| 4 — Multi-Agent Deployment | Two SLM agents, partial observability, no comms, joint reward | ⏳ Pending |

---

## 3. Environment — exactly what the agent sees and does

### Domain
- **CityLearn 2.6.0b2** (Gymnasium-compatible), dataset `citylearn_challenge_2022_phase_all`.
- 6 buildings indexed `[0..5]` available; subset used per phase (see below).
- 8,760 hourly steps per episode (1 year). Episode terminates at t=8759.
- `central_agent=False` everywhere — env returns **list of 6 per-building obs / actions / rewards**. Policy count is independent of this flag.

### Building split (current convention)
| Constant | Indices | Use |
|---|---|---|
| `TRAINING_BUILDINGS` | `[0, 1, 2]` | Phase 3 train + in-dist eval; also Phase 4 agent α |
| `HELDOUT_BUILDINGS`  | `[3, 4, 5]` | Unseen-buildings generalization test; also Phase 4 agent β |
| `BUILDINGS`          | `[0..5]`     | Full district (Phase 4 dual-agent rollout) |
| `UNSEEN_BUILDINGS`   | `[6..11]`    | Out-of-distribution gen test (other 2022-dataset buildings) |

### Observation set (real-time only, **no oracle forecasts** by design)
9 variables per building, from `snapshot_state(env)`:
```
month, day_type, hour,
electrical_storage_soc,            ← 0..1
electricity_pricing,                ← $/kWh, mostly binary 0.22/0.54
carbon_intensity,                   ← kgCO₂/kWh
solar_generation,                   ← kWh
non_shiftable_load,                 ← kWh
net_electricity_consumption_last    ← kWh, last step
```

The CityLearn dataset exposes oracle forecasts (+6h/+12h price, +6h solar) but we **deliberately exclude them** — they're perfect look-ahead reads from the simulation tape, not realistic at deployment. The agent must anticipate from `hour`/`day_type`/trend alone.

### Action space
- Continuous: `[-1.0, +1.0]` per building, controlling battery charge (+) or discharge (-).
- Battery: ~6.4 kWh capacity, 5 kW nominal, 90% efficiency.
- **SLM action vocabulary (11 buckets, 20% steps):** `CHARGE_100, ..., CHARGE_20, IDLE, DISCHARGE_20, ..., DISCHARGE_100`. Mapped to floats by `src.sft.action_to_token` / `parse_actions`.
- Battery dynamics gotcha: charge and discharge are roughly symmetric in ΔSoC. There is **no asymmetric hardware cap** on discharge. `±0.2` ≈ ±15 pp/step; `±1.0` ≈ ±70 pp/step.

### Reward
- **`MERLINReward`** (default): SoC-aware net-consumption reward (Nweye et al., 2024, *Applied Energy*).
  `reward_i = −(1 + sign(net_i)·SoC_i) · |net_i|` per building.
  Returns a **list of 6 per-building rewards**. The per-building sum is what we typically train against.
- Alternative: `EcoPeakBatteryReward` (cost+carbon+peak with normalization). Not used in distillation.
- All rewards are negative (costs); lower magnitude = better.

### KPIs (CityLearn 2.6 `evaluate_v2()`)
Ratios to a no-battery baseline (1.0 = no improvement, < 1.0 = better):
- `C` = district cost ratio
- `G` = district carbon emissions ratio
- `R` = ramping (grid-shape quality)
- `1-L` = inverse load factor (grid-shape quality)
- `Phase I = (C + G) / 2`  ← **the primary thesis metric**
- `Combined = (C + G + D) / 3`, where `D = (R + (1-L))/2`
- Plus ZNE ratio and self-consumption ratio

---

## 4. What's currently working (artifacts + numbers ready for RL)

### Notebooks
- `01_env_setup.ipynb` — env factory + SAC/RBC baselines on 6 buildings. Pedagogical (analytical defs live here).
- `02_llm_policy.ipynb` — single-agent zero-shot via remote APIs (Anthropic, DeepSeek, Kimi, OpenAI) on `TRAINING_BUILDINGS` for 300 steps at `t=3624` (summer week).
- `03_slm_colab.ipynb` — single-agent zero-shot via local SLMs on Colab GPU (Gemma, Qwen, Phi, Llama). Same window.
- `04_sac_distill_dataset.ipynb` — SAC teacher trains on full 6-bldg district, then dumps full-year (state_text, action_token) pairs with two 3-bldg slices per env step. **Produced JSONL: 17,520 rows, ~10 MB.**
- `05_sft_gemma_colab.ipynb` — LoRA SFT on Unsloth Gemma-4 E4B using the JSONL. Completion-only loss via TRL `<0.20` collator. **About to run.**
- `06_eval_generalization.ipynb` — load LoRA → eval across (season × building-subset) grid → generalisation gap.

### `src/` layout (single source of truth)
- `src/env.py` — `make_env`, `snapshot_state`, `MERLINReward`, `EcoPeakBatteryReward`, building-set constants, `OBSERVATIONS`.
- `src/agent.py` — buckets, `render_state`, `parse_actions`, **`ACTION_RE`**, **canonical CoT prompt** `make_minimal_prompt`, `make_policy_llm`, reference policies (`policy_noop`, `policy_random`, `policy_rbc`).
- `src/providers.py` — `APIProvider` (Anthropic / OpenAI-compat), `LocalHFProvider` (HF causal LMs on GPU).
- `src/rollout.py` — `run_policy` (single-agent), `run_policy_dual_agent` (**Phase 4 only**), summaries.
- `src/sft.py` — `action_to_token`, `format_action_block`, `make_sft_prompt` (no-CoT, **SFT-only variant**), `dump_sac_trajectory_jsonl`, `filter_uninformative_rows`. Re-exports state/parse helpers from `src.agent` (one source of truth).
- `src/eval.py` — `evaluate(env, label) → EvalResult`, `comparison_table`, `challenge_score`, `zne_metric`, `generalisation_gap`. All `evaluate_v2()`-based (CityLearn 2.6).

### Design decisions already taken
1. **Single agent through Phase 3, dual-agent only at Phase 4 deployment.** One SLM call per step during SFT *and* RL (halves inference cost vs dual-agent training). Same fine-tuned LoRA loads into both Phase 4 agents.
2. **Group-centralized over 3 buildings** (one policy sees `TRAINING_BUILDINGS`, emits 3 action tokens in one prompt). Intra-group coordination is the SLM's strength; the Phase 4 research question is about *inter-group* implicit coordination.
3. **Building-agnostic 3-bldg policy** via 2-slice distillation (`[0,1,2]` and `[3,4,5]` both go into the dataset). LoRA hot-swaps into Phase 4 agents α and β without retraining.
4. **Two prompts on purpose:**
   - `src.agent.make_minimal_prompt(n=3)` — **canonical CoT prompt** with `<thought>` block, used at zero-shot and at RL/inference.
   - `src.sft.make_sft_prompt(n=3)` — **SFT-only no-CoT variant** because the SAC teacher provides no rationales. **Drift between SFT-time and eval-time prompts caused a historical KPI blowup** — they must match at SFT-eval time.
5. **No oracle forecasts** — see §3.
6. **`central_agent=False`** everywhere; joint reward (Phase 4) is computed in the rollout loop by summing the per-building reward list.

### Validation gate before any Phase 4 work
The fine-tuned single SLM must reach **≥70% of SAC Phase I** on `TRAINING_BUILDINGS` (in-distribution).  
Generalization gap (`TRAINING → HELDOUT`) reported separately for RQ2.

---

## 5. Compute budget

| Env | Hardware | Use |
|---|---|---|
| MacBook Air | Apple Silicon, CPU | Local dev, debug, dataset gen, analysis |
| Google Colab Pro | T4 / V100 / A100 | SAC training, SFT, **RL** |
| NVIDIA DGX Spark ×2 | (future) | Phase 4 distributed deployment |

Constraints:
- Colab sessions can disconnect → **must checkpoint to Drive every N updates**.
- SLM inference on T4 (Gemma-4 E4B, 4-bit) is ~5–10 sec/call. **Budget RL accordingly** — see §6.

---

## 6. What we need help with: the RL phase

We need to choose and design an **online RL stage on top of the SFT'd SLM** that:

1. **Improves on SFT KPIs** (the SAC teacher had only 20 episodes, so it's a weak teacher — there's headroom).
2. **Respects the inference budget** (each rollout step = 1 SLM forward pass; T4 ~5s/call; one CityLearn year ≈ 8,760 steps → infeasible at full episode length).
3. **Preserves the base SLM** — only LoRA weights update; **base model frozen**.
4. **KL-regularizes against the reference model** to prevent catastrophic forgetting.
5. **Works with our text-action vocabulary** (parse failures default to 0.0 / IDLE, see `src.sft.parse_actions`).
6. **Is hardware-compatible with TRL / Unsloth on Colab** (we already have a working LoRA + 4-bit pipeline with completion-only loss).

### Specific questions

1. **PPO vs GRPO** — pros/cons for this setup. GRPO needs multiple candidate generations per observation (expensive at 5s/call); PPO needs a value head (extra LoRA adapter?).
2. **Reward shaping** — should we train against the raw `MERLINReward` sum, or against a KPI-derived signal (e.g. `−Phase I`)? The KPI is only well-defined at episode end; the per-step MERLIN reward is dense but noisy.
3. **Episode length for RL** — full year is too long; what's a defensible sliding-window scheme (e.g. random 7-day windows? seasonal-balanced sampling?) that doesn't bias the policy toward summer-only behavior?
4. **Action discretization at RL time** — should we keep the 11-bucket vocabulary (matches SFT, but constrains the policy) or sample logits over a finer grid? The CityLearn action space is continuous in `[-1, 1]`; SAC outputs floats. SFT teaches buckets.
5. **CoT during RL** — the canonical prompt asks for `<thought>` blocks. Does GRPO/PPO benefit from including the rationale tokens in the trajectory, or should we mask them out of the loss?
6. **KL coefficient + baseline** — sensible starting values for a 4B-parameter LoRA-tuned SLM? Recipes from RLHF (e.g. β=0.01–0.1) — do they transfer?
7. **Validation gate logistics** — when does the RL loop check the ≥70%-of-SAC gate? Every N updates? Should we keep the best checkpoint by validation KPI rather than by training reward?
8. **Phase 4 implications** — anything in the RL recipe that would prevent the trained LoRA from being hot-loaded into two agents at deployment? (E.g. if we use a value head, do both agents share it or need independent ones?)
9. **Wandb + checkpoint cadence** — recommended logging schema for an RL run on Colab (intermittent disconnects).
10. **Library choice** — TRL `PPOTrainer` / `GRPOTrainer` (we already use TRL for SFT, so dependency footprint is unchanged) vs CleanRL-style hand-rolled (more control, more code). What does the recent literature do for similar text-action RL setups?

### What to *not* propose
- **Per-token RL with PPO on raw text** (e.g. RLHF-style with a reward model). Our reward comes from CityLearn env, not a learned model. Treat the entire generated action block as the "action."
- **Replacing the LoRA-SFT'd base.** SFT is the prior; RL must stay close.
- **Multi-agent RL** at this stage. We're explicitly single-agent until Phase 4.
- **Curriculum / dataset augmentation** that requires re-distilling — we want online RL on the existing SFT'd checkpoint.

---

## 7. Concrete deliverables we'd like back

1. **Recommendation: PPO or GRPO**, with one-paragraph justification given §6.
2. **A reference pipeline sketch**: data flow (env → text → SLM → text → action → env → reward → update), naming the TRL classes / hyperparameters we'd configure.
3. **A scaffolded `notebooks/07_rl_<algo>_colab.ipynb`** outline (cell-level, not code) — what each cell does, in what order, and where the Drive checkpoint cadence sits.
4. **A `src/rl.py` module sketch** — what helpers belong there vs. in the notebook. We already have `src/sft.py` as the model for how a phase-specific helper module looks.
5. **Reward-shaping recommendation** (per-step MERLIN vs episode-end KPI vs a hybrid), with the trade-offs spelled out.
6. **A go/no-go checklist** before kicking off Phase 4 (what evidence convinces us the RL'd LoRA is ready to drop into two agents).

---

## 8. Useful citations (already used in the codebase)

- Nweye et al. (2024) — *Merlin: Multi-agent reinforcement learning for energy systems on real-world data.* Applied Energy 358, 121958. The reward function is from here.
- CityLearn 2022 Challenge specification (KPI definitions, Appendix A).
- Unsloth + TRL for the SFT pipeline.

---

## 9. Repo layout (for orientation)

```
eclipse-thesis/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── docs/
│   ├── CONTEXT.md
│   ├── PROGRESS.md             ← living changelog (read for full history)
│   ├── CITYLEARN_API.md
│   └── CITYLEARN_INSIGHTS.md
├── notebooks/
│   ├── 01_env_setup.ipynb
│   ├── 02_llm_policy.ipynb
│   ├── 03_slm_colab.ipynb
│   ├── 04_sac_distill_dataset.ipynb
│   ├── 05_sft_gemma_colab.ipynb
│   ├── 06_eval_generalization.ipynb
│   └── artifacts/
│       └── sft_datasets/
│           └── sac_merlin_distill_20260512_212359.jsonl   ← 17,520 rows, 10 MB
├── src/
│   ├── env.py
│   ├── agent.py
│   ├── providers.py
│   ├── rollout.py
│   ├── sft.py
│   └── eval.py
└── configs/
    └── experiment.yaml
```

GitHub: https://github.com/antonisbast/eclipse-thesis

---

*End of handoff. Treat this as the working ground truth as of 2026-05-13. Any contradiction with older docs/CONTEXT.md content should defer to this file.*
