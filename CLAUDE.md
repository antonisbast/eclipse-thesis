# ECLIPSE Thesis — SLM-Based Energy Management

## What this project is

MSc thesis at University of West Attica (AIDL program), implementing a subset of the ECLIPSE project.

The core objective is to investigate whether Small Language Model (SLM)-based agents can effectively
manage building energy in CityLearn, and how distributed SLM agents can cooperate under partial
observability without a central orchestrator.

**Primary research questions:**
1. How does the SLM agent compare to rule-based and RL baselines (SAC) on energy KPIs?
2. How well does the SLM agent generalize to unseen buildings and weather conditions?
3. Can the SLM provide interpretable natural-language rationales for its control actions?

**Secondary research question:** Can SLM-based agents develop implicit coordination through
behavioral observation alone, without explicit communication?

Supervisor: Dr. Panagiotis Kasnesis
Student: Antonios Bastoulis

## Current phase

> **UPDATE THIS** after each work session.

**Phase 1 + Phase 2 zero-shot complete.** Notebooks 01 (env, RBC, SAC), 02 (remote-API LLM-as-policy, dual-agent), and 03 (local SLM-as-policy on Colab) are working. Reusable code extracted to `src/`: `env.py`, `agent.py`, `providers.py`, `rollout.py`, `eval.py`.

**In progress — Phase 2→3 transition: SAC→SLM behavior-cloning distillation.** Notebook 04 (`04_sac_distill_dataset.ipynb`) generates `(state_text, action_token)` JSONL from a trained SAC rollout; notebook 05 (`05_sft_gemma_colab.ipynb`) runs LoRA SFT on Gemma in Colab via Unsloth. Pipeline helpers live in `src/sft.py`. **Experiments not yet completed** — dataset generation and fine-tuning runs still pending.

**Design decision (2026-05-12):** Phases 1–3 train a SINGLE group-centralized agent over 3 buildings (`TRAINING_BUILDINGS=[0,1,2]`), not the dual-agent setup of nb 02/03. One SLM call per step during SFT/RL. Phase 4 deployment still uses two agents — the same fine-tuned LoRA loads into both (α on B0–2, β on B3–5) without retraining. SAC teacher remains trained on the full 6-building district (`central_agent=False`, per-building policies); nb 04 slices the rollout into two 3-building rows per env step (`[0,1,2]` and `[3,4,5]`) for 2× SFT data and building-agnosticism within the 3-bldg shape. See `docs/PROGRESS.md` 2026-05-12 entry for full rationale.

## Four-phase plan

| Phase | Goal | Compute |
|-------|------|---------|
| 1 — Expert Baselines | Train SAC; benchmark RBC vs no-op vs LLM-as-policy | Colab GPU |
| 2 — SLM Integration | Translate observations to text; parse SLM actions | MacBook + Colab |
| 3 — SLM Fine-Tuning | LoRA / GRPO fine-tuning on CityLearn rollouts | Colab / DGX Spark |
| 4 — Multi-Agent Deployment | Two SLM agents, partial observability, coordination | DGX Spark ×2 |

## Project structure

```
eclipse-thesis/
├── CLAUDE.md              ← you are here
├── README.md
├── .gitignore
├── requirements.txt
├── docs/
│   ├── CONTEXT.md         ← full thesis background, read this first
│   ├── PROGRESS.md        ← living changelog, CHECK BEFORE EVERY SESSION
│   ├── CITYLEARN_API.md   ← CityLearn v2 API reference (imports, wrappers, boilerplate)
│   └── CITYLEARN_INSIGHTS.md  ← observation quirks, battery dynamics, prompting tips
├── sandbox/               ← one-off exploration scripts (not imported anywhere)
│   ├── 01–05_*.py         ← env exploration, baselines, obs-to-text experiments
│   └── _env_helpers.py
├── notebooks/             ← narrative + small-scale demos, run LOCALLY on CPU or COLAB GPU
│   ├── 01_env_setup.ipynb           ← env setup, RBC/SAC baselines, KPI evaluation (Phase 1)
│   ├── 02_llm_policy.ipynb          ← dual-agent LLM-as-policy, remote APIs (Phase 2)
│   ├── 03_slm_colab.ipynb           ← local SLM inference on Colab GPU (Phase 2)
│   ├── 04_sac_distill_dataset.ipynb ← SAC rollout → (state_text, action_token) JSONL [IN PROGRESS]
│   └── 05_sft_gemma_colab.ipynb     ← LoRA SFT on Gemma via Unsloth (Colab) [IN PROGRESS]
├── scripts/               ← full-scale training, run on COLAB or DGX SPARK (empty stubs)
├── src/                   ← reusable modules
│   ├── env.py             ← env factory, reward functions, snapshot_state()
│   ├── eval.py            ← KPI evaluation: evaluate(), comparison_table(), generalisation_gap()
│   ├── agent.py           ← prompt construction, render_state, parse_actions, reference policies
│   ├── providers.py       ← APIProvider (remote) + LocalHFProvider (local HF) — same .step() interface
│   ├── rollout.py         ← run_policy, run_policy_dual_agent, summary helpers
│   └── sft.py             ← SAC→SLM distillation helpers: action_to_token, dump JSONL, SFT prompt
└── configs/
    └── experiment.yaml    ← all hyperparameters, never hardcode them
```

## Compute environments

| Environment      | Hardware           | Use for                                            |
|------------------|--------------------|----------------------------------------------------|
| MacBook Air      | Apple Silicon, CPU | Local dev, debugging, toy-scale tests, analysis    |
| Google Colab Pro | A100/V100 GPU      | SAC training, SLM fine-tuning, full experiments    |
| DGX Spark (×2)   | NVIDIA GPU         | Distributed multi-agent deployment (Phase 4)       |

**Development workflow:**
1. Develop and debug on MacBook (CPU, toy scale: 2 buildings, 10 episodes)
2. Push to GitHub
3. Pull into Colab, run full-scale training/experiments on GPU
4. Results logged to wandb automatically
5. Analyze results back on MacBook

**Code must be hardware-agnostic.** Use `configs/experiment.yaml` to switch between:
- `device: cpu` (local) vs `device: cuda` (Colab/DGX)
- `scale: debug` (2 buildings, 10 episodes, 1 seed) vs `scale: full` (6 buildings, 300 episodes, 5 seeds)

## Tech stack

- Python 3.10+, PyTorch 2.x
- CityLearn v2 (Gymnasium-compatible)
- Stable Baselines3 — SAC expert baseline (Phase 1)
- HuggingFace PEFT + TRL or Unsloth — LoRA fine-tuning (Phase 3)
- SLM candidates: Qwen3-4B/8B, LLaMA-3.1-8B, Nemotron-3-Nano
- NVIDIA TensorRT-LLM + Triton for inference serving (Phase 4, DGX only)
- wandb for experiment tracking and artifact storage

## Rules

### Before starting work
- ALWAYS read `docs/PROGRESS.md` first to understand the current state
- If unsure about project context, read `docs/CONTEXT.md`
- For CityLearn API questions, consult `docs/CITYLEARN_API.md`
- For observation/battery/prompting quirks, consult `docs/CITYLEARN_INSIGHTS.md`

### Code style
- Type hints on all function signatures
- Docstrings on all public functions and classes (Google style)
- Keep notebooks thin: import from `src/`, don't write core logic in cells
- All experiment parameters go in `configs/experiment.yaml`, never hardcoded
- Use `logging` module, not print statements

### Notebooks vs scripts
- `notebooks/` are for narrative, visualization, and small-scale demos — run locally on CPU
- `scripts/` are for full-scale GPU workloads — run on Colab or DGX Spark
- Both import from `src/` — shared logic lives ONLY in `src/`
- Scripts MUST have checkpointing (save state every N episodes) for Colab resilience
- Scripts MUST log to wandb for experiment tracking

### Git workflow
- Commit after each working change with descriptive messages
- Branch for experimental features (`main` stays stable)
- Never commit large files (models, checkpoints, datasets, wandb logs)
- Pin dependencies with exact versions in `requirements.txt`

### Reproducibility
- Every experiment config MUST include a `seed` field
- Set seeds for Python, NumPy, PyTorch, and CityLearn at the start of every run
- Log all hyperparameters to wandb at experiment start
- Save final configs alongside results for full traceability

### After finishing work
- UPDATE `docs/PROGRESS.md` with what changed in this session
- Update the "Current phase" section in this file if phase changed
- Run `pip freeze > requirements.txt` if any packages were added

## Key constraints

- **Single agent through Phase 3** (group-centralized over `TRAINING_BUILDINGS=[0,1,2]`); dual-agent setup is Phase 4 only. Same LoRA loads into both Phase 4 agents — no retraining.
- Joint reward: both agents must receive the SAME reward at each timestep (Phase 4)
- Partial observability: Agent α sees buildings {0,1,2}, Agent β sees {3,4,5} — never crossed (Phase 4)
- Action-Only condition: NO explicit communication channel between agents (Phase 4)
- `central_agent=False` everywhere (Phases 1–4) — the flag controls env I/O shape, not policy count. Joint reward at Phase 4 is computed in the rollout loop by summing the per-building reward list.
- Online RL updates LoRA weights only; base SLM weights stay frozen (Phase 3+)
- KL penalty against reference model required to prevent catastrophic forgetting (Phase 3+)
- Validation gate before Phase 4: single-agent SLM must reach ≥70% of SAC expert performance

## Gotchas

- CityLearn v2 API changed from v1 — use `citylearn.citylearn.CityLearnEnv`, not the old interface
- `electrical_storage_soc` in the raw obs vector is bugged — read from `building.electrical_storage.soc[t]` directly
- CityLearn rewards are negative (costs), lower is better — don't flip the sign
- Battery charge and discharge are roughly symmetric in ΔSoC: `±0.20` ≈ ±14–17 pp/step, `±1.0` ≈ ±70 pp/step. There is **no asymmetric hardware cap** on discharge — the older "1.5 kWh/hr discharge cap" claim was wrong (verified in nb 03 on `citylearn_challenge_2022_phase_all`)
- `+1.0` charging across all buildings simultaneously spikes district demand — use small actions (0.1–0.3) when charging
- Discharging is action-driven, not load-driven: same `|action|` produces the same SoC drop regardless of building load. Load only affects whether surplus is exported. Size discharge to roughly match `load − solar` during peak; over-discharge exports cheaply
- SLM action parsing can fail — always have a fallback (default to 0.0 / no-op)
- GRPO needs multiple candidate generations per observation — budget for inference time
- Non-stationarity when both agents learn simultaneously — use slower learning rates
- Colab sessions can disconnect — ALWAYS checkpoint to Google Drive during long runs
- Test everything locally at debug scale before running full-scale on Colab
