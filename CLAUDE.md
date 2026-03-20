# ECLIPSE Thesis — Multi-Agent Energy Management

## What this project is

MSc thesis at University of West Attica (AIDL program), implementing a subset of the ECLIPSE project.
Two SLM-based agents manage energy storage across buildings in CityLearn v2 under partial observability.
Agents perform online RL (not frozen after fine-tuning) and coordinate through three paradigms:
Action-Only (primary), Experience Sharing (ceiling), and PPO/SAC baseline (reference).

Supervisor: Dr. Panagiotis Kasnesis
Student: Antonios Bastoulis

## Current phase

> **UPDATE THIS** after each work session. Example:
> Phase 1, Month 1 — Setting up CityLearn environment, configuring obs/action spaces.

Phase: NOT STARTED

## Project structure

```
eclipse-thesis/
├── CLAUDE.md              ← you are here
├── README.md
├── .gitignore
├── requirements.txt
├── docs/
│   ├── CONTEXT.md         ← full project background, read this first
│   └── PROGRESS.md        ← living changelog, CHECK BEFORE EVERY SESSION
├── notebooks/             ← narrative + small-scale demos, run LOCALLY on CPU
│   ├── 01_env_setup.ipynb
│   ├── 02_rl_baselines.ipynb
│   ├── 03_rationales.ipynb
│   ├── 04_finetuning.ipynb
│   ├── 05_online_rl.ipynb
│   └── 06_analysis.ipynb
├── scripts/               ← full-scale training, run on COLAB or DGX SPARK
│   ├── train_baseline.py  ← PPO/SAC training (Phase 1, Colab GPU)
│   ├── finetune_slm.py    ← LoRA fine-tuning (Phase 1, Colab GPU)
│   └── run_experiment.py  ← multi-agent experiments (Phase 2, Colab/DGX)
├── src/
│   ├── env.py             ← CityLearn wrappers, observation encoding
│   ├── agent.py           ← SLM agent class, prompt construction, action parsing
│   ├── rl.py              ← GRPO/PPO online RL logic, replay buffer
│   ├── coordination.py    ← experience sharing, action-only logic
│   └── utils.py           ← shared helpers, logging, config loading
└── configs/
    └── experiment.yaml    ← all hyperparameters, never hardcode them
```

## Compute environments

| Environment      | Hardware           | Use for                                           |
|------------------|--------------------|-------------------------------------------------  |
| MacBook Air      | Apple Silicon, CPU | Local dev, debugging, toy-scale tests, analysis   |
| Google Colab Pro | A100/V100 GPU      | RL training, SLM fine-tuning, full experiments    |
| DGX Spark (×2)   | NVIDIA GPU         | Distributed multi-agent deployment (when available)|

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
- Stable-Baselines3 for PPO/SAC baselines
- Unsloth or HuggingFace PEFT + TRL for LoRA fine-tuning
- SLM candidates: Qwen3-4B/8B, LLaMA-3.1-8B, Nemotron-3-Nano
- NVIDIA TensorRT-LLM + Triton for inference serving (Phase 2, DGX only)
- wandb for experiment tracking and artifact storage
- Google Colab Pro for GPU compute
- Hardware target: 2× NVIDIA DGX Spark (ECLIPSE grant, when available)

## Rules

### Before starting work
- ALWAYS read `docs/PROGRESS.md` first to understand the current state
- If unsure about project context, read `docs/CONTEXT.md`

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

- Joint reward function: both agents must receive the SAME reward at each timestep
- Partial observability: Agent α sees buildings {1,2,3}, Agent β sees {4,5,6} — never crossed
- Action-Only condition: NO explicit communication channel between agents
- Online RL updates LoRA weights only, base model weights stay frozen
- KL penalty against reference model is required to prevent catastrophic forgetting
- Validation gate before Phase 2: single-agent SLM must reach ≥70% of expert RL performance

## Gotchas

- CityLearn v2 API changed from v1 — use `citylearn.citylearn.CityLearnEnv`, not the old interface
- CityLearn rewards are negative (costs), lower is better — don't flip the sign
- SLM action parsing can fail — always have a fallback (default to 0.0 / no-op)
- GRPO needs multiple candidate generations per observation — budget for inference time
- When both agents learn simultaneously, non-stationarity can cause instability — use slower learning rates
- Colab sessions can disconnect — ALWAYS checkpoint to Google Drive during long runs
- Colab Pro gives A100 access but sessions have time limits — design scripts to be resumable
- Test everything locally at debug scale before running full-scale on Colab
