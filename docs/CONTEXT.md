# Project Context — ECLIPSE MSc Thesis

This document provides the full background for the ECLIPSE thesis project.
Claude Code should read this when it needs deeper understanding of the project.

## The ECLIPSE project

ECLIPSE (Edge Coordination via Learning In Partially observable Shared Environments)
is a research project led by Dr. Panagiotis Kasnesis at the University of West Attica.
It investigates how SLM-based agents on separate edge nodes can learn to cooperate
under partial observability through online reinforcement learning.

The full ECLIPSE project covers two domains (traffic via SUMO, energy via CityLearn)
and three coordination paradigms (Gradient Sharing, Experience Sharing, Action-Only).
Hardware: 2× NVIDIA DGX Spark requested from NVIDIA academic grant.

## This thesis scope

This thesis implements the CityLearn (energy) domain only, with two of the three
coordination paradigms (Experience Sharing and Action-Only), plus a classical RL baseline.
Gradient Sharing is excluded from the thesis scope (acknowledged as future work).

## Research question

Can SLM-based agents develop implicit coordination protocols through behavioral
observation alone (Action-Only), without any explicit communication, in a distributed
energy management setting? How does this compare to explicit Experience Sharing?

## CityLearn environment

CityLearn v2 is a Gymnasium-compatible simulation of a neighborhood of grid-interactive
buildings. Each building has:
- Solar PV generation
- Battery energy storage system (BESS)
- Dynamic electricity loads (weather + occupancy driven)
- Connection to a shared electricity grid with time-of-use pricing

Agent action: continuous [-1, 1] per building (battery charge/discharge fraction).
Agent observation: solar gen, net load, battery SoC, temperature, hour, day type, price signal.
Reward: joint negative cost combining peak demand, electricity cost, and ramping penalty.

The environment runs in hourly timesteps. One episode = 1 simulated year = 8,760 steps.

## Three-phase plan

### Phase 1: Foundation (Months 1-2)
1. Configure CityLearn with 4-6 buildings, define obs/action spaces and reward
2. Train PPO and SAC expert policies via Stable-Baselines3
3. Roll out expert policies, convert trajectories to natural language rationales
4. Fine-tune SLM via LoRA on rationale dataset (30-50K examples)
5. Validate single-agent SLM passes performance gate (≥70% of expert)

### Phase 2: Online Multi-Agent RL (Months 3-4)
1. Deploy two SLM agents, each observing a subset of buildings
2. Implement online RL loop: observation → prompt → SLM generation → action → reward → GRPO update
3. Run three conditions × 5 random seeds:
   - Action-Only: agents share nothing, observe only grid-level effects
   - Experience Sharing: agents exchange trajectory buffers every 10 episodes
   - PPO/SAC Baseline: classical RL under same partial observability
4. Metrics: peak demand reduction, energy cost savings, grid stability, convergence rate

### Phase 3: Analysis (Months 5-6)
1. Action correlation analysis over training episodes
2. Mutual information estimation between agents' actions
3. Strategy clustering and counterfactual experiments
4. Gap quantification: Action-Only vs Experience Sharing
5. Reasoning chain inspection for evidence of implicit coordination
6. Thesis writing and submission

## Compute strategy

### Where things run

**MacBook Air (Apple Silicon, CPU):** The development machine. All code is written and
debugged here using Claude Code. Run toy-scale tests (2 buildings, 10 episodes) to
verify code correctness before pushing to GPU. Also used for Phase 3 analysis and
visualization — plotting, statistics, reasoning chain inspection all run on CPU.

**Google Colab Pro (A100/V100 GPU):** The training machine. All GPU-intensive work
runs here: PPO/SAC baseline training, SLM fine-tuning via LoRA, and full-scale
multi-agent experiments. Colab Pro provides A100 access and extended runtime.

**DGX Spark × 2 (when available from ECLIPSE grant):** The deployment target.
Each DGX Spark node hosts one SLM agent for the true distributed multi-agent setup.
Until hardware arrives, multi-agent experiments run on single Colab GPU with both
agents in the same process (simulating distribution).

### Code <-> Compute flow

1. Develop on MacBook → push to GitHub (private repo)
2. Pull into Colab from GitHub (or from Google Drive persistent clone)
3. Run scripts on Colab GPU → results logged to wandb
4. Analyze results on MacBook by pulling from wandb

### Colab session setup
```python
# Option A: fresh clone each session
!git clone https://github.com/USERNAME/eclipse-thesis.git
%cd eclipse-thesis
!pip install -r requirements.txt

# Option B: persistent clone on Google Drive (preferred)
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/eclipse-thesis
!git pull
!pip install -r requirements.txt
```

### Colab resilience
Colab sessions can disconnect. All scripts in `scripts/` must:
- Checkpoint model state to Google Drive every N episodes
- Log metrics to wandb continuously (not just at end)
- Support `--resume` flag to continue from last checkpoint
- Print progress to stdout so Colab shows it in the cell output

## Key design decisions

- Online RL, not frozen deployment: agents continue learning through GRPO during experiments
- LoRA-only updates: base SLM weights stay frozen, only adapter weights are updated by RL
- KL penalty: prevents policy collapse / catastrophic forgetting of imitation knowledge
- Joint reward: creates cooperative incentive without requiring explicit communication
- GRPO preferred over PPO for online RL: no value network needed, works on full generations

## Models under consideration

| Model            | Params | Notes                                        |
|------------------|--------|----------------------------------------------|
| Qwen3-4B         | 4B    | Fits easily on Colab/DGX, strong reasoning    |
| Qwen3-8B         | 8B    | Primary candidate if memory allows            |
| LLaMA-3.1-8B     | 8B    | Well-studied, good baseline comparison        |
| Nemotron-3-Nano  | ~4B   | NVIDIA-native, TensorRT-LLM optimized         |

## Key references

- CityLearn v2: Nweye et al., 2024 (arXiv:2405.03848)
- LLMLight (trajectory-to-rationale method): Liang et al., 2024 (arXiv:2312.16044)
- LLaRP (online RL with LLMs): Szot et al., 2024 (arXiv:2310.17722)
- LLM Multi-Agent survey: Guo et al., 2024 (arXiv:2402.01680)
- LoRA: Hu et al., 2021 (arXiv:2106.09685)
- DeepSeek-R1 (GRPO): DeepSeek, 2025

## File-by-file guide

### src/ modules

- `env.py`: CityLearn wrapper that handles building partitioning (which agent sees which
  buildings), observation-to-text encoding, action parsing, and reward computation.
  This is the bridge between CityLearn's numeric API and the SLM's text interface.

- `agent.py`: The SLM agent class. Handles prompt construction from observations,
  SLM inference (generation), action extraction from generated text, and the fallback
  parser for malformed outputs. Also manages the LoRA adapter loading/saving.

- `rl.py`: Online RL logic. Contains the GRPO implementation (multiple candidate
  generations, reward ranking, policy gradient update on LoRA weights). Also contains
  the KL penalty computation against the reference model. Replay buffer management.

- `coordination.py`: Implements the coordination paradigms. For Action-Only: just the
  environment coupling (no explicit sharing). For Experience Sharing: trajectory buffer
  exchange every N episodes, merging partner data into local replay buffer.

- `utils.py`: Config loading from YAML, logging setup, wandb integration, seed setting,
  metric computation helpers (peak demand, cost, ramping, convergence rate).

### scripts/ (GPU workloads)

- `train_baseline.py`: Trains PPO and SAC expert policies on CityLearn using
  Stable-Baselines3. Saves trained models and generates trajectory rollouts.
  Supports `--resume` for Colab resilience. Logs training curves to wandb.

- `finetune_slm.py`: Fine-tunes SLM via LoRA on the trajectory-to-rationale dataset.
  Saves LoRA adapter checkpoints to Google Drive. Validates against performance gate.
  Supports `--resume` for Colab resilience.

- `run_experiment.py`: Runs the full multi-agent experiment suite. Takes a config
  specifying condition (action_only / experience_sharing / baseline), seed, and scale.
  Checkpoints every N episodes. Logs all metrics to wandb. Supports `--resume`.

### notebooks/ (narrative + demos)

- `01_env_setup.ipynb`: CityLearn setup, observation/action space exploration, reward
  function design. Runs locally with 2 buildings for demonstration.

- `02_rl_baselines.ipynb`: Demonstrates PPO/SAC training at small scale. Shows
  learning curves, evaluates trained policies. Full training happens via scripts.

- `03_rationales.ipynb`: Trajectory-to-rationale conversion pipeline. Shows examples
  of how numeric trajectories become natural language. Quality analysis of rationales.

- `04_finetuning.ipynb`: LoRA fine-tuning demonstration at small scale. Shows prompt
  format, generation examples, validation gate evaluation. Full training via scripts.

- `05_online_rl.ipynb`: Online RL loop demonstration. Shows single-episode walkthrough
  of observation → prompt → generation → action → reward → GRPO update cycle.

- `06_analysis.ipynb`: Full analysis suite. Action correlation, mutual information,
  strategy clustering, gap quantification. Pulls results from wandb. Generates all
  thesis figures and tables.

### configs/

- `experiment.yaml`: Master config file. Contains all hyperparameters organized by
  phase and component. Has `scale` presets (`debug` for local, `full` for Colab/DGX).
  Every value that could change between experiments lives here, not in code.
