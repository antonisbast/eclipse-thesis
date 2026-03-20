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

## Compute strategy

### Where things run

**MacBook Air (Apple Silicon, CPU):** Development machine. All code is written and
debugged here. Run toy-scale tests (2 buildings, 10 episodes) to verify correctness.
Also used for Phase 3 analysis and visualization.

**Google Colab Pro (A100/V100 GPU):** Training machine. PPO/SAC baseline training,
SLM fine-tuning via LoRA, and full-scale multi-agent experiments.

**DGX Spark × 2 (when available from ECLIPSE grant):** Deployment target for true
distributed multi-agent setup. Until hardware arrives, both agents run on Colab.

### Code <-> Compute flow

1. Develop on MacBook → push to GitHub (private repo)
2. Pull into Colab from GitHub (or from Google Drive persistent clone)
3. Run scripts on Colab GPU → results logged to wandb
4. Analyze results on MacBook by pulling from wandb

### Colab session setup
```python
# Option A: fresh clone each session
!git clone https://github.com/antonisbast/eclipse-thesis.git
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

## File guide

### src/ — reusable Python modules
- `env.py`: CityLearn wrappers and observation/action handling
- `agent.py`: SLM agent class
- `rl.py`: RL training logic (baselines and online RL)
- `utils.py`: config loading, seeding, logging helpers
- Add new modules as needed — this list is a starting point, not a constraint

### scripts/ — GPU workloads for Colab/DGX
Full-scale training and experiments. Each script must support `--resume` for
Colab resilience and log to wandb. Add scripts as phases progress.

### notebooks/ — narrative and small-scale demos
One per thesis chapter. Run locally on CPU with debug-scale configs.
These tell the story of the thesis with working code examples.

### configs/ — YAML experiment parameters
All hyperparameters live here. Code reads from config, never hardcodes values.

> This structure will evolve. If a module gets too big, split it.
> If a new concern emerges, create a new file. Update this doc to match.

## Key references

- CityLearn v2: Nweye et al., 2024 (arXiv:2405.03848)
- LLMLight (trajectory-to-rationale method): Liang et al., 2024 (arXiv:2312.16044)
- LLaRP (online RL with LLMs): Szot et al., 2024 (arXiv:2310.17722)
- LLM Multi-Agent survey: Guo et al., 2024 (arXiv:2402.01680)
- LoRA: Hu et al., 2021 (arXiv:2106.09685)
- DeepSeek-R1 (GRPO): DeepSeek, 2025
