# Project Context —  MSc Thesis

This document provides the full background for the  thesis project.
Claude Code should read this when it needs deeper understanding of the project.

## The  project

ECLIPSE (Edge Coordination via Learning In Partially observable Shared Environments)
is a research project led by Dr. Panagiotis Kasnesis at the University of West Attica.
It investigates how SLM-based agents on separate edge nodes can learn to cooperate
under partial observability through online reinforcement learning.

The full ECLIPSE project covers energy domain via CityLearn
and three coordination paradigms (Gradient Sharing, Experience Sharing, Action-Only).
Hardware: 2× NVIDIA DGX Spark received from NVIDIA academic grant.

## This thesis scope

The core objective is to investigate whether Small Language Model (SLM)-based agents (such as Qwen3 or LLaMA) can effectively manage building energy within
a Reinforcement Learning (RL) environment like CityLearn. Furthermore, it explores how these agents can cooperatively manage complex systems when physically
distributed on edge hardware (NVIDIA DGX Spark nodes), operating with only partial observations of a shared environment and lacking a central orchestrator.

## Research question

The **primary research questions** I will focus on are:
1. How does the effectiveness of the SLM agent (measured in specific KPIs) compare to traditional rule-based and RL methods?
2. What is the generalization capability of the SLM agent across unseen buildings and varying weather conditions?
3. How effective is the SLM agent at providing interpretable natural language rationales for its selected control actions?

The **secondary research question** is: Can LLM-based agents develop implicit coordination protocols through behavioral observation alone, without explicit communication, in a distributed energy management setting? 

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

## Four-phase plan

* **Phase 1 (Expert Baselines):** I will first configure CityLearn and train classical RL baselines (SAC) to act as benchmarks and potentially as warm-up policies for the SLMs.
* **Phase 2 (SLM Integration):** I will integrate the SLM to communicate with the RL environment. State observations will be translated into natural language to be fed into the SLM. The SLM will then generate actions that are passed back to the RL environment.
* **Phase 3 (SLM Fine-Tuning):** I will fine-tune the SLM to optimize its performance in the CityLearn environment. The specific fine-tuning methodology (e.g., LoRA, RLHF,GRPO) remains open for exploration.
* **Phase 4 (Multi-Agent Deployment):** I will deploy two fine-tuned SLM-based agents, each observing only a subset of buildings (partial observability). Possibly these agents will continue to learn on online-learning. 


## Key design decisions

- Online RL, not frozen deployment: agents continue learning through GRPO during experiments
- LoRA-only updates: base SLM weights stay frozen, only adapter weights are updated by RL
- KL penalty: prevents policy collapse / catastrophic forgetting of imitation knowledge
- Joint reward: creates cooperative incentive without requiring explicit communication
- GRPO preferred over PPO for online RL: no value network needed, works on full generations

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
