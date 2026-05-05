# ECLIPSE Thesis — Multi-Agent Energy Management

MSc thesis at the University of West Attica (AIDL program), implementing a subset of the
[ECLIPSE](https://docs.google.com/document/d/1) project. Two SLM-based agents manage battery
energy storage across buildings in CityLearn v2 under partial observability.

## Research question

Can SLM-based agents develop implicit coordination protocols through behavioral observation
alone (Action-Only), without explicit communication, in a distributed energy management
setting? How does this compare to explicit Experience Sharing?

## Approach

1. **Phase 1 — Foundation:** Train PPO/SAC expert policies on CityLearn, convert trajectories
   to natural-language rationales, fine-tune an SLM via LoRA, and validate against a
   performance gate (>=70% of expert).
2. **Phase 2 — Online Multi-Agent RL:** Deploy two SLM agents with partial observability,
   run three conditions (Action-Only, Experience Sharing, PPO/SAC baseline) across 5 seeds.
3. **Phase 3 — Analysis:** Action correlation, mutual information, strategy clustering,
   and reasoning chain inspection for evidence of implicit coordination.

## Project structure

```
src/           Core modules (env wrapper, agent, RL, coordination, utils)
scripts/       Full-scale GPU training scripts (Colab / DGX Spark)
notebooks/     Narrative demos and analysis (local CPU)
configs/       Experiment hyperparameters (YAML)
docs/          Project context and progress log
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

See `docs/CONTEXT.md` for full project background and `docs/PROGRESS.md` for current status.

## Supervisor

Dr. Panagiotis Kasnesis — University of West Attica
