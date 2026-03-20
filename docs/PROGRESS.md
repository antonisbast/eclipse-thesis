# Progress Log

> Update this file after EVERY work session. Claude Code reads this first.

## Current status

**Phase:** Not started
**Working on:** Project initialization
**Blockers:** None
**Compute:** MacBook Air (local dev), Google Colab Pro (GPU training)
**Next step:** Initialize project repository, install CityLearn, run first example

---

## Log

### [DATE] — Session title
- What was done
- What worked / what didn't
- Where it ran (local / Colab / DGX)
- Next steps for the following session

<!-- Example entry:
### 2026-04-01 — CityLearn environment setup [LOCAL]
- Installed CityLearn v2 via pip, confirmed Gymnasium compatibility
- Configured 4-building scenario with default weather data
- Defined observation and action spaces in src/env.py
- Tested with debug scale (2 buildings, 10 episodes) — works
- Issue: reward function returns NaN when battery SoC hits bounds — needs clipping
- Next: fix reward clipping, start PPO training in notebook 02

### 2026-04-05 — PPO baseline training [COLAB]
- Pushed code to GitHub, pulled into Colab
- Ran scripts/train_baseline.py with PPO on A100
- Training: 500K timesteps, took ~45 min
- Results logged to wandb (run: ppo-baseline-v1)
- PPO achieves 18% peak demand reduction vs no-storage baseline
- Checkpoints saved to Google Drive
- Next: train SAC, compare performance
-->
