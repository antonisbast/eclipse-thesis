# GRPO for SLM Policies in CityLearn — Thesis Reference

**Audience.** This document is the technical reference companion for notebook
`07_grpo_colab.ipynb`. It is written in thesis tone — formal definitions,
citations with URLs, derivations spelled out — so that chunks can be lifted
nearly verbatim into the manuscript. The notebook stays pragmatic; this
document is the place to read when something in the notebook is unclear.

**Reading order if you are new to RL on LLMs.** §§ 1 → 2 → 3 → 4 → 8 first; the
proof-heavy § 5 and the comparison in § 6 are useful for the thesis chapter
but not required to run the notebook.

---

## 1. Motivation: why RL on top of SFT

The pipeline up to notebook 05 trains the SLM by **behavior cloning** on
trajectories from a SAC teacher: minimize the cross-entropy of the SLM's
distribution against the teacher's discretized actions. This is supervised
fine-tuning (SFT) and it has two known ceilings.

1. **The student inherits the teacher's mistakes.** The SAC teacher was
   trained for only 20 episodes. Wherever the teacher is suboptimal, the SLM
   will be at best equally suboptimal — SFT cannot exceed the data
   distribution.
2. **SFT has no signal beyond "match the teacher's action."** It has no idea
   *why* an action was good. Two different actions with very different
   downstream consequences look equally good to SFT if neither matches the
   teacher.

Reinforcement learning replaces "match the teacher" with **"earn more reward
in the actual environment."** The environment (CityLearn 2.6 with
`MERLINReward`) becomes the ground truth. The agent rolls out, the env
returns a scalar, and the policy is updated to make high-reward rollouts more
likely.

The standard pipeline in modern post-training (DeepSeek-R1, RAGEN, ReFT) is
**SFT first, then RL** — SFT puts the policy in the rough neighborhood of
sensible behavior so RL doesn't waste compute on random exploration. This
is exactly the recipe followed in nb 05 → nb 07.

---

## 2. The RL framing in CityLearn

### 2.1 The Markov decision process

| Element | In our setup |
|---|---|
| State $s_t$ | The rendered text block: time-of-day, price, carbon, per-building SoC, load, last net consumption, solar bucket — for 3 buildings |
| Action $a_t$ | A text block of 3 lines, each `<action building=i>TOKEN</action>` with TOKEN ∈ {CHARGE_100, …, IDLE, …, DISCHARGE_100}. Optionally preceded by a `<thought>` block. |
| Policy $\pi_\theta(a\mid s)$ | The LoRA-adapted Gemma-4 E4B SLM — conditional distribution over completion tokens given the prompt |
| Reward $r_t$ | The district-summed MERLIN reward: $r_t = \sum_b -(1 + \mathrm{sign}(\mathrm{net}_b)\cdot \mathrm{SoC}_b)\cdot \|\mathrm{net}_b\|$. Dense, negative (cost), one per env step. |
| Return $R$ | Sum of rewards over a window of $W$ steps. The full-year return is the canonical thesis metric, but full-year rollouts are too expensive per RL update. |

### 2.2 Why this is different from typical LLM RL

The widespread LLM-RL literature (RLHF, DeepSeek-R1, GSM8K-style reasoning)
deals with **one-shot** problems: prompt in, response out, one scalar reward,
done. CityLearn is fundamentally a **multi-step control** problem: each
action changes the state, the next state depends on the action, and the
return is summed over a trajectory.

This distinction matters because the credit-assignment problem becomes harder:
when a window earns a high reward, *which actions* deserve credit? Naïvely
applying a one-shot RL algorithm (one advantage per trajectory) to a
multi-step problem leads to the well-documented *Echo Trap* failure mode
(Wang et al., 2025): the policy collapses to one trajectory and stops
exploring.

The notebook handles this by treating a short window (e.g. W=48 steps) as
the unit of credit assignment — long enough that battery decisions have
visible consequences, short enough that exploration cost is bounded. Every
action token within a window receives the same group-normalized advantage
(see § 5). This is the same compromise RAGEN/StarPO, ArCHer, and ReFT
converge on for multi-turn LLM agents.

---

## 3. Policy gradients, variance, and baselines

The fundamental object in policy-gradient RL is

$$
\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_t R(\tau)\,\nabla_\theta \log \pi_\theta(a_t \mid s_t)\right].
$$

In plain words: collect a rollout $\tau$, compute its total reward $R(\tau)$,
and shift the policy so that the actions taken in that rollout become more
likely if $R$ was high (and less likely if $R$ was low). This is the
REINFORCE algorithm (Williams, 1992).

**The variance problem.** This estimator is unbiased but has catastrophic
variance. Two rollouts with rewards 100 and 102 are nearly identical in
quality, but the estimator multiplies their log-probabilities by 100 and 102
respectively — the gradient is dominated by the *level* of the reward, not
its *quality* relative to other actions.

**Baselines fix this.** Replace $R(\tau)$ with $R(\tau) - b$ where $b$ is any
function that does not depend on the action. The estimator stays unbiased
(expectation over actions of $b \cdot \nabla \log \pi = 0$) but variance drops
dramatically when $b$ tracks $R$ well.

Classical PPO learns $b$ as a value function $V_\phi(s)$ via a second neural
network (the **critic**). This is powerful but adds a whole second model to
train. **GRPO, RLOO, and REINFORCE-with-mean-baseline drop the critic
entirely** and use a sample-based baseline instead. This is the family of
"critic-free" methods that has become standard for LLM post-training.

---

## 4. GRPO — the algorithm

GRPO (Group Relative Policy Optimization, Shao et al. 2024) is the
critic-free policy-gradient method introduced by DeepSeekMath. Its
single idea:

> For each prompt, sample $G$ completions from the current policy. Use
> the **group mean** as the baseline and the **group standard deviation**
> as a normalizer.

Formally, for a group of $G$ rollouts with returns $R_1, \ldots, R_G$, define
the **group-normalized advantage**

$$
A_g \;=\; \frac{R_g - \bar R}{\sigma_R + \epsilon}, \qquad \bar R = \frac{1}{G}\sum_g R_g,\quad \sigma_R = \mathrm{std}(R_1, \ldots, R_G).
$$

The GRPO loss is then

$$
\mathcal{L}_\mathrm{GRPO}(\theta) \;=\; -\,\mathbb{E}_{g,t}\!\left[A_g \cdot \log \pi_\theta(c_{g,t} \mid s_{g,t})\right] \;+\; \beta \cdot \mathrm{KL}\!\left(\pi_\theta \,\Vert\, \pi_\mathrm{ref}\right).
$$

Two terms:

1. **Policy-gradient term** $-A_g \log \pi$. Makes high-advantage completions
   more likely, low-advantage ones less likely. The standardisation by
   $\sigma_R$ means we don't care about the absolute scale of the reward —
   only the *relative* quality within the group matters.
2. **KL term** $\beta \cdot \mathrm{KL}(\pi_\theta \Vert \pi_\mathrm{ref})$.
   Keeps the policy close to a frozen reference $\pi_\mathrm{ref}$ (typically
   the SFT model). This prevents *catastrophic forgetting* — RL must improve
   on the SFT prior without destroying its language and reasoning abilities.

The hyperparameter $\beta$ controls the trust region:
- Too small → policy drifts arbitrarily far from $\pi_\mathrm{ref}$, can
  collapse into degenerate outputs.
- Too large → policy can barely move from $\pi_\mathrm{ref}$ even when it
  finds high-reward strategies.
- Standard starting value for LoRA-tuned LLMs: $\beta \in [0.01, 0.1]$.

### 4.1 What GRPO *does not* have

| | PPO | GRPO |
|---|---|---|
| Value head (critic) | required | **not used** |
| Importance-sampling clip | required (ε-clip) | optional (used in DeepSeek-R1; we omit) |
| KL to reference | added separately | **part of the loss** |
| Samples per prompt | 1 | **G (typically 4–8)** |

The drop of the value head is the practical win on small hardware: a
second adapter or a separate head doubles trainable parameters and
materially complicates 4-bit training. GRPO's group baseline is "free" in
the sense that it requires nothing beyond sampling $G$ times — useful when
inference is cheap, which is the regime modern LLM serving stacks (Unsloth,
vLLM) are optimized for.

### 4.2 Adapting GRPO to a multi-step environment

Vanilla GRPO assumes one completion → one reward. CityLearn gives us a
reward at *every* env step. The notebook follows RAGEN's adaptation:

1. Define a **window** of $W$ env steps as the unit of "trajectory."
2. From a single starting state $s_{t_0}$, sample $G$ parallel rollouts of
   length $W$. (Same $t_0$ across the group is what makes the group mean a
   valid baseline.)
3. Return for rollout $g$ is $R_g = \sum_{t=t_0}^{t_0+W-1} r_{g,t}$.
4. Compute $A_g$ as above (one scalar per rollout).
5. The advantage is broadcast across **every action and every token** in
   that rollout. All completion tokens — including the `<thought>` block —
   receive the same $A_g$ multiplier in the loss.

This is a coarse but practical credit assignment. Step-level credit
assignment in this regime is an active research area (GiGPO, SALT 2025);
the windowed approach is the documented robust baseline.

---

## 5. Worked numerical example

Suppose at update 17 the seasonal sampler picks $t_0 = 5{,}120$ (a Sunday
morning in August). The training loop performs the following sequence —
all numbers below are illustrative.

**Step 1 — collect G=2 rollouts of W=3 steps.**

| Rollout | step | $r_t$ (MERLIN) | parse failure? |
|---|---|---|---|
| 1 | 5120 | $-1.20$ | no |
| 1 | 5121 | $-0.80$ | no |
| 1 | 5122 | $-1.40$ | no |
| **1** | **return** | $\mathbf{R_1 = -3.40}$ | |
| 2 | 5120 | $-1.10$ | no |
| 2 | 5121 | $-2.40$ | yes (penalty already applied) |
| 2 | 5122 | $-1.00$ | no |
| **2** | **return** | $\mathbf{R_2 = -4.50}$ | |

**Step 2 — group statistics.**
$$\bar R = \tfrac{1}{2}(-3.40 + -4.50) = -3.95, \quad \sigma_R = 0.55$$

**Step 3 — group-normalized advantages.**
$$A_1 = \frac{-3.40 - (-3.95)}{0.55 + 10^{-6}} = +1.0, \qquad A_2 = -1.0.$$

Rollout 1 was the *relatively* better one in this group, so it gets a
positive advantage; rollout 2 a negative one. **Both rollouts have negative
returns** (MERLIN is a cost), yet GRPO correctly identifies the better one
as positive-advantage because it standardises *within the group*.

**Step 4 — per-token loss contribution.** For each step in each rollout we
have already recorded the prompt tokens and the generated tokens. Suppose
rollout 1, step 5120 generated 25 tokens whose summed policy log-probability
under the *current* policy is $-30.0$ nats. Its policy-gradient
contribution to the loss is
$$-A_1 \cdot \sum_t \log \pi_\theta(c_t \mid s_t) \;=\; -1.0 \cdot (-30.0) \;=\; +30.0.$$
Minimising this term (adding it with a negative sign because we're
*minimising* the loss but want to *maximise* expected reward) increases
$\sum \log \pi$, i.e. makes that completion more likely. Conversely a
$-1.0$ advantage on rollout 2 makes its tokens *less* likely.

**Step 5 — KL term.** For the same 25 tokens, we also compute their log
probabilities under the frozen reference (the SFT'd LoRA, accessed via
`model.disable_adapter()` in the notebook). Suppose those sum to $-32.0$
nats. The per-rollout KL contribution is
$$\mathrm{KL}_g \approx \frac{1}{25}\sum_t \left(\log\pi_\theta - \log\pi_\mathrm{ref}\right) = \frac{-30.0 - (-32.0)}{25} = +0.08 \text{ nats/token}.$$

**Step 6 — total loss.** Average policy-gradient term over all
(rollout, step) pairs; average KL term similarly; add $\beta$-weighted KL.
$$\mathcal{L} = \mathrm{mean}(\text{pg terms}) + 0.04 \cdot \mathrm{mean}(\text{KL terms}).$$
Call `loss.backward()`, clip gradients, `optimizer.step()`. Done.

This six-step pattern — collect, normalize, score, KL, sum, step — is what
notebook 07 implements in code in §§ 7–9. The toy code cell in § 0.6 of the
notebook runs exactly this on hand-crafted numbers so you can see every
intermediate quantity.

---

## 6. Why GRPO and not PPO or RLOO

| Aspect | PPO | RLOO | **GRPO (chosen)** |
|---|---|---|---|
| Critic / value head | required | none | **none** |
| Samples per prompt | 1 | K (similar to GRPO's G) | G (4–8 typical) |
| Stability on small LoRA | fragile (critic instability) | good | **good** |
| Tooling: env-coupled loops | TRL `PPOTrainer` rigid for env-reward | `RLOOTrainer` single-prompt-oriented | **first-class in recent TRL; Unsloth integration** |
| Used by recent SoTA | RLHF era | not yet widespread | DeepSeek-R1, RAGEN, ArCHer, ReFT |
| Compute on 4-bit Gemma | high (two models) | low | **low** |

**The decisive factor for this thesis is the critic.** PPO's value head is
a second neural network. Training it stably on a 4-bit Gemma backbone with
LoRA is a substantial engineering project on its own. GRPO removes it.
RLOO removes it too and is algorithmically close to GRPO; we chose GRPO
because TRL's GRPO tooling and Unsloth's GRPO integration are more mature
and the recent SoTA examples (DeepSeek-R1, RAGEN) are all GRPO-based — so
the design choices and pitfalls are better documented.

---

## 7. Hyperparameter rationale

| Hyperparameter | Notebook value | What it controls | When to change |
|---|---|---|---|
| **Group size G** | 4 (full) / 2 (toy) | Variance of the group baseline. G=1 collapses to plain REINFORCE; G≥4 is the sweet spot in DeepSeekMath and RLOO ablations. | Increase if reward std is noisy across updates; capped by inference budget (G × W generations per update). |
| **Window W** | 48 (full) / 12 (toy) | Credit-assignment horizon. Too short → policy can't see consequences of a charge action (battery dynamics are hours-long). Too long → exploration cost explodes. 48h covers one diurnal solar/load cycle plus SoC carry-over. | Lengthen to 72–96 if policy seems myopic (e.g. discharges fully every peak even with empty solar forecast). |
| **KL β** | 0.04 | Trust-region size around the reference policy. Standard RLHF range is 0.01–0.1. | Halve if KL grows unboundedly; double if reward variance collapses. |
| **LR (LoRA)** | 5e-6 | Optimizer step size. Unsloth's published default for Gemma-4 GRPO. | Drop 2× if loss is noisy; raise to 1e-5 after 50 stable updates if convergence is slow. |
| **Rollout temperature** | 0.7 | Exploration vs. exploitation at rollout time. Greedy at eval. | Raise to 0.9 if rollouts within a group are too similar (group std → 0); drop if outputs become incoherent. |
| **Rollout top-p** | 0.9 | Nucleus sampling cutoff. | Rarely needs tuning. |
| **Fallback penalty** | 0.5 | Added penalty when action parsing fails (treated as IDLE). | Raise if parse failures persist past update ~50; the model should learn to avoid them. |
| **Grad clip** | 1.0 | Protects against rare exploding-gradient updates on long sequences. RAGEN/StarPO-S recommendation. | Rarely needs tuning. |
| **Updates** | 300 (full) / 8 (toy) | Total optimizer steps. At G=4, W=48 this is ~60k env steps total (~7 simulated years). | Increase if KPI is still trending; stop early if it plateaus for 50+ updates. |

---

## 8. Pitfalls and how to detect them

The agentic-RL literature documents a handful of recurring failure modes.
The notebook's § 11 plots are designed so you can read these off the curves.

### 8.1 Echo Trap (RAGEN / StarPO 2025)
**What it looks like:** Mean return improves rapidly for a few dozen updates,
then plateaus; reward variance within each group collapses toward zero;
the policy starts producing identical or near-identical completions across
rollouts.
**What's happening:** The policy has found one strategy that earns
positive advantage and is concentrating all its probability mass there.
With no exploration, $\sigma_R \to 0$, advantages are undefined (handled
by the $\epsilon$ in the denominator), and learning stalls.
**Mitigations:** Raise rollout temperature; raise KL β (forces policy to
stay closer to a more diverse reference); shorten the window so the
sampler sees more diverse states; add a small entropy bonus.

### 8.2 KL blow-up
**What it looks like:** KL term in § 11 plot grows monotonically and
unboundedly; the policy starts producing nonsense.
**What's happening:** Policy is escaping the trust region. Either β is
too small or the reference is a poor anchor (e.g. the LoRA-disabled base
model rather than the SFT'd model — the simplification used in the
notebook by default).
**Mitigations:** Double β; reload from the last good checkpoint; if
problem persists, upgrade the reference policy to the SFT'd snapshot.

### 8.3 Fallback collapse
**What it looks like:** Parse-failure rate (§ 11 right plot) stays high
or grows; rationale tokens become gibberish; action tags missing entirely
from completions.
**What's happening:** Policy has discovered that random text is "free"
under the policy-gradient loss (with $A < 0$ it's discouraged, but only
weakly if returns are all comparable). The output structure is being
eroded.
**Mitigations:** Raise `FALLBACK_PENALTY`; raise KL β; verify the
reference policy still produces well-structured completions
(`model.disable_adapter()` smoke test).

### 8.4 CoT collapse
**What it looks like:** `<thought>` blocks become empty, single-token, or
clearly degenerate ("ok ok ok").
**What's happening:** Thought tokens are receiving the same advantage as
action tokens, but they don't *directly* affect reward — so the policy
learns to spend the minimum probability mass on them. KL term should
catch this if the reference produces non-trivial thoughts.
**Mitigations:** This is why the notebook recommends a **CoT warm-restart
SFT** before this notebook if RQ3 (interpretable rationales) matters. If
the SFT adapter has zero `<thought>` mass, RL cannot conjure it from
nothing.

### 8.5 Seasonal bias
**What it looks like:** Eval KPI on summer weeks improves; winter / shoulder
KPIs degrade.
**What's happening:** Bias in the $t_0$ sampler (or in which windows happen
to get sampled by chance).
**Mitigations:** The notebook's `T0_POOL_SEASONS` ensures all four seasons
are sampled uniformly. If a particular season is overrepresented in early
updates, the policy might overfit; verify the sampling distribution in
the JSONL log.

---

## 9. Annotated reading order

If you have a few hours and want to learn the field from the most relevant
papers in priority order:

### Tier 1 — read in full

1. **DeepSeekMath / GRPO** — Shao, Wang, Zhu, Liu, Xu, et al. (2024).
   [arXiv:2402.03300](https://arxiv.org/abs/2402.03300).
   *§ 4 (Group Relative Policy Optimization)* is the algorithm. Read the
   formula and the pseudo-code. § 5.2 has the ablation that justifies
   G=4–8. Skim § 3 (math-specific data pipeline) — not relevant to us.

2. **RAGEN / StarPO** — Wang, Wang, Zhao, et al. (2025).
   [arXiv:2504.20073](https://arxiv.org/abs/2504.20073).
   This is the closest paper to our setup. Read §§ 1–4 carefully. The
   "Echo Trap" diagnosis (§ 4) and "StarPO-S" stabilizers (trajectory
   filtering, grad clipping) are directly applied in the notebook.

### Tier 2 — read sections

3. **ReFT** — Trung, Luong, Phan, Hoi (2024). ACL 2024.
   [arXiv:2401.08967](https://arxiv.org/abs/2401.08967).
   Read § 3. Establishes the SFT-warmup → PPO/RL recipe on chain-of-thought
   rationales with env-derived reward. Conceptual blueprint for our
   nb 05 → nb 07 pipeline.

4. **ArCHer** — Zhou, Aytar, Misra, et al. (2024). ICML 2024.
   [arXiv:2402.19446](https://arxiv.org/abs/2402.19446).
   Read § 2 (problem formulation) and § 3 (hierarchical TD critic).
   Useful for understanding *why* we don't currently implement step-level
   credit assignment, and what would be required to upgrade.

### Tier 3 — skim

5. **RLOO ("Back to Basics")** — Ahmadian, Cremer, Gallé, et al. (2024).
   ACL 2024. [arXiv:2402.14740](https://arxiv.org/abs/2402.14740).
   Justifies the critic-free family in general. Argues for the simpler
   REINFORCE-with-mean-baseline; algorithmically very close to GRPO. Useful
   if you ever want to swap algorithms — the changes are minimal.

6. **MERLIN reward** — Nweye, Liu, Stone, Nagy (2024). *Applied Energy* 358.
   [doi:10.1016/j.apenergy.2023.121958](https://doi.org/10.1016/j.apenergy.2023.121958).
   Already cited in the thesis. Read Table 3 for the reward
   parameterisation we use.

### Tier 4 — optional / further reading

7. **GiGPO / SALT** — recent (2025) work on step-level credit assignment for
   multi-turn agentic RL. Useful if window-level credit proves too coarse
   and you need to upgrade. Out of scope for the current notebook.

### Mapping to thesis chapters

| Notebook concept | Thesis chapter / section |
|---|---|
| The MDP framing in § 2 of this doc | Background — RL in CityLearn |
| Policy gradients, baselines (§ 3) | Background — Policy Gradient Methods |
| GRPO formal definition (§ 4) | Methods — RL Fine-Tuning |
| Windowed adaptation (§ 4.2) | Methods — Multi-Step Credit Assignment |
| Worked example (§ 5) | Methods — Worked Example (in appendix) |
| Algorithm comparison (§ 6) | Methods — Algorithm Selection |
| Hyperparameter table (§ 7) | Methods — Hyperparameters |
| Pitfalls (§ 8) | Results — Training Dynamics |

---

## References

- Ahmadian et al. (2024). *Back to Basics: Revisiting REINFORCE-Style
  Optimization for Learning from Human Feedback in LLMs.* ACL 2024.
  [arXiv:2402.14740](https://arxiv.org/abs/2402.14740).
- Luong (Trung) et al. (2024). *ReFT: Reasoning with Reinforced Fine-Tuning.*
  ACL 2024. [arXiv:2401.08967](https://arxiv.org/abs/2401.08967).
- Nweye et al. (2024). *MERLIN: Multi-agent offline and transfer learning
  for occupant-centric operation of grid-interactive communities.* Applied
  Energy 358, 121958.
- Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical
  Reasoning in Open Language Models.*
  [arXiv:2402.03300](https://arxiv.org/abs/2402.03300).
- Wang et al. (2025). *RAGEN: Understanding Self-Evolution in LLM Agents via
  Multi-Turn Reinforcement Learning.*
  [arXiv:2504.20073](https://arxiv.org/abs/2504.20073).
- Williams (1992). *Simple Statistical Gradient-Following Algorithms for
  Connectionist Reinforcement Learning.* Machine Learning 8, 229–256.
- Zhou et al. (2024). *ArCHer: Training Language Model Agents via
  Hierarchical Multi-Turn RL.* ICML 2024.
  [arXiv:2402.19446](https://arxiv.org/abs/2402.19446).
- TRL documentation, GRPO trainer:
  [huggingface.co/docs/trl/main/en/grpo_trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- Unsloth Gemma-4 training guide:
  [unsloth.ai/docs/models/gemma-4/train](https://unsloth.ai/docs/models/gemma-4/train)
