# Progress Log

> Update this file after EVERY work session. Claude Code reads this first.

## Current status

**Phase:** Phase 1 + Phase 2 zero-shot complete; transitioning into Phase 3 (fine-tuning) via SAC→SLM distillation.
- Phase 1 — `src/env.py`, `src/eval.py` populated; RBC + SAC baselines benchmarked in `01_env_setup.ipynb`
- Phase 2 — zero-shot LLM-as-policy working end-to-end: remote APIs in `02_llm_policy.ipynb` (Anthropic, DeepSeek, Kimi, NVIDIA NIM, Gemma); local SLMs on Colab in `03_slm_colab.ipynb` (Qwen, Phi, Llama, Gemma). Reusable code in `src/agent.py`, `src/providers.py`, `src/rollout.py`.

**Working on (IN PROGRESS, not completed):**
- `04_sac_distill_dataset.ipynb` — full-year SAC rollout dumped to `(state_text, action_token)` JSONL via `src/sft.py`
- `05_sft_gemma_colab.ipynb` — LoRA SFT on Gemma in Colab via Unsloth, consuming the JSONL above
- `src/sft.py` — `action_to_token` (11-bucket discretisation), `dump_sac_trajectory_jsonl`, `make_sft_prompt`

**Blockers:** None
**Compute:** MacBook Air (local dev / nb 04 dataset gen), Google Colab Pro (SAC full-scale training, SFT in nb 05)
**Next step:** Run distillation dataset generation end-to-end → run LoRA SFT on Colab → evaluate fine-tuned SLM in CityLearn against zero-shot SLM and SAC baselines (validation gate: ≥70 % of SAC).

---

## Log

### 2026-05-15 — SFT dataset diagnosis + no-op relabel [LOCAL]
- **Why the SFT distillation gave bad results:** ~50 % of all labels in the
  SAC-distill JSONL were physical no-ops. 77 % of the teacher's DISCHARGE
  labels were discharges from an empty battery (SoC≤2 %) — clipped to zero by
  CityLearn, so identical to IDLE. Discretisation turned a clipped float into
  a confident `DISCHARGE_40` token. The SLM minimised loss by always emitting
  `DISCHARGE_20` (42.6 % majority class) → degenerate distilled policy.
- **Root cause upstream:** the SAC teacher is undertrained (30 episodes,
  Phase I 0.824 vs RBC 0.942) and barely cycles the batteries — mean SoC
  14–29 %/building, empty 48–72 % of the time.
- **Fix #1 done — `src/sft.py`:** `action_to_token` gained a `soc` arg;
  `format_action_block` a `socs` arg; `dump_sac_trajectory_jsonl` a
  `relabel_noops=True` flag (+ `n_relabeled` stat). A discharge at SoC≤3 %
  or a charge at SoC≥97 % is now written as IDLE — physically exact, and
  stops cloning a token that is harmful in non-empty/full states. Raw SAC
  float still kept in `actions_float`.
- **Effect (measured on existing JSONL):** post-`filter_uninformative_rows`
  token mix goes from DISCHARGE_20-dominated/no-op-poisoned to IDLE 48 %,
  CHARGE_20 15 %, CHARGE_40 11 %, DISCHARGE_20 18 %, DISCHARGE_40 7 % —
  honest labels. IDLE share is still high because the teacher genuinely
  idles that much.
- **Still open (not done):** retrain SAC much longer (the real ceiling fix);
  consider class-balancing IDLE; gate teacher quality before dumping.
- **Re-run scope:** nb 04 must be re-run to regenerate the JSONL with the
  relabel (no SAC retrain needed — reuse the pickle via § 4b).

### 2026-05-14 — Project-wide code review + 22 fixes [LOCAL]
- **Scope:** end-to-end review of `src/*.py`, `configs/experiment.yaml`,
  and notebooks 01 / 04 / 05 / 06 / 07 ahead of the Phase 3 RL run.
  Full findings table and per-file changes in [REVIEW_2026-05-14.md](REVIEW_2026-05-14.md).
- **Critical fixes (will corrupt results or crash a run):**
  - **nb 05 N_BUILDINGS=3** (#1): previous setup told the model "manage 6
    buildings" while every JSONL row was 3-bldg state + 3-line response,
    AND eval rolled out on a 6-bldg env the SLM had never seen — headline
    Phase I was uninterpretable. `make_colab_env` now defaults to
    `TRAINING_BUILDINGS=[0,1,2]` and routes through `src.env.make_env`.
  - **nb 06 `NameError: _ACTION_RE`** (#2): cherry-picked the e543a387 fix
    that was applied to nb 05 but missed nb 06.
  - **nb 07 LoRA warm-start was silent** (#6): old `try/except` could
    swallow a `load_adapter` failure and degrade "RL from SFT" to "RL from
    base". Now stamps the SFT state_dict via `set_peft_model_state_dict`
    and **hard-asserts** that a probe LoRA-A weight changed.
  - **nb 07 KL term could go negative** (#7): `(lp_pol - lp_ref).mean()`
    rewards drift for finite samples. Replaced with the k3 estimator
    (`exp(log_ratio) - 1 - log_ratio`), element-wise ≥ 0.
  - **nb 07 prompt was OOD vs SFT** (#8): used `make_minimal_prompt` (CoT)
    against an SFT'd model trained on `make_sft_prompt` (no-CoT). Same
    failure mode as the historical nb05 § 19 blowup.
  - **prompt buckets didn't match real labels** (#5): `make_minimal_prompt`
    and `make_sft_prompt` advertised `price (LOW / MID / PEAK)` and
    `solar (NONE / LOW / MID / HIGH)`, but the bucket fns only emit 2 and
    3 levels respectively. Prompts now match reality.
- **High-impact fixes:**
  - **action_to_token banker's rounding bias** (#10): SAC actions at
    0.50/0.70/0.90 were squeezed into wrong buckets (e.g. 0.50→CHARGE_40,
    should be CHARGE_60). Replaced with integer round-half-up — uniform
    0.20-wide buckets, symmetric for charge/discharge.
  - **nb 07 rollout window off-by-one** (#9): `end=t0+WINDOW_STEPS` →
    `end=t0+WINDOW_STEPS-1` (CityLearn end is inclusive).
  - **nb 01 SIM_END 8758→8759** (#11): one step short of a year; aligned
    with `src/env.py`.
  - **nb 01 EcoPeakBatteryReward** (#4, #23): inline copy diverged from
    `src/env.py` (penalised exports). Both copies now use a
    **district-level** peak term (sum first, then square, clamped
    non-negative), distributed across buildings.
  - **ZNE column validation** (#15): `src.eval.zne_metric` now raises
    `KeyError` if expected CityLearn columns are missing rather than
    silently defaulting `imp = 1.0`.
- **Medium fixes:** dump_sac_trajectory seed thread (#17); nb 04 render
  flag removed for dump-only env (#18); nb 06 pip version pins quoted
  (#19); nb 06 TRAIN_BUILDINGS aligned to TRAINING_BUILDINGS (#20); nb 06
  uses `src.env.make_env` for matching schema source (#21); configs/
  experiment.yaml rewritten to current state (#22).
- **Cleanup:** irradiance constants + `irradiance_bucket` removed (#24) —
  the observation is no longer used anywhere in the pipeline.
- **Retracted (#16):** initial claim that SAC trained on a stale obs
  vector (SoC always 0). Verified in `citylearn==2.5.0` source — bug is
  real on 2.5. **Fixed in 2.6.0b2** via `endogenous_t = max(t-1, 0)` in
  `BuildingOpsService.get_observations_data()`. Project pins 2.6.0b2 in
  every install cell so SAC training was correct all along. My test ran
  against system-Python 2.5.0, not the Colab 2.6.0b2 the actual pipeline
  uses. `docs/CITYLEARN_INSIGHTS.md` § 1 updated to mark this as
  2.5-only and the workaround as version-independent.
- **Open items:**
  - **Gemma `RESPONSE_TEMPLATE`** (#13) — `"<|turn>model\n"` doesn't look
    like a Gemma chat marker (real is `<start_of_turn>model\n`). If the
    marker is wrong, every prior SFT run was effectively training on
    prompt boilerplate. **Awaiting cell 19 paste-outputs from user
    before editing.**
  - **#3 sys_p_cot undefined** in nb 05 cells 44/46 — user planned to
    delete those cells.
- **Re-run scope:**
  - nb 04 — recommended (rounding fix changes ~10–15% of JSONL rows). Can
    re-use the existing SAC pickle (no SAC retrain).
  - nb 05 — required (N_BUILDINGS=3 fundamental fix).
  - nb 06 — required (was crashing).
  - nb 07 — not yet run; use fixed version.
  - nb 01 — optional (SIM_END off-by-one and eco-reward fix shift Phase I
    by <10⁻³; only retrain for bit-exact match).

### 2026-05-13 — Colab CityLearn-version fixes + SAC distillation dataset pushed [LOCAL]
- Aligned `WEEK_START = 3624` across nb 02 and nb 03 (nb 03 had 2624) so remote-API and local-SLM zero-shot results are on the same window for direct comparison.
- Pinned CityLearn 2.6.0b2 in nb 03 and nb 05 install cells (both were resolving to 2.5, which only has the legacy `env.evaluate()` and crashed `src.eval` calls). All three Colab notebooks (03, 05, 06) now use the same install pattern: `CITYLEARN_VERSION = "2.6.0b2"` + `pip install --pre --no-deps` + `startswith("2.6")` assertion.
- Ran nb 04 end-to-end on MacBook: SAC trained on 6-building district, full-year rollout dumped → **17,520 JSONL rows** (8,760 env steps × 2 slices: `[0,1,2]` and `[3,4,5]`), 10 MB, committed at `notebooks/artifacts/sft_datasets/sac_merlin_distill_20260512_212359.jsonl`. nb 05 on Colab picks up the newest matching file via its glob.
- Committed nb 01–04 post-run outputs.
- **Next:** run nb 05 SFT on Colab → adapter to Drive → nb 06 generalization eval → validation gate (≥70% of SAC Phase I) before any RL phase.

### 2026-05-12 — src/ + notebook consistency pass: single source of truth [LOCAL]
- **Goal:** notebooks define things analytically once (nb 01) and import from `src/` thereafter. No silent duplication.
- **`src/` deduplication:**
  - `src/sft.py` now re-exports `render_state`, bucket fns, thresholds, `parse_actions`, `ACTION_RE` from `src.agent` (was full inline duplicate). `_ACTION_RE` kept as legacy alias. Single source of truth for state rendering + action parsing.
  - `district_kpis` removed from `src/rollout.py` — the only canonical one is `src.eval.district_kpis` (evaluate_v2 based, CityLearn 2.6+).
  - `OBSERVATIONS_LLM` and `OBSERVATIONS_SAC` collapsed into one `OBSERVATIONS` constant in `src/env.py`. Old names kept as aliases for back-compat.
- **Notebooks slimmed (~25 KB inline code removed):**
  - **nb 02** (`02_llm_policy`) — dropped ~14 KB inline cell (buckets, render_state, parse_actions, make_minimal_prompt, APIProvider, make_policy_llm, reference policies). All imports from `src.agent`, `src.providers`, `src.eval`.
  - **nb 03** (`03_slm_colab`) — same pattern; dropped ~17 KB (incl. inline `LocalHFProvider`, ~6 KB). `make_colab_env` kept for Colab schema auto-download.
  - **nb 05** — dropped inline `make_minimal_prompt`; CoT prompt now imported from `src.agent` for OOD comparison eval (§ 19). Renamed local `_ACTION_RE` → `_ACTION_TAG_RE` to disambiguate from the canonical `src.agent.ACTION_RE` (opening-tag check vs full parser).
  - **nb 06** — same `_ACTION_TAG_RE` rename; `OBSERVATIONS_LLM` → `OBSERVATIONS`.
- **Prompt policy:** `src.agent.make_minimal_prompt` is the **canonical CoT prompt** (with `<thought>` block). `src.sft.make_sft_prompt` is the **SFT-only no-CoT variant** because the SAC-distilled JSONL has no rationales. Both docstrings updated. Cross-prompt drift caused the nb05 CoT eval blowup — never apply make_minimal_prompt to an SLM fine-tuned on make_sft_prompt without re-distilling with synthesised thoughts.
- **Eval everywhere:** every notebook except 01 imports `from src.eval import evaluate, comparison_table` and uses `evaluate_v2()` under the hood (CityLearn 2.6+). nb 05/06 v1 deleted; v2 → canonical.
- **Verification:** `render_state`, `parse_actions`, `ACTION_RE` are the *same Python objects* across `src.agent` and `src.sft` (`is` check passes). `OBSERVATIONS_*` aliases all resolve to the same list.
- **Remaining "redefs" are intentional thin wrappers:** nb 01's `challenge_score`/`zne_metric` (pedagogical analytical defs); nb 02/03 `summarize_district(df, label)` (binds `n_buildings=N_BLDGS`, calls `src.rollout.summarize_district`); nb 02/03 `run_policy` (binds `env_factory`); nb 06 `make_env` (Colab schema-by-name).

### 2026-05-12 — Single-agent design decision: Phase 3 trains ONE SLM on 3 buildings [LOCAL]
- **Decision (after supervisor discussion):** until Phase 4, all SLM training and
  evaluation uses a SINGLE group-centralized agent over 3 buildings, not the
  dual-agent setup of nb 02/03. One inference call per step instead of two
  during Phase 3 SFT + RL. Phase 4 deployment still uses two agents; the same
  trained LoRA loads into both — no retraining.
- **Building split:**
  - `TRAINING_BUILDINGS = [0, 1, 2]` — Phase 3 train + in-distribution eval; also Phase 4 agent α
  - `HELDOUT_BUILDINGS  = [3, 4, 5]` — unseen-buildings generalization test (RQ2); also Phase 4 agent β
  - `BUILDINGS = [0..5]` — full district, Phase 4 dual-agent rollout
  - `UNSEEN_BUILDINGS = [6..11]` — OOD generalization (different buildings, same 2022 dataset)
- **SAC teacher retraining: NOT NEEDED.** SAC was trained `central_agent=False`
  (per-building policies, independent) → slicing rollouts to {0,1,2} or {3,4,5}
  is distribution-clean. We dump the existing 6-building SAC's trajectory and
  emit two 3-building rows per env step ({0,1,2} + {3,4,5}) → 2× SFT data, and
  the SLM becomes building-agnostic within the 3-building shape.
- **Why group-centralized over 3 instead of fully decentralized 1/bldg:**
  intra-group coordination is the SLM's strength (multi-building context in
  one prompt), inference cost is 1 call/step not 3, and this matches the
  Phase 4 research question (implicit coordination *across* the group boundary
  with no comms, while each agent coordinates *within* its group).
- **Env config:** `central_agent=False` everywhere (Phase 3 and Phase 4) — the
  flag controls env I/O shape, not policy count. Joint reward at Phase 4 is
  computed in the rollout loop by summing the per-building reward list.
- **Code changes (this commit):**
  - `src/env.py` — added `TRAINING_BUILDINGS`, `HELDOUT_BUILDINGS` constants;
    documented building-set conventions; `make_env` default unchanged (still
    `BUILDINGS`) — new code opts in explicitly via `buildings=TRAINING_BUILDINGS`.
  - `src/rollout.py` — `run_policy_dual_agent` docstring updated: PHASE 4 ONLY.
    `run_policy` (single-agent) is the default through Phase 3. No behavior changes.
  - `src/sft.py` — `dump_sac_trajectory_jsonl` gained `building_slices` arg
    (list of index-lists); SAC still acts on the full env, but JSONL output is
    sliced per row. New rows include a `slice` field. `make_sft_prompt` default
    changed from 6 → 3 buildings.
  - `notebooks/04_sac_distill_dataset.ipynb` — SAC still trains/evaluates on
    6 buildings; dump cell now passes `building_slices=[TRAINING_BUILDINGS,
    HELDOUT_BUILDINGS]` → JSONL has 2× rows, each with 3 buildings. Prompt
    template display switched to `make_sft_prompt(3)`. Sanity cell shows one
    row from each slice.
- **Notebooks 01/02/03 (Phase 1/2 zero-shot) left as-is** — they are complete
  experiments. Any rerun would need to pass `buildings=BUILDINGS` explicitly
  (still the `make_env` default, so they work unchanged).
- **Next:** rerun nb 04 end-to-end → produce the 17,520-row JSONL → push for nb 05 SFT on Colab.

### 2026-05-10 — Phase 2→3 transition: SAC→SLM distillation pipeline scaffolded [LOCAL]
- Confirmed Phase 1 + Phase 2 zero-shot are complete and stable: notebooks 01/02/03 work end-to-end; `src/` cleanly hosts env, agent, providers, rollout, eval.
- New work-in-progress (commits `cca9eb11`, `c943a802`):
  - `notebooks/04_sac_distill_dataset.ipynb` — runs trained SAC for one full CityLearn year and dumps per-step `(state_text, action_token)` pairs as JSONL for SFT.
  - `notebooks/05_sft_gemma_colab.ipynb` — Colab notebook for LoRA SFT on Gemma using Unsloth, consuming the JSONL produced by nb 04.
  - `src/sft.py` — distillation helpers: `action_to_token` discretises continuous SAC actions in `[-1, 1]` into the same 11-bucket vocabulary the inference prompt uses (`CHARGE_20…100`, `IDLE`, `DISCHARGE_20…100`, 20% steps); `dump_sac_trajectory_jsonl` for dataset emission; `make_sft_prompt` (drops `<thought>` block — distilling without rationales).
- **Status: pipeline scaffolded but experiments not yet run** — dataset generation and Colab fine-tuning runs still pending.
- Updated `CLAUDE.md` (Current phase + project-structure tree) and this file to reflect the actual state of `src/` (6 modules, not the 5 originally planned: `agent.py`, `env.py`, `eval.py`, `providers.py`, `rollout.py`, `sft.py`; planned `rl.py`/`utils.py` not yet created).

### 2026-05-07 — src/eval.py: standardised evaluation module [LOCAL]
- Created `src/eval.py` — all KPI logic extracted from `notebooks/01_env_setup.ipynb`:
  - `CHALLENGE_KPIS` — mapping of short names to CityLearn v2 `evaluate_v2()` column names
  - `district_kpis(env)` — pulls district-level rows from `evaluate_v2()` as a Series
  - `challenge_score(env, label)` — computes C, G, R, 1-L, Phase I `(C+G)/2`, Combined `(C+G+D)/3`
  - `zne_metric(env, label)` — solar generation, grid import, ZNE ratio, self-consumption ratio
  - `evaluate(env, label)` — runs both above in one call, returns an `EvalResult` dataclass
  - `comparison_table(results)` — builds challenge + ZNE DataFrames from a list of `EvalResult`s
  - `generalisation_gap(train, unseen)` — Phase I and Combined gap between two `EvalResult`s
  - `EvalResult` dataclass with `.phase1` and `.combined` convenience properties
- `01_env_setup.ipynb` is now fully reflected in `src/`: env factory in `env.py`, KPIs in `eval.py`
- Future notebooks/scripts use: `from src.eval import evaluate, comparison_table, generalisation_gap`

### 2026-05-05 — 03_slm_colab: self-contained notebook, minimal prompt [LOCAL]
- **Architectural shift**: `03_slm_colab.ipynb` is now fully self-contained for SLM
  experimentation — mirrors the Phase 1 pattern (experiment in notebook → promote to
  src/ only when mature)
- Removed all imports from `src/agent.py`; defined inline in notebook:
  - `PRICE_PEAK_THRESHOLD`, `IRRADIANCE_LOW/HIGH_THRESHOLD`
  - `price_bucket`, `carbon_bucket`, `solar_bucket`, `irradiance_bucket`
  - `render_state()` — converts snapshot dict list to LLM prompt string
  - `_ACTION_RE`, `parse_actions()` — XML tag action extraction with [-1,1] clip
  - `make_policy_llm()` — binds LocalHFProvider into rollout-compatible policy fn
- Added `make_minimal_prompt(n_buildings)` — the new default prompt:
  - Task context + state variable meanings + output format only
  - NO prescribed rules — SLM decides its own strategy
  - ~120 words vs ~190 for rules-based prompt
- Added `make_rules_prompt(n_buildings)` — kept as comparison baseline:
  - Numbered priority rules (first-match-wins), same as old `make_slm_system_prompt`
  - Easy swap: uncomment one line in § 10 `run-slm` cell
- `LocalHFProvider.step()` now defaults to `make_minimal_prompt` (not `make_system_prompt`)
- `src/env.py` is still imported (SEED, BUILDINGS, snapshot_state, reward fns) — stable
- `src/agent.py` is NOT imported by notebook 03 at all
- § 6b updated: Minimal vs Rules comparison table (dropped Full API prompt comparison)
- `warmup` cell uses `make_minimal_prompt` for accurate per-call timing estimate
- Title updated to reflect Phase 2 + self-contained design philosophy

### 2026-05-05 — 03_slm_colab merge: V2 Colab fixes + prompt/timing fixes [LOCAL]
- Merged 03_slm_colabV2.ipynb (user's working Colab version) into 03_slm_colab.ipynb
- V2 changes preserved (necessary to run on Colab):
  - CityLearn install split into two steps: deps first (numpy/gymnasium/doe-xstock/
    nrel-pysam), then `citylearn --no-deps` to avoid pip resolver conflicts
  - `LocalHFProvider.complete()`: Gemma system-role workaround (Gemma rejects "system"
    role — system prompt merged into user message); `return_dict=True` in
    `apply_chat_template`; handles both Tensor and BatchEncoding return types
  - `_is_gemma` flag added to `__init__`
  - Real GitHub URL: `https://github.com/antonisbast/eclipse-thesis`
  - Model: `meta-llama/Meta-Llama-3-8B-Instruct` with `LOAD_IN_4BIT=True`
  - Utility cells: rm-rf (fresh clone), debug paths (verify clone), Drive mount
  - `MOUNT_DRIVE=True` (user always mounts Drive)
- Our fixes also applied:
  - `make_slm_system_prompt` imported and used in run-slm cell
  - `MAX_NEW_TOKENS=150` with runtime tradeoff comment (was 400)
  - Warmup cell uses realistic state prompt for accurate timing estimate
  - § 6b prompt comparison table + code cell
  - `|` syntax typo fixed in timing-analysis cell
  - VRAM table updated to include Llama-3-8B row
- 03_slm_colabV2.ipynb kept as reference; 03_slm_colab.ipynb is the canonical version

### 2026-05-05 — SLM prompt + timing fixes [LOCAL]
- `src/agent.py`: added `make_slm_system_prompt(n_buildings)` — compact prompt for ≤4B models:
  - No "think step by step" REASONING PROTOCOL (was the main cause of 35-min runtime)
  - 7 numbered priority rules (first-match-wins) instead of prose strategy section
  - Output instruction: "these N lines only, nothing else" (stronger than "strict")
  - ~40% fewer prompt tokens → faster prefill on every call
  - Designed for MAX_NEW_TOKENS ≤ 150; generates ~40-80 tokens vs 100-300 with full prompt
- `notebooks/03_slm_colab.ipynb` updated:
  - `config`: MAX_NEW_TOKENS 400 → 150 (with explanation of runtime tradeoff)
  - `imports`: added `make_slm_system_prompt` import
  - `warmup`: fixed timing estimate — now uses realistic state + SLM system prompt instead
    of trivial "Say READY" (which generated 1 token and gave 10× optimistic estimate)
  - New § 6b: comparison table (full vs SLM prompt) + code cell showing both
  - `run-slm`: now passes `system=make_slm_system_prompt(3)` to both agents
  - Expected improvement: 35 min → ~10 min for 168-step dual-agent rollout on T4
- Confirmed KPI evaluation is correct for dual-agent: both agents' actions combine into
  a single env.step() call on the shared 6-building env; env.evaluate() sees the full
  trajectory independent of the agent split.

### 2026-05-05 — 03_slm_colab.ipynb: local SLM inference on Colab GPU [LOCAL]
- Created `notebooks/03_slm_colab.ipynb` — fully self-contained Colab notebook
- **LocalHFProvider** class defined inline: same `.complete()` / `.step()` interface as
  `LLMProvider` so `make_policy_llm()` and all rollout functions work unchanged
  - Greedy decoding (`do_sample=False`) — deterministic, reproducible
  - No timeout needed (local GPU calls finish in < 2 s)
  - Qwen3 thinking mode disabled (`enable_thinking=False`) — 2× faster
  - Retry logic: up to `max_retries=2` on missing action tags, then zeros
- **make_colab_env()** passes dataset name as string to `CityLearnEnv` → auto-download
- Dual-agent setup: α controls B0-2, β controls B3-5, same as `02_llm_policy`
- Model presets in config cell (Qwen2.5-1.5B default, Qwen3-4B, Phi-3.5-mini, Qwen3-8B)
- 4-bit quantization support (bitsandbytes) for ≥7B models on T4
- Drive mount option — set MOUNT_DRIVE=True to persist results across sessions
- § 13.4 timing analysis: tokens/call, tokens/s, fallback rate
- Estimated rollout time: ~2 min (1.5B on T4) vs 1+ hour for remote API calls

### 2026-05-05 — Smoke-test split + forecast labels in notebook [LOCAL]
- `src/agent.py`: fixed `ThreadPoolExecutor` shutdown bug in `complete()` — replaced
  `with ThreadPoolExecutor() as ex:` (calls `shutdown(wait=True)` on any exception, blocking
  indefinitely) with explicit executor + `executor.shutdown(wait=False)` in all code paths
  (success, TimeoutError, other exception). Hung NVIDIA/slow calls now return in ≤ timeout_s.
- `notebooks/02_llm_policy.ipynb` — § 2, § 3, § 5 updated:
  - `env-check` (§ 2): expanded to show all 12 snapshot fields (9 real-time + 3 forecasts),
    prints forecast availability at t=0, shows price+6h and irr+6h per building
  - `s3-header` (§ 3): updated to mention 12-field snapshot, added example of the
    `Forecast: price+6h=X  price+12h=Y  solar+6h=Z` line now shown in rendered state
  - `s5-header` (§ 5): rewritten to explain per-provider cell structure
  - `provider-setup` (§ 5, cell 1): now only initialises `PROVIDER_OBJS = {}`
  - Added 5 individual smoke-test cells (one per provider: anthropic, deepseek, kimi,
    nvidia, gemma) — each independently interruptible; NVIDIA gets 30 s (cold-start headroom)

### 2026-05-05 — Google Gemma + forecast variables in state/prompt [LOCAL]
- `src/env.py` — `snapshot_state()` extended with 3 forecast fields:
  - `electricity_pricing_predicted_1` — price +6 h ($/kWh), via `b.pricing.electricity_pricing_predicted_1[t]`
  - `electricity_pricing_predicted_2` — price +12 h ($/kWh)
  - `solar_irradiance_predicted_1` — diffuse+direct irradiance +6 h (W/m²), via private
    `b.weather._diffuse/direct_solar_irradiance_predicted_1[t]`; reads wrapped in try/except
    so missing forecast columns degrade gracefully to None
- `src/agent.py`:
  - Added `IRRADIANCE_LOW_THRESHOLD=50`, `IRRADIANCE_HIGH_THRESHOLD=600` (W/m²)
  - Added `irradiance_bucket(v)` → NONE/LOW/HIGH (returns '?' on None)
  - `render_state()` now inserts `Forecast: price+6h=X  price+12h=Y  solar+6h=Z` between
    header and buildings; uses `price_bucket()` for price forecasts (same $/kWh scale),
    `irradiance_bucket()` for solar irradiance (W/m²)
  - `make_system_prompt(n)` rewritten: FORECAST VARIABLES section, expanded STRATEGY RULES
    to 6 rules (4 price-regime combos + solar headroom + limits), updated REASONING PROTOCOL
    Step 1 to read forecasts before deciding actions
- `notebooks/02_llm_policy.ipynb`:
  - Added Google Gemma (`gemma-3-12b-it`, Google AI Studio OpenAI-compat, `GOOGLE_API_KEY`)
    as 5th provider; standalone § 13 cell; sections renumbered § 13→14 through § 16→17

### 2026-05-05 — Dual-agent notebook + timeout + NVIDIA NIM [LOCAL]
- `src/agent.py` updated:
  - `make_system_prompt(n_buildings)` — parametric prompt; action-format block and peak-demand
    estimates scale automatically (used for both 3-building agents and 6-building single-agent)
  - `SYSTEM_PROMPT` kept as `make_system_prompt(6)` for backward compatibility
  - `complete()` gains `timeout_s` param via `ThreadPoolExecutor.result(timeout=...)`;
    raises `TimeoutError` on expiry, cancels the in-flight thread best-effort
  - `step()` breaks immediately on `TimeoutError` (no retry — a hung endpoint stays hung);
    API errors still retry up to `max_retries` times with 1 s backoff
  - `make_policy_llm()` gains `n_buildings`, `agent_label`, `system`, `timeout_s` params;
    `agent_label` ("α"/"β") appears in every verbose print line
- `notebooks/02_llm_policy.ipynb` rewritten as dual-agent experiment:
  - Agent α controls B0-B2, Agent β controls B3-B5 (partial observability, mirrors Phase 4)
  - Two LLM calls per timestep; actions combined in global building-index order before env.step()
  - NVIDIA NIM added as 4th provider (`meta/llama-3.1-8b-instruct`, `https://integrate.api.nvidia.com/v1`)
  - `LLM_TIMEOUT_S = 45.0` config constant; each call hard-stops and returns zeros on expiry
  - **One cell per provider** (§ 9–12) — interrupt a hung cell without losing other results
  - `llm_runs` dict initialised in § 8b; provider cells append to it; results cells gracefully
    handle any subset (empty, partial, full)
  - § 14 per-agent breakdown: reward split α/β, fallback counts per agent, mean SoC, peak net
  - § 15 diagnostics: SoC coloured by agent group (blue=α, red=β), district net load, behaviour
    table with sync_rate/fallback/rule-violations per agent, raw response sample

### 2026-05-04 — src/ modules + 02_llm_policy notebook [LOCAL]
- Populated `src/env.py`:
  - `MERLINReward`, `EcoPeakBatteryReward` (extracted from 01 notebook)
  - `make_env()` — supports `start`/`end` windowing, `obs_set` (`sac`=13 vars, `llm`=9 vars), `reward_fn`
  - `snapshot_state()` — bypasses obs-vector SoC bug by reading building objects directly
  - `OBSERVATIONS_SAC` (13 vars, with forecasts) and `OBSERVATIONS_LLM` (9 real-time, no forecasts)
  - Absolute `DATASET_ROOT` via `Path(__file__).parent.parent` — importable from any notebook
- Populated `src/agent.py`:
  - `price_bucket`, `carbon_bucket`, `solar_bucket`, `render_state()` — state-to-text pipeline
  - `SYSTEM_PROMPT` — battery physics + strategy rules + strict XML output format
  - `LLMProvider` — uniform Anthropic / OpenAI-compat wrapper with `.complete()` and `.step()`
  - `parse_actions()` — regex parser with per-building fallback to 0.0 and [-1,1] clip
  - `make_policy_llm()` — binds a provider into a rollout-compatible policy function
- Created `notebooks/02_llm_policy.ipynb`:
  - Imports env/agent entirely from `src/` (no logic in cells)
  - 11 sections: config → imports → env → renderer → LLM interface → reference policies → baselines → LLM runs → results → diagnostics → save
  - Same 1-week window (t=3624, 168 steps) as `04_llm_policy_clean.ipynb` for direct comparison
  - Saves rollout CSV, KPI CSV, behaviour CSV, raw JSON logs per provider

### 2026-05-04 — Project cleanup + docs reorganization [LOCAL]
- Updated `docs/CONTEXT.md` with accurate thesis scope and four-phase plan
- Renamed `CLAUDE_CITYLEARN_INSTRUCTIONS.md` → `docs/CITYLEARN_API.md` (CityLearn v2 API reference)
- Renamed `citylearn_insights.md` → `docs/CITYLEARN_INSIGHTS.md` (observation quirks, battery dynamics, prompting tips)
- Rewrote `CLAUDE.md` to align with revised research questions and actual folder structure
- `src/` and `scripts/` are confirmed empty stubs — to be populated in Phase 1 (SAC) and Phase 2 (SLM)

### 2026-04-30 — Cleanup + archive of exploratory work [LOCAL]
- Built `notebooks/04_llm_policy_clean.ipynb` — clean rewrite of `04_llm_policyV3` with:
  - Single config cell, no hardcoded API keys (env var only)
  - Solar bucket added to state renderer (NONE/LOW/HIGH)
  - System prompt updated with battery asymmetry insight (charge small, discharge full = safe)
  - `make_env(start, length)` parametric (no global mutation)
  - RBC baseline added alongside no-op/random/LLM
  - Clean model routing — no broken `responses` API fallback
- Moved old work to `archive/`:
  - `archive/notebooks/` ← 8 versioned notebooks (`01_environment_and_experts` … `04_llm_policyV3`)
  - `archive/root_notebooks/` ← `ECLIPSE_Meeting1_v3`, `ECLIPSE_diagnostic`, `citylearn_ccai_tutorial`
  - `archive/notebook_generators/` ← 4 `_gen_*.py` scripts
  - `archive/CLAUDE_CODE_PROMPT.md` — one-off prompt file
- `notebooks/` now contains only `04_llm_policy_clean.ipynb` (LLM-as-policy baseline)
- See `archive/README.md` for index of archived files



### 2026-03-21 — CityLearn v2 environment exploration [LOCAL]
- Installed CityLearn v2.5.0 (latest) on Python 3.13 — works despite version pin mismatches
- Created `sandbox/` folder with 5 exploration scripts + shared `_env_helpers.py`
- **01_basic_env.py**: 4-building env, 28 obs dims per building, 1 action (battery charge/discharge [-1,1])
- **02_explore_observations.py**: Full year (8760 steps) with plots of solar, SoC, net load, pricing
- **03_simple_rules.py**: Do-nothing vs peak-shaving baselines. Peak shaving saves ~5% on cost
- **04_ppo_baseline.py**: PPO via SB3 with CityLearn wrapper. 50K steps (5 episodes) — not enough to beat rules
- **05_observation_to_text.py**: 3 prompt formats (terse/medium/verbose) + robust action parser with 12 test cases

Key findings:
- CityLearn v2.5 `electrical_storage_soc` in observations shows next-step initial value (always 0), NOT current SoC — must read from `building.electrical_storage.soc[t]` directly
- Observations are not raw values — some are normalized/transformed. Use building objects for decision-making
- Battery works correctly (6.4 kWh capacity, 5 kW nominal power, 90% efficiency)
- CityLearn data is cached locally after first download — use schema path to avoid GitHub API rate limits
- Dataset: `citylearn_challenge_2022_phase_all`, pricing has binary low/high ($0.22/$0.54)
- Episode terminates at step 8759 (not 8760) — handle terminated/truncated signals
- PPO needs significantly more than 50K steps to learn (only ~5 episodes with 8760-step episodes)

Plots saved in `sandbox/plots/`:
- solar_generation.png, battery_soc.png, net_electricity_load.png, electricity_pricing.png
- ppo_training_curve.png, strategy_comparison.png

Next steps:
- Build proper Gymnasium wrapper in `src/env.py` based on the CityLearnWrapper from script 04
- Start notebook 01 with environment setup narrative
- Scale up PPO training on Colab (500K+ timesteps)

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
