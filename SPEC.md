# Phase 1: Base Model Selection + SFT Cold Start — τ³-bench

**Owner:** Debdoot Mukherjee
**Executor:** Claude Code (autonomous with checkpoints)
**Duration:** ~3–5 weeks (base-model bake-off: 3–5 days; teacher rollouts: 5–10 days; SFT training + eval: 5–10 days)
**Compute target:** 1× H200 (141 GB) for serving + LoRA SFT, OR 2× H100 (160 GB) for full-parameter SFT on Qwen3-8B

---

## 1. Context & Goal

This spec covers Phase 1 of the broader plan to top τ³-bench. Phase 0 (`/home/jovyan/tau-bench-phase0`) established baselines for Qwen3.6-35B-A3B at airline pass^4 = 0.680, retail = 0.632, telecom = 0.974 (inflated; strict subset 0.850). Phase 1 has two objectives:

1. **Pick the right base model.** For an open-weight leaderboard run, base-model choice dominates. We bake off Qwen3-32B vs GLM-4.5-Air vs Qwen3-8B on the Phase 0 harness and pick one.
2. **SFT cold start.** Generate 20–50K telecom-heavy trajectories from a strong teacher (Claude Opus / GPT-5 / DeepSeek-R1) against the τ³-bench gym interface. Apply dual verification. Run SFT on the chosen base. Produce a checkpoint that beats Phase 0's Qwen3.6-35B-A3B baseline on telecom pass^4 by ≥ 5 percentage points.

The exit criterion of Phase 1 is a markdown report (`phase1_results.md`) containing (a) the base-model bake-off table, (b) the SFT data-curation breakdown (yields per teacher, dual-verification pass rates), (c) the SFT'd checkpoint's pass@1 / pass^4 on all three domains, and (d) a frozen artifact ready for Phase 2 RL training.

**This is the cold start for the entire RL effort. Do not skip the verification gates. Garbage trajectories at Phase 1 contaminate every downstream phase.**

---

## 2. Phase 1 in Context: Roadmap for Phases 2–4

Phase 1 produces an SFT'd checkpoint. Phases 2–4 then take that checkpoint to leaderboard performance. Brief, here for handoff continuity — full operational SPECs will be written when each phase begins.

### Phase 2 — User simulator (the silent bottleneck)
- Bias rollouts toward the **Hard persona** (low tech literacy, ambiguous complaints).
- **Rotate user-sim model** across GPT-4.1-mini, Claude Sonnet, Qwen, Llama during training to hedge against overfitting to one LLM's quirks.
- Inject realistic noise beyond τ²'s defaults: misread status bars, skipped instructions, contradictions, silence.

### Phase 3 — RL algorithm: multi-turn GRPO
- **MT-GRPO** turn-level advantages (Zeng et al., arxiv 2505.11821) using outcome + intermediate verifiable rewards (tool-call success, state advance, correct user instruction issued, correct DB mutation).
- **Loss masking is critical** (per MUA-RL): mask gradients on tool-execution outputs and user-simulator messages.
- **GiGPO / RTMC** for step-level credit grouping across rollouts sharing state anchors.
- **DAPO extensions**: Clip-Higher (preserve exploration), dynamic sampling, overlong reward shaping.
- Hyperparams: KL β ∈ [0.001, 0.01], 8–16 rollouts/prompt, rollout temperature 1.0.

### Phase 4 — Reward design for pass^k
- **Group-consistency reward**: sample k trajectories per task; reward each by `group_success_rate × individual_success` (cf. ProxMO). Directly optimizes pass^k surface.
- **Calibration term**: judge-model score on reasoning quality, gated to apply only when outcome is positive — penalizes brittle correct paths without distorting the signal on failures.

### Per-MUA-RL guidance applied across Phase 1+
- For SFT (Phase 1): keep natural-language guidance from teacher in trajectories.
- For RL reward (Phase 3): strip dialogue-content-requirements; reward task-completion only.

---

## 3. Prerequisites & Assumptions

### Hardware
- One H200 (141 GB VRAM) is the default. Sufficient for:
  - Serving any of the three candidate base models (Qwen3-32B at bf16 ≈ 64 GB; GLM-4.5-Air ≈ 90 GB; Qwen3-8B ≈ 16 GB).
  - LoRA SFT on Qwen3-32B or GLM-4.5-Air (adapter only, frozen base).
  - Full-parameter SFT on Qwen3-8B with optimizer state offload.
- Full-parameter SFT on a 32B model needs at least 4× H100 with FSDP — flag for budget approval before electing.
- 500 GB+ free disk on `/workspace` for: model weights × 3 candidates (~150 GB), teacher rollouts (~80 GB raw, ~20 GB after dedupe), SFT checkpoints (~100 GB).
- 128 GB+ system RAM (teacher rollouts spike memory during dataset materialization).

### Software baseline
- Same as Phase 0: Ubuntu 22.04 / 24.04, Python 3.11, CUDA 12.4 (avoid 13.2 — Qwen3 gibberish bug).
- vLLM ≥ 0.19 (matches Phase 0).
- `tau2-bench` v1.0 cloned and installed with the `gym` extra: `uv sync --extra gym`.
- SFT framework: **default to LLaMA-Factory** for ease of LoRA + chat-template handling. Alternatives if LLaMA-Factory blocks: TRL `SFTTrainer`, Axolotl. Decide at Task 7 based on chat-template compatibility with the chosen base.

### Credentials required
- `OPENAI_API_KEY` — for `gpt-4.1` user simulator (leaderboard standard, do not substitute) AND for GPT-5 teacher if selected. Budget: $200–$400 for user-sim across rollouts; $1.5K–$5K for GPT-5 teacher rollouts depending on volume.
- `HF_TOKEN` — for downloading base + comparison weights.
- `ANTHROPIC_API_KEY` — if Claude Opus is the chosen teacher. Budget: $2K–$6K for 20–50K trajectories.
- `DEEPSEEK_API_KEY` — if DeepSeek-R1 is the chosen teacher. Budget: $300–$1K (much cheaper than Opus/GPT-5; trade-off is reasoning quality).
- `WANDB_API_KEY` — strongly recommended for SFT runs. No reason to skip.

### Environment variables to set before starting
```bash
export OPENAI_API_KEY="sk-..."
export HF_TOKEN="hf_..."
export ANTHROPIC_API_KEY="..."   # if Claude is teacher
export DEEPSEEK_API_KEY="..."    # if DeepSeek-R1 is teacher
export WANDB_API_KEY="..."

export WORKSPACE="/workspace/phase1"
mkdir -p $WORKSPACE
cd $WORKSPACE
```

### Decisions confirmed with Debdoot (2026-05-10)

These three choices have order-of-magnitude cost/effort consequences. All confirmed before Task 2 start.

1. **Base model — CONFIRMED: run the bake-off.** Evaluate Qwen3-32B, GLM-4.5-Air, and Qwen3-8B per Task 2 (~30 GPU-h, pass@1 only on 25-task subset). Pick the winner per § Task 2 decision rule. Fallback default if bake-off is interrupted: **Qwen3-8B** (cheapest Phase 3 RL rollouts; MUA-RL precedent). Qwen3-32B is the safer fallback if Qwen3-8B's airline pass@1 trails by >5 pp.
2. **Teacher model for SFT — CONFIRMED: Claude Opus.** Best reasoning quality at telecom-style multi-turn. Budget envelope $2–6K for 30K trajectories. Contingencies: GPT-5 if Anthropic Tier-4 rate limits bind during rollouts; DeepSeek-R1 if budget overruns force a switch (accept lower verification yield).
3. **SFT trajectory volume — CONFIRMED: 30K target, 20K floor.** Telecom-heavy split: 60% telecom / 25% retail / 15% airline. Generate 30K raw; accept the SFT run at Task 6 if dual-verification yield reaches 20K usable. Generate up to 50K only if mid-run yield gate (Task 6.3) shows <50% pass rate.

---

## 4. Success Criteria

Phase 1 is complete when **all** of the following exist:

- [ ] `$WORKSPACE/base_eval/` contains pass@1 / pass^4 numbers for at least the chosen base model on all three Phase 0 domains, evaluated on the Phase 0 harness with no fine-tuning.
- [ ] `$WORKSPACE/sft_data/raw/` contains ≥ 20K teacher trajectories generated against the τ³-bench gym interface.
- [ ] `$WORKSPACE/sft_data/verified/` contains the dual-verification-passed subset (typical yield: 50–70% of raw).
- [ ] `$WORKSPACE/sft_data/dataset.jsonl` is a single training-ready file (chat-template formatted, ≥ 15K turns after dedupe).
- [ ] `$WORKSPACE/checkpoints/sft_v1/` contains a complete SFT checkpoint (LoRA adapter or full weights).
- [ ] `$WORKSPACE/sft_eval/` contains pass@1 / pass^4 for the SFT'd checkpoint on all three Phase 0 domains.
- [ ] `$WORKSPACE/phase1_results.md` exists, with: bake-off table, data-curation funnel, SFT eval table, comparison to Phase 0 baseline, qualitative failure-mode shifts, and reproducibility one-liners.
- [ ] `$WORKSPACE/run_phase1.sh` exists and re-running it from scratch reproduces the SFT'd checkpoint (modulo teacher non-determinism).

**Headline target**: SFT'd checkpoint achieves telecom pass^4 ≥ 0.730 (Phase 0 Qwen3.6-35B-A3B Config A telecom strict subset = 0.850, but on a 35B param model; Phase 1 chosen base may be smaller, so the gate is on improvement against the same base, not against Phase 0). Re-define this concretely after Task 2 once the base is picked.

---

## 5. Tasks (Execute in Order)

### Task 1: Environment Setup

**Goal:** Working Python environment with vLLM, tau2-bench (gym extra), and SFT framework.

1.1. Mirror the Phase 0 venv (or create fresh under `$WORKSPACE/venv`):
```bash
cd $WORKSPACE
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install "vllm>=0.19" openai anthropic transformers accelerate datasets pandas numpy rich wandb
```

1.2. Clone and install τ³-bench with the gym extra (this is the new requirement vs Phase 0):
```bash
cd $WORKSPACE
git clone https://github.com/sierra-research/tau2-bench.git
cd tau2-bench
uv sync --extra gym   # the gym extra is what enables programmatic rollout; required for SFT data gen
```

1.3. Install SFT framework (default LLaMA-Factory):
```bash
pip install llamafactory  # or follow upstream install if pip wheel is stale
# Alternatives: pip install trl axolotl
```

1.4. Verify gym interface importable:
```bash
python -c "from tau2.gym import make_env; print('gym OK')"
```
If the import path is different at the time of execution, search for it: `grep -r "class.*Env" $WORKSPACE/tau2-bench/src/tau2/gym/ | head` and update Task 5 commands to match.

**Acceptance:** All imports succeed; gym smoke import returns no error.

---

### Task 2: Base Model Bake-off

**Goal:** Pick the base model. Run a minimal eval (pass@1 only, single trial) of each candidate on the Phase 0 telecom + airline tasks to compare tool-use priors and thinking-mode behavior. We don't need pass^4 here — just enough signal to pick.

2.1. Download all three candidate weights (only the ones you have time/disk for; minimum is two). Allocate disk before starting:
```bash
mkdir -p $WORKSPACE/models
huggingface-cli download Qwen/Qwen3-32B            --local-dir $WORKSPACE/models/qwen3-32b      &
huggingface-cli download zai-org/GLM-4.5-Air        --local-dir $WORKSPACE/models/glm-4.5-air    &
huggingface-cli download Qwen/Qwen3-8B              --local-dir $WORKSPACE/models/qwen3-8b       &
wait
```
Verify exact HF repo paths at execution time — names and orgs change.

2.2. For each candidate, launch vLLM (in tmux), run airline + telecom on Phase 0's tau2-bench install with `--num-trials 1` and a 25-task subset. Use the wrapper:
```bash
bash scripts/run_base_eval.sh qwen3-32b
bash scripts/run_base_eval.sh glm-4.5-air
bash scripts/run_base_eval.sh qwen3-8b
```
The wrapper writes to `$WORKSPACE/base_eval/<model>/` and computes pass@1 via `tau2 view`.

2.3. **Decision gate.** Pick a base. Record in `$WORKSPACE/phase1_results.md` § "Base model selection" with:
- Per-model pass@1 on airline + telecom
- Tokens-per-second observed (impacts both rollout cost and downstream RL throughput)
- Subjective quality of reasoning traces on 3 sampled telecom trajectories
- Final choice + rationale

**Default decision rule if numbers are within 3 pp:** prefer the smaller model. Smaller base means cheaper rollouts in Phase 3 RL, where rollout cost dominates total compute. Qwen3-8B is acceptable if its airline pass@1 is within 5 pp of the larger candidates.

**Acceptance:** Bake-off table in `phase1_results.md`; `$WORKSPACE/CHOSEN_BASE` file exists containing the model name (consumed by later scripts).

**Failure mode to watch:** if a candidate's pass@1 on telecom is < 0.10, suspect chat-template / tool-parser misconfiguration before concluding the model is bad. Inspect a trajectory; verify both agent and user have tool calls.

---

### Task 3: Stand up Chosen Base on vLLM (production config)

**Goal:** Long-running vLLM server for the chosen base model, identical config to what SFT data gen and downstream eval will use. Reproducibility matters here.

3.1. Read the chosen base from `$WORKSPACE/CHOSEN_BASE`. Launch in tmux:
```bash
CHOSEN=$(cat $WORKSPACE/CHOSEN_BASE)
tmux new -s vllm
source $WORKSPACE/venv/bin/activate

vllm serve $WORKSPACE/models/$CHOSEN \
  --served-model-name $CHOSEN \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --trust-remote-code
```
For GLM-4.5-Air swap `--reasoning-parser` and `--tool-call-parser` per its model card. Verify before serving.

3.2. Smoke test (extends Phase 0's `test_toolcall.py`):
```bash
python scripts/test_serving.py --base-url http://localhost:8000/v1 --model $CHOSEN
```
Acceptance: structured tool call returned; thinking-mode toggle works.

---

### Task 4: τ³-bench Gym Interface — Smoke Test

**Goal:** Verify we can drive the τ³-bench environment programmatically (not via the `tau2 run` CLI). This is the substrate for both SFT data gen and Phase 3 RL rollouts.

4.1. Write a 30-line script that:
- Calls `tau2.gym.make_env("telecom")` (or whatever the actual API is — check at execution time).
- Runs one rollout with a hard-coded sequence of 3 dummy actions.
- Asserts that `step()` returns `(obs, reward, done, info)` in some form.
- Prints the final `info` dict so we can see what state-checking signals are available.

4.2. The output `info` dict should include the per-step verifiable signals we need for Phase 3 reward shaping (DB-state delta, communication-info checks, tool-call validity). Document what's actually exposed in `phase1_results.md` § "Gym signals available".

**Acceptance:** Smoke script runs to completion; `info` dict structure recorded.

**Failure mode:** if the gym extra doesn't expose the per-step signals we need, two options — (a) extend the gym wrapper in tau2-bench (upstream PR), or (b) wrap `tau2 run` JSON logs and re-derive signals offline. (a) is preferred but adds 1–2 days to Phase 1.

---

### Task 5: SFT Trajectory Generation

**Goal:** 20–50K teacher trajectories against τ³-bench gym, telecom-heavy.

5.1. Choose the teacher per § 3 decisions. Configure `scripts/run_sft_gen.sh` with:
- `TEACHER_MODEL` (e.g. `claude-opus-4-5`, `gpt-5`, `deepseek-r1`)
- `TARGET_TRAJECTORIES` (default 30000)
- Per-domain split: 60% telecom / 25% retail / 15% airline
- User simulator: `gpt-4.1` (leaderboard standard, do not change)
- Persona mix: bias 60% Hard, 30% Medium, 10% Easy (Phase 2 will train *against* Hard, but we want SFT data covering the distribution)
- Concurrency: tune to teacher rate limits. Anthropic Tier 4 supports ~30 concurrent Opus calls; OpenAI Tier 5 supports ~50 concurrent GPT-5.
- Output: one JSONL per task batch under `$WORKSPACE/sft_data/raw/` with full message history + tool-call traces + final `info` dict.

5.2. Run:
```bash
bash scripts/run_sft_gen.sh
```
Expected wall-clock: 3–7 days at 30K trajectories with 30 concurrency, depending on teacher latency. Monitor and resume — the script must be idempotent on partial output.

5.3. **Mid-run gate (after first 1K trajectories):** sample 20 by hand, eyeball quality. If success rate < 0.70 or trajectories show repeated tool-arg errors, stop and either swap teacher or fix prompt.

**Acceptance:** ≥ 20K raw trajectories under `sft_data/raw/`; per-domain counts written to `sft_data/raw_summary.json`.

---

### Task 6: Dual Verification

**Goal:** Filter raw trajectories to keep only those passing **both** (a) the τ³-bench DB-state + comm-info checks AND (b) a second judge model rating reasoning as coherent. MUA-RL guidance applies: keep natural-language guidance in the kept trajectories — we strip dialogue-content-requirements only at Phase 3 reward time, not now.

6.1. Implement `scripts/dual_verify.py`:
- Input: `$WORKSPACE/sft_data/raw/*.jsonl`
- Check (a): re-run the trajectory's final state through tau2-bench's evaluator. Trajectory passes (a) iff reward = 1 AND comm-info checks all pass. Most teacher trajectories will already have this in `info`; just gate on it.
- Check (b): pass the trajectory's reasoning + final answer to a judge model. Default judge: `gpt-4.1` (cheap, fast, well-calibrated for τ²-style judgment). Prompt template lives at `scripts/judge_prompt.txt`. Judge returns a 1–5 score; trajectory passes (b) iff score ≥ 4.
- Output: `$WORKSPACE/sft_data/verified/*.jsonl` (passing both checks); `$WORKSPACE/sft_data/verification_summary.json` with per-domain pass rates and reasons-for-rejection histogram.

6.2. Run:
```bash
bash scripts/run_dual_verify.sh
```

6.3. **Yield gate:** if verified subset is < 15K trajectories (dataset.jsonl below threshold), either:
- Generate more raw trajectories (return to Task 5),
- Loosen the judge threshold to ≥ 3 (document the tradeoff),
- Or accept and proceed with the smaller dataset (document expected SFT-strength impact).

**Acceptance:** verified JSONL files exist; per-domain pass rate ≥ 50%; total verified ≥ 15K.

6.4. Materialize the SFT-ready dataset:
```bash
python scripts/build_sft_dataset.py \
  --in $WORKSPACE/sft_data/verified \
  --out $WORKSPACE/sft_data/dataset.jsonl \
  --chat-template $WORKSPACE/models/$CHOSEN/tokenizer_config.json \
  --dedupe-by-task
```
Acceptance: single dataset.jsonl exists, chat-template formatted, deduped.

---

### Task 7: SFT Training Run

**Goal:** SFT the chosen base on the verified dataset. Produces `checkpoints/sft_v1/`.

7.1. Decide LoRA vs full-parameter:
- **Default LoRA** (rank 64, alpha 128) for Qwen3-32B and GLM-4.5-Air. Fits on 1× H200. Fast, cheap, easy to merge or stack adapters in Phase 3.
- **Full-parameter** for Qwen3-8B if chosen. Better priors propagation into RL; H200 single-card with optimizer state offload + bf16 should fit.
- **Do not** attempt full-parameter on a 32B base on 1× H200 — request multi-GPU compute first.

7.2. Hyperparameters (defaults; adjust per LLaMA-Factory config or TRL `SFTConfig`):
- Epochs: 2 (LoRA) or 1 (full-parameter)
- LR: 1e-4 (LoRA) or 5e-6 (full-parameter)
- Batch: effective 32 (per-device 4 × grad-accum 8 on 1 H200)
- Sequence length: 16384 (covers ~99% of telecom multi-turn trajectories)
- LR schedule: cosine with 3% warmup
- Optimizer: AdamW (8-bit if memory tight)
- Mask loss on `tool` role messages (matches MUA-RL guidance, even though we keep them in the prompt context). Verify framework supports per-role masking.

7.3. Run:
```bash
bash scripts/run_sft_train.sh
```
Expected wall-clock: 12–48 hours depending on dataset size, base size, LoRA vs full.

**Acceptance:** Checkpoint exists; final eval loss decreased monotonically over training; W&B run logged.

---

### Task 8: Eval SFT'd Checkpoint

**Goal:** Run the Phase 0 eval pipeline on the SFT'd checkpoint. Compare against (a) the chosen base's pre-SFT numbers from Task 2 and (b) Phase 0's Qwen3.6-35B-A3B baseline.

8.1. Serve the SFT'd checkpoint via vLLM. For LoRA:
```bash
vllm serve $WORKSPACE/models/$CHOSEN \
  --enable-lora \
  --lora-modules sft_v1=$WORKSPACE/checkpoints/sft_v1 \
  --served-model-name $CHOSEN-sft-v1 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --trust-remote-code
```
For full-parameter: `vllm serve $WORKSPACE/checkpoints/sft_v1 ...` directly.

8.2. Run Phase 0 harness (`tau2 run`) with `--num-trials 4` for pass^4 on all three domains. Output to `$WORKSPACE/sft_eval/`.

8.3. Run `tau2 review` (LLM-judge error attribution) on the SFT eval trajectories. Compare fault-type distribution to Phase 0's distribution — we want to see *guideline violation* and *missed required action* drop, since those were Phase 0's dominant failure modes.

**Acceptance:** Three pass@1 / pass^4 numbers exist; comparison table written to `phase1_results.md`.

---

### Task 9: Compile Phase 1 Results Document

**Goal:** Single markdown report. Mirror Phase 0's `phase0_results.md` structure.

9.1. Sections:
- **Summary** (2–3 sentences)
- **Base model selection** (bake-off table, decision rationale)
- **SFT data curation** (teacher used, raw count, verification yield, per-domain breakdown, dataset stats)
- **SFT training** (LoRA vs full, hyperparameters, training curves screenshot)
- **Eval results** (pass@1 / pass^4 table; pre-SFT vs post-SFT vs Phase 0)
- **Failure mode analysis** (`tau2 review` output diff vs Phase 0)
- **Implications for Phase 2 (RL)** (which fault types remain; which signals from gym `info` will drive reward shaping)
- **Reproducibility** (one-liners; pinned commits; dataset HF link)

9.2. Push the verified dataset to HuggingFace Datasets under `debdootmiitd/tau3-bench-sft-cold-start-v1` (mirror Phase 0's pattern). Trajectories are the moat — back them up off-machine.

**Acceptance:** `phase1_results.md` exists; dataset uploaded; numbers in the report match files in `sft_eval/`.

---

## 6. Failure Modes & Mitigations

| Symptom | Likely cause | Fix |
|---|---|---|
| Bake-off candidate has pass@1 < 0.10 on telecom | Wrong tool-call parser for that model | Check model card; GLM and Qwen use different parser names |
| Teacher rate limits cripple Task 5 throughput | Single-key concurrency cap | Spread across orgs/keys, or switch teacher mid-run (document) |
| Verified yield < 50% | Teacher trajectories don't satisfy comm-info checks | Inspect rejection histogram; if mostly comm-info, the teacher's prompt is missing required dialogue cues — revise system prompt and regen |
| SFT loss plateaus immediately | LR too low, or chat template mismatch | Verify dataset rendered template matches model's expected template character-for-character; sanity-check by training on 100 examples and confirming overfitting |
| SFT'd checkpoint *worse* than base on retail | SFT overfit to telecom; catastrophic forgetting | Reduce telecom share to 50%; add 5% airline + retail; or apply sample-weighting |
| LoRA adapter eval much worse than base eval | vLLM not loading adapter | Check `--enable-lora`; confirm adapter path; verify with a no-tool-call prompt that adapter is actually being applied |
| pass^4 unchanged but pass@1 up significantly | SFT improved capability but not consistency — expected pre-RL | This is fine for Phase 1; consistency is Phase 4's job |
| `info` dict from gym lacks per-step signals | Gym wrapper exposes only final state | See Task 4 failure mode — extend wrapper or fall back to log-derived signals |

---

## 7. Checkpoints Requiring Human (Debdoot) Approval

Pause and request confirmation before proceeding past these points:

- **After § 3 decisions confirmed:** before any model downloads (cost gate on disk + teacher API budget).
- **After Task 2 bake-off:** confirm base-model choice. Once chosen, all downstream compute targets it.
- **After Task 5 mid-run gate (1K trajectories):** show 5 sampled teacher trajectories; confirm quality before letting the remaining 19K–49K rollouts run.
- **After Task 6:** before launching SFT, confirm dataset stats (size, per-domain split, dedupe rate) look sensible.
- **After Task 8:** before publishing dataset to HF and closing Phase 1, confirm the eval numbers and the failure-mode-shift narrative.

---

## 8. Estimated Resource Budget

| Resource | Estimate |
|---|---|
| GPU hours (H200) | 200–400 hours wall clock (bake-off 30h, SFT data gen idle for serving 0h, SFT training 12–48h, eval 60h × 1.5 configs) |
| OpenAI API spend | $400–$800 (user simulator across rollouts + judge in dual verification + final eval) |
| Anthropic API spend (if Opus teacher) | $2K–$6K |
| OpenAI GPT-5 spend (if GPT-5 teacher) | $1.5K–$5K |
| DeepSeek-R1 spend (if R1 teacher) | $300–$1K |
| Disk | ~500 GB |
| Engineering time | 3–5 weeks including debugging |

If actuals exceed 1.5× estimate on any line, stop and report rather than burning budget.

---

## 9. Out of Scope for Phase 1

The following are explicitly **not** part of Phase 1:

- Any RL training (GRPO, MT-GRPO, DAPO) — that is Phase 3.
- User simulator robustness work (persona injection, sim-model rotation) — that is Phase 2.
- Reward design experiments (group-consistency, calibration term) — that is Phase 4.
- Leaderboard submission — that is Phase 5.
- Comvera-specific spillover evaluation — separate workstream.
- Test-time scaling (Best-of-N, SABER scaffold) — Phase 1 evaluates the SFT'd checkpoint at temp 0.6, single-pass per trial, no scaffolding.

If during execution it becomes clear one of these is required to complete Phase 1, stop and escalate.

---

## 10. Handoff to Phase 2

At Phase 1 exit, the following must be ready for Phase 2 (user simulator + RL prep):

- SFT'd checkpoint accessible (LoRA adapter or full weights) and one-command serveable via vLLM.
- `phase1_results.md` numbers frozen — used as the bar for "RL gain" in Phase 3.
- Verified SFT dataset published to HuggingFace; URL in README.
- Documented `info` dict structure from τ³-bench gym (Phase 3 reward shaping uses these signals).
- `CHOSEN_BASE` file checked in; downstream phase repos read this to stay consistent.
- Open issues / known limitations from Task 8 failure-mode analysis listed at end of `phase1_results.md` — these become Phase 2's targeting list.

---

*End of Phase 1 spec. Update this document with any deviations encountered during execution; the live document is the source of truth for the eventual write-up.*
