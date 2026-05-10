# Phase 1 — Progress journal

Chronological record of decisions, deviations from `SPEC.md`, and incidents during execution. Append-only; do not rewrite history. Mirror Phase 0's `progress.md` style.

---

## 2026-05-10 — Repo scaffolded

- Repo created at `/home/jovyan/tau-bench-phase1` from `tau-bench-phase0` template.
- `SPEC.md` operationalizes the Phase 1+ training roadmap (base model + SFT cold start). Phases 2–4 are sketched in SPEC § 2 for context but will get their own repos.
- Open § 3 decisions awaiting Debdoot:
  - Base model (default plan: bake off Qwen3-32B vs GLM-4.5-Air vs Qwen3-8B; fallback default Qwen3-32B).
  - Teacher (default Claude Opus; alternatives GPT-5, DeepSeek-R1).
  - Trajectory volume (default 30K, telecom-heavy 60/25/15).
- No code or model downloads yet. Scripts in `scripts/` are stubs with documented contracts — see each file header.

---

## 2026-05-10 — § 3 decisions confirmed

All three open SPEC § 3 decisions locked in:

- **Base model:** run the Task 2 bake-off across Qwen3-32B, GLM-4.5-Air, Qwen3-8B. Fallback if interrupted: Qwen3-8B (cheapest Phase 3 RL rollouts; MUA-RL precedent). Switch to Qwen3-32B only if Qwen3-8B trails airline pass@1 by >5 pp.
- **Teacher:** Claude Opus. Budget $2–6K for 30K trajectories. GPT-5 contingency for rate limits, DeepSeek-R1 contingency for budget overruns.
- **Trajectory volume:** 30K target / 20K floor / 50K cap (only if mid-run yield <50%). Domain split 60% telecom / 25% retail / 15% airline.

SPEC § 3 updated to reflect confirmation. Ready for Task 1 (env setup) once the compute box is provisioned.

---

## 2026-05-10 — Task 1 (env setup) complete

Executed Task 1 on the H200 box (143 GB VRAM, CUDA 12.4, driver 550.127.08, 340 GB free disk).

**Deviations from SPEC drafted earlier:**
- `$WORKSPACE` switched to repo root (`/home/jovyan/tau-bench-phase1`) per user direction — was `/workspace/phase1`. SPEC § 3 + .gitignore updated. tau2-bench stays as a sibling clone at `/home/jovyan/tau2-bench`, NOT nested under `$WORKSPACE`.
- Python 3.11 wasn't in apt; used `uv venv --python 3.11 venv` which auto-fetched cpython-3.11.14.
- vllm 0.20.2 (newer than SPEC's `>=0.19` floor) installed cleanly with all base deps.
- LLaMA-Factory 0.9.4 installed into the same venv as vLLM. **Side effect:** transformers downgraded from 5.8.0 → 4.57.1. Both vllm and llamafactory import OK at this version, so single-venv setup retained. If vLLM serve later breaks on a transformers-5 API, will split to `venv-vllm` + `venv-sft`.
- Task 1.4 smoke test exposed that the gym API is **not** `tau2.gym.make_env(domain)` (what the SPEC originally guessed). Real API is `gymnasium.make(TAU_BENCH_ENV_ID, domain=..., task_id=...)` after calling `register_gym_agent()`. Env IDs: `tau-bench-v0` (agent side), `tau-bench-user-v0` (user side). `step()` returns the gymnasium 5-tuple `(obs, reward, terminated, truncated, info)`; `info` includes `tools`, `policy`, `simulation_run`. SPEC Task 1.4, Task 4, and `run_sft_gen.sh` outline updated.
- Available gym domains: mock, airline, retail, telecom, banking_knowledge — plus telecom variants (`telecom_full`, `telecom_small`, `telecom-workflow`) which we may want to use for SFT data diversity.

**State after Task 1:**
- `venv/` — Python 3.11.14, vllm 0.20.2, transformers 4.57.1, trl 0.24.0, llamafactory 0.9.4, peft 0.17.1, torch (CUDA 12.x, sees the H200).
- `/home/jovyan/tau2-bench` — uv-managed, gym extra installed, tau2 v1.0.0.
- `huggingface-cli` and `uv 0.9.7` already on box.
- Disk flag: 340 GB free. Three candidate models (~150 GB) + rollouts + checkpoints will be tight. Will delete bake-off losers post-Task 2 and stay on LoRA SFT.

Next: Task 2 (base-model bake-off), gated on Debdoot's go-ahead and `.env` populated with API keys.

<!-- Append future entries below. Format: ## YYYY-MM-DD — short title -->
