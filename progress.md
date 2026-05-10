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

<!-- Append future entries below. Format: ## YYYY-MM-DD — short title -->
