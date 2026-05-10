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

<!-- Append future entries below. Format: ## YYYY-MM-DD — short title -->
