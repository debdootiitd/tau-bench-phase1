# tau-bench-phase1

Phase 1 of the τ³-bench leaderboard run: **base-model selection + SFT cold start**.
Picks up from [tau-bench-phase0](https://github.com/debdootiitd/tau-bench-phase0)
(Qwen3.6-35B-A3B baselines on [τ³-bench](https://github.com/sierra-research/tau2-bench)).

Produces an SFT'd checkpoint that Phase 2 (RL) will start training from.

## What's in the box

| Path | Purpose |
|---|---|
| `SPEC.md` | The Phase 1 plan — base-model bake-off, teacher rollouts, dual verification, SFT, eval |
| `phase1_results.md` | Final report — bake-off table, data-curation funnel, SFT eval (post-execution) |
| `progress.md` | Chronological journal of decisions, deviations, and incidents |
| `scripts/run_phase1.sh` | One-command end-to-end orchestrator (idempotent, auto-resumes) |
| `scripts/run_base_eval.sh` | Per-candidate base-model bake-off wrapper (Task 2) |
| `scripts/run_sft_gen.sh` | Teacher rollout against τ³-bench gym interface (Task 5) |
| `scripts/run_dual_verify.sh` | Filter rollouts via DB-state checks + judge-model coherence (Task 6) |
| `scripts/dual_verify.py` | Verification implementation |
| `scripts/build_sft_dataset.py` | Materialize verified rollouts into chat-template JSONL |
| `scripts/run_sft_train.sh` | SFT (LoRA default; full-parameter for 8B base) (Task 7) |
| `scripts/test_serving.py` | vLLM tool-calling + thinking-mode smoke test |
| `.env.example` | Template for `OPENAI_API_KEY`, `HF_TOKEN`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `WANDB_API_KEY` |

## Trajectory data + SFT checkpoint

Will be hosted on Hugging Face (links populated post-execution):
- Verified SFT dataset: `debdootmiitd/tau3-bench-sft-cold-start-v1`
- SFT'd checkpoint: TBD (depends on Task 2 base-model decision)

## Hardware

- **Default:** 1× NVIDIA H200 NVL (143 GB) — sufficient for serving any candidate base, LoRA SFT, and eval.
- **Required for full-parameter SFT on 32B:** 4× H100 with FSDP. Flag for budget approval before electing — see SPEC § 3.
- ~500 GB disk for model weights × 3 candidates + raw rollouts + verified subset + checkpoints.
- API spend depends on teacher choice — see SPEC § 8 budget table. Worst case (Opus, 50K trajectories): ~$6K.

## Reproducing

1. Provision an H200 (or 2× H100) Vast.ai box. 500 GB disk, 128 GB RAM minimum.
2. Clone this repo. Clone `sierra-research/tau2-bench` next to it (Phase 0 instance is fine to reuse).
3. Install vLLM into a venv. `uv sync --extra gym` inside `tau2-bench` (the `gym` extra is the new requirement vs Phase 0).
4. Set up `.env` with all four API keys.
5. **Confirm § 3 decisions** in `SPEC.md` (base model, teacher, trajectory volume).
6. `bash scripts/run_phase1.sh` — orchestrator runs Tasks 2 → 8 with explicit human checkpoints between. Auto-resumes on partial output.
7. `tau2 submit prepare` (only after Phase 4) for leaderboard submission.

## Relationship to Phase 0

Phase 0 measured a strong off-the-shelf model (Qwen3.6-35B-A3B). Phase 1 *trains* a new model. Two upstream dependencies:
- Phase 0's eval harness (`tau2 run` with the same user-sim and parser config) is reused verbatim for the Task 2 bake-off and Task 8 final eval.
- Phase 0's documented deviations (e.g. `qwen3_xml` parser vs spec's `qwen3_coder`, pass^k formula) carry over.

Phases 2–4 (user-sim, RL algorithm, reward design) are sketched in SPEC § 2 and will get their own repos when each phase begins.
