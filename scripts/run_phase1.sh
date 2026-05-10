#!/usr/bin/env bash
# run_phase1.sh — Phase 1 end-to-end orchestrator.
#
# Idempotent: each task checks for its expected output and skips if present.
# Pauses at human-approval checkpoints (see SPEC.md § 7).
#
# Inputs (env, sourced from $WORKSPACE/.env):
#   OPENAI_API_KEY     — user simulator (gpt-4.1) + judge (gpt-4.1) + optional GPT-5 teacher
#   HF_TOKEN           — base model + comparison weights
#   ANTHROPIC_API_KEY  — if Claude Opus is teacher (set after § 3 decision)
#   DEEPSEEK_API_KEY   — if DeepSeek-R1 is teacher
#   WANDB_API_KEY      — recommended for SFT runs
#   WORKSPACE          — defaults to /workspace/phase1
#
# Outputs (under $WORKSPACE):
#   base_eval/<model>/                        — Task 2 bake-off pass@1 numbers
#   CHOSEN_BASE                               — Task 2 decision
#   sft_data/raw/*.jsonl                      — Task 5 teacher rollouts
#   sft_data/verified/*.jsonl                 — Task 6 dual-verified subset
#   sft_data/dataset.jsonl                    — Task 6 SFT-ready dataset
#   checkpoints/sft_v1/                       — Task 7 SFT'd checkpoint
#   sft_eval/{airline,retail,telecom}/        — Task 8 final eval
#   phase1_results.md                         — Task 9 report
#
# Usage:
#   bash scripts/run_phase1.sh             # run from current task
#   bash scripts/run_phase1.sh --from N    # force restart from Task N

set -euo pipefail

echo "run_phase1.sh: NOT YET IMPLEMENTED"
echo "See SPEC.md for the task-by-task plan. The orchestrator is the last thing to wire up,"
echo "after each per-task script (run_base_eval.sh, run_sft_gen.sh, etc.) is working standalone."
exit 1
