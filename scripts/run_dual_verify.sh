#!/usr/bin/env bash
# run_dual_verify.sh — Task 6: filter raw teacher trajectories with two checks.
#
# Check (a): τ³-bench DB-state + comm-info evaluator (re-run trajectory,
#            require reward=1 AND all comm-info checks pass).
# Check (b): judge model rates reasoning coherence on 1-5 scale; require ≥ 4.
#
# Per MUA-RL: keep natural-language guidance in the kept trajectories. We only
# strip dialogue-content-requirements at Phase 3 reward time, NOT here.
#
# Reads:
#   $WORKSPACE/sft_data/raw/**/*.jsonl
#
# Writes:
#   $WORKSPACE/sft_data/verified/**/*.jsonl
#   $WORKSPACE/sft_data/verification_summary.json   — per-domain pass rates,
#                                                     reasons-for-rejection histogram
#
# Implementation lives in scripts/dual_verify.py — this wrapper just sets env
# and invokes it.
#
# Yield gate: if verified count < 15K, see SPEC § 6.3 (regenerate, loosen, or
# accept and document).

set -euo pipefail

echo "run_dual_verify.sh: NOT YET IMPLEMENTED"
echo "See scripts/dual_verify.py for the implementation contract."
echo "See SPEC.md Task 6 for the pipeline."
exit 1
