#!/usr/bin/env bash
# run_base_eval.sh — Task 2 wrapper: evaluate one candidate base model on Phase 0 harness.
#
# Reuses the Phase 0 tau2-bench install. Runs pass@1 (single trial) on a 25-task
# subset of airline + telecom — enough signal to compare candidates without
# burning the full bake-off budget.
#
# Args:
#   $1  — model short name (one of: qwen3-32b, glm-4.5-air, qwen3-8b)
#
# Reads:
#   $WORKSPACE/models/<model>/                — pre-downloaded weights
#
# Writes:
#   $WORKSPACE/base_eval/<model>/             — trajectories + passk.json
#   $WORKSPACE/base_eval/<model>/serving.cmd  — exact vLLM serve command used (reproducibility)
#
# Implementation outline (see SPEC.md Task 2):
#   1. Launch vLLM in a detached tmux session targeting models/<model>.
#      Reasoning + tool-call parser are model-specific — Qwen uses qwen3 / qwen3_xml,
#      GLM uses its own (check model card at execution time).
#   2. Wait for /v1/models to respond.
#   3. Run `tau2 run --domain airline --num-tasks 25 --num-trials 1 \
#                    --agent-llm <model> --agent-llm-base-url http://localhost:8000/v1 \
#                    --user-llm gpt-4.1 --output-dir $WORKSPACE/base_eval/<model>/airline`
#   4. Same for telecom.
#   5. Compute pass@1 via `tau2 view`.
#   6. Tear down vLLM.

set -euo pipefail

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 <model-short-name>" >&2
  echo "  one of: qwen3-32b, glm-4.5-air, qwen3-8b" >&2
  exit 2
fi

echo "run_base_eval.sh: NOT YET IMPLEMENTED for model=$MODEL"
echo "See SPEC.md Task 2 for the full pipeline."
exit 1
