#!/usr/bin/env bash
# run_sft_train.sh — Task 7: SFT the chosen base on the verified dataset.
#
# Default: LoRA (rank 64, alpha 128) for Qwen3-32B and GLM-4.5-Air.
#          Full-parameter for Qwen3-8B.
# Override via env: SFT_MODE=lora|full
#
# Config (env, defaults shown):
#   SFT_MODE=lora                    # lora | full (auto-picked from CHOSEN_BASE if unset)
#   SFT_FRAMEWORK=llamafactory       # llamafactory | trl | axolotl
#   EPOCHS=2                         # 1 if SFT_MODE=full
#   LR=1e-4                          # 5e-6 if SFT_MODE=full
#   PER_DEVICE_BATCH=4
#   GRAD_ACCUM=8                     # effective batch 32 on 1 GPU
#   SEQ_LEN=16384
#   LORA_R=64
#   LORA_ALPHA=128
#
# Reads:
#   $WORKSPACE/CHOSEN_BASE                    — base model short name
#   $WORKSPACE/models/<CHOSEN>/               — base weights
#   $WORKSPACE/sft_data/dataset.jsonl         — training-ready dataset
#
# Writes:
#   $WORKSPACE/checkpoints/sft_v1/            — adapter (lora) or full weights
#   $WORKSPACE/checkpoints/sft_v1/training_log.json
#   W&B run (if WANDB_API_KEY set)
#
# Implementation outline (see SPEC.md Task 7):
#   1. Generate framework-specific config YAML (LLaMA-Factory's format by default).
#   2. Set role mask: loss=False on `tool` role messages.
#   3. Launch training. Single H200 should fit:
#        - LoRA on 32B: ~12-24h for 30K trajectories × 2 epochs
#        - Full-param on 8B: ~24-48h
#   4. On finish: copy final checkpoint to checkpoints/sft_v1/, upload to W&B.

set -euo pipefail

echo "run_sft_train.sh: NOT YET IMPLEMENTED"
echo "Blocked on Task 6 (need dataset.jsonl) and Task 2 (need CHOSEN_BASE)."
echo "See SPEC.md Task 7 for the full pipeline."
exit 1
