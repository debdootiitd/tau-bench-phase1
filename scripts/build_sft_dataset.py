"""build_sft_dataset.py — Task 6.4: materialize verified trajectories into SFT-ready JSONL.

Input:  one or more JSONL files of verified trajectories (output of dual_verify.py)
Output: single chat-template-formatted JSONL where each line is one training example
        in the format expected by LLaMA-Factory / TRL SFTTrainer.

Steps:
  1. Load all verified trajectories from --in dir.
  2. Dedupe by task_id (--dedupe-by-task) — keep the highest-scoring trajectory per task.
     Without dedupe, telecom's small task pool would be over-represented and SFT would
     overfit to memorizing tasks rather than learning policy.
  3. Render each trajectory through the chosen base model's chat template.
     Read template from --chat-template (the model's tokenizer_config.json).
  4. Mark `tool` role messages as loss-masked (per MUA-RL guidance and Task 7.2).
  5. Emit one JSONL line per trajectory.

CLI:
    python build_sft_dataset.py \
        --in            $WORKSPACE/sft_data/verified \
        --out           $WORKSPACE/sft_data/dataset.jsonl \
        --chat-template $WORKSPACE/models/$CHOSEN/tokenizer_config.json \
        --dedupe-by-task

NOT YET IMPLEMENTED — contract only.
"""

import sys

if __name__ == "__main__":
    print("build_sft_dataset.py: NOT YET IMPLEMENTED — see SPEC.md Task 6.4.", file=sys.stderr)
    sys.exit(1)
