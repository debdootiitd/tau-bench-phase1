"""dual_verify.py — Task 6 implementation: filter raw teacher trajectories.

Two checks, both required:
  (a) τ³-bench DB-state + comm-info evaluator: re-run final state through
      tau2-bench's evaluator. Pass iff reward == 1 AND all comm-info checks pass.
      For most teacher trajectories this is already in the trajectory's `info`
      dict — we just gate on it. Re-run only when `info` is missing.
  (b) Judge model coherence: send (system policy, user goal, agent reasoning,
      final answer) to a judge LLM. Default judge: gpt-4.1. Returns int 1-5.
      Pass iff score >= JUDGE_THRESHOLD (default 4).

CLI:
    python dual_verify.py \
        --raw-dir   $WORKSPACE/sft_data/raw \
        --out-dir   $WORKSPACE/sft_data/verified \
        --summary   $WORKSPACE/sft_data/verification_summary.json \
        --judge-model gpt-4.1 \
        --judge-threshold 4 \
        --concurrency 16

Output JSONL preserves the original trajectory schema 1:1, plus:
    "verification": {
        "check_a_pass": bool,
        "check_b_score": int,
        "judge_rationale": str
    }

Summary JSON shape:
    {
      "per_domain": {
         "telecom": {"raw": int, "passed_a": int, "passed_b": int, "passed_both": int}
      },
      "rejection_reasons": {
         "comm_info_failed": int,
         "reward_zero": int,
         "judge_score_low": int,
         ...
      }
    }

NOT YET IMPLEMENTED — this file is a contract.

Implementation notes for whoever wires this up:
  - Use tau2-bench's evaluator directly (it's importable). Don't reimplement.
  - Judge prompt template lives at scripts/judge_prompt.txt (also a stub).
  - Concurrency on judge calls: respect OpenAI rate limits; gpt-4.1 Tier 5 is
    ~50 RPM at full concurrency. Backoff on 429.
  - Cost estimate: ~$0.02 per judge call × 30K trajectories = ~$600. Budget.
"""

import sys

if __name__ == "__main__":
    print("dual_verify.py: NOT YET IMPLEMENTED — see SPEC.md Task 6.", file=sys.stderr)
    sys.exit(1)
