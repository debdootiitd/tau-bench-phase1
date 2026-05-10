#!/usr/bin/env bash
# run_sft_gen.sh — Task 5: generate SFT trajectories from a strong teacher
#                  against the τ³-bench gym interface.
#
# Idempotent: skips task batches whose JSONL already exists in sft_data/raw/.
# Resumable: kill -INT to checkpoint mid-batch.
#
# Config (env, defaults shown):
#   TEACHER_MODEL=claude-opus-4-5     # or gpt-5, deepseek-r1
#   TARGET_TRAJECTORIES=30000         # SPEC § 3 default
#   DOMAIN_SPLIT="telecom:0.6,retail:0.25,airline:0.15"
#   PERSONA_MIX="hard:0.6,medium:0.3,easy:0.1"
#   USER_SIMULATOR=gpt-4.1            # leaderboard standard, do not change
#   CONCURRENCY=30                    # tune to teacher rate limits
#
# Reads:
#   $WORKSPACE/tau2-bench/                    — gym extra installed
#
# Writes:
#   $WORKSPACE/sft_data/raw/<domain>/<batch_id>.jsonl
#   $WORKSPACE/sft_data/raw_summary.json      — per-domain counts, runtime, token spend
#
# Implementation outline (see SPEC.md Task 5):
#   1. Resolve teacher SDK + auth from env.
#   2. Compute per-domain quotas from DOMAIN_SPLIT × TARGET_TRAJECTORIES.
#   3. For each domain: spawn CONCURRENCY workers, each calls
#        import gymnasium as gym
#        from tau2.gym import register_gym_agent, TAU_BENCH_ENV_ID
#        register_gym_agent()  # once per process
#        env = gym.make(TAU_BENCH_ENV_ID, domain=<domain>, task_id=<task>,
#                       user_llm="gpt-4.1", user_llm_args={"temperature": 0.7})
#      and runs teacher policy → step → log. step() returns the gymnasium
#      5-tuple (obs, reward, terminated, truncated, info). Persona is sampled
#      per-rollout per PERSONA_MIX (encoded into task_id, e.g.
#      "[mobile_data_issue]...[PERSONA:Hard]" — see tau2-bench/src/tau2/gym/README.md).
#   4. Each worker writes one JSONL line per completed trajectory (full message
#      history, tool calls, final info dict).
#   5. After first 1K trajectories: pause and request human review (SPEC § 7
#      mid-run gate).
#   6. On completion: write raw_summary.json with per-domain counts + costs.

set -euo pipefail

echo "run_sft_gen.sh: NOT YET IMPLEMENTED"
echo "Blocked on Task 4 (gym smoke test) — confirm tau2.gym.make_env() API surface first."
echo "See SPEC.md Task 5 for the full pipeline."
exit 1
