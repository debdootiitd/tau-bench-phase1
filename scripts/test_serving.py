"""test_serving.py — vLLM tool-calling + thinking-mode smoke test.

Generalizes Phase 0's test_toolcall.py to run against any candidate base model.
Used in Task 2 (after each bake-off serve) and Task 3 (after production serve).

Usage:
    python scripts/test_serving.py \
        --base-url http://localhost:8000/v1 \
        --model    qwen3-32b

Exits 0 on success. Exits non-zero with diagnostic on:
  - server unreachable
  - tool call returned as text instead of structured tool_calls
  - thinking mode toggle ineffective
"""

import argparse
import sys

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: `openai` package not installed. `pip install openai`.", file=sys.stderr)
    sys.exit(2)


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True, help="vLLM endpoint, e.g. http://localhost:8000/v1")
    p.add_argument("--model", required=True, help="--served-model-name as registered with vLLM")
    p.add_argument("--skip-thinking-toggle", action="store_true",
                   help="Skip the thinking-mode toggle test (use for non-Qwen models)")
    args = p.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    # Test 1: structured tool call
    print(f"[1/2] Tool-call test against {args.model} ...")
    resp = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "What's the weather in Tokyo? Use the tool."}],
        tools=[WEATHER_TOOL],
    )
    msg = resp.choices[0].message
    if not getattr(msg, "tool_calls", None):
        print("FAIL: model returned text instead of a structured tool_call.", file=sys.stderr)
        print(f"  content: {msg.content!r}", file=sys.stderr)
        print("  → check vLLM was launched with --enable-auto-tool-choice and a matching"
              " --tool-call-parser for this model family.", file=sys.stderr)
        return 1
    tc = msg.tool_calls[0]
    print(f"  OK — tool_call: {tc.function.name}({tc.function.arguments})")

    # Test 2: thinking-mode toggle (Qwen3-family only)
    if args.skip_thinking_toggle:
        print("[2/2] Skipping thinking-mode toggle (--skip-thinking-toggle).")
        return 0

    print(f"[2/2] Thinking-mode toggle test ...")
    no_think = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "What is 2+2?"}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    think = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "What is 2+2?"}],
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    no_think_text = no_think.choices[0].message.content or ""
    think_msg = think.choices[0].message
    has_reasoning = bool(getattr(think_msg, "reasoning_content", None))

    print(f"  no-think output ({len(no_think_text)} chars): {no_think_text[:80]!r}")
    print(f"  thinking returned reasoning_content: {has_reasoning}")

    if not has_reasoning:
        print("WARN: thinking-mode response had no reasoning_content field.", file=sys.stderr)
        print("  → check vLLM was launched with --reasoning-parser qwen3 (or model-specific equiv).",
              file=sys.stderr)
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
