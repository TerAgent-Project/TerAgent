#!/usr/bin/env python3
"""TerAgent Quickstart — 5-minute getting started example

A standalone script that demonstrates the core teragent APIs.
Uses MockAdapter so it runs without any API keys.

Run directly:
    python examples/quickstart.py

What you'll learn:
    1. Creating a provider (Compiler + Adapter composition)
    2. Sending a TAP request (execute_tap)
    3. Simple chat (bypasses TAP compilation)
    4. Streaming responses (stream_tap)
    5. Cost tracking (get_cost_summary)
    6. Using different compilers for the same request
"""

from __future__ import annotations

import asyncio
import sys

# ── Ensure the project root is on sys.path when running standalone ──
# When invoked as `python examples/quickstart.py` from the project root,
# the teragent package may not be on the import path.  This snippet adds
# the parent directory so `import teragent` works without installation.
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import teragent
from teragent import TAPRequest, create_provider


# ======================================================================
# 1. Create a provider
# ======================================================================

def step1_create_provider() -> teragent.ModelProvider:
    """Create a ModelProvider with MockAdapter (no API keys needed)."""
    print("\n" + "=" * 60)
    print("Step 1: Create a Provider")
    print("=" * 60)

    # create_provider() combines a Compiler (prompt optimization strategy)
    # with an Adapter (HTTP protocol) into a ModelProvider.
    #
    # With adapter="mock", no network I/O occurs — perfect for testing
    # and learning the API without real API keys.
    provider = create_provider(
        compiler="default",     # Prompt compilation strategy
        adapter="mock",         # Mock adapter (no API calls)
        model="mock-model",     # Model identifier
    )

    print(f"  Provider: {provider}")
    print(f"  Capabilities: {provider.capabilities}")
    return provider


# ======================================================================
# 2. Execute a TAP request
# ======================================================================

async def step2_execute_tap(provider: teragent.ModelProvider) -> None:
    """Execute a TAP request — the core teragent API."""
    print("\n" + "=" * 60)
    print("Step 2: Execute a TAP Request")
    print("=" * 60)

    # TAPRequest is the unified input format (IR) that captures *what* you
    # want, not *how* to ask the model.  The Compiler decides the best
    # prompt format for each model.
    request = TAPRequest(
        meta={"task_id": "demo-1", "intent": "code_generation"},
        instruction="Write a Python function that checks if a string is a palindrome",
        constraints=["Python 3.10+", "Include type hints"],
        output_format_hint="<file path='palindrome.py'>complete code</file>",
    )

    print(f"  Request intent: {request.meta['intent']}")
    print(f"  Instruction: {request.instruction}")
    print(f"  Constraints: {request.constraints}")

    # execute_tap() compiles the request into model-specific prompts,
    # sends them via the adapter, and returns a TAPResponse.
    response = await provider.execute_tap(request)

    print(f"\n  Response ({len(response.raw_text or '')} chars):")
    # Show first few lines of the response
    for line in (response.raw_text or "").split("\n")[:8]:
        print(f"    {line}")
    print(f"  Tokens: prompt={response.prompt_tokens}, completion={response.completion_tokens}")


# ======================================================================
# 3. Simple chat
# ======================================================================

async def step3_chat(provider: teragent.ModelProvider) -> None:
    """Simple chat — bypasses TAP compilation, sends raw messages."""
    print("\n" + "=" * 60)
    print("Step 3: Simple Chat")
    print("=" * 60)

    # chat() is a convenience method that sends raw messages without
    # TAP compilation.  Useful for quick conversational exchanges.
    result = await provider.chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
    )

    print(f"  Response: {result['content'][:200]}")
    print(f"  Finish reason: {result.get('finish_reason', 'N/A')}")


# ======================================================================
# 4. Streaming
# ======================================================================

async def step4_streaming(provider: teragent.ModelProvider) -> None:
    """Stream a TAP request — receive chunks as they arrive."""
    print("\n" + "=" * 60)
    print("Step 4: Streaming (stream_tap)")
    print("=" * 60)

    request = TAPRequest(
        meta={"task_id": "stream-1", "intent": "chat"},
        instruction="Explain the difference between a list and a tuple in Python.",
    )

    # stream_tap() compiles the request and yields text chunks as they
    # arrive from the model — ideal for real-time UI updates.
    print("  Streaming response: ", end="", flush=True)
    chunk_count = 0
    async for chunk in provider.stream_tap(request):
        print(chunk, end="", flush=True)
        chunk_count += 1
    print(f"\n  [Received {chunk_count} chunks]")


# ======================================================================
# 5. Cost tracking
# ======================================================================

async def step5_cost_tracking(provider: teragent.ModelProvider) -> None:
    """Track costs across multiple TAP calls."""
    print("\n" + "=" * 60)
    print("Step 5: Cost Tracking")
    print("=" * 60)

    # Execute a few more requests to accumulate cost data
    for i in range(3):
        request = TAPRequest(
            meta={"task_id": f"cost-{i}", "intent": "code_generation"},
            instruction=f"Write a function for task {i}",
        )
        # Use execute_tap_with_retry for automatic retry and cost recording
        await provider.execute_tap_with_retry(request, max_retries=1)

    # Get the cost summary — shows total tokens, call count, and per-provider breakdown
    summary = provider.get_cost_summary()
    print(f"  Total calls:          {summary['total_calls']}")
    print(f"  Total prompt tokens:  {summary['total_prompt_tokens']}")
    print(f"  Total completion tokens: {summary['total_completion_tokens']}")
    print(f"  By provider:")
    for name, stats in summary["by_provider"].items():
        print(f"    {name}: {stats['calls']} calls, "
              f"{stats['prompt_tokens']} prompt, "
              f"{stats['completion_tokens']} completion tokens")


# ======================================================================
# 6. Multiple compilers
# ======================================================================

async def step6_multiple_compilers() -> None:
    """Show how different compilers produce different prompts for the same request."""
    print("\n" + "=" * 60)
    print("Step 6: Multiple Compilers (Same Request, Different Prompts)")
    print("=" * 60)

    # The same TAPRequest can be compiled differently depending on the
    # target model.  Each Compiler optimizes the prompt for its model.
    request = TAPRequest(
        meta={"task_id": "multi-1", "intent": "design"},
        instruction="Design a REST API for a task manager",
        constraints=["Use OpenAPI 3.0 spec"],
    )

    compiler_names = ["default", "glm", "glm_5", "deepseek"]
    for name in compiler_names:
        try:
            provider = create_provider(
                compiler=name,
                adapter="mock",
                model=f"mock-{name}",
            )
            response = await provider.execute_tap(request)
            print(f"\n  Compiler: {name}")
            print(f"    Response: {(response.raw_text or '')[:80]}...")
        except Exception as e:
            print(f"\n  Compiler: {name} — Error: {e}")


# ======================================================================
# Main
# ======================================================================

async def main() -> None:
    """Run all quickstart steps."""
    print("=" * 60)
    print("TerAgent Quickstart")
    print("5-minute getting started guide (MockAdapter — no API keys)")
    print("=" * 60)

    # Steps 1-5 use the same provider instance
    provider = step1_create_provider()

    await step2_execute_tap(provider)
    await step3_chat(provider)
    await step4_streaming(provider)
    await step5_cost_tracking(provider)
    await step6_multiple_compilers()

    # Clean up
    await provider.close()

    print("\n" + "=" * 60)
    print("Quickstart Complete!")
    print("=" * 60)
    print("  Next steps:")
    print("    - Replace adapter='mock' with adapter='openai_compatible'")
    print("      and provide api_key_env='GLM_API_KEY' for real LLM calls")
    print("    - Try different compilers: 'glm_5', 'deepseek_v4', 'minimax_m3'")
    print("    - Explore EventBus orchestration in examples/full_agent/")
    print()


if __name__ == "__main__":
    asyncio.run(main())
