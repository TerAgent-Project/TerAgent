"""examples/multi_model — Multi-model collaboration example using teragent

This example demonstrates teragent's core feature: Compiler/Adapter
orthogonal composition. Different models get different compilation
strategies even through the same protocol adapter.

Usage:
    export GLM_API_KEY=your_key_here
    export OPENROUTER_API_KEY=your_key_here
    python -m examples.multi_model
"""

from __future__ import annotations

import asyncio

import teragent


async def main() -> None:
    # Model 1: GLM via direct API — GLM compiler + OpenAI adapter
    glm_provider = teragent.create_provider(
        compiler="glm",
        adapter="openai_compatible",
        model="glm-4-flash",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="GLM_API_KEY",
    )

    # Model 2: Claude via OpenRouter — Anthropic compiler + OpenAI adapter
    # Key insight: same protocol (OpenAI), different compiler (Anthropic XML)
    claude_provider = teragent.create_provider(
        compiler="anthropic",
        adapter="openai_compatible",
        model="anthropic/claude-sonnet-4-20250514",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    )

    # Model 3: DeepSeek via direct API — DeepSeek compiler + OpenAI adapter
    deepseek_provider = teragent.create_provider(
        compiler="deepseek",
        adapter="openai_compatible",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
    )

    # Same request, different compilers → different optimized prompts
    tap_request = teragent.TAPRequest(
        meta={"task_id": "multi-1", "intent": "execute"},
        instruction="Write a function to check if a string is a palindrome",
        constraints=["Python 3.10+", "Include type hints", "Handle edge cases"],
        output_format_hint="<file path='palindrome.py'>complete code</file>",
    )

    # Execute with each model
    providers = [
        ("GLM (recency effect optimization)", glm_provider),
        ("Claude (XML tag optimization)", claude_provider),
        ("DeepSeek (minimalist optimization)", deepseek_provider),
    ]

    for name, provider in providers:
        try:
            response = await provider.execute_tap(tap_request)
            # Extract files from response
            files = teragent.extract_files_from_response(
                response.raw_text, task_id="multi-1"
            )
            print(f"\n{'='*60}")
            print(f"Model: {name}")
            print(f"Files extracted: {list(files.keys())}")
            if files:
                first_file = list(files.values())[0]
                print(f"Code preview:\n{first_file[:500]}")
        except Exception as e:
            print(f"\nModel: {name} — Error: {e}")

    # Demonstrate Compiler/Adapter orthogonality
    print("\n" + "="*60)
    print("Compiler/Adapter Orthogonality:")
    print("  Same Adapter (OpenAI) + Different Compilers = Different Prompts")
    print("  Same Compiler (GLM) + Different Adapters = Same Prompt, Different Transport")


if __name__ == "__main__":
    asyncio.run(main())
