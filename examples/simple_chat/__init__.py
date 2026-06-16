"""examples/simple_chat — Minimal chat example using teragent

This example shows the simplest way to create a provider and
have a conversation with an LLM through teragent's TAP pipeline.

Usage:
    export GLM_API_KEY=your_key_here
    python -m examples.simple_chat
"""

from __future__ import annotations

import asyncio

import teragent


async def main() -> None:
    # Create a provider with GLM compiler + OpenAI-compatible adapter
    provider = teragent.create_provider(
        compiler="glm_5",
        adapter="openai_compatible",
        model="glm-5",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="GLM_API_KEY",
    )

    # Simple chat (bypasses TAP compilation, sends raw messages)
    result = await provider.chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! What is the capital of France?"},
        ]
    )
    print(f"Chat response: {result['content']}")

    # TAP-based execution (goes through Compiler for model-specific optimization)
    tap_response = await provider.execute_tap(
        teragent.TAPRequest(
            meta={"task_id": "chat-1", "intent": "chat"},
            instruction="Explain the difference between a list and a tuple in Python.",
            constraints=["Be concise", "Include a code example"],
            output_format_hint="Plain text with code block",
        )
    )
    print(f"\nTAP response:\n{tap_response.raw_text}")

    # Clean up — release any persistent HTTP connections held by the adapter.
    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
