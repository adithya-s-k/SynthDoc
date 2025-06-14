#!/usr/bin/env python3
"""
Example usage of SynthDoc with LiteLLM integration.

This example demonstrates how to use different LLM providers with SynthDoc
using LiteLLM's unified interface.
"""

import os
from synthdoc import SynthDoc


def main():
    """Demonstrate SynthDoc with different LLM providers."""

    # Example 1: Using OpenAI GPT-3.5-turbo
    print("=== Example 1: OpenAI GPT-3.5-turbo ===")
    synth_openai = SynthDoc(
        llm_model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),  # Set this environment variable
    )

    # Generate raw documents
    docs = synth_openai.generate_raw_docs(
        language="en",
        num_pages=2,
        prompt="Generate technical documentation about machine learning",
    )

    print(f"Generated {len(docs)} documents")
    for i, doc in enumerate(docs):
        print(f"Document {i + 1} preview: {doc['content'][:100]}...")

    # Generate VQA dataset
    vqa_data = synth_openai.generate_vqa(
        source_documents=docs,
        question_types=["factual", "reasoning"],
        difficulty_levels=["easy", "medium"],
    )

    print(f"Generated {len(vqa_data['questions'])} VQA pairs")

    # Example 2: Using Claude (if you have Anthropic API key)
    print("\n=== Example 2: Anthropic Claude ===")
    if os.getenv("ANTHROPIC_API_KEY"):
        synth_claude = SynthDoc(
            llm_model="claude-3-sonnet-20240229", api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        docs_claude = synth_claude.generate_raw_docs(
            language="en",
            num_pages=1,
            prompt="Generate a research paper abstract about computer vision",
        )

        print(f"Claude generated: {docs_claude[0]['content'][:100]}...")
    else:
        print("ANTHROPIC_API_KEY not set, skipping Claude example")

    # Example 3: Using local Ollama model (if available)
    print("\n=== Example 3: Local Ollama ===")
    try:
        synth_ollama = SynthDoc(
            llm_model="ollama/llama2",  # Assumes you have llama2 installed locally
            api_key=None,  # No API key needed for local models
        )

        docs_ollama = synth_ollama.generate_raw_docs(
            language="en",
            num_pages=1,
            prompt="Generate a simple document about renewable energy",
        )

        print(f"Ollama generated: {docs_ollama[0]['content'][:100]}...")
    except Exception as e:
        print(f"Ollama not available: {e}")

    # Example 4: Working without LLM (fallback mode)
    print("\n=== Example 4: Fallback mode (no LLM) ===")
    synth_fallback = SynthDoc(
        llm_model="gpt-3.5-turbo",
        api_key=None,  # No API key provided
    )

    docs_fallback = synth_fallback.generate_raw_docs(
        language="en", num_pages=1, prompt="This will use fallback content"
    )

    print(f"Fallback content: {docs_fallback[0]['content']}")


if __name__ == "__main__":
    main()
