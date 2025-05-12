"""Tests for the Anthropic LLM client implementation."""
import os
import pytest
from cort_sdk.llm.anthropic_client import AnthropicClient


def test_anthropic_client():
    """Test the AnthropicClient with a simple prompt."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")

    client = AnthropicClient(api_key=api_key)
    response = client.generate("Hello, how are you?")

    assert response
    assert isinstance(response, str)
    assert len(response) > 0
