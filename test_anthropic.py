import os
import pytest
from thinkthread_sdk.llm.anthropic_client import AnthropicClient


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY environment variable not set",
)
def test_anthropic_client():
    """Test the Anthropic client with a real API key (skipped if key not available)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = AnthropicClient(api_key=api_key)

    response = client.generate("Hello, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0
