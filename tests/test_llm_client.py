import pytest
from typing import AsyncIterator
from thinkthread.llm import LLMClient


def test_llm_client_is_abstract():
    """Test that LLMClient cannot be instantiated directly."""
    with pytest.raises(TypeError):
        LLMClient()


class MockLLMClient(LLMClient):
    """Mock implementation of LLMClient for testing."""

    def _generate_uncached(self, prompt: str, **kwargs) -> str:
        """Mock implementation that returns a fixed response."""
        return f"Mock response to: {prompt}"

    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Mock implementation of astream that yields a fixed response."""
        yield f"Mock streaming response to: {prompt}"


def test_llm_client_implementation():
    """Test that a concrete implementation can be instantiated and used."""
    client = MockLLMClient(model_name="mock-model")

    assert client.model_name == "mock-model"

    response = client.generate("Hello")
    assert response == "Mock response to: Hello"

    response = client.generate("Hello", temperature=0.7)
    assert response == "Mock response to: Hello"
