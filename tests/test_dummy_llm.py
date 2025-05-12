import pytest
from thinkthread_sdk.llm.dummy import DummyLLMClient


def test_dummy_llm_default_behavior():
    """Test the default behavior of DummyLLMClient."""
    client = DummyLLMClient(model_name="dummy-model")

    assert client.model_name == "dummy-model"

    assert client.call_count == 0

    response1 = client.generate("Hello")
    assert response1 == "Dummy response #1 to: 'Hello'"
    assert client.call_count == 1

    response2 = client.generate("World")
    assert response2 == "Dummy response #2 to: 'World'"
    assert client.call_count == 2

    assert response1 != response2


def test_dummy_llm_with_predefined_responses():
    """Test DummyLLMClient with a list of predefined responses."""
    responses = ["First response", "Second response", "Third response"]
    client = DummyLLMClient(responses=responses)

    assert client.generate("Any prompt") == "First response"
    assert client.generate("Any prompt") == "Second response"
    assert client.generate("Any prompt") == "Third response"

    assert client.generate("Any prompt") == "First response"


def test_dummy_llm_with_custom_generator():
    """Test DummyLLMClient with a custom response generator function."""

    def custom_generator(prompt: str, count: int) -> str:
        return f"Custom response {count}: {prompt.upper()}"

    client = DummyLLMClient(response_generator=custom_generator)

    assert client.generate("test") == "Custom response 1: TEST"
    assert client.generate("another") == "Custom response 2: ANOTHER"


def test_dummy_llm_reset():
    """Test that the call counter can be reset."""
    client = DummyLLMClient()

    client.generate("First")
    client.generate("Second")
    assert client.call_count == 2

    client.reset()
    assert client.call_count == 0

    assert client.generate("After reset") == "Dummy response #1 to: 'After reset'"


def test_dummy_llm_with_kwargs():
    """Test that kwargs are accepted but ignored."""
    client = DummyLLMClient()

    response1 = client.generate("Test")
    response2 = client.generate("Test", temperature=0.7, max_tokens=100)

    assert response1 == "Dummy response #1 to: 'Test'"
    assert response2 == "Dummy response #2 to: 'Test'"
