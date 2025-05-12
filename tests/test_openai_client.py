from unittest.mock import patch, MagicMock

from thinkthread_sdk.llm.openai_client import OpenAIClient
from openai import OpenAIError


def test_openai_client_init():
    """Test that the OpenAIClient initializes correctly with default and custom values."""
    client = OpenAIClient(api_key="test_key")
    assert client.model == "gpt-4"
    assert client.api_key == "test_key"

    client = OpenAIClient(api_key="test_key", model="gpt-3.5-turbo")
    assert client.model == "gpt-3.5-turbo"

    client = OpenAIClient(api_key="test_key", temperature=0.7, max_tokens=100)
    assert client.opts.get("temperature") == 0.7
    assert client.opts.get("max_tokens") == 100


@patch("openai.OpenAI")
def test_generate_success(mock_openai):
    """Test successful text generation."""
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    mock_chat = MagicMock()
    mock_client.chat.completions.create.return_value = mock_chat

    mock_choice = MagicMock()
    mock_chat.choices = [mock_choice]
    mock_choice.message.content = "Test response"

    client = OpenAIClient(api_key="test_key")
    response = client.generate("Test prompt")

    assert response == "Test response"

    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "gpt-4"
    assert call_args["messages"] == [{"role": "user", "content": "Test prompt"}]


@patch("openai.OpenAI")
def test_generate_with_options(mock_openai):
    """Test generation with custom options."""
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    mock_chat = MagicMock()
    mock_client.chat.completions.create.return_value = mock_chat

    mock_choice = MagicMock()
    mock_chat.choices = [mock_choice]
    mock_choice.message.content = "Test response"

    client = OpenAIClient(api_key="test_key", temperature=0.7)

    client.generate("Test prompt", temperature=0.2, max_tokens=50)

    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["temperature"] == 0.2
    assert call_args["max_tokens"] == 50


@patch("openai.OpenAI")
def test_generate_api_error(mock_openai):
    """Test handling of API errors."""
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.side_effect = OpenAIError("API error")

    client = OpenAIClient(api_key="test_key")
    response = client.generate("Test prompt")

    assert "OpenAI API error" in response


def test_manual_example():
    """Manual test example (not automatically run).

    This test demonstrates how to use the OpenAIClient with a real API key.
    To run this test, uncomment the code and set your API key.
    """
    #     client = OpenAIClient(api_key=api_key, model="gpt-3.5-turbo")
    #     response = client.generate("Hello, how are you?")
