import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

from thinkthread_sdk.llm.openai_client import OpenAIClient


class AsyncIteratorMock:
    """Helper class to mock an async iterator."""
    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.items):
            item = self.items[self.index]
            self.index += 1
            return item
        raise StopAsyncIteration


@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
async def test_astream_parameters(mock_async_openai):
    """Test that the astream method correctly sets the stream parameter."""
    mock_client = AsyncMock()
    mock_async_openai.return_value = mock_client
    
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Hello"
    
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = " world"
    
    mock_stream = AsyncIteratorMock([chunk1, chunk2])
    mock_client.chat.completions.create.return_value = mock_stream
    
    client = OpenAIClient(api_key="test_key")
    result = []
    async for token in client.astream("Test prompt"):
        result.append(token)
    
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4"
    assert call_kwargs["messages"] == [{"role": "user", "content": "Test prompt"}]
    assert call_kwargs["stream"] is True
    
    assert result == ["Hello", " world"]


@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
async def test_astream_with_options(mock_async_openai):
    """Test streaming with custom options."""
    mock_client = AsyncMock()
    mock_async_openai.return_value = mock_client
    
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = "Test response"
    
    mock_stream = AsyncIteratorMock([chunk])
    mock_client.chat.completions.create.return_value = mock_stream
    
    client = OpenAIClient(api_key="test_key", temperature=0.7)
    result = []
    async for token in client.astream("Test prompt", temperature=0.2, max_tokens=50):
        result.append(token)
    
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["temperature"] == 0.2
    assert call_kwargs["max_tokens"] == 50
    assert call_kwargs["stream"] is True  # Stream should always be True
    
    assert result == ["Test response"]


@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
async def test_astream_api_error(mock_async_openai):
    """Test handling of API errors in streaming mode."""
    from openai import OpenAIError
    
    mock_client = AsyncMock()
    mock_async_openai.return_value = mock_client
    mock_client.chat.completions.create.side_effect = OpenAIError("API error")
    
    client = OpenAIClient(api_key="test_key")
    result = []
    async for token in client.astream("Test prompt"):
        result.append(token)
    
    assert len(result) == 1
    assert "OpenAI API error" in result[0]


@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
async def test_astream_empty_response(mock_async_openai):
    """Test handling of empty responses in streaming mode."""
    mock_client = AsyncMock()
    mock_async_openai.return_value = mock_client
    
    # Create chunks for the mock stream with empty content
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = None  # Empty content
    
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = ""  # Empty string
    
    chunk3 = MagicMock()
    chunk3.choices = [MagicMock()]
    chunk3.choices[0].delta.content = "Content"  # Valid content
    
    mock_stream = AsyncIteratorMock([chunk1, chunk2, chunk3])
    mock_client.chat.completions.create.return_value = mock_stream
    
    client = OpenAIClient(api_key="test_key")
    result = []
    async for token in client.astream("Test prompt"):
        result.append(token)
    
    assert result == ["Content"]
