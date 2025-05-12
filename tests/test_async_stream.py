import pytest
import asyncio
from typing import AsyncIterator, List

from cort_sdk.llm.base import LLMClient
from cort_sdk.llm.dummy import DummyLLMClient
from cort_sdk.cort_session import CoRTSession
from cort_sdk.config import CoRTConfig


class CustomStreamingClient(DummyLLMClient):
    """Custom streaming client that yields tokens with controlled timing."""
    
    def __init__(self, tokens: List[str], delay: float = 0.01):
        super().__init__()
        self.tokens = tokens
        self.delay = delay
    
    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream tokens with a controlled delay."""
        for token in self.tokens:
            await asyncio.sleep(self.delay)
            yield token
    
    async def acomplete(self, prompt: str, **kwargs) -> str:
        """Return the full concatenated response."""
        return "".join(self.tokens)


@pytest.mark.asyncio
async def test_streaming_tokens():
    """Test that streaming yields tokens correctly."""
    tokens = ["Hello", ", ", "world", "!"]
    client = CustomStreamingClient(tokens)
    
    collected = []
    async for token in client.astream("Test prompt"):
        collected.append(token)
    
    assert collected == tokens
    
    complete = await client.acomplete("Test prompt")
    assert complete == "Hello, world!"


@pytest.mark.asyncio
async def test_empty_stream():
    """Test streaming with no tokens."""
    client = CustomStreamingClient([])
    
    collected = []
    async for token in client.astream("Test prompt"):
        collected.append(token)
    
    assert len(collected) == 0


@pytest.mark.asyncio
async def test_large_stream():
    """Test streaming with a large number of tokens."""
    tokens = [f"Token {i}" for i in range(100)]
    client = CustomStreamingClient(tokens, delay=0.001)
    
    collected = []
    async for token in client.astream("Test prompt"):
        collected.append(token)
    
    assert len(collected) == 100
    assert collected[0] == "Token 0"
    assert collected[-1] == "Token 99"


@pytest.mark.asyncio
async def test_async_reasoning_loop():
    """Test async reasoning loop with streaming results."""
    config = CoRTConfig(use_pairwise_evaluation=False)
    
    responses = [
        "Initial answer",
        "Alternative 1",
        "The best answer is Answer 2"  # Select Alternative 1
    ]
    
    client = DummyLLMClient(responses=responses)
    
    session = CoRTSession(
        llm_client=client,
        max_rounds=1,
        alternatives=1,
        config=config
    )
    
    result = await session.run_async("Test question")
    
    assert result == "Alternative 1"
    assert client.call_count == 3  # Initial + 1 alternative + 1 evaluation


@pytest.mark.asyncio
async def test_streaming_collect():
    """Test collecting streamed tokens into a final result."""
    tokens = ["This ", "is ", "a ", "streamed ", "response."]
    client = CustomStreamingClient(tokens)
    
    result = []
    async for token in client.astream("Test prompt"):
        result.append(token)
    
    complete = "".join(result)
    assert complete == "This is a streamed response."
