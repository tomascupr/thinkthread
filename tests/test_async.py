import pytest
import asyncio
from typing import List

from cort_sdk.llm.dummy import DummyLLMClient
from cort_sdk.cort_session import CoRTSession
from cort_sdk.prompting import TemplateManager
from cort_sdk.config import CoRTConfig


@pytest.fixture
def mock_template_manager():
    """Provide a mock template manager that returns the template context as a string."""
    from unittest.mock import MagicMock
    
    mock = MagicMock(spec=TemplateManager)
    
    def render_side_effect(template_name, context):
        if template_name == "initial_prompt.j2":
            return f"Initial prompt for question: {context['question']}"
        elif template_name == "alternative_prompt.j2":
            return f"Alternative prompt for question: {context['question']}, current: {context['current_answer']}"
        elif template_name == "evaluation_prompt.j2":
            return f"Evaluation prompt for question: {context['question']}, answers: {context['formatted_answers']}"
        elif template_name == "final_answer.j2":
            return f"Final answer to {context['question']}: {context['answer']}"
        return f"Unknown template: {template_name}"
        
    mock.render_template.side_effect = render_side_effect
    return mock


@pytest.fixture
def mock_config():
    """Provide a mock config object."""
    return CoRTConfig()


@pytest.mark.asyncio
async def test_dummy_client_acomplete():
    """Test that DummyLLMClient.acomplete works correctly."""
    client = DummyLLMClient(responses=["Test response"])
    
    response = await client.acomplete("Test prompt")
    
    assert response == "Test response"
    assert client.call_count == 1


@pytest.mark.asyncio
async def test_dummy_client_astream():
    """Test that DummyLLMClient.astream works correctly."""
    client = DummyLLMClient(responses=["This is a test response"])
    
    chunks = []
    async for chunk in client.astream("Test prompt"):
        chunks.append(chunk)
    
    result = "".join(chunks)
    assert "This is a test response" in result
    assert client.call_count == 1
    
    assert len(chunks) > 1


@pytest.mark.asyncio
async def test_cort_session_run_async(mock_template_manager, mock_config):
    """Test that CoRTSession.run_async works correctly."""
    initial_answer = "Initial answer"
    alt1 = "Alternative 1"
    alt2 = "Alternative 2"
    alt3 = "Alternative 3"
    
    responses = [
        initial_answer,  # Initial answer
        alt1, alt2, alt3,  # Round 1 alternatives
        "The best answer is Answer 2",  # Evaluation selects alt1 (index 1 in the list)
    ]
    
    client = DummyLLMClient(responses=responses)
    
    mock_config.use_pairwise_evaluation = False
    
    session = CoRTSession(
        llm_client=client, 
        template_manager=mock_template_manager,
        max_rounds=1,
        config=mock_config
    )
    
    result = await session.run_async("What is the meaning of life?")
    
    assert result == alt1
    assert client.call_count == 5  # Initial + 3 alternatives + 1 evaluation


@pytest.mark.asyncio
async def test_sync_vs_async_results(mock_template_manager, mock_config):
    """Test that sync and async methods produce the same results."""
    responses = ["Response 1", "Response 2", "Response 3", "Response 4", "Response 5"]
    client = DummyLLMClient(responses=responses)
    
    mock_config.use_pairwise_evaluation = False
    
    session = CoRTSession(
        llm_client=client, 
        template_manager=mock_template_manager,
        max_rounds=1,
        config=mock_config
    )
    
    client.reset()
    sync_result = session.run("Test question")
    
    client.reset()
    async_result = await session.run_async("Test question")
    
    assert sync_result == async_result


@pytest.mark.asyncio
async def test_streaming_output():
    """Test that streaming output works correctly."""
    client = DummyLLMClient(responses=["This is a streaming test response"])
    
    chunks = []
    async for chunk in client.astream("Test prompt"):
        chunks.append(chunk)
    
    full_text = "".join(chunks)
    assert "This is a streaming test response" in full_text
    assert len(chunks) > 1
