import pytest
from unittest.mock import MagicMock

from cort_sdk.cort_session import CoRTSession
from cort_sdk.config import CoRTConfig
from cort_sdk.llm.dummy import DummyLLMClient
from cort_sdk.prompting import TemplateManager
from tests.test_evaluator import SimpleEvaluator
from cort_sdk.llm import LLMClient


@pytest.fixture
def mock_template_manager():
    """Provide a mock template manager that returns the template context as a string."""
    mock = MagicMock(spec=TemplateManager)
    
    def render_side_effect(template_name, context):
        if template_name == "initial_prompt.j2":
            return f"Initial prompt for question: {context['question']}"
        elif template_name == "alternative_prompt.j2":
            return f"Alternative prompt for question: {context['question']}, current: {context['current_answer']}"
        elif template_name == "pairwise_prompt.j2":
            return f"Pairwise prompt for question: {context['question']}, prev: {context['prev_answer']}, new: {context['new_answer']}"
        return f"Unknown template: {template_name}"
        
    mock.render_template.side_effect = render_side_effect
    return mock


@pytest.fixture
def mock_config():
    """Provide a mock config object with pairwise evaluation enabled."""
    config = CoRTConfig()
    config.use_pairwise_evaluation = True
    return config


def test_cort_session_with_pairwise_evaluation_prefer_new(mock_template_manager, mock_config):
    """Test CoRT session with pairwise evaluation that prefers new answers."""
    initial_answer = "Initial answer"
    alt1 = "Alternative 1"
    alt2 = "Alternative 2"
    alt3 = "Alternative 3"
    
    responses = [
        initial_answer,  # Initial answer
        alt1, alt2, alt3,  # Round 1 alternatives
    ]
    
    client = DummyLLMClient(responses=responses)
    
    evaluator = SimpleEvaluator(should_prefer_new=True)
    
    session = CoRTSession(
        llm_client=client, 
        template_manager=mock_template_manager,
        config=mock_config,
        max_rounds=1,
        evaluator=evaluator
    )
    
    result = session.run("What is AI?")
    
    assert result == alt3
    
    assert client.call_count == 4
    assert mock_template_manager.render_template.call_count == 4


def test_cort_session_with_pairwise_evaluation_prefer_previous(mock_template_manager, mock_config):
    """Test CoRT session with pairwise evaluation that prefers previous answers."""
    
    initial_answer = "Initial answer"
    alt1 = "Alternative 1"
    alt2 = "Alternative 2"
    alt3 = "Alternative 3"
    
    responses = [
        initial_answer,  # Initial answer
        alt1, alt2, alt3,  # Round 1 alternatives
    ]
    
    client = DummyLLMClient(responses=responses)
    
    evaluator = SimpleEvaluator(should_prefer_new=False)
    
    session = CoRTSession(
        llm_client=client, 
        template_manager=mock_template_manager,
        config=mock_config,
        max_rounds=1,
        evaluator=evaluator
    )
    
    result = session.run("What is AI?")
    
    assert result == initial_answer
    
    assert client.call_count == 4
    assert mock_template_manager.render_template.call_count == 4


def test_cort_session_with_model_evaluator(mock_template_manager, mock_config):
    """Test CoRT session with the ModelEvaluator."""
    from cort_sdk.evaluation import ModelEvaluator
    
    initial_answer = "Initial answer"
    alt1 = "Alternative 1"
    alt2 = "Alternative 2"
    alt3 = "Alternative 3"
    
    eval_responses = [
        "After comparing both answers, the new answer is better.",
        "After comparing both answers, the previous answer is better.",
        "After comparing both answers, the new answer is better."
    ]
    
    responses = [
        initial_answer,  # Initial answer
        alt1, alt2, alt3,  # Round 1 alternatives
    ] + eval_responses  # Evaluation responses
    
    client = DummyLLMClient(responses=responses)
    
    session = CoRTSession(
        llm_client=client, 
        template_manager=mock_template_manager,
        config=mock_config,
        max_rounds=1
    )
    
    result = session.run("What is AI?")
    
    assert result == alt3
    
    assert client.call_count == 7
    assert mock_template_manager.render_template.call_count == 7
