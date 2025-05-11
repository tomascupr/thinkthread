import pytest
from unittest.mock import patch, call, MagicMock

from cort_sdk.cort_session import CoRTSession
from cort_sdk.llm.dummy import DummyLLMClient
from cort_sdk.prompting import TemplateManager


@pytest.fixture
def mock_template_manager():
    """Provide a mock template manager that returns the template context as a string."""
    mock = MagicMock(spec=TemplateManager)
    
    def render_side_effect(template_name, context):
        if template_name == "initial_prompt.j2":
            return f"Initial prompt for question: {context['question']}"
        elif template_name == "alternative_prompt.j2":
            return f"Alternative prompt for question: {context['question']}, current: {context['current_answer']}"
        elif template_name == "evaluation_prompt.j2":
            return f"Evaluation prompt for question: {context['question']}, answers: {context['formatted_answers']}"
        return f"Unknown template: {template_name}"
        
    mock.render_template.side_effect = render_side_effect
    return mock

@pytest.fixture
def mock_config():
    """Provide a mock config object."""
    from cort_sdk.config import CoRTConfig
    return CoRTConfig()


def test_cort_session_init(mock_template_manager, mock_config):
    """Test that CoRTSession initializes with correct default values."""
    client = DummyLLMClient()
    session = CoRTSession(llm_client=client, template_manager=mock_template_manager, config=mock_config)
    
    assert session.llm_client == client
    assert session.alternatives == 3
    assert session.rounds == 2
    assert session.template_manager == mock_template_manager


def test_cort_session_custom_params(mock_template_manager, mock_config):
    """Test that CoRTSession initializes with custom parameter values."""
    client = DummyLLMClient()
    session = CoRTSession(llm_client=client, alternatives=5, rounds=3, template_manager=mock_template_manager, config=mock_config)
    
    assert session.llm_client == client
    assert session.alternatives == 5
    assert session.rounds == 3
    assert session.template_manager == mock_template_manager


def test_cort_session_zero_rounds(mock_template_manager, mock_config):
    """Test that when rounds=0, the session returns the initial answer unchanged."""
    initial_answer = "This is the initial answer."
    client = DummyLLMClient(responses=[initial_answer])
    
    session = CoRTSession(llm_client=client, rounds=0, template_manager=mock_template_manager, config=mock_config)
    
    result = session.run("What is the meaning of life?")
    
    assert result == initial_answer
    assert mock_template_manager.render_template.call_count == 1


def test_cort_session_normal_case(mock_template_manager, mock_config):
    """Test the normal case with rounds=2 default."""
    initial_answer = "Initial answer"
    alt1_round1 = "Alternative 1 (round 1)"
    alt2_round1 = "Alternative 2 (round 1)"
    alt3_round1 = "Alternative 3 (round 1)"
    
    alt1_round2 = "Alternative 1 (round 2)"
    alt2_round2 = "Alternative 2 (round 2)"
    alt3_round2 = "Alternative 3 (round 2)"
    
    responses = [
        initial_answer,  # Initial answer
        alt1_round1, alt2_round1, alt3_round1,  # Round 1 alternatives
        "The best answer is Answer 2",  # Evaluation selects alt2_round1 (index 1 in the list)
        alt1_round2, alt2_round2, alt3_round2,  # Round 2 alternatives
        "The best answer is Answer 4"   # Evaluation selects alt3_round2 (index 3 in the list)
    ]
    
    client = DummyLLMClient(responses=responses)
    
    session = CoRTSession(llm_client=client, template_manager=mock_template_manager, config=mock_config)
    
    result = session.run("What is the meaning of life?")
    
    assert result == alt3_round2
    
    assert client.call_count == 9
    assert mock_template_manager.render_template.call_count == 9  # 1 initial + 2*3 alternatives + 2 evals


def test_cort_session_same_answer_selected(mock_template_manager, mock_config):
    """Test the case where the model's evaluation picks the same answer as current."""
    initial_answer = "Initial answer"
    alt1 = "Alternative 1"
    alt2 = "Alternative 2"
    alt3 = "Alternative 3"
    
    responses = [
        initial_answer,  # Initial answer
        alt1, alt2, alt3,  # Round 1 alternatives
        "The best answer is Answer 1",  # Evaluation selects initial_answer (index 0 in the list)
        alt1, alt2, alt3,  # Round 2 alternatives
        "The best answer is Answer 1"   # Evaluation selects initial_answer again (index 0 in the list)
    ]
    
    client = DummyLLMClient(responses=responses)
    
    session = CoRTSession(llm_client=client, template_manager=mock_template_manager, config=mock_config)
    
    result = session.run("What is the meaning of life?")
    
    assert result == initial_answer
    
    assert client.call_count == 9
    assert mock_template_manager.render_template.call_count == 9  # 1 initial + 2*3 alternatives + 2 evals


def test_cort_session_parse_evaluation(mock_template_manager, mock_config):
    """Test the _parse_evaluation method with different evaluation texts."""
    client = DummyLLMClient()
    session = CoRTSession(llm_client=client, template_manager=mock_template_manager, config=mock_config)
    
    assert session._parse_evaluation("After careful consideration, the best answer is Answer 2.", 3) == 1
    assert session._parse_evaluation("I think the Best answer is Answer 3.", 3) == 2
    
    assert session._parse_evaluation("Answer 1 is the best because it's more comprehensive.", 3) == 0
    assert session._parse_evaluation("I would select Answer 3 as it provides more detail.", 3) == 2
    
    assert session._parse_evaluation("All answers have merit.", 3) == 0


def test_cort_session_exception_handling(mock_template_manager, mock_config):
    """Test that no exceptions are raised during normal operation."""
    class ExceptionClient(DummyLLMClient):
        def generate(self, prompt: str, **kwargs) -> str:
            if self._call_count == 4:  # 0-based index, so this is the 5th call
                raise ValueError("Test exception")
            return super().generate(prompt, **kwargs)
    
    client = ExceptionClient(responses=["Answer"] * 10)
    session = CoRTSession(llm_client=client, template_manager=mock_template_manager, config=mock_config)
    
    with pytest.raises(ValueError):
        session.run("What is the meaning of life?")
