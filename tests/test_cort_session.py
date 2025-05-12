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
    assert session.max_rounds == 2  # max_rounds defaults to rounds
    assert session.template_manager == mock_template_manager


def test_cort_session_custom_params(mock_template_manager, mock_config):
    """Test that CoRTSession initializes with custom parameter values."""
    client = DummyLLMClient()
    session = CoRTSession(
        llm_client=client, 
        alternatives=5, 
        rounds=3, 
        max_rounds=4,
        template_manager=mock_template_manager, 
        config=mock_config
    )
    
    assert session.llm_client == client
    assert session.alternatives == 5
    assert session.rounds == 3
    assert session.max_rounds == 4  # max_rounds is set explicitly
    assert session.template_manager == mock_template_manager


def test_cort_session_zero_rounds(mock_template_manager, mock_config):
    """Test that when max_rounds=0, the session returns the initial answer unchanged."""
    initial_answer = "This is the initial answer."
    client = DummyLLMClient(responses=[initial_answer])
    
    session = CoRTSession(llm_client=client, max_rounds=0, template_manager=mock_template_manager, config=mock_config)
    
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
    
    # Disable pairwise evaluation for this test to use the DefaultEvaluationStrategy
    mock_config.use_pairwise_evaluation = False
    
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
    
    # Disable pairwise evaluation for this test to use the DefaultEvaluationStrategy
    mock_config.use_pairwise_evaluation = False
    
    session = CoRTSession(llm_client=client, template_manager=mock_template_manager, config=mock_config)
    
    result = session.run("What is the meaning of life?")
    
    assert result == initial_answer
    
    assert client.call_count == 9
    assert mock_template_manager.render_template.call_count == 9  # 1 initial + 2*3 alternatives + 2 evals


def test_evaluation_strategy_parse_evaluation(mock_template_manager, mock_config):
    """Test the _parse_evaluation method of DefaultEvaluationStrategy with different evaluation texts."""
    from cort_sdk.evaluation import DefaultEvaluationStrategy
    
    strategy = DefaultEvaluationStrategy()
    
    assert strategy._parse_evaluation("After careful consideration, the best answer is Answer 2.", 3) == 1
    assert strategy._parse_evaluation("I think the Best answer is Answer 3.", 3) == 2
    
    assert strategy._parse_evaluation("Answer 1 is the best because it's more comprehensive.", 3) == 0
    assert strategy._parse_evaluation("I would select Answer 3 as it provides more detail.", 3) == 2
    
    assert strategy._parse_evaluation("All answers have merit.", 3) == 0


def test_cort_session_max_rounds(mock_template_manager, mock_config):
    """Test that max_rounds parameter works correctly."""
    initial_answer = "Initial answer"
    alt1_round1 = "Alternative 1 (round 1)"
    alt2_round1 = "Alternative 2 (round 1)"
    alt3_round1 = "Alternative 3 (round 1)"
    
    responses = [
        initial_answer,  # Initial answer
        alt1_round1, alt2_round1, alt3_round1,  # Round 1 alternatives
        "The best answer is Answer 2",  # Evaluation selects alt1_round1 (index 1 in the list)
    ]
    
    client = DummyLLMClient(responses=responses)
    
    # Set max_rounds=1 explicitly and disable pairwise evaluation
    mock_config.use_pairwise_evaluation = False
    
    session = CoRTSession(
        llm_client=client, 
        template_manager=mock_template_manager, 
        max_rounds=1,
        config=mock_config
    )
    
    result = session.run("What is the meaning of life?")
    
    assert result == alt1_round1
    assert client.call_count == 5  # Initial + 3 alternatives + 1 evaluation
    assert mock_template_manager.render_template.call_count == 5


def test_cort_session_custom_evaluation_strategy(mock_template_manager, mock_config):
    """Test that a custom evaluation strategy works correctly."""
    from tests.test_strategies import SimpleEvaluationStrategy
    
    initial_answer = "Initial answer"
    alt1 = "Alternative 1"
    alt2 = "Alternative 2"
    alt3 = "Alternative 3"
    
    responses = [
        initial_answer,  # Initial answer
        alt1, alt2, alt3,  # Alternatives
    ]
    
    client = DummyLLMClient(responses=responses)
    
    custom_strategy = SimpleEvaluationStrategy(index_to_select=2)
    
    # Disable pairwise evaluation for this test to use the custom evaluation strategy
    mock_config.use_pairwise_evaluation = False
    
    session = CoRTSession(
        llm_client=client, 
        template_manager=mock_template_manager,
        max_rounds=1,
        evaluation_strategy=custom_strategy,
        config=mock_config
    )
    
    result = session.run("What is the meaning of life?")
    
    assert result == alt2
    
    assert client.call_count == 4  # Initial + 3 alternatives
    assert mock_template_manager.render_template.call_count == 4  # Initial + 3 alternatives


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
