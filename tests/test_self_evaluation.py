"""Tests for the self-evaluation functionality in CoRT sessions."""

import pytest
from unittest.mock import MagicMock

from thinkthread_sdk.cort_session import ThinkThreadSession
from thinkthread_sdk.config import ThinkThreadConfig
from thinkthread_sdk.llm.dummy import DummyLLMClient
from thinkthread_sdk.prompting import TemplateManager
from tests.test_evaluator import SimpleEvaluator


@pytest.fixture
def mock_template_manager():
    """Provide a mock template manager that returns the template context as a string."""
    mock = MagicMock(spec=TemplateManager)

    def render_side_effect(template_name, context):
        if template_name == "initial_prompt.j2":
            return f"Initial prompt for question: {context['question']}"
        elif template_name == "alternative_prompt.j2":
            return f"Alternative prompt for question: {context['question']}, current: {context['current_answer']}"
        elif template_name == "evaluate_prompt.j2":
            return f"Evaluation prompt for question: {context['question']}, prev: {context['prev_answer']}, new: {context['new_answer']}"
        elif template_name == "pairwise_prompt.j2":
            return f"Pairwise prompt for question: {context['question']}, prev: {context['prev_answer']}, new: {context['new_answer']}"
        return f"Unknown template: {template_name}"

    mock.render_template.side_effect = render_side_effect
    return mock


@pytest.fixture
def mock_config():
    """Provide a mock config object with self-evaluation enabled."""
    config = ThinkThreadConfig()
    config.use_pairwise_evaluation = False
    config.use_self_evaluation = True
    return config


def test_cort_session_with_self_evaluation_prefer_new(
    mock_template_manager, mock_config
):
    """Test CoRT session with self-evaluation that prefers new answers."""
    initial_answer = "Initial answer"
    alt1 = "Alternative 1"
    alt2 = "Alternative 2"
    alt3 = "Alternative 3"

    responses = [
        initial_answer,  # Initial answer
        alt1,
        alt2,
        alt3,  # Round 1 alternatives
    ]

    client = DummyLLMClient(responses=responses)

    evaluator = SimpleEvaluator(should_prefer_new=True)

    session = ThinkThreadSession(
        llm_client=client,
        template_manager=mock_template_manager,
        config=mock_config,
        max_rounds=1,
        evaluator=evaluator,
    )

    result = session.run("What is AI?")

    assert result == alt3

    assert client.call_count == 4
    assert mock_template_manager.render_template.call_count == 4


def test_cort_session_with_self_evaluation_prefer_previous(
    mock_template_manager, mock_config
):
    """Test CoRT session with self-evaluation that prefers previous answers."""
    initial_answer = "Initial answer"
    alt1 = "Alternative 1"
    alt2 = "Alternative 2"
    alt3 = "Alternative 3"

    responses = [
        initial_answer,  # Initial answer
        alt1,
        alt2,
        alt3,  # Round 1 alternatives
    ]

    client = DummyLLMClient(responses=responses)

    evaluator = SimpleEvaluator(should_prefer_new=False)

    session = ThinkThreadSession(
        llm_client=client,
        template_manager=mock_template_manager,
        config=mock_config,
        max_rounds=1,
        evaluator=evaluator,
    )

    result = session.run("What is AI?")

    assert result == initial_answer

    assert client.call_count == 4
    assert mock_template_manager.render_template.call_count == 4


def test_cort_session_with_model_evaluator(mock_template_manager, mock_config):
    """Test CoRT session with the ModelEvaluator."""

    initial_answer = "Initial answer"
    alt1 = "Alternative 1"
    alt2 = "Alternative 2"
    alt3 = "Alternative 3"

    eval_responses = [
        "After comparing both answers, the new answer is better.",
        "After comparing both answers, the previous answer is better.",
        "After comparing both answers, the new answer is better.",
    ]

    responses = [
        initial_answer,  # Initial answer
        alt1,
        alt2,
        alt3,  # Round 1 alternatives
    ] + eval_responses  # Evaluation responses

    client = DummyLLMClient(responses=responses)

    session = ThinkThreadSession(
        llm_client=client,
        template_manager=mock_template_manager,
        config=mock_config,
        max_rounds=1,
    )

    result = session.run("What is AI?")

    assert result == alt3

    assert client.call_count == 7
    assert mock_template_manager.render_template.call_count == 7


def test_cort_session_multiple_rounds_with_self_evaluation(
    mock_template_manager, mock_config
):
    """Test the self-evaluation over multiple rounds to verify improvement."""

    initial_answer = "Initial answer (round 0)"

    alt1_round1 = "Alternative 1 (round 1)"
    alt2_round1 = "Alternative 2 (round 1)"
    alt3_round1 = "Alternative 3 (round 1)"

    alt1_round2 = "Alternative 1 (round 2)"
    alt2_round2 = "Alternative 2 (round 2)"
    alt3_round2 = "Alternative 3 (round 2)"

    eval_responses_round1 = [
        "After comparing, the previous answer is better.",  # Initial vs Alt1 Round1
        "After comparing, the previous answer is better.",  # Initial vs Alt2 Round1
        "After comparing, the new answer is better.",  # Initial vs Alt3 Round1
    ]

    eval_responses_round2 = [
        "After comparing, the previous answer is better.",  # Alt3 Round1 vs Alt1 Round2
        "After comparing, the new answer is better.",  # Alt3 Round1 vs Alt2 Round2
        "After comparing, the previous answer is better.",  # Alt2 Round2 vs Alt3 Round2
    ]

    responses = (
        [
            initial_answer,  # Initial answer
            alt1_round1,
            alt2_round1,
            alt3_round1,  # Round 1 alternatives
        ]
        + eval_responses_round1
        + [
            alt1_round2,
            alt2_round2,
            alt3_round2,  # Round 2 alternatives
        ]
        + eval_responses_round2
    )

    client = DummyLLMClient(responses=responses)

    session = ThinkThreadSession(
        llm_client=client,
        template_manager=mock_template_manager,
        config=mock_config,
        max_rounds=2,
    )

    result = session.run("What is AI?")

    assert result == alt2_round2

    assert client.call_count == 13
    assert mock_template_manager.render_template.call_count == 13
