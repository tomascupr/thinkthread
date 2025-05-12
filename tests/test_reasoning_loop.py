import pytest
from unittest.mock import MagicMock

from cort_sdk.cort_session import CoRTSession
from cort_sdk.llm.dummy import DummyLLMClient
from cort_sdk.prompting import TemplateManager
from cort_sdk.config import CoRTConfig


@pytest.fixture
def mock_template_manager():
    """Provide a mock template manager for testing."""
    mock = MagicMock(spec=TemplateManager)

    def render_side_effect(template_name, context):
        if template_name == "initial_prompt.j2":
            return f"Initial: {context['question']}"
        elif template_name == "alternative_prompt.j2":
            return f"Alt: {context['question']} Current: {context['current_answer']}"
        elif template_name == "evaluation_prompt.j2":
            return (
                f"Eval: {context['question']} Answers: {context['formatted_answers']}"
            )
        return f"Unknown: {template_name}"

    mock.render_template.side_effect = render_side_effect
    return mock


class RoundAwareDummyClient(DummyLLMClient):
    """A dummy client that returns different responses based on the round number in the prompt."""

    def generate(self, prompt, **kwargs):
        self._call_count += 1
        # Handle initial prompt
        if "Answer the following question" in prompt:
            return "Initial answer"
        elif "Generate a different possible answer" in prompt:
            if "Current answer: Initial answer" in prompt:
                return "Round 1 alternative"
            elif "Current answer: Round 1 alternative" in prompt:
                return "Round 2 alternative"
            else:
                return f"Alternative for unknown round: {prompt}"
        elif "Evaluate each of the above answers" in prompt:
            if "Initial answer" in prompt and "Round 1 alternative" in prompt:
                return "The best answer is Answer 2"  # Select Round 1 alternative
            elif "Round 1 alternative" in prompt and "Round 2 alternative" in prompt:
                return "The best answer is Answer 2"  # Select Round 2 alternative
            else:
                return "The best answer is Answer 1"  # Default
        elif "Initial:" in prompt:
            return "Initial answer"
        elif "Alt:" in prompt:
            if "Current: Initial answer" in prompt:
                return "Round 1 alternative"
            elif "Current: Round 1 alternative" in prompt:
                return "Round 2 alternative"
            else:
                return f"Alternative for unknown round: {prompt}"
        elif "Eval:" in prompt:
            if "Initial answer" in prompt and "Round 1 alternative" in prompt:
                return "The best answer is Answer 2"  # Select Round 1 alternative
            elif "Round 1 alternative" in prompt and "Round 2 alternative" in prompt:
                return "The best answer is Answer 2"  # Select Round 2 alternative
            else:
                return "The best answer is Answer 1"  # Default
        return f"Unrecognized prompt: {prompt}"


def test_zero_rounds():
    """Test that when max_rounds=0, the initial answer is returned unchanged."""
    client = DummyLLMClient(responses=["Initial answer"])
    config = CoRTConfig(use_pairwise_evaluation=False)

    session = CoRTSession(llm_client=client, max_rounds=0, config=config)

    result = session.run("Test question")

    assert result == "Initial answer"
    assert client.call_count == 1  # Only the initial prompt


def test_single_round():
    """Test a single round of recursive reasoning."""
    client = RoundAwareDummyClient()
    config = CoRTConfig(use_pairwise_evaluation=False)

    session = CoRTSession(
        llm_client=client, max_rounds=1, alternatives=1, config=config
    )

    result = session.run("Test question")

    assert result == "Round 1 alternative"
    assert client.call_count == 3  # Initial + 1 alternative + 1 evaluation


def test_multiple_rounds():
    """Test multiple rounds of recursive reasoning."""
    client = RoundAwareDummyClient()
    config = CoRTConfig(use_pairwise_evaluation=False)

    session = CoRTSession(
        llm_client=client, max_rounds=2, alternatives=1, config=config
    )

    result = session.run("Test question")

    assert result == "Round 2 alternative"
    assert client.call_count == 5  # Initial + 2 alternatives + 2 evaluations


def test_evaluation_selects_best_answer():
    """Test that the evaluation selects the best answer from alternatives."""
    initial_answer = "Initial answer"
    alt1 = "Alternative 1"
    alt2 = "Alternative 2"

    responses = [
        initial_answer,  # Initial
        alt1,
        alt2,  # Alternatives
        "The best answer is Answer 3",  # Evaluation selects alt2
    ]

    client = DummyLLMClient(responses=responses)
    config = CoRTConfig(use_pairwise_evaluation=False)

    session = CoRTSession(
        llm_client=client, max_rounds=1, alternatives=2, config=config
    )

    result = session.run("Test question")

    assert result == alt2
    assert client.call_count == 4  # Initial + 2 alternatives + 1 evaluation


def test_with_mock_template_manager(mock_template_manager):
    """Test the reasoning loop with a mock template manager."""
    responses = [
        "Initial answer",
        "Alternative 1",
        "The best answer is Answer 2",  # Select Alternative 1
    ]

    client = DummyLLMClient(responses=responses)
    config = CoRTConfig(use_pairwise_evaluation=False)

    session = CoRTSession(
        llm_client=client,
        template_manager=mock_template_manager,
        max_rounds=1,
        alternatives=1,
        config=config,
    )

    result = session.run("Test question")

    assert result == "Alternative 1"
    assert client.call_count == 3  # Initial + 1 alternative + 1 evaluation
    assert mock_template_manager.render_template.call_count == 3


def test_self_evaluation():
    """Test recursive reasoning with self-evaluation."""
    from cort_sdk.evaluation import ModelEvaluator

    class TestEvaluator(ModelEvaluator):
        def evaluate(
            self, question, prev_answer, new_answer, llm_client, template_manager
        ):
            return True

    initial_answer = "Initial answer"
    alt1 = "Alternative 1"
    alt2 = "Alternative 2"

    responses = [
        initial_answer,  # Initial
        alt1,
        alt2,  # Alternatives
    ]

    client = DummyLLMClient(responses=responses)
    config = CoRTConfig(use_self_evaluation=True, use_pairwise_evaluation=False)

    evaluator = TestEvaluator()

    session = CoRTSession(
        llm_client=client,
        max_rounds=1,
        alternatives=2,
        evaluator=evaluator,
        config=config,
    )

    result = session.run("Test question")

    assert result == alt2  # Last alternative with self-evaluation always preferring new
    assert (
        client.call_count == 3
    )  # Initial + 2 alternatives (no separate evaluation calls)


def test_very_short_question():
    """Test that the system handles very short questions properly."""
    client = DummyLLMClient(responses=["Short response for a short question"])
    config = CoRTConfig(use_pairwise_evaluation=False)

    session = CoRTSession(
        llm_client=client, max_rounds=1, alternatives=1, config=config
    )

    result = session.run("Why?")

    assert "Short response for a short question" in result
    assert client.call_count == 3  # Initial + 1 alternative + 1 evaluation


def test_long_answer_handling():
    """Test that the system handles potentially long answers properly."""
    long_response = "This is a long response. " * 500

    client = DummyLLMClient(
        responses=[long_response, "Short alternative", "The best answer is Answer 1"]
    )
    config = CoRTConfig(use_pairwise_evaluation=False)

    session = CoRTSession(
        llm_client=client, max_rounds=1, alternatives=1, config=config
    )

    result = session.run("Generate a very long explanation")

    assert result == long_response
    assert client.call_count == 3  # Initial + 1 alternative + 1 evaluation
