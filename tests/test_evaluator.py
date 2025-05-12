import pytest
from unittest.mock import MagicMock

from cort_sdk.evaluation import Evaluator, ModelEvaluator, DefaultEvaluationStrategy
from cort_sdk.llm.dummy import DummyLLMClient
from cort_sdk.prompting import TemplateManager
from cort_sdk.llm import LLMClient


class SimpleEvaluator(Evaluator):
    """Simple evaluator that always returns a predetermined result."""

    def __init__(self, should_prefer_new: bool = True):
        self.should_prefer_new = should_prefer_new

    def evaluate(
        self,
        question: str,
        prev_answer: str,
        new_answer: str,
        llm_client: LLMClient,
        template_manager: TemplateManager,
    ) -> bool:
        """Always returns the configured preference."""
        return self.should_prefer_new


@pytest.fixture
def mock_template_manager():
    """Provide a mock template manager that returns the template context as a string."""
    mock = MagicMock(spec=TemplateManager)

    def render_side_effect(template_name, context):
        if template_name == "pairwise_prompt.j2":
            return f"Pairwise prompt for question: {context['question']}, prev: {context['prev_answer']}, new: {context['new_answer']}"
        return f"Unknown template: {template_name}"

    mock.render_template.side_effect = render_side_effect
    return mock


def test_model_evaluator_parse_evaluation():
    """Test the _parse_evaluation method of ModelEvaluator with different evaluation texts."""
    evaluator = ModelEvaluator()

    assert (
        evaluator._parse_evaluation(
            "After careful consideration, the new answer is better."
        )
        == True
    )
    assert (
        evaluator._parse_evaluation(
            "I prefer the second answer because it's more comprehensive."
        )
        == True
    )
    assert (
        evaluator._parse_evaluation(
            "The new answer is more accurate and should replace the previous one."
        )
        == True
    )

    assert evaluator._parse_evaluation("The previous answer is more accurate.") == False
    assert evaluator._parse_evaluation("I prefer the first answer.") == False
    assert evaluator._parse_evaluation("Both answers have merit.") == False


def test_model_evaluator_evaluate(mock_template_manager):
    """Test the evaluate method of ModelEvaluator."""
    question = "What is AI?"
    prev_answer = "AI is artificial intelligence."
    new_answer = "AI is the simulation of human intelligence in machines."

    responses = [
        "After comparing both answers, the new answer is better because it's more comprehensive."
    ]

    client = DummyLLMClient(responses=responses)

    evaluator = ModelEvaluator()
    result = evaluator.evaluate(
        question, prev_answer, new_answer, client, mock_template_manager
    )

    assert result == True
    assert client.call_count == 1
    assert mock_template_manager.render_template.call_count == 1


class AlwaysSelectIndexEvaluator(DefaultEvaluationStrategy):
    """Evaluator that always selects a specific index."""

    def __init__(self, index_to_select):
        self.index_to_select = index_to_select

    def evaluate(self, question, answers, llm_client, template_manager):
        return self.index_to_select


def test_default_evaluation_strategy_specific_responses():
    """Test DefaultEvaluationStrategy with specific evaluation responses."""
    evaluator = DefaultEvaluationStrategy()

    test_cases = [
        ("After analyzing all answers, the best answer is Answer 2.", 1),
        ("I think Answer 3 is the best because it's more comprehensive.", 2),
        ("Answer 1 is clearly superior to the others.", 0),
        ("The best answer is Answer 4 as it provides more detail.", 3),
    ]

    for response, expected_index in test_cases:
        assert evaluator._parse_evaluation(response, 4) == expected_index


def test_default_evaluation_strategy_full_flow():
    """Test the full flow of DefaultEvaluationStrategy."""
    question = "What is AI?"
    answers = [
        "AI is artificial intelligence.",
        "AI stands for artificial intelligence, which is the simulation of human intelligence in machines.",
        "Artificial Intelligence refers to systems that can perform tasks requiring human-like intelligence.",
    ]

    client = DummyLLMClient(responses=["The best answer is Answer 2."])
    template_manager = MagicMock()
    template_manager.render_template.return_value = "Evaluation prompt"

    evaluator = DefaultEvaluationStrategy()
    result = evaluator.evaluate(question, answers, client, template_manager)

    assert result == 1  # Should select index 1 (Answer 2)
    assert client.call_count == 1
    assert template_manager.render_template.call_count == 1


def test_model_evaluator_edge_cases():
    """Test ModelEvaluator with edge case evaluation responses."""
    evaluator = ModelEvaluator()

    assert evaluator._parse_evaluation("I can't decide which is better.") == False
    assert (
        evaluator._parse_evaluation("Both answers have merit but are flawed.") == False
    )
    assert (
        evaluator._parse_evaluation(
            "The previous answer was good, but the new one is not better."
        )
        == False
    )

    assert evaluator._parse_evaluation("The new answer is not better.") == False
    assert evaluator._parse_evaluation("I cannot say the new answer is better.") == True

    assert (
        evaluator._parse_evaluation(
            "The new answer is significantly better and more comprehensive."
        )
        == False
    )
    assert (
        evaluator._parse_evaluation(
            "The second answer corrects errors in the first and adds more depth."
        )
        == False
    )


def test_model_evaluator_with_different_llm_responses():
    """Test ModelEvaluator with different LLM responses."""
    question = "What is the meaning of life?"
    prev_answer = "The meaning of life is subjective."
    new_answer = "The meaning of life is to seek happiness and fulfillment."

    client1 = DummyLLMClient(responses=["After analysis, the new answer is better."])
    template_manager = MagicMock()
    template_manager.render_template.return_value = "Pairwise evaluation prompt"

    evaluator = ModelEvaluator()
    result1 = evaluator.evaluate(
        question, prev_answer, new_answer, client1, template_manager
    )

    assert result1 == True

    client2 = DummyLLMClient(
        responses=["The previous answer is more concise and better."]
    )
    result2 = evaluator.evaluate(
        question, prev_answer, new_answer, client2, template_manager
    )

    assert result2 == False
