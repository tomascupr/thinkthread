import pytest
from unittest.mock import MagicMock

from cort_sdk.evaluation import Evaluator, ModelEvaluator
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
        template_manager: TemplateManager
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
    
    assert evaluator._parse_evaluation("After careful consideration, the new answer is better.") == True
    assert evaluator._parse_evaluation("I prefer the second answer because it's more comprehensive.") == True
    assert evaluator._parse_evaluation("The new answer is more accurate and should replace the previous one.") == True
    
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
    result = evaluator.evaluate(question, prev_answer, new_answer, client, mock_template_manager)
    
    assert result == True
    assert client.call_count == 1
    assert mock_template_manager.render_template.call_count == 1
