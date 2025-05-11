from typing import List, Optional
from abc import ABC, abstractmethod

from cort_sdk.llm import LLMClient
from cort_sdk.prompting import TemplateManager


class EvaluationStrategy(ABC):
    """
    Abstract base class for evaluation strategies.
    
    This defines the interface for evaluating and selecting the best answer
    from a list of candidate answers.
    """
    
    @abstractmethod
    def evaluate(
        self, 
        question: str, 
        answers: List[str], 
        llm_client: LLMClient,
        template_manager: TemplateManager
    ) -> int:
        """
        Evaluate the answers and return the index of the best one.
        
        Args:
            question: The original question
            answers: List of candidate answers
            llm_client: LLM client to use for evaluation
            template_manager: Template manager for prompt templates
            
        Returns:
            Index of the best answer in the answers list
        """
        pass


class DefaultEvaluationStrategy(EvaluationStrategy):
    """
    Default implementation of the evaluation strategy.
    
    Uses an LLM to evaluate and select the best answer.
    """
    
    def evaluate(
        self, 
        question: str, 
        answers: List[str], 
        llm_client: LLMClient,
        template_manager: TemplateManager
    ) -> int:
        """
        Evaluate answers using the LLM and prompt template.
        
        Args:
            question: The original question
            answers: List of candidate answers
            llm_client: LLM client to use for evaluation
            template_manager: Template manager for prompt templates
            
        Returns:
            Index of the best answer in the answers list
        """
        formatted_answers = "\n\n".join([
            f"Answer {i+1}:\n{answer}" for i, answer in enumerate(answers)
        ])
        
        prompt = template_manager.render_template(
            "evaluation_prompt.j2",
            {
                "question": question,
                "formatted_answers": formatted_answers,
                "num_answers": len(answers)
            }
        )
        
        evaluation = llm_client.generate(prompt, temperature=0.2)
        
        return self._parse_evaluation(evaluation, len(answers))
    
    def _parse_evaluation(self, evaluation: str, num_answers: int) -> int:
        """
        Parse the evaluation text to determine which answer was selected as best.
        
        Args:
            evaluation: The evaluation text from the LLM
            num_answers: The number of answers that were evaluated
            
        Returns:
            Index of the best answer (0 to num_answers-1)
        """
        for i in range(1, num_answers + 1):
            if f"best answer is Answer {i}" in evaluation or f"Best answer is Answer {i}" in evaluation:
                return i - 1  # Convert to 0-based index
        
        for i in range(1, num_answers + 1):
            indicators = [
                f"Answer {i} is the best",
                f"select Answer {i}",
                f"choose Answer {i}",
                f"prefer Answer {i}",
            ]
            for indicator in indicators:
                if indicator in evaluation:
                    return i - 1
        
        return 0
