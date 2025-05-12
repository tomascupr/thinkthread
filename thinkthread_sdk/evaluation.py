"""Evaluation strategies for comparing and selecting the best answers.

This module provides evaluation strategies that determine which answer is best
among a set of alternatives in the CoRT reasoning process.
"""

from typing import List
from abc import ABC, abstractmethod
import re

from thinkthread_sdk.llm import LLMClient
from thinkthread_sdk.prompting import TemplateManager


class EvaluationStrategy(ABC):
    """Abstract base class for evaluation strategies.

    This defines the interface for evaluating and selecting the best answer
    from a list of candidate answers.
    """

    @abstractmethod
    def evaluate(
        self,
        question: str,
        answers: List[str],
        llm_client: LLMClient,
        template_manager: TemplateManager,
    ) -> int:
        """Evaluate the answers and return the index of the best one.

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
    """Default implementation of the evaluation strategy.

    Uses an LLM to evaluate and select the best answer.
    """

    def evaluate(
        self,
        question: str,
        answers: List[str],
        llm_client: LLMClient,
        template_manager: TemplateManager,
    ) -> int:
        """Evaluate answers using the LLM and prompt template.

        Args:
            question: The original question
            answers: List of candidate answers
            llm_client: LLM client to use for evaluation
            template_manager: Template manager for prompt templates

        Returns:
            Index of the best answer in the answers list

        """
        formatted_answers = "\n\n".join(
            [f"Answer {i + 1}:\n{answer}" for i, answer in enumerate(answers)]
        )

        prompt = template_manager.render_template(
            "evaluation_prompt.j2",
            {
                "question": question,
                "formatted_answers": formatted_answers,
                "num_answers": len(answers),
            },
        )

        evaluation = llm_client.generate(prompt, temperature=0.2)

        return self._parse_evaluation(evaluation, len(answers))

    def _parse_evaluation(self, evaluation: str, num_answers: int) -> int:
        """Parse the evaluation text to determine which answer was selected as best.

        Args:
            evaluation: The evaluation text from the LLM
            num_answers: The number of answers that were evaluated

        Returns:
            Index of the best answer (0 to num_answers-1)

        """
        for i in range(1, num_answers + 1):
            if (
                f"best answer is Answer {i}" in evaluation
                or f"Best answer is Answer {i}" in evaluation
            ):
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


class Evaluator(ABC):
    """Abstract base class for pairwise answer evaluation.

    This defines the interface for evaluating whether a new answer
    is better than the previous answer.
    """

    @abstractmethod
    def evaluate(
        self,
        question: str,
        prev_answer: str,
        new_answer: str,
        llm_client: LLMClient,
        template_manager: TemplateManager,
    ) -> bool:
        """Evaluate whether the new answer is better than the previous answer.

        Args:
            question: The original question
            prev_answer: The previous answer
            new_answer: The new answer to evaluate
            llm_client: LLM client to use for evaluation
            template_manager: Template manager for prompt templates

        Returns:
            True if the new answer is better, False otherwise

        """
        pass


class ModelEvaluator(Evaluator):
    """Default implementation of the pairwise evaluator.

    Uses an LLM to evaluate whether a new answer is better than the previous one.
    """

    def evaluate(
        self,
        question: str,
        prev_answer: str,
        new_answer: str,
        llm_client: LLMClient,
        template_manager: TemplateManager,
    ) -> bool:
        """Evaluate using the LLM and prompt template to determine if the new answer is better.

        Args:
            question: The original question
            prev_answer: The previous answer
            new_answer: The new answer to evaluate
            llm_client: LLM client to use for evaluation
            template_manager: Template manager for prompt templates

        Returns:
            True if the new answer is better, False otherwise

        """
        prompt = template_manager.render_template(
            "pairwise_prompt.j2",
            {
                "question": question,
                "prev_answer": prev_answer,
                "new_answer": new_answer,
            },
        )

        evaluation = llm_client.generate(prompt, temperature=0.2)

        return self._parse_evaluation(evaluation)

    def _parse_evaluation(self, evaluation: str) -> bool:
        """Parse the evaluation text to determine if the new answer should be selected.

        Uses regex patterns to robustly detect whether the LLM evaluation indicates
        the new answer is better than the previous one, handling variations in phrasing
        and properly accounting for negations.

        Args:
            evaluation: The evaluation text from the LLM

        Returns:
            True if the new answer is better, False otherwise

        """
        positive_patterns = [
            r"(?i)(?<!not\s)(?:new|second)\s+answer\s+(?:is|seems|appears|was)\s+better",
            r"(?i)prefer\s+(?:the\s+)?(?:new|second)\s+answer",
            r"(?i)(?:new|second)\s+answer\s+(?:is|seems|appears)\s+more\s+(?:accurate|complete|helpful|comprehensive)",
            r"(?i)(?:new|second)\s+answer\s+should\s+replace",
            r"(?i)(?:the\s+)?(?:new|second)\s+one\s+(?:is|seems|appears)\s+better",
        ]

        negative_patterns = [
            r"(?i)(?<!not\s)(?:previous|first|old)\s+answer\s+(?:is|seems|appears|was)\s+better",
            r"(?i)prefer\s+(?:the\s+)?(?:previous|first|old)\s+answer",
            r"(?i)(?:the\s+)?(?:previous|first|old)\s+one\s+(?:is|seems|appears)\s+better",
        ]

        for pattern in negative_patterns:
            if re.search(pattern, evaluation):
                return False

        for pattern in positive_patterns:
            if re.search(pattern, evaluation):
                return True

        return False
