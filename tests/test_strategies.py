"""Tests for evaluation strategies in the ThinkThread SDK."""

from typing import List

from thinkthread.llm import LLMClient
from thinkthread.prompting import TemplateManager
from thinkthread.evaluation import EvaluationStrategy


class SimpleEvaluationStrategy(EvaluationStrategy):
    """Simple evaluation strategy that always selects a specific index."""

    def __init__(self, index_to_select: int = 0):
        self.index_to_select = index_to_select

    def evaluate(
        self,
        question: str,
        answers: List[str],
        llm_client: LLMClient,
        template_manager: TemplateManager,
    ) -> int:
        """Always select the configured index if it's within range.

        Args:
            question: The original question
            answers: List of candidate answers
            llm_client: LLM client to use for evaluation
            template_manager: Template manager for prompt templates

        Returns:
            Index of the best answer in the answers list

        """
        if self.index_to_select < len(answers):
            return self.index_to_select
        return 0
