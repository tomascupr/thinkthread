"""High-level utilities for common ThinkThread SDK usage patterns.

This module provides simplified abstractions for common reasoning patterns,
making it easier to use the powerful reasoning capabilities of ThinkThread SDK
with minimal code.
"""

from typing import Optional, Dict, Any, Union, Tuple

from thinkthread_sdk.llm import LLMClient
from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.tree_thinker import TreeThinker
from thinkthread_sdk.config import ThinkThreadConfig, create_config
from thinkthread_sdk.prompting import TemplateManager


class ThinkThreadUtils:
    """Utility class providing high-level abstractions for common reasoning patterns.

    This class wraps the more complex ThinkThreadSession and TreeThinker classes
    to provide simple, one-line methods for common reasoning patterns like
    self-refinement and n-best brainstorming.
    """

    def __init__(
        self, llm_client: LLMClient, config: Optional[ThinkThreadConfig] = None
    ):
        """Initialize a ThinkThreadUtils instance.

        Args:
            llm_client: LLM client to use for generation and evaluation
            config: Optional configuration object
        """
        self.llm_client = llm_client
        self.config = config or create_config()
        self.template_manager = TemplateManager(self.config.prompt_dir)

    def self_refine(
        self,
        question: str,
        initial_answer: str,
        rounds: int = 2,
        alternatives: int = 3,
        return_metadata: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """Refine an answer through multiple rounds of critique and revision.

        This method implements a generate → critique → revise loop that iteratively
        improves an initial answer through self-refinement. It leverages the
        Chain-of-Recursive-Thoughts (CoRT) technique from ThinkThreadSession.

        Args:
            question: The question or prompt to refine an answer for
            initial_answer: The initial answer to refine
            rounds: Number of refinement rounds to perform
            alternatives: Number of alternative answers to generate per round
            return_metadata: Whether to return additional metadata

        Returns:
            The refined answer, or a dictionary with the answer and metadata if
            return_metadata is True
        """
        session = ThinkThreadSession(
            llm_client=self.llm_client,
            alternatives=alternatives,
            rounds=rounds,
            template_manager=self.template_manager,
            config=self.config,
        )

        current_answer = initial_answer

        for round_num in range(1, rounds + 1):
            alternatives_list = session._generate_alternatives(question, current_answer)

            all_answers = [current_answer] + alternatives_list

            best_index = session.evaluation_strategy.evaluate(
                question, all_answers, self.llm_client, self.template_manager
            )

            current_answer = all_answers[best_index]

        if return_metadata:
            return {
                "question": question,
                "initial_answer": initial_answer,
                "final_answer": current_answer,
                "rounds": rounds,
                "alternatives_per_round": alternatives,
            }

        return current_answer

    async def self_refine_async(
        self,
        question: str,
        initial_answer: str,
        rounds: int = 2,
        alternatives: int = 3,
        return_metadata: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """Asynchronous version of self_refine.

        Args:
            question: The question or prompt to refine an answer for
            initial_answer: The initial answer to refine
            rounds: Number of refinement rounds to perform
            alternatives: Number of alternative answers to generate per round
            return_metadata: Whether to return additional metadata

        Returns:
            The refined answer, or a dictionary with the answer and metadata if
            return_metadata is True
        """
        session = ThinkThreadSession(
            llm_client=self.llm_client,
            alternatives=alternatives,
            rounds=rounds,
            template_manager=self.template_manager,
            config=self.config,
        )

        current_answer = initial_answer

        for round_num in range(1, rounds + 1):
            alternatives_list = await session._generate_alternatives_async(
                question, current_answer
            )

            all_answers = [current_answer] + alternatives_list

            best_index = await session._evaluate_all_async(question, all_answers)

            current_answer = all_answers[best_index]

        if return_metadata:
            return {
                "question": question,
                "initial_answer": initial_answer,
                "final_answer": current_answer,
                "rounds": rounds,
                "alternatives_per_round": alternatives,
            }

        return current_answer

    def _find_best_answer(
        self, tree_thinker: TreeThinker
    ) -> Tuple[str, Optional[str], float]:
        """Find the best answer from a tree thinker's threads.

        Args:
            tree_thinker: TreeThinker instance with populated threads

        Returns:
            Tuple containing (best_answer, best_node_id, best_score)
        """
        best_node_id = None
        best_score = -1.0

        for node_id, node in tree_thinker.threads.items():
            if node.score > best_score:
                best_score = node.score
                best_node_id = node_id

        if not best_node_id:
            return "No answer found", None, -1.0

        best_node = tree_thinker.threads[best_node_id]
        best_answer = best_node.state.get("current_answer", "No answer found")

        return best_answer, best_node_id, best_score

    def n_best_brainstorm(
        self,
        question: str,
        n: int = 5,
        max_depth: int = 1,
        return_metadata: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """Generate multiple answers and select the best one.

        This method implements a shallow Tree-of-Thoughts approach that generates
        multiple answers in parallel and evaluates them to select the best one.

        Args:
            question: The question or prompt to generate answers for
            n: Number of answers to generate
            max_depth: Maximum depth of the thinking tree (should be shallow for brainstorming)
            return_metadata: Whether to return additional metadata

        Returns:
            The best answer, or a dictionary with the answer and metadata if
            return_metadata is True
        """
        tree_thinker = TreeThinker(
            llm_client=self.llm_client,
            max_tree_depth=max_depth,
            branching_factor=n,
            template_manager=self.template_manager,
            config=self.config,
        )

        result = tree_thinker.solve(question, beam_width=n, max_iterations=max_depth)

        best_answer, best_node_id, best_score = self._find_best_answer(tree_thinker)

        if return_metadata and best_node_id:
            return {
                "question": question,
                "best_answer": best_answer,
                "best_score": best_score,
                "best_node_id": best_node_id,
                "n": n,
                "max_depth": max_depth,
                "all_node_ids": list(tree_thinker.threads.keys()),
            }

        return best_answer

    async def n_best_brainstorm_async(
        self,
        question: str,
        n: int = 5,
        max_depth: int = 1,
        return_metadata: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """Asynchronous version of n_best_brainstorm.

        Args:
            question: The question or prompt to generate answers for
            n: Number of answers to generate
            max_depth: Maximum depth of the thinking tree (should be shallow for brainstorming)
            return_metadata: Whether to return additional metadata

        Returns:
            The best answer, or a dictionary with the answer and metadata if
            return_metadata is True
        """
        tree_thinker = TreeThinker(
            llm_client=self.llm_client,
            max_tree_depth=max_depth,
            branching_factor=n,
            template_manager=self.template_manager,
            config=self.config,
        )

        result = await tree_thinker.solve_async(
            question, beam_width=n, max_iterations=max_depth
        )

        best_answer, best_node_id, best_score = self._find_best_answer(tree_thinker)

        if return_metadata and best_node_id:
            return {
                "question": question,
                "best_answer": best_answer,
                "best_score": best_score,
                "best_node_id": best_node_id,
                "n": n,
                "max_depth": max_depth,
                "all_node_ids": list(tree_thinker.threads.keys()),
            }

        return best_answer
