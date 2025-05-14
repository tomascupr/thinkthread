"""Tree-of-Thoughts solver implementation.

This module contains the TreeThinker class that implements tree-based search
for exploring multiple reasoning paths using ThinkThreadSession instances.
"""

from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
import re
import asyncio

from thinkthread_sdk.llm import LLMClient
from thinkthread_sdk.prompting import TemplateManager
from thinkthread_sdk.config import ThinkThreadConfig, create_config
from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.evaluation import Evaluator, ModelEvaluator
from thinkthread_sdk.base_reasoner import BaseReasoner
from thinkthread_sdk.reasoning_utils import (
    generate_alternatives,
    generate_alternatives_async,
)


@dataclass
class ThinkThreadNode:
    """Node in the thinking tree representing a single reasoning path.

    This class stores the session, state, and evaluation score for a single
    reasoning path in the tree-of-thoughts search process.
    """

    session: ThinkThreadSession
    state: Dict[str, Any]
    score: float = 0.0
    parent_id: Optional[str] = None
    node_id: str = ""
    depth: int = 0
    children: List[str] = None  # type: ignore # Will be initialized in __post_init__

    def __post_init__(self):
        """Initialize empty lists for None values."""
        if self.children is None:
            self.children = []


class TreeThinker(BaseReasoner):
    """Tree-of-Thoughts solver.

    This class implements a tree-based search approach for exploring multiple
    reasoning paths using ThinkThreadSession instances. It can manage
    multiple thinking threads and evaluate them to find the best solution.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_tree_depth: int = 3,
        branching_factor: int = 3,
        template_manager: Optional[TemplateManager] = None,
        config: Optional[ThinkThreadConfig] = None,
        evaluator: Optional[Evaluator] = None,
        scoring_function: Optional[Callable[[str, Dict[str, Any]], float]] = None,
    ) -> None:
        """Initialize a TreeThinker instance.

        Args:
            llm_client: LLM client to use for generating and evaluating thoughts
            max_tree_depth: Maximum depth of the thinking tree
            branching_factor: Number of branches to explore at each node
            template_manager: Optional template manager for prompt templates
            config: Optional configuration object
            evaluator: Optional evaluator for scoring thought branches
            scoring_function: Optional custom function for scoring thought branches
        """
        super().__init__(llm_client, template_manager, config)
        self.max_tree_depth = max_tree_depth
        self.branching_factor = branching_factor
        self.evaluator = evaluator or ModelEvaluator()
        self.scoring_function = scoring_function
        self.threads: Dict[str, ThinkThreadNode] = {}
        self.current_layer: List[str] = []

    def solve(
        self, problem: str, beam_width: int = 1, max_iterations: int = 10, **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """Solve a problem using tree-of-thoughts approach.

        This method initiates the tree search process to find the best solution
        to the given problem. It explores multiple reasoning paths by creating
        and managing ThinkThreadSession instances.

        Args:
            problem: The problem to solve
            beam_width: Number of parallel thought threads to create
            max_iterations: Maximum number of iterations to perform
            **kwargs: Additional parameters for the solving process

        Returns:
            The best solution found or a dictionary containing the solution and metadata
        """
        self.threads.clear()
        self.current_layer = []

        for i in range(beam_width):
            session = ThinkThreadSession(
                llm_client=self.llm_client,
                template_manager=self.template_manager,
                config=self.config,
            )

            node_id = f"root_{i}" if i > 0 else "root"
            node = ThinkThreadNode(
                session=session,
                state={"problem": problem},
                node_id=node_id,
                depth=0,
            )

            self.threads[node_id] = node
            self.current_layer.append(node_id)

        if max_iterations > 0:
            expansion_results = self.expand_threads(beam_width=beam_width)

            return {
                "status": "expanded",
                "message": f"Created {beam_width} parallel thought threads and expanded them with beam search (width={beam_width})",
                "thread_count": len(self.threads),
                "root_threads": self.current_layer[:beam_width],
                "expanded_threads": expansion_results["new_nodes"],
                "expansion_count": expansion_results["count"],
                "pruned_count": expansion_results.get("pruned_count", 0),
                "pruned_out_count": expansion_results.get("pruned_out_count", 0),
            }

        return {
            "status": "initialized",
            "message": f"Created {beam_width} parallel thought threads for problem: {problem}",
            "thread_count": beam_width,
            "thread_ids": self.current_layer,
        }

    def expand_threads(
        self,
        nodes_to_expand: Optional[List[str]] = None,
        beam_width: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Expand the specified thought threads by generating the next thought for each.

        This method takes the current active thought threads and expands each one
        by generating alternative continuations. Each new continuation becomes a
        child node in the thinking tree. After expansion, only the top N branches
        (where N is the beam width) are kept for further expansion.

        Args:
            nodes_to_expand: List of node IDs to expand. If None, expands all nodes in the current layer.
            beam_width: Number of top branches to keep after expansion. If None, uses the branching_factor.

        Returns:
            Dictionary containing information about the expansion results
        """
        if nodes_to_expand is None:
            nodes_to_expand = self.current_layer.copy()

        if beam_width is None:
            beam_width = self.branching_factor

        new_layer = []
        new_nodes = []
        all_expanded_nodes = []

        for node_id in nodes_to_expand:
            if node_id not in self.threads:
                continue

            parent_node = self.threads[node_id]

            if parent_node.depth >= self.max_tree_depth:
                continue

            parent_session = parent_node.session
            problem = parent_node.state.get("problem", "")
            current_answer = parent_node.state.get("current_answer", "")

            if not current_answer:
                initial_prompt = self.template_manager.render_template(
                    "initial_prompt.j2", {"question": problem}
                )
                current_answer = self.llm_client.generate(
                    initial_prompt, temperature=0.7
                )
                parent_node.state["current_answer"] = current_answer

            alternatives = self._generate_continuations(
                parent_session, problem, current_answer
            )

            for i, alternative in enumerate(alternatives):
                child_id = f"{node_id}_child_{i}"

                child_session = ThinkThreadSession(
                    llm_client=self.llm_client,
                    template_manager=self.template_manager,
                    config=self.config,
                )

                child_state = parent_node.state.copy()
                child_state["current_answer"] = alternative

                child_node = ThinkThreadNode(
                    session=child_session,
                    state=child_state,
                    parent_id=node_id,
                    node_id=child_id,
                    depth=parent_node.depth + 1,
                )

                score = self._score_node(problem, child_node)
                child_node.score = score

                parent_node.children.append(child_id)

                self.threads[child_id] = child_node
                all_expanded_nodes.append(child_id)
                new_nodes.append(child_id)

        if all_expanded_nodes:
            sorted_nodes = sorted(
                all_expanded_nodes,
                key=lambda node_id: self.threads[node_id].score,
                reverse=True,
            )

            pruned_nodes = sorted_nodes[:beam_width]
            pruned_out = sorted_nodes[beam_width:]

            self.current_layer = pruned_nodes
            new_layer = pruned_nodes

            pruned_scores = {
                node_id: self.threads[node_id].score for node_id in pruned_nodes
            }
            pruned_out_scores = {
                node_id: self.threads[node_id].score for node_id in pruned_out
            }
        else:
            self.current_layer = []
            new_layer = []
            pruned_scores = {}
            pruned_out_scores = {}

        return {
            "count": len(new_nodes),
            "new_nodes": new_nodes,
            "new_layer": new_layer,
            "pruned_count": len(new_layer),
            "pruned_out_count": len(pruned_out) if "pruned_out" in locals() else 0,
            "scores": pruned_scores,
            "pruned_out_scores": pruned_out_scores,
        }

    def _score_node(self, problem: str, node: ThinkThreadNode) -> float:
        """Score a node based on the quality of its thought.

        This method evaluates the promise of a thought branch using one of three approaches:
        1. A custom scoring function provided during initialization
        2. LLM-based evaluation using ModelEvaluator if a reference answer is available
        3. A heuristic based on answer quality metrics as a fallback

        Args:
            problem: The original problem
            node: The node to score

        Returns:
            A score between 0.0 and 1.0 indicating the quality of the thought
        """
        try:
            if self.scoring_function:
                return self.scoring_function(problem, node.state)

            current_answer = node.state.get("current_answer", "")
            if not current_answer:
                return 0.0

            base_score = self._calculate_base_score(current_answer)

            # If this is a child node, compare with parent using ModelEvaluator
            if node.parent_id and node.parent_id in self.threads:
                parent_node = self.threads[node.parent_id]
                parent_answer = parent_node.state.get("current_answer", "")
                parent_score = parent_node.score

                if parent_answer and current_answer:
                    try:
                        comparison_result = self.evaluator.evaluate(
                            problem,
                            parent_answer,
                            current_answer,
                            self.llm_client,
                            self.template_manager,
                        )

                        improvement_factor = 0.05 + (base_score * 0.2)

                        if comparison_result:
                            import random

                            random_factor = random.uniform(0.01, 0.05)
                            return min(
                                parent_score + improvement_factor + random_factor, 1.0
                            )
                        else:
                            return max(parent_score - (improvement_factor / 2), 0.0)
                    except Exception:
                        return min(
                            max(base_score, parent_score - 0.1), parent_score + 0.1
                        )

            import random

            random_factor = random.uniform(-0.05, 0.05)
            return max(min(base_score + random_factor, 1.0), 0.0)

        except Exception:
            import random

            return 0.4 + random.uniform(0.0, 0.2)

    def _calculate_base_score(self, answer: str) -> float:
        """Calculate a base score for an answer based on various quality metrics.

        Args:
            answer: The answer to score

        Returns:
            A score between 0.0 and 1.0 indicating the base quality
        """
        try:
            length = len(answer)
            length_score = min(length / 1000, 1.0)  # Cap at 1000 chars

            sentences = answer.split(". ")
            sentence_count = len(sentences)

            words = answer.lower().split()
            word_count = len(words)
            unique_words = len(set(words))

            vocabulary_richness = 0.0
            if word_count > 0:
                vocabulary_richness = min(unique_words / word_count, 1.0)

            has_structure = bool(re.search(r"^\s*[\d\.\-\*]+\s+", answer, re.MULTILINE))
            structure_bonus = 0.1 if has_structure else 0.0

            has_examples = bool(
                re.search(
                    r"example|instance|case|e\.g\.|i\.e\.|for instance", answer.lower()
                )
            )
            examples_bonus = 0.1 if has_examples else 0.0

            avg_sentence_length = word_count / max(sentence_count, 1)
            sentence_complexity = min(
                avg_sentence_length / 20, 1.0
            )  # Optimal ~20 words

            score = (
                (0.3 * length_score)
                + (0.2 * vocabulary_richness)
                + (0.2 * sentence_complexity)
                + (
                    0.1 * min(sentence_count / 10, 1.0)
                )  # Reward multiple sentences up to 10
                + structure_bonus
                + examples_bonus
            )

            return min(score, 1.0)  # Ensure score is at most 1.0

        except Exception:
            return 0.5

    def _generate_continuations(
        self, session: ThinkThreadSession, problem: str, current_answer: str
    ) -> List[str]:
        """Generate continuations for a thought thread.

        This method uses the session's ability to generate alternatives to create
        continuations for the current thought thread. It includes error handling
        to gracefully handle LLM API failures.

        Args:
            session: The ThinkThreadSession to use for generation
            problem: The original problem
            current_answer: The current answer or thought

        Returns:
            List of alternative continuations
        """
        num_continuations = min(self.branching_factor, 5)  # Limit to 5 for efficiency
        alternatives = []

        try:
            alternatives = generate_alternatives(
                problem,
                current_answer,
                self.llm_client,
                self.template_manager,
                count=num_continuations,
                temperature=0.9,
            )
        except Exception:
            # Fall back to manual generation with error handling
            for i in range(num_continuations):
                try:
                    prompt = self.template_manager.render_template(
                        "alternative_prompt.j2",
                        {"question": problem, "current_answer": current_answer},
                    )

                    alternative = self.llm_client.generate(prompt, temperature=0.9)
                    alternatives.append(alternative)
                except Exception:
                    fallback_alternative = f"{current_answer}\n\nAdditional thoughts: Unable to generate continuation due to an error."
                    alternatives.append(fallback_alternative)

                    if not alternatives:
                        alternatives.append(current_answer)

                    if i == 0:
                        break

        return alternatives

    async def _generate_continuations_async(
        self, session: ThinkThreadSession, problem: str, current_answer: str
    ) -> List[str]:
        """Asynchronously generate continuations for a thought thread.

        This method uses the session's ability to generate alternatives to create
        continuations for the current thought thread. It leverages async LLM calls
        to generate multiple continuations in parallel. Includes robust error handling
        to gracefully handle LLM API failures.

        Args:
            session: The ThinkThreadSession to use for generation
            problem: The original problem
            current_answer: The current answer or thought

        Returns:
            List of alternative continuations
        """
        num_continuations = min(self.branching_factor, 5)  # Limit to 5 for efficiency
        alternatives = []

        try:
            parallel = (
                hasattr(self.config, "parallel_alternatives")
                and self.config.parallel_alternatives
            )
            alternatives = await generate_alternatives_async(
                problem,
                current_answer,
                self.llm_client,
                self.template_manager,
                count=num_continuations,
                temperature=0.9,
                parallel=parallel,
            )
            return alternatives
        except Exception:
            # Fall back to original implementation with error handling
            try:
                # Create prompts for all continuations
                prompts = []
                for i in range(num_continuations):
                    try:
                        prompt = self.template_manager.render_template(
                            "alternative_prompt.j2",
                            {"question": problem, "current_answer": current_answer},
                        )
                        prompts.append(prompt)
                    except Exception:
                        prompts.append(f"Continue this thought: {current_answer}")

                # Try batch completion if available
                if hasattr(self.llm_client, "acomplete_batch"):
                    try:
                        alternatives = await self.llm_client.acomplete_batch(
                            prompts, temperature=0.9
                        )
                        return alternatives
                    except Exception:
                        pass

                # Individual completions with error handling
                async def generate_alternative(prompt):
                    try:
                        return await self.llm_client.acomplete(prompt, temperature=0.9)
                    except Exception:
                        return f"{current_answer}\n\nAdditional thoughts: Unable to generate continuation due to an error."

                alternatives = await asyncio.gather(
                    *[generate_alternative(prompt) for prompt in prompts]
                )

                return alternatives
            except Exception:
                # If all else fails, return a single fallback alternative
                return [current_answer]

    async def expand_threads_async(
        self,
        nodes_to_expand: Optional[List[str]] = None,
        beam_width: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Asynchronously expand the specified thought threads by generating the next thought for each.

        This method takes the current active thought threads and expands each one
        by generating alternative continuations in parallel. Each new continuation becomes a
        child node in the thinking tree. After expansion, only the top N branches
        (where N is the beam width) are kept for further expansion.

        Args:
            nodes_to_expand: List of node IDs to expand. If None, expands all nodes in the current layer.
            beam_width: Number of top branches to keep after expansion. If None, uses the branching_factor.

        Returns:
            Dictionary containing information about the expansion results
        """
        if nodes_to_expand is None:
            nodes_to_expand = self.current_layer.copy()

        if beam_width is None:
            beam_width = self.branching_factor

        new_layer = []
        new_nodes = []
        all_expanded_nodes = []

        async def process_node(node_id):
            if node_id not in self.threads:
                return []

            parent_node = self.threads[node_id]

            if parent_node.depth >= self.max_tree_depth:
                return []

            parent_session = parent_node.session
            problem = parent_node.state.get("problem", "")
            current_answer = parent_node.state.get("current_answer", "")

            if not current_answer:
                initial_prompt = self.template_manager.render_template(
                    "initial_prompt.j2", {"question": problem}
                )
                current_answer = await self.llm_client.acomplete(
                    initial_prompt, temperature=0.7
                )
                parent_node.state["current_answer"] = current_answer

            alternatives = await self._generate_continuations_async(
                parent_session, problem, current_answer
            )

            node_results = []

            for i, alternative in enumerate(alternatives):
                child_id = f"{node_id}_child_{i}"

                child_session = ThinkThreadSession(
                    llm_client=self.llm_client,
                    template_manager=self.template_manager,
                    config=self.config,
                )

                child_state = parent_node.state.copy()
                child_state["current_answer"] = alternative

                child_node = ThinkThreadNode(
                    session=child_session,
                    state=child_state,
                    parent_id=node_id,
                    node_id=child_id,
                    depth=parent_node.depth + 1,
                )

                score = self._score_node(problem, child_node)
                child_node.score = score

                parent_node.children.append(child_id)

                self.threads[child_id] = child_node
                node_results.append(child_id)

            return node_results

        node_results = await asyncio.gather(
            *[process_node(node_id) for node_id in nodes_to_expand]
        )

        for result in node_results:
            all_expanded_nodes.extend(result)
            new_nodes.extend(result)

        # Implement beam search pruning: keep only the top N branches
        if all_expanded_nodes:
            sorted_nodes = sorted(
                all_expanded_nodes,
                key=lambda node_id: self.threads[node_id].score,
                reverse=True,
            )

            pruned_nodes = sorted_nodes[:beam_width]
            pruned_out = sorted_nodes[beam_width:]

            self.current_layer = pruned_nodes
            new_layer = pruned_nodes

            pruned_scores = {
                node_id: self.threads[node_id].score for node_id in pruned_nodes
            }
            pruned_out_scores = {
                node_id: self.threads[node_id].score for node_id in pruned_out
            }
        else:
            self.current_layer = []
            new_layer = []
            pruned_scores = {}
            pruned_out_scores = {}

        return {
            "count": len(new_nodes),
            "new_nodes": new_nodes,
            "new_layer": new_layer,
            "pruned_count": len(new_layer),
            "pruned_out_count": len(pruned_out) if "pruned_out" in locals() else 0,
            "scores": pruned_scores,
            "pruned_out_scores": pruned_out_scores,
        }

    def run(self, question: str) -> str:
        """Execute the reasoning process on a question.

        Args:
            question: The question to answer

        Returns:
            The final answer after reasoning
        """
        result = self.solve(question, beam_width=3, max_iterations=3)

        if isinstance(result, dict):
            best_node_id = None
            best_score = -1.0

            for node_id, node in self.threads.items():
                if node.score > best_score:
                    best_score = node.score
                    best_node_id = node_id

            if best_node_id:
                best_node = self.threads[best_node_id]
                return best_node.state.get("current_answer", "No answer found")

            return "No answer found"

        return result

    async def run_async(self, question: str) -> str:
        """Execute the reasoning process asynchronously on a question.

        Args:
            question: The question to answer

        Returns:
            The final answer after reasoning
        """
        result = await self._solve_async(question, beam_width=3, max_iterations=3)

        if isinstance(result, dict):
            best_node_id = None
            best_score = -1.0

            for node_id, node in self.threads.items():
                if node.score > best_score:
                    best_score = node.score
                    best_node_id = node_id

            if best_node_id:
                best_node = self.threads[best_node_id]
                return best_node.state.get("current_answer", "No answer found")

            return "No answer found"

        return result

    async def solve_async(
        self, problem: str, beam_width: int = 1, max_iterations: int = 10, **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """Asynchronously solve a problem using tree-of-thoughts approach.

        This method initiates the tree search process to find the best solution
        to the given problem. It explores multiple reasoning paths by creating
        and managing ThinkThreadSession instances in parallel.

        Args:
            problem: The problem to solve
            beam_width: Number of parallel thought threads to create
            max_iterations: Maximum number of iterations to perform
            **kwargs: Additional parameters for the solving process

        Returns:
            The best solution found or a dictionary containing the solution and metadata
        """
        self.threads.clear()
        self.current_layer = []

        async def create_root_node(i):
            session = ThinkThreadSession(
                llm_client=self.llm_client,
                template_manager=self.template_manager,
                config=self.config,
            )

            node_id = f"root_{i}" if i > 0 else "root"
            node = ThinkThreadNode(
                session=session,
                state={"problem": problem},
                node_id=node_id,
                depth=0,
            )

            return node_id, node

        root_nodes = await asyncio.gather(
            *[create_root_node(i) for i in range(beam_width)]
        )

        for node_id, node in root_nodes:
            self.threads[node_id] = node
            self.current_layer.append(node_id)

        if max_iterations > 0:
            expansion_results = await self.expand_threads_async(beam_width=beam_width)

            return {
                "status": "expanded",
                "message": f"Created {beam_width} parallel thought threads and expanded them asynchronously with beam search (width={beam_width})",
                "thread_count": len(self.threads),
                "root_threads": self.current_layer[:beam_width],
                "expanded_threads": expansion_results["new_nodes"],
                "expansion_count": expansion_results["count"],
                "pruned_count": expansion_results.get("pruned_count", 0),
                "pruned_out_count": expansion_results.get("pruned_out_count", 0),
            }

        return {
            "status": "initialized",
            "message": f"Created {beam_width} parallel thought threads for problem: {problem}",
            "thread_count": beam_width,
            "thread_ids": self.current_layer,
        }
