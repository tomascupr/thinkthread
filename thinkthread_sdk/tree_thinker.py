"""Tree-of-Thoughts solver implementation.

This module contains the TreeThinker class that implements tree-based search
for exploring multiple reasoning paths using ThinkThreadSession instances.
"""

from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
import uuid
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

from thinkthread_sdk.llm import LLMClient
from thinkthread_sdk.prompting import TemplateManager
from thinkthread_sdk.config import ThinkThreadConfig, create_config
from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.evaluation import Evaluator, ModelEvaluator


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


class TreeThinker:
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
        self.llm_client = llm_client
        self.max_tree_depth = max_tree_depth
        self.branching_factor = branching_factor

        self.config = config or create_config()
        self.template_manager = template_manager or TemplateManager(
            self.config.prompt_dir
        )
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

        This method evaluates the promise of a thought branch using either
        a custom scoring function or a simple heuristic based on answer length
        and complexity. This is a placeholder implementation that can be enhanced
        with more sophisticated evaluation strategies.

        Args:
            problem: The original problem
            node: The node to score

        Returns:
            A score between 0.0 and 1.0 indicating the quality of the thought
        """
        if self.scoring_function:
            return self.scoring_function(problem, node.state)

        current_answer = node.state.get("current_answer", "")

        length_score = min(len(current_answer) / 1000, 1.0)

        sentences = current_answer.split(". ")
        sentence_count = len(sentences)
        unique_words = len(set(current_answer.lower().split()))
        word_count = len(current_answer.split())

        complexity_score = 0.0
        if word_count > 0:
            complexity_score = min(
                (unique_words / word_count) * (sentence_count / 5), 1.0
            )

        score = (0.7 * length_score) + (0.3 * complexity_score)

        return score

    def _generate_continuations(
        self, session: ThinkThreadSession, problem: str, current_answer: str
    ) -> List[str]:
        """Generate continuations for a thought thread.

        This method uses the session's ability to generate alternatives to create
        continuations for the current thought thread.

        Args:
            session: The ThinkThreadSession to use for generation
            problem: The original problem
            current_answer: The current answer or thought

        Returns:
            List of alternative continuations
        """
        num_continuations = min(self.branching_factor, 5)  # Limit to 5 for efficiency

        alternatives = []

        for i in range(num_continuations):
            prompt = self.template_manager.render_template(
                "alternative_prompt.j2",
                {"question": problem, "current_answer": current_answer},
            )

            alternative = self.llm_client.generate(prompt, temperature=0.9)
            alternatives.append(alternative)

        return alternatives

    async def _generate_continuations_async(
        self, session: ThinkThreadSession, problem: str, current_answer: str
    ) -> List[str]:
        """Asynchronously generate continuations for a thought thread.

        This method uses the session's ability to generate alternatives to create
        continuations for the current thought thread. It leverages async LLM calls
        to generate multiple continuations in parallel.

        Args:
            session: The ThinkThreadSession to use for generation
            problem: The original problem
            current_answer: The current answer or thought

        Returns:
            List of alternative continuations
        """
        num_continuations = min(self.branching_factor, 5)  # Limit to 5 for efficiency

        # Create prompts for all continuations
        prompts = []
        for i in range(num_continuations):
            prompt = self.template_manager.render_template(
                "alternative_prompt.j2",
                {"question": problem, "current_answer": current_answer},
            )
            prompts.append(prompt)

        if hasattr(self.llm_client, "acomplete_batch"):
            alternatives = await self.llm_client.acomplete_batch(
                prompts, temperature=0.9
            )
            return alternatives

        async def generate_alternative(prompt):
            return await self.llm_client.acomplete(prompt, temperature=0.9)

        alternatives = await asyncio.gather(
            *[generate_alternative(prompt) for prompt in prompts]
        )

        return alternatives

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
