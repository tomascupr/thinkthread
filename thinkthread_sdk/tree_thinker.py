"""Tree-of-Thoughts solver implementation.

This module contains the TreeThinker class that implements tree-based search
for exploring multiple reasoning paths using ThinkThreadSession instances.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import uuid

from thinkthread_sdk.llm import LLMClient
from thinkthread_sdk.prompting import TemplateManager
from thinkthread_sdk.config import ThinkThreadConfig, create_config
from thinkthread_sdk.session import ThinkThreadSession


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
    ) -> None:
        """Initialize a TreeThinker instance.

        Args:
            llm_client: LLM client to use for generating and evaluating thoughts
            max_tree_depth: Maximum depth of the thinking tree
            branching_factor: Number of branches to explore at each node
            template_manager: Optional template manager for prompt templates
            config: Optional configuration object
        """
        self.llm_client = llm_client
        self.max_tree_depth = max_tree_depth
        self.branching_factor = branching_factor

        self.config = config or create_config()
        self.template_manager = template_manager or TemplateManager(
            self.config.prompt_dir
        )

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
            expansion_results = self.expand_threads()

            return {
                "status": "expanded",
                "message": f"Created {beam_width} parallel thought threads and expanded them",
                "thread_count": len(self.threads),
                "root_threads": self.current_layer[:beam_width],
                "expanded_threads": expansion_results["new_nodes"],
                "expansion_count": expansion_results["count"],
            }

        return {
            "status": "initialized",
            "message": f"Created {beam_width} parallel thought threads for problem: {problem}",
            "thread_count": beam_width,
            "thread_ids": self.current_layer,
        }

    def expand_threads(
        self, nodes_to_expand: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Expand the specified thought threads by generating the next thought for each.

        This method takes the current active thought threads and expands each one
        by generating alternative continuations. Each new continuation becomes a
        child node in the thinking tree.

        Args:
            nodes_to_expand: List of node IDs to expand. If None, expands all nodes in the current layer.

        Returns:
            Dictionary containing information about the expansion results
        """
        if nodes_to_expand is None:
            nodes_to_expand = self.current_layer.copy()

        new_layer = []
        new_nodes = []

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

                parent_node.children.append(child_id)

                self.threads[child_id] = child_node
                new_layer.append(child_id)
                new_nodes.append(child_id)

        self.current_layer = new_layer

        return {"count": len(new_nodes), "new_nodes": new_nodes, "new_layer": new_layer}

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
