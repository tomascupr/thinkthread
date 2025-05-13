"""Tree-of-Thoughts solver implementation.

This module contains the TreeThinker class that implements tree-based search
for exploring multiple reasoning paths using ThinkThreadSession instances.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

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
            )

            self.threads[node_id] = node

        return {
            "status": "initialized",
            "message": f"Created {beam_width} parallel thought threads for problem: {problem}",
            "thread_count": beam_width,
            "thread_ids": list(self.threads.keys()),
        }
