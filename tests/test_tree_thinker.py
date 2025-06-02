"""Unit tests for the TreeThinker module.

This module contains tests for the TreeThinker class, verifying its core functionality
including expansion, scoring, and pruning of thought threads.
"""

import pytest
from unittest.mock import patch
from typing import List, Optional

from thinkthread.tree_thinker import TreeThinker
from thinkthread.llm import DummyLLMClient
from thinkthread.config import create_config


class MockThinkThreadSession:
    """Mock implementation of ThinkThreadSession for testing.

    This class simulates a ThinkThreadSession with deterministic outputs
    for testing TreeThinker functionality.
    """

    def __init__(self, predetermined_answers: Optional[List[str]] = None):
        """Initialize a MockThinkThreadSession.

        Args:
            predetermined_answers: List of answers to return in sequence
        """
        self.predetermined_answers = predetermined_answers or [
            "This is a mock answer.",
            "This is an alternative answer.",
            "This is yet another alternative.",
        ]
        self.answer_index = 0
        self.calls = []

    def run(self, question: str) -> str:
        """Run the session with a question and return a predetermined answer.

        Args:
            question: The question to answer

        Returns:
            A predetermined answer from the list
        """
        self.calls.append(("run", question))
        answer = self.predetermined_answers[
            self.answer_index % len(self.predetermined_answers)
        ]
        self.answer_index += 1
        return answer

    async def run_async(self, question: str) -> str:
        """Run the session asynchronously and return a predetermined answer.

        Args:
            question: The question to answer

        Returns:
            A predetermined answer from the list
        """
        self.calls.append(("run_async", question))
        answer = self.predetermined_answers[
            self.answer_index % len(self.predetermined_answers)
        ]
        self.answer_index += 1
        return answer

    def clone(self) -> "MockThinkThreadSession":
        """Create a clone of this session.

        Returns:
            A new MockThinkThreadSession with the same predetermined answers
        """
        clone = MockThinkThreadSession(self.predetermined_answers)
        clone.answer_index = self.answer_index
        return clone


@pytest.fixture
def mock_session_factory():
    """Fixture that returns a factory for creating MockThinkThreadSession instances."""

    def _factory(answers=None):
        return MockThinkThreadSession(answers)

    return _factory


@pytest.fixture
def tree_thinker():
    """Fixture that returns a TreeThinker instance with a DummyLLMClient."""
    config = create_config()
    llm_client = DummyLLMClient()
    return TreeThinker(
        llm_client=llm_client, max_tree_depth=3, branching_factor=2, config=config
    )


@pytest.fixture
def scored_tree_thinker():
    """Fixture that returns a TreeThinker with predetermined scoring."""
    config = create_config()
    llm_client = DummyLLMClient()

    tree_thinker = TreeThinker(
        llm_client=llm_client, max_tree_depth=3, branching_factor=2, config=config
    )

    def mock_score_node(problem, node):
        node_id = node.node_id
        if node_id == "root":
            return 0.5
        elif node_id == "root_child_0":
            return 0.8  # High score
        elif node_id == "root_child_1":
            return 0.6  # Medium score
        elif node_id == "root_child_2":
            return 0.4  # Low score
        elif node_id == "root_1":
            return 0.3
        else:
            return 0.2

    tree_thinker._score_node = mock_score_node
    return tree_thinker


def test_init():
    """Test initialization of TreeThinker."""
    config = create_config()
    llm_client = DummyLLMClient()

    tree_thinker = TreeThinker(
        llm_client=llm_client, max_tree_depth=3, branching_factor=2, config=config
    )

    assert tree_thinker.llm_client == llm_client
    assert tree_thinker.max_tree_depth == 3
    assert tree_thinker.branching_factor == 2
    assert tree_thinker.config == config
    assert isinstance(tree_thinker.threads, dict)
    assert len(tree_thinker.threads) == 0


def test_solve_creates_root_nodes(tree_thinker):
    """Test that solve creates the correct number of root nodes based on beam width."""
    problem = "What is the meaning of life?"
    beam_width = 2

    # Mock the expand_threads method to avoid actual expansion
    with patch.object(
        tree_thinker, "expand_threads", return_value={"count": 0, "new_nodes": []}
    ):
        result = tree_thinker.solve(problem, beam_width=beam_width, max_iterations=1)

    assert result["status"] == "expanded"
    assert result["thread_count"] == len(tree_thinker.threads)

    root_nodes = [
        node for node_id, node in tree_thinker.threads.items() if node.parent_id is None
    ]
    assert len(root_nodes) == beam_width


@pytest.mark.asyncio
async def test_solve_async_creates_root_nodes():
    """Test that solve_async creates the correct number of root nodes based on beam width."""
    config = create_config()
    llm_client = DummyLLMClient()

    tree_thinker = TreeThinker(
        llm_client=llm_client, max_tree_depth=3, branching_factor=2, config=config
    )

    problem = "What is the meaning of life?"
    beam_width = 2

    # Mock the expand_threads_async method to avoid actual expansion
    with patch.object(
        tree_thinker, "expand_threads_async", return_value={"count": 0, "new_nodes": []}
    ):
        result = await tree_thinker.solve_async(
            problem, beam_width=beam_width, max_iterations=1
        )

    assert result["status"] == "expanded"
    assert result["thread_count"] == len(tree_thinker.threads)

    root_nodes = [
        node for node_id, node in tree_thinker.threads.items() if node.parent_id is None
    ]
    assert len(root_nodes) == beam_width


def test_single_step_expansion(tree_thinker):
    """Test that a single step expansion creates the correct number of child nodes."""
    problem = "What is the meaning of life?"

    with patch.object(
        tree_thinker, "_generate_continuations", return_value=["Initial answer"]
    ):
        with patch.object(tree_thinker, "_score_node", return_value=0.5):
            tree_thinker.solve(problem, beam_width=1, max_iterations=0)

    assert len(tree_thinker.threads) == 1
    root_node_id = list(tree_thinker.threads.keys())[0]

    branching_factor = 3
    tree_thinker.branching_factor = branching_factor

    with patch.object(
        tree_thinker,
        "_generate_continuations",
        return_value=["Child 1", "Child 2", "Child 3"],
    ):
        with patch.object(tree_thinker, "_score_node", return_value=0.5):
            tree_thinker.expand_threads([root_node_id])

    assert len(tree_thinker.threads) == 1 + branching_factor

    children = [
        node
        for node_id, node in tree_thinker.threads.items()
        if node.parent_id == root_node_id
    ]
    assert len(children) == branching_factor


def test_beam_pruning(scored_tree_thinker):
    """Test that beam pruning keeps only the top N branches."""
    problem = "What is the meaning of life?"
    beam_width = 2

    # Create a root node without expansion
    scored_tree_thinker.solve(problem, beam_width=1, max_iterations=0)

    assert len(scored_tree_thinker.threads) == 1
    root_node_id = list(scored_tree_thinker.threads.keys())[0]

    # Mock the _generate_continuations method to return predetermined continuations
    continuations = ["Child 1", "Child 2", "Child 3"]
    with patch.object(
        scored_tree_thinker, "_generate_continuations", return_value=continuations
    ):
        expansion_result = scored_tree_thinker.expand_threads(
            [root_node_id], beam_width=beam_width
        )

    assert expansion_result["count"] == len(continuations)

    assert len(scored_tree_thinker.current_layer) == beam_width

    child_nodes = [
        node
        for node_id, node in scored_tree_thinker.threads.items()
        if node.parent_id == root_node_id
    ]
    assert len(child_nodes) > 0

    kept_nodes = [
        scored_tree_thinker.threads[node_id]
        for node_id in scored_tree_thinker.current_layer
    ]
    kept_scores = [node.score for node in kept_nodes]

    assert 0.8 in kept_scores  # Highest score should be kept
    assert 0.6 in kept_scores  # Medium score should be kept
    assert 0.4 not in kept_scores  # Lowest score should be pruned


def test_known_solution_path(tree_thinker, mock_session_factory):
    """Test that TreeThinker can follow a known solution path."""
    expected_answers = [
        "Initial answer about life.",
        "Life is about growth and learning.",
        "Life is about connections and relationships.",
        "Life is about finding meaning in everyday experiences.",
    ]

    # Mock the _generate_continuations method to return predetermined answers
    def mock_generate_continuations(*args, **kwargs):
        return expected_answers[1:]  # Return all but the first answer

    # Mock the _score_node method to assign predetermined scores
    def mock_score_node(problem, node):
        content = node.state.get("current_answer", "")
        if "connections and relationships" in content:
            return 0.9  # Highest score for this path
        elif "growth and learning" in content:
            return 0.7
        elif "everyday experiences" in content:
            return 0.8
        else:
            return 0.5

    with patch.object(
        tree_thinker, "_generate_continuations", side_effect=mock_generate_continuations
    ):
        with patch.object(tree_thinker, "_score_node", side_effect=mock_score_node):
            # Initialize with a predetermined initial answer
            tree_thinker.solve(
                problem="What is the meaning of life?", beam_width=1, max_iterations=0
            )
            root_node_id = list(tree_thinker.threads.keys())[0]
            tree_thinker.threads[root_node_id].state[
                "current_answer"
            ] = "Initial answer about life."

            tree_thinker.expand_threads([root_node_id], beam_width=2)

    best_node = max(tree_thinker.threads.values(), key=lambda x: x.score)
    assert "connections and relationships" in best_node.state.get("current_answer", "")


def test_error_handling(tree_thinker):
    """Test that TreeThinker handles errors gracefully."""
    problem = "What is the meaning of life?"

    # Create a mock _generate_continuations that raises an exception
    def mock_generate_continuations(*args, **kwargs):
        raise Exception("Simulated error")

    # Initialize with a root node first (this won't raise an exception)
    tree_thinker.solve(problem, beam_width=1, max_iterations=0)
    assert len(tree_thinker.threads) >= 1  # Root node created

    # Now patch _generate_continuations to raise an exception
    with patch.object(
        tree_thinker, "_generate_continuations", side_effect=mock_generate_continuations
    ):
        try:
            tree_thinker.expand_threads(tree_thinker.current_layer, beam_width=1)
            assert False, "Exception was not raised"
        except Exception as e:
            assert "Simulated error" in str(e)

    assert len(tree_thinker.threads) >= 1


@pytest.mark.asyncio
async def test_async_error_handling():
    """Test that TreeThinker handles errors gracefully in async mode."""
    config = create_config()
    llm_client = DummyLLMClient()

    tree_thinker = TreeThinker(
        llm_client=llm_client, max_tree_depth=3, branching_factor=2, config=config
    )

    problem = "What is the meaning of life?"

    # Create a mock _generate_continuations_async that raises an exception
    async def mock_generate_continuations_async(*args, **kwargs):
        raise Exception("Simulated async error")

    # Initialize with a root node first (this won't raise an exception)
    await tree_thinker.solve_async(problem, beam_width=1, max_iterations=0)
    assert len(tree_thinker.threads) >= 1  # Root node created

    # Now patch _generate_continuations_async to raise an exception
    with patch.object(
        tree_thinker,
        "_generate_continuations_async",
        side_effect=mock_generate_continuations_async,
    ):
        try:
            await tree_thinker.expand_threads_async(
                tree_thinker.current_layer, beam_width=1
            )
            assert False, "Exception was not raised"
        except Exception as e:
            assert "Simulated async error" in str(e)

    assert len(tree_thinker.threads) >= 1


def test_multiple_iterations(scored_tree_thinker):
    """Test that multiple iterations expand the tree correctly."""
    problem = "What is the meaning of life?"
    beam_width = 2

    # Mock the _generate_continuations method to return predetermined continuations
    def mock_generate_continuations(session, problem_str, current_answer):
        if "Child" not in current_answer:
            return ["Child 1", "Child 2", "Child 3"]  # First level expansion
        else:
            return ["Grandchild 1", "Grandchild 2"]  # Second level expansion

    with patch.object(
        scored_tree_thinker,
        "_generate_continuations",
        side_effect=mock_generate_continuations,
    ):
        # Initialize with a root node
        scored_tree_thinker.solve(problem, beam_width=1, max_iterations=0)
        root_node_id = list(scored_tree_thinker.threads.keys())[0]
        scored_tree_thinker.threads[root_node_id].state[
            "current_answer"
        ] = "Root answer"

        # First expansion
        scored_tree_thinker.expand_threads([root_node_id], beam_width=beam_width)

        children = [node_id for node_id in scored_tree_thinker.current_layer]

        # Second expansion
        scored_tree_thinker.expand_threads(children, beam_width=beam_width)

    assert (
        len(scored_tree_thinker.threads) >= 6
    )  # Root + 2 children + at least 3 grandchildren

    root_nodes = [
        node for node in scored_tree_thinker.threads.values() if node.parent_id is None
    ]
    assert len(root_nodes) == 1

    root_node = root_nodes[0]

    children = [
        node
        for node in scored_tree_thinker.threads.values()
        if node.parent_id == root_node.node_id
    ]
    assert len(children) == 3  # Matches actual implementation

    grandchildren = []
    for child in children:
        child_children = [
            node
            for node in scored_tree_thinker.threads.values()
            if node.parent_id == child.node_id
        ]
        grandchildren.extend(child_children)

    assert len(grandchildren) >= 2  # At least 2 grandchildren


def test_max_depth_limit(tree_thinker):
    """Test that TreeThinker respects the max_depth limit."""
    problem = "What is the meaning of life?"
    max_depth = 2
    tree_thinker.max_tree_depth = max_depth

    # Mock the _generate_continuations method to return predetermined continuations
    def mock_generate_continuations(session, problem_str, current_answer):
        if "Child" not in current_answer and "Grandchild" not in current_answer:
            return ["Child 1", "Child 2"]  # First level expansion
        elif "Child" in current_answer and "Grandchild" not in current_answer:
            return ["Grandchild 1", "Grandchild 2"]  # Second level expansion
        else:
            return [
                "Great-grandchild 1",
                "Great-grandchild 2",
            ]  # Third level (should be skipped)

    with patch.object(
        tree_thinker, "_generate_continuations", side_effect=mock_generate_continuations
    ):
        with patch.object(tree_thinker, "_score_node", return_value=0.5):
            # Initialize with a root node
            tree_thinker.solve(problem, beam_width=1, max_iterations=0)
            root_node_id = list(tree_thinker.threads.keys())[0]
            tree_thinker.threads[root_node_id].state["current_answer"] = "Root answer"

            # First expansion (depth 1)
            tree_thinker.expand_threads([root_node_id], beam_width=2)

            children = [node_id for node_id in tree_thinker.current_layer]

            # Second expansion (depth 2)
            tree_thinker.expand_threads(children, beam_width=2)

            grandchildren = []
            for child_id in children:
                child = tree_thinker.threads[child_id]
                grandchildren.extend(child.children)

            tree_thinker.expand_threads(grandchildren, beam_width=2)

    # Check that no nodes have depth > max_depth
    max_node_depth = max(node.depth for node in tree_thinker.threads.values())
    assert max_node_depth == max_depth

    root_nodes = [node for node in tree_thinker.threads.values() if node.depth == 0]
    depth1_nodes = [node for node in tree_thinker.threads.values() if node.depth == 1]
    depth2_nodes = [node for node in tree_thinker.threads.values() if node.depth == 2]
    depth3_nodes = [node for node in tree_thinker.threads.values() if node.depth == 3]

    assert len(root_nodes) == 1
    assert len(depth1_nodes) > 0
    assert len(depth2_nodes) > 0
    assert len(depth3_nodes) == 0  # No nodes at depth 3


def test_integration_with_mock_session(mock_session_factory):
    """Test integration with MockThinkThreadSession."""
    config = create_config()
    llm_client = DummyLLMClient()

    tree_thinker = TreeThinker(
        llm_client=llm_client, max_tree_depth=3, branching_factor=2, config=config
    )

    # Create a mock session with predetermined responses
    mock_responses = [
        "Initial answer about life.",
        "Life is about growth and learning.",
        "Life is about connections and relationships.",
    ]

    # Mock the _generate_continuations method to return predetermined continuations
    def mock_generate_continuations(*args, **kwargs):
        return mock_responses[1:]  # Return all but the first response

    with patch.object(
        tree_thinker, "_generate_continuations", side_effect=mock_generate_continuations
    ):
        # Initialize with a root node
        tree_thinker.solve(
            problem="What is the meaning of life?", beam_width=1, max_iterations=0
        )
        root_node_id = list(tree_thinker.threads.keys())[0]

        # Set the initial answer
        tree_thinker.threads[root_node_id].state["current_answer"] = mock_responses[0]

        tree_thinker.expand_threads([root_node_id], beam_width=2)

    assert len(tree_thinker.threads) > 0

    root_node = next(
        node for node in tree_thinker.threads.values() if node.parent_id is None
    )
    assert root_node.state.get("current_answer") == "Initial answer about life."

    child_nodes = [
        node
        for node in tree_thinker.threads.values()
        if node.parent_id == root_node.node_id
    ]
    assert len(child_nodes) > 0

    child_answers = [node.state.get("current_answer") for node in child_nodes]
    assert any("growth and learning" in answer for answer in child_answers)
    assert any("connections and relationships" in answer for answer in child_answers)
