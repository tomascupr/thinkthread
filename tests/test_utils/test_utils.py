"""Tests for the ThinkThreadUtils class."""

import pytest
from unittest.mock import MagicMock, patch

from thinkthread_sdk.utils import ThinkThreadUtils
from thinkthread_sdk.llm import DummyLLMClient
from thinkthread_sdk.config import create_config


@pytest.fixture
def dummy_client():
    """Create a dummy LLM client for testing."""
    return DummyLLMClient()


@pytest.fixture
def utils(dummy_client):
    """Create a ThinkThreadUtils instance for testing."""
    return ThinkThreadUtils(llm_client=dummy_client)


class TestThinkThreadUtils:
    """Test suite for ThinkThreadUtils class."""

    def test_init(self, dummy_client):
        """Test initialization of ThinkThreadUtils."""
        utils = ThinkThreadUtils(llm_client=dummy_client)
        assert utils.llm_client == dummy_client
        assert utils.config is not None
        assert utils.template_manager is not None

    @patch("thinkthread_sdk.session.ThinkThreadSession._generate_alternatives")
    @patch("thinkthread_sdk.session.ThinkThreadSession.evaluation_strategy.evaluate")
    def test_self_refine(self, mock_evaluate, mock_generate_alternatives, utils):
        """Test self_refine method."""
        question = "What is the meaning of life?"
        initial_answer = "42"
        mock_generate_alternatives.return_value = ["43", "44"]
        mock_evaluate.return_value = 0  # Index of the best answer (initial_answer)

        result = utils.self_refine(question, initial_answer, rounds=1)

        assert result == initial_answer
        mock_generate_alternatives.assert_called_once()
        mock_evaluate.assert_called_once()

    @patch("thinkthread_sdk.session.ThinkThreadSession._generate_alternatives")
    @patch("thinkthread_sdk.session.ThinkThreadSession.evaluation_strategy.evaluate")
    def test_self_refine_with_metadata(self, mock_evaluate, mock_generate_alternatives, utils):
        """Test self_refine method with metadata."""
        question = "What is the meaning of life?"
        initial_answer = "42"
        mock_generate_alternatives.return_value = ["43", "44"]
        mock_evaluate.return_value = 0  # Index of the best answer (initial_answer)

        result = utils.self_refine(question, initial_answer, rounds=1, return_metadata=True)

        assert isinstance(result, dict)
        assert result["question"] == question
        assert result["initial_answer"] == initial_answer
        assert result["final_answer"] == initial_answer
        assert result["rounds"] == 1
        assert result["alternatives_per_round"] == 3
        mock_generate_alternatives.assert_called_once()
        mock_evaluate.assert_called_once()

    @patch("thinkthread_sdk.session.ThinkThreadSession._generate_alternatives_async")
    @patch("thinkthread_sdk.session.ThinkThreadSession._evaluate_all_async")
    @pytest.mark.asyncio
    async def test_self_refine_async(self, mock_evaluate_async, mock_generate_alternatives_async, utils):
        """Test self_refine_async method."""
        question = "What is the meaning of life?"
        initial_answer = "42"
        mock_generate_alternatives_async.return_value = ["43", "44"]
        mock_evaluate_async.return_value = 0  # Index of the best answer (initial_answer)

        result = await utils.self_refine_async(question, initial_answer, rounds=1)

        assert result == initial_answer
        mock_generate_alternatives_async.assert_called_once()
        mock_evaluate_async.assert_called_once()

    @patch("thinkthread_sdk.tree_thinker.TreeThinker.solve")
    def test_n_best_brainstorm(self, mock_solve, utils):
        """Test n_best_brainstorm method."""
        question = "What is the meaning of life?"
        mock_solve.return_value = None  # Not used in the method
        
        utils.tree_thinker = MagicMock()
        node1 = MagicMock()
        node1.score = 0.8
        node1.state = {"current_answer": "42"}
        
        node2 = MagicMock()
        node2.score = 0.9
        node2.state = {"current_answer": "The meaning of life is to be happy"}
        
        utils.tree_thinker.threads = {
            "node1": node1,
            "node2": node2
        }

        with patch.object(utils, "tree_thinker", create=True) as mock_tree_thinker:
            mock_tree_thinker.threads = {"node1": node1, "node2": node2}
            mock_tree_thinker.solve.return_value = None
            result = utils.n_best_brainstorm(question, n=2)

        assert result == "The meaning of life is to be happy"

    @patch("thinkthread_sdk.tree_thinker.TreeThinker.solve_async")
    @pytest.mark.asyncio
    async def test_n_best_brainstorm_async(self, mock_solve_async, utils):
        """Test n_best_brainstorm_async method."""
        question = "What is the meaning of life?"
        mock_solve_async.return_value = None  # Not used in the method
        
        utils.tree_thinker = MagicMock()
        node1 = MagicMock()
        node1.score = 0.8
        node1.state = {"current_answer": "42"}
        
        node2 = MagicMock()
        node2.score = 0.9
        node2.state = {"current_answer": "The meaning of life is to be happy"}
        
        utils.tree_thinker.threads = {
            "node1": node1,
            "node2": node2
        }

        with patch.object(utils, "tree_thinker", create=True) as mock_tree_thinker:
            mock_tree_thinker.threads = {"node1": node1, "node2": node2}
            mock_tree_thinker.solve_async.return_value = None
            result = await utils.n_best_brainstorm_async(question, n=2)

        assert result == "The meaning of life is to be happy"
