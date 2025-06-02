"""Tests for the ThinkThreadUtils class."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from thinkthread.utils import ThinkThreadUtils
from thinkthread.llm import DummyLLMClient
from thinkthread.config import create_config
from thinkthread.evaluation import DefaultEvaluationStrategy


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

    def test_self_refine(self, utils):
        """Test self_refine method."""
        question = "What is the meaning of life?"
        initial_answer = "42"

        mock_session = MagicMock()
        mock_session._generate_alternatives.return_value = ["43", "44"]
        mock_session.evaluation_strategy = MagicMock(spec=DefaultEvaluationStrategy)
        mock_session.evaluation_strategy.evaluate.return_value = (
            0  # Index of the best answer
        )

        with patch(
            "thinkthread.utils.ThinkThreadSession", return_value=mock_session
        ):
            result = utils.self_refine(question, initial_answer, rounds=1)

            assert result == initial_answer
            mock_session._generate_alternatives.assert_called_once()
            mock_session.evaluation_strategy.evaluate.assert_called_once()

    def test_self_refine_with_metadata(self, utils):
        """Test self_refine method with metadata."""
        question = "What is the meaning of life?"
        initial_answer = "42"

        mock_session = MagicMock()
        mock_session._generate_alternatives.return_value = ["43", "44"]
        mock_session.evaluation_strategy = MagicMock(spec=DefaultEvaluationStrategy)
        mock_session.evaluation_strategy.evaluate.return_value = (
            0  # Index of the best answer
        )

        with patch(
            "thinkthread.utils.ThinkThreadSession", return_value=mock_session
        ):
            result = utils.self_refine(
                question, initial_answer, rounds=1, return_metadata=True
            )

            assert isinstance(result, dict)
            assert result["question"] == question
            assert result["initial_answer"] == initial_answer
            assert result["final_answer"] == initial_answer
            assert result["rounds"] == 1
            assert result["alternatives_per_round"] == 3
            mock_session._generate_alternatives.assert_called_once()
            mock_session.evaluation_strategy.evaluate.assert_called_once()

    @pytest.mark.asyncio
    async def test_self_refine_async(self, utils):
        """Test self_refine_async method."""
        question = "What is the meaning of life?"
        initial_answer = "42"

        mock_session = MagicMock()
        mock_session._generate_alternatives_async = AsyncMock(return_value=["43", "44"])
        mock_session._evaluate_all_async = AsyncMock(
            return_value=0
        )  # Index of the best answer

        with patch(
            "thinkthread.utils.ThinkThreadSession", return_value=mock_session
        ):
            result = await utils.self_refine_async(question, initial_answer, rounds=1)

            assert result == initial_answer
            mock_session._generate_alternatives_async.assert_called_once()
            mock_session._evaluate_all_async.assert_called_once()

    def test_n_best_brainstorm(self, utils):
        """Test n_best_brainstorm method."""
        question = "What is the meaning of life?"

        node1 = MagicMock()
        node1.score = 0.8
        node1.state = {"current_answer": "42"}

        node2 = MagicMock()
        node2.score = 0.9
        node2.state = {"current_answer": "The meaning of life is to be happy"}

        mock_tree_thinker = MagicMock()
        mock_tree_thinker.threads = {"node1": node1, "node2": node2}
        mock_tree_thinker.solve.return_value = None

        with patch("thinkthread.utils.TreeThinker", return_value=mock_tree_thinker):
            result = utils.n_best_brainstorm(question, n=2)

            assert result == "The meaning of life is to be happy"
            mock_tree_thinker.solve.assert_called_once()

    @pytest.mark.asyncio
    async def test_n_best_brainstorm_async(self, utils):
        """Test n_best_brainstorm_async method."""
        question = "What is the meaning of life?"

        node1 = MagicMock()
        node1.score = 0.8
        node1.state = {"current_answer": "42"}

        node2 = MagicMock()
        node2.score = 0.9
        node2.state = {"current_answer": "The meaning of life is to be happy"}

        mock_tree_thinker = MagicMock()
        mock_tree_thinker.threads = {"node1": node1, "node2": node2}
        mock_tree_thinker.solve_async = AsyncMock(return_value=None)

        with patch("thinkthread.utils.TreeThinker", return_value=mock_tree_thinker):
            result = await utils.n_best_brainstorm_async(question, n=2)

            assert result == "The meaning of life is to be happy"
            mock_tree_thinker.solve_async.assert_called_once()
