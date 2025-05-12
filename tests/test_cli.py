import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from thinkthread_sdk.cli import app


@pytest.fixture
def runner():
    """Provide a CLI runner for testing."""
    return CliRunner()


def test_version_command(runner):
    """Test the version command."""
    from thinkthread_sdk import __version__

    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert f"ThinkThread SDK version: {__version__}" in result.stdout


@patch("thinkthread_sdk.cli.create_config")
@patch("thinkthread_sdk.cli.DummyLLMClient")
@patch("thinkthread_sdk.cli.CoRTSession")
@patch("thinkthread_sdk.cli.asyncio.run")
def test_ask_command(
    mock_asyncio_run, mock_session, mock_client, mock_create_config, runner
):
    """Test the ask command."""
    mock_config = MagicMock()
    mock_create_config.return_value = mock_config
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_session_instance = MagicMock()
    mock_session.return_value = mock_session_instance

    result = runner.invoke(app, ["ask", "What is AI?", "--provider", "dummy"])

    assert result.exit_code == 0
    mock_client.assert_called_once_with(model_name="dummy-model")
    mock_session.assert_called_once_with(
        llm_client=mock_client_instance, alternatives=3, rounds=2, config=mock_config
    )
    mock_asyncio_run.assert_called_once()


@patch("thinkthread_sdk.cli.create_config")
@patch("thinkthread_sdk.cli.DummyLLMClient")
@patch("thinkthread_sdk.cli.CoRTSession")
@patch("thinkthread_sdk.cli.asyncio.run")
def test_run_command(
    mock_asyncio_run, mock_session, mock_client, mock_create_config, runner
):
    """Test the run command."""
    mock_config = MagicMock()
    mock_create_config.return_value = mock_config
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_session_instance = MagicMock()
    mock_session.return_value = mock_session_instance

    result = runner.invoke(
        app,
        [
            "run",
            "What is AI?",
            "--provider",
            "dummy",
            "--rounds",
            "3",
            "--alternatives",
            "2",
        ],
    )

    assert result.exit_code == 0
    mock_client.assert_called_once_with(model_name="dummy-model")
    mock_session.assert_called_once_with(
        llm_client=mock_client_instance, alternatives=2, rounds=3, config=mock_config
    )
    mock_asyncio_run.assert_called_once()


@patch("thinkthread_sdk.cli.create_config")
@patch("thinkthread_sdk.cli.OpenAIClient")
@patch("thinkthread_sdk.cli.CoRTSession")
@patch("thinkthread_sdk.cli.asyncio.run")
def test_provider_selection(
    mock_asyncio_run, mock_session, mock_openai, mock_create_config, runner
):
    """Test provider selection in CLI commands."""
    mock_config = MagicMock()
    mock_config.openai_api_key = "test_key"
    mock_config.openai_model = "gpt-4"
    mock_create_config.return_value = mock_config
    mock_client_instance = MagicMock()
    mock_openai.return_value = mock_client_instance
    mock_session_instance = MagicMock()
    mock_session.return_value = mock_session_instance

    result = runner.invoke(app, ["run", "What is AI?", "--provider", "openai"])

    assert result.exit_code == 0
    mock_openai.assert_called_once_with(api_key="test_key", model_name="gpt-4")


def test_help_output(runner):
    """Test the help command output."""
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "ThinkThread SDK - Command Line Interface" in result.stdout

    result_run = runner.invoke(app, ["run", "--help"])
    assert result_run.exit_code == 0
    assert "Run recursive reasoning on a question" in result_run.stdout

    result_ask = runner.invoke(app, ["ask", "--help"])
    assert result_ask.exit_code == 0
    assert "Ask a question and get an answer" in result_ask.stdout


@patch("thinkthread_sdk.cli.create_config")
@patch("thinkthread_sdk.cli.DummyLLMClient")
@patch("thinkthread_sdk.cli.CoRTSession")
@patch("thinkthread_sdk.cli.asyncio.run")
def test_cli_with_options(
    mock_asyncio_run, mock_session, mock_client, mock_create_config, runner
):
    """Test CLI with various options."""
    mock_config = MagicMock()
    mock_create_config.return_value = mock_config
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_session_instance = MagicMock()
    mock_session.return_value = mock_session_instance

    result = runner.invoke(
        app,
        [
            "run",
            "What is AI?",
            "--provider",
            "dummy",
            "--model",
            "custom-model",
        ],
    )

    assert result.exit_code == 0
    mock_client.assert_called_with(model_name="custom-model")

    mock_client.reset_mock()
    mock_session.reset_mock()
    mock_asyncio_run.reset_mock()

    result = runner.invoke(
        app,
        [
            "run",
            "What is AI?",
            "--provider",
            "dummy",
            "--self-evaluation",
        ],
    )

    assert result.exit_code == 0
    assert mock_config.use_self_evaluation
