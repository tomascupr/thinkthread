import os
import pytest
from tempfile import NamedTemporaryFile

from thinkthread_sdk.config import CoRTConfig, create_config


def test_default_values():
    """Test that default values are set correctly when no environment variables are present."""
    for env_var in CoRTConfig._env_vars.values():
        if env_var in os.environ:
            del os.environ[env_var]

    config = create_config(env_file=None)

    assert config.openai_api_key is None
    assert config.anthropic_api_key is None
    assert config.hf_api_token is None

    assert config.provider == "openai"

    assert config.openai_model == "gpt-4"
    assert config.anthropic_model == "claude-2"
    assert config.hf_model == "gpt2"

    assert config.alternatives == 3
    assert config.rounds == 2


def test_environment_variables(monkeypatch):
    """Test that environment variables are correctly loaded into the config."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test456")
    monkeypatch.setenv("HF_API_TOKEN", "hf-test789")
    monkeypatch.setenv("PROVIDER", "anthropic")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-3.5-turbo")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3-opus")
    monkeypatch.setenv("HF_MODEL", "mistral-7b")
    monkeypatch.setenv("ALTERNATIVES", "5")
    monkeypatch.setenv("ROUNDS", "3")

    config = create_config(env_file=None)

    assert config.openai_api_key == "sk-test123"
    assert config.anthropic_api_key == "sk-ant-test456"
    assert config.hf_api_token == "hf-test789"
    assert config.provider == "anthropic"
    assert config.openai_model == "gpt-3.5-turbo"
    assert config.anthropic_model == "claude-3-opus"
    assert config.hf_model == "mistral-7b"
    assert config.alternatives == 5
    assert config.rounds == 3


def test_env_file():
    """Test loading configuration from a .env file."""
    env_content = """
OPENAI_API_KEY=sk-env-test123
ANTHROPIC_API_KEY=sk-ant-env-test456
PROVIDER=hf
ALTERNATIVES=4
"""

    with NamedTemporaryFile(mode="w+") as env_file:
        env_file.write(env_content)
        env_file.flush()

        config = create_config(env_file=env_file.name)

        assert config.openai_api_key == "sk-env-test123"
        assert config.anthropic_api_key == "sk-ant-env-test456"
        assert config.provider == "hf"
        assert config.alternatives == 4

        assert config.hf_api_token is None
        assert config.openai_model == "gpt-4"
        assert config.rounds == 2


def test_type_validation():
    """Test that type validation works correctly."""
    with pytest.raises(ValueError):
        os.environ["ALTERNATIVES"] = "not_an_integer"
        create_config(env_file=None)
