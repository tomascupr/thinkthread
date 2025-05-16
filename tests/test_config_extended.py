import os
import pytest
from pydantic import ValidationError
from thinkthread_sdk.config import ThinkThreadConfig, create_config, load_dotenv


def test_thinkthread_config_defaults():
    """Test that ThinkThreadConfig has the expected default values."""
    config = ThinkThreadConfig()
    assert config.provider == "openai"
    assert config.openai_model == "gpt-4"
    assert config.anthropic_model == "claude-2"
    assert config.hf_model == "gpt2"
    assert config.alternatives == 3
    assert config.rounds == 2
    assert config.max_rounds == 3
    assert config.use_pairwise_evaluation is True
    assert config.use_self_evaluation is False


def test_validate_provider():
    """Test provider validation."""
    config = ThinkThreadConfig(provider="openai")
    assert config.provider == "openai"
    
    config = ThinkThreadConfig(provider="anthropic")
    assert config.provider == "anthropic"
    
    config = ThinkThreadConfig(provider="hf")
    assert config.provider == "hf"
    
    config = ThinkThreadConfig(provider="dummy")
    assert config.provider == "dummy"
    
    with pytest.raises(ValidationError):
        ThinkThreadConfig(provider="invalid")


def test_validate_int_fields():
    """Test integer field validation."""
    config = ThinkThreadConfig(alternatives=5, rounds=3, max_rounds=7)
    assert config.alternatives == 5
    assert config.rounds == 3
    assert config.max_rounds == 7
    
    config = ThinkThreadConfig(alternatives="5", rounds="3", max_rounds="7")
    assert config.alternatives == 5
    assert config.rounds == 3
    assert config.max_rounds == 7
    
    with pytest.raises(ValidationError):
        ThinkThreadConfig(alternatives="invalid")
    
    with pytest.raises(ValidationError):
        ThinkThreadConfig(alternatives=-1)


def test_load_dotenv(tmp_path):
    """Test loading variables from a .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=test_key\n"
        "PROVIDER=anthropic\n"
        "ALTERNATIVES=5\n"
        "# Comment line\n"
        "EMPTY_LINE=\n"
    )
    
    env_vars = load_dotenv(env_file)
    
    assert env_vars["OPENAI_API_KEY"] == "test_key"
    assert env_vars["PROVIDER"] == "anthropic"
    assert env_vars["ALTERNATIVES"] == "5"
    assert env_vars["EMPTY_LINE"] == ""
    assert "# Comment line" not in env_vars


def test_create_config(tmp_path, monkeypatch):
    """Test creating config from environment variables and .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=env_file_key\n"
        "PROVIDER=anthropic\n"
    )
    
    monkeypatch.setenv("OPENAI_API_KEY", "env_var_key")
    monkeypatch.setenv("ALTERNATIVES", "7")
    
    config = create_config(env_file=str(env_file))
    
    assert config.openai_api_key == "env_var_key"
    assert config.provider == "anthropic"
    assert config.alternatives == 7
    assert config.rounds == 2
