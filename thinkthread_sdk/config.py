"""Configuration handling for ThinkThread SDK.

This module manages configuration options for the ThinkThread SDK, including API keys,
model names, and algorithm parameters.
"""

import os
from typing import Optional, Dict, Any, Union, ClassVar
from pathlib import Path
from pydantic import BaseModel, field_validator


class ThinkThreadConfig(BaseModel):
    """Configuration for the ThinkThread SDK.

    This class manages settings like API keys, model choices, and default ThinkThread parameters.
    Values are loaded from environment variables or a .env file if present.
    """

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    hf_api_token: Optional[str] = None

    provider: str = "openai"

    openai_model: str = "gpt-4"
    anthropic_model: str = "claude-2"
    hf_model: str = "gpt2"

    alternatives: int = 3
    rounds: int = 2
    max_rounds: int = 3

    use_pairwise_evaluation: bool = True
    use_self_evaluation: bool = False

    prompt_dir: Optional[str] = None

    _env_vars: ClassVar[Dict[str, str]] = {
        "openai_api_key": "OPENAI_API_KEY",
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "hf_api_token": "HF_API_TOKEN",
        "provider": "PROVIDER",
        "openai_model": "OPENAI_MODEL",
        "anthropic_model": "ANTHROPIC_MODEL",
        "hf_model": "HF_MODEL",
        "alternatives": "ALTERNATIVES",
        "rounds": "ROUNDS",
        "max_rounds": "MAX_ROUNDS",
        "use_pairwise_evaluation": "USE_PAIRWISE_EVALUATION",
        "use_self_evaluation": "USE_SELF_EVALUATION",
        "prompt_dir": "PROMPT_DIR",
    }

    @field_validator("alternatives", "rounds", "max_rounds", mode="before")
    @classmethod
    def validate_int_fields(cls, v: Union[str, int]) -> int:
        """Validate integer fields to ensure they are valid non-negative integers."""
        if isinstance(v, str):
            try:
                v = int(v)
            except ValueError:
                raise ValueError(f"Value must be a valid integer, got {v}")

        if v < 0:
            raise ValueError(f"Value must be non-negative, got {v}")

        return v

    @field_validator("provider", mode="before")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider name against supported providers."""
        valid_providers = ["openai", "anthropic", "hf", "dummy"]
        if v.lower() not in valid_providers:
            raise ValueError(
                f"Unknown provider: {v}. Valid providers are: {', '.join(valid_providers)}"
            )
        return v.lower()


def load_dotenv(env_file: Union[str, Path]) -> Dict[str, str]:
    """Load environment variables from a .env file.

    Args:
        env_file: Path to the .env file

    Returns:
        Dictionary of environment variables

    """
    env_vars: Dict[str, str] = {}

    if not os.path.exists(env_file):
        return env_vars

    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            key, value = line.split("=", 1)
            env_vars[key.strip()] = value.strip()

    return env_vars


def create_config(env_file: Optional[str] = ".env") -> ThinkThreadConfig:
    """Create a ThinkThreadConfig instance with values from environment variables and .env file.

    Args:
        env_file: Path to the .env file (default: ".env")

    Returns:
        ThinkThreadConfig instance

    """
    env_vars: Dict[str, str] = {}
    if env_file and os.path.exists(env_file):
        env_vars = load_dotenv(env_file)

    config_data: Dict[str, Any] = {}

    for field_name, env_var in ThinkThreadConfig._env_vars.items():
        if env_var in env_vars:
            config_data[field_name] = env_vars[env_var]

        if env_var in os.environ:
            config_data[field_name] = os.environ[env_var]

    return ThinkThreadConfig(**config_data)
