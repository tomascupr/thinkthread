"""Base reasoner implementation.

This module contains the BaseReasoner abstract class that defines the common interface
for different reasoning approaches like Chain-of-Recursive-Thoughts and Tree-of-Thoughts.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

from thinkthread_sdk.llm import LLMClient
from thinkthread_sdk.prompting import TemplateManager
from thinkthread_sdk.config import ThinkThreadConfig, create_config


class BaseReasoner(ABC):
    """Base class for all reasoning implementations.

    This abstract class defines the common interface and shared functionality
    for different reasoning approaches like Chain-of-Recursive-Thoughts and
    Tree-of-Thoughts.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        template_manager: Optional[TemplateManager] = None,
        config: Optional[ThinkThreadConfig] = None,
    ) -> None:
        """Initialize a BaseReasoner instance.

        Args:
            llm_client: LLM client to use for generating and evaluating thoughts
            template_manager: Optional template manager for prompt templates
            config: Optional configuration object
        """
        self.llm_client = llm_client
        self.config = config or create_config()
        self.template_manager = template_manager or TemplateManager(
            self.config.prompt_dir
        )

    @abstractmethod
    def run(self, question: str) -> str:
        """Execute the reasoning process on a question.

        Args:
            question: The question to answer

        Returns:
            The final answer after reasoning
        """
        pass

    @abstractmethod
    async def run_async(self, question: str) -> str:
        """Execute the reasoning process asynchronously on a question.

        Args:
            question: The question to answer

        Returns:
            The final answer after reasoning
        """
        pass

    def generate_initial_answer(self, question: str, temperature: float = 0.7) -> str:
        """Generate an initial answer to a question.

        Args:
            question: The question to answer
            temperature: Temperature for generation

        Returns:
            The initial answer
        """
        initial_prompt = self.template_manager.render_template(
            "initial_prompt.j2", {"question": question}
        )
        return self.llm_client.generate(initial_prompt, temperature=temperature)

    async def generate_initial_answer_async(
        self, question: str, temperature: float = 0.7
    ) -> str:
        """Generate an initial answer to a question asynchronously.

        Args:
            question: The question to answer
            temperature: Temperature for generation

        Returns:
            The initial answer
        """
        initial_prompt = self.template_manager.render_template(
            "initial_prompt.j2", {"question": question}
        )
        return await self.llm_client.acomplete(initial_prompt, temperature=temperature)
