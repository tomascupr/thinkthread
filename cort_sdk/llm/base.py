from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """
    Abstract base class for Large Language Model clients.

    This class defines the essential interface for interacting with different
    LLM providers. Concrete subclasses should implement the generate method
    for specific providers.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the LLM client.

        Args:
            model_name: Optional name of the model to use
        """
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from the language model based on the provided prompt.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional model-specific parameters

        Returns:
            The generated text response from the model
        """
        pass
