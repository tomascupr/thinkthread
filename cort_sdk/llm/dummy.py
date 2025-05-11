from typing import Callable, List, Optional, Union

from .base import LLMClient


class DummyLLMClient(LLMClient):
    """
    A dummy implementation of LLMClient for testing purposes.
    
    This class provides deterministic responses for testing CoRT logic
    without calling external APIs. It can be configured to return responses
    from a predefined list, use a counter-based approach, or use a custom
    response generator function.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        responses: Optional[List[str]] = None,
        response_generator: Optional[Callable[[str, int], str]] = None,
    ):
        """
        Initialize the DummyLLMClient.

        Args:
            model_name: Optional name of the model to use
            responses: Optional list of predefined responses to cycle through
            response_generator: Optional function to generate responses based on
                                prompt and call count
        """
        super().__init__(model_name=model_name)
        self._call_count = 0
        self._responses = responses or []
        self._response_generator = response_generator

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a deterministic response based on configuration.

        If responses were provided during initialization, cycles through them.
        If a response_generator was provided, uses it to generate a response.
        Otherwise, returns a simple counter-based response that includes the prompt.

        Args:
            prompt: The input text
            **kwargs: Additional parameters (ignored in this implementation)

        Returns:
            A deterministic response string
        """
        self._call_count += 1

        if self._response_generator:
            return self._response_generator(prompt, self._call_count)

        if self._responses:
            index = (self._call_count - 1) % len(self._responses)
            return self._responses[index]

        return f"Dummy response #{self._call_count} to: '{prompt}'"

    @property
    def call_count(self) -> int:
        """
        Get the number of times the generate method has been called.

        Returns:
            The call count
        """
        return self._call_count

    def reset(self) -> None:
        """
        Reset the call counter to zero.
        """
        self._call_count = 0
