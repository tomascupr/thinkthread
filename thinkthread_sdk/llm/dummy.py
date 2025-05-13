"""Dummy LLM client implementation for testing purposes.

This module provides a deterministic LLM client that can be used for testing
without making API calls to actual providers.
"""

from typing import Callable, List, Optional, AsyncIterator, Dict, Any
import asyncio

from .base import LLMClient


class DummyLLMClient(LLMClient):
    """A dummy implementation of LLMClient for testing purposes.

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
    ) -> None:
        """Initialize the DummyLLMClient.

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

    def _generate_uncached(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """Generate a deterministic response based on configuration without using cache.

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

    def generate(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """Generate a deterministic response based on configuration.

        Uses the base class implementation which handles caching.

        Args:
            prompt: The input text
            **kwargs: Additional parameters (ignored in this implementation)

        Returns:
            A deterministic response string

        """
        return super().generate(prompt, **kwargs)

    @property
    def call_count(self) -> int:
        """Get the number of times the generate method has been called.

        Returns:
            The call count

        """
        return self._call_count

    def reset(self) -> None:
        """Reset the call counter to zero."""
        self._call_count = 0

    async def acomplete(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """Asynchronously generate a deterministic response based on configuration.

        This method provides the same functionality as the synchronous `generate`
        method but in an asynchronous context. It's useful for testing async
        workflows and simulating real LLM behavior in async applications without
        making actual API calls.

        The implementation increments the call counter and returns a response
        based on the client's configuration (predefined responses, generator
        function, or default counter-based response).

        Args:
            prompt: The input text
            **kwargs: Additional parameters (ignored in this implementation)

        Returns:
            A deterministic response string, identical to what would be returned
            by the synchronous `generate` method with the same inputs

        Note:
            This implementation is thread-safe and can be called concurrently
            from multiple tasks.

        """
        return await super().acomplete(prompt, **kwargs)

    async def astream(
        self, prompt: str, **kwargs: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """Asynchronously stream a response in chunks to simulate streaming responses.

        This method demonstrates how streaming works by splitting the full response
        into words and yielding them one by one with a small delay. It's useful for:

        1. Testing streaming UI components without real API calls
        2. Simulating different streaming speeds by adjusting the delay
        3. Developing and testing streaming handlers in your application
        4. Demonstrating the benefits of streaming in educational contexts

        The implementation first gets the complete response using `acomplete`,
        then splits it into words and yields each word with a delay to simulate
        network latency.

        Args:
            prompt: The input text
            **kwargs: Additional parameters (ignored in this implementation)

        Yields:
            Chunks of the response string (words with spaces)

        Note:
            The artificial delay (0.1s per word) can be adjusted to simulate
            different network conditions or model generation speeds.

        """
        full_response = await self.acomplete(prompt, **kwargs)
        words = full_response.split()

        for word in words:
            await asyncio.sleep(0.1)  # Simulate network delay
            yield word + " "
