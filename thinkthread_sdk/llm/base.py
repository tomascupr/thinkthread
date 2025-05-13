"""Abstract base class for LLM client implementations.

This module defines the interface that all LLM client implementations must follow,
providing a consistent API for different language model providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, AsyncIterator, Any, Dict, Tuple
import asyncio
import hashlib
import json
from functools import wraps


class LLMClient(ABC):
    """Abstract base class for Large Language Model clients.

    This class defines the essential interface for interacting with different
    LLM providers. Concrete subclasses should implement the generate method
    for specific providers.

    The class also provides async methods for non-blocking operation and
    streaming responses, with proper resource management through the aclose
    method for cleaning up resources when the client is no longer needed.
    """

    _cache: Dict[str, str] = {}

    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize the LLM client.

        Args:
            model_name: Optional name of the model to use

        """
        self.model_name = model_name
        self._use_cache = False
        self._semaphore = None

    def enable_cache(self, enabled: bool = True) -> None:
        """Enable or disable caching for this client.

        Args:
            enabled: Whether to enable caching
        """
        self._use_cache = enabled

    def set_concurrency_limit(self, limit: int) -> None:
        """Set the concurrency limit for API calls.

        Args:
            limit: Maximum number of concurrent API calls
        """
        self._semaphore = asyncio.Semaphore(limit) if limit > 0 else None
        
    def _get_cache_key(self, prompt: str, **kwargs: Any) -> str:
        """Generate a cache key for the given prompt and parameters.

        The cache key includes the model name, prompt, and all parameters
        that affect the output (e.g., temperature, max_tokens).

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional model-specific parameters

        Returns:
            A unique string representing the cache key
        """
        cache_dict = {
            "model": self.model_name,
            "prompt": prompt,
        }
        
        for key, value in kwargs.items():
            if key not in ["stream", "timeout"]:  # Skip parameters that don't affect the output
                cache_dict[key] = value
        
        cache_json = json.dumps(cache_dict, sort_keys=True)
        return hashlib.md5(cache_json.encode()).hexdigest()

    @abstractmethod
    def _generate_uncached(self, prompt: str, **kwargs: Any) -> str:
        """Generate text without using the cache.

        This method should be implemented by subclasses to provide
        the actual generation logic.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional model-specific parameters

        Returns:
            The generated text response from the model
        """
        pass

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text with optional caching.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional model-specific parameters

        Returns:
            The generated text response from the model
        """
        if self._use_cache:
            cache_key = self._get_cache_key(prompt, **kwargs)
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            response = self._generate_uncached(prompt, **kwargs)
            self._cache[cache_key] = response
            return response
        
        return self._generate_uncached(prompt, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously generate text from the language model.

        This method provides a non-blocking way to generate text from the LLM,
        allowing the calling application to perform other tasks while waiting
        for the model's response. Use this method in async applications or when
        you need to make multiple concurrent LLM calls.

        If caching is enabled, this method will check the cache before
        making an API call. When concurrency limits are set, it will use
        semaphores to limit the number of concurrent API calls.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional model-specific parameters which may include:
                - temperature: Controls randomness (higher = more random)
                - max_tokens: Maximum number of tokens to generate
                - stop: Sequences where the API will stop generating further tokens

        Returns:
            The generated text response from the model

        Raises:
            Various exceptions may be raised depending on the implementation,
            including network errors, authentication issues, or rate limiting.
            Implementations should document their specific error handling behavior.

        """
        if self._use_cache:
            cache_key = self._get_cache_key(prompt, **kwargs)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        if self._semaphore is not None:
            async with self._semaphore:
                response = await self._acomplete_uncached(prompt, **kwargs)
        else:
            response = await self._acomplete_uncached(prompt, **kwargs)
        
        if self._use_cache:
            self._cache[cache_key] = response
        
        return response
    
    async def _acomplete_uncached(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously generate text without using the cache.

        By default, this wraps the synchronous _generate_uncached method using asyncio.to_thread,
        but subclasses should override this with a native async implementation
        when possible for better performance and resource management.
        
        Args:
            prompt: The input text to send to the model
            **kwargs: Additional model-specific parameters
            
        Returns:
            The generated text response from the model
        """
        return await asyncio.to_thread(self._generate_uncached, prompt, **kwargs)

    @abstractmethod
    def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Asynchronously stream text generation from the language model.

        This method yields chunks of the generated text as they become available,
        rather than waiting for the complete response. This is particularly useful for:

        1. Providing real-time feedback to users as text is being generated
        2. Processing very long responses without waiting for completion
        3. Implementing responsive UIs that display partial results
        4. Handling early termination of generation if needed

        Implementations should ensure proper resource cleanup even if the caller
        stops consuming the stream before it's complete.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional model-specific parameters which may include:
                - temperature: Controls randomness (higher = more random)
                - max_tokens: Maximum number of tokens to generate
                - stop: Sequences where the API will stop generating further tokens

        Yields:
            Chunks of the generated text response from the model. The exact
            chunking behavior depends on the implementation (e.g., by tokens,
            by words, or by sentences).

        Raises:
            Various exceptions may be raised depending on the implementation,
            including network errors, authentication issues, or rate limiting.
            Implementations should document their specific error handling behavior.

        """
        pass

    async def aclose(self) -> None:
        """Asynchronously close the client and clean up resources.

        This method ensures that all resources used by the async client are
        properly released when the client is no longer needed. It should be
        called when you're done using the client to prevent resource leaks.

        Implementations should override this method if they use resources that
        need to be explicitly cleaned up, such as HTTP sessions, database
        connections, or file handles.

        Example usage:
            ```python
            client = SomeLLMClient(api_key="your-api-key")
            try:
                result = await client.acomplete("Hello, world!")
                print(result)
            finally:
                await client.aclose()
            ```

        Returns:
            None

        """
        pass
