"""OpenAI implementation of the LLMClient interface.

This module provides a client for interacting with OpenAI's models
through their Chat Completion API.
"""

from typing import Any, AsyncIterator, Optional, List
import time
import openai
import asyncio
from openai import OpenAIError

from .base import LLMClient


class OpenAIClient(LLMClient):
    """OpenAI implementation of LLMClient.

    This class provides an interface to OpenAI's API for generating text
    using models like GPT-4 or GPT-3.5-turbo through the Chat Completion endpoint.

    The implementation uses both synchronous and asynchronous OpenAI clients
    for optimal performance in different contexts. It properly manages the
    lifecycle of these clients to ensure resources are cleaned up appropriately.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4",
        model: Optional[str] = None,
        **opts: Any,
    ) -> None:
        """Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key
            model_name: Model name to use (default: "gpt-4")
            model: Alternative parameter name for model_name (for compatibility)
            **opts: Additional options to pass to the API (e.g., temperature, max_tokens)

        """
        model_to_use = model if model is not None else model_name
        super().__init__(model_name=model_to_use)
        self.api_key = api_key
        self.model = model_to_use
        self.opts = opts

        # Initialize the synchronous OpenAI client
        self.client = openai.OpenAI(api_key=api_key)

        # Initialize the asynchronous OpenAI client
        self.async_client = openai.AsyncOpenAI(api_key=api_key)

        self._last_call_time: float = 0.0

    def _generate_uncached(self, prompt: str, **kwargs: Any) -> str:
        """Generate text without using the cache.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters to override the default options

        Returns:
            The generated text response from the model
        """
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        if time_since_last_call < 0.5:  # 500ms minimum between calls
            time.sleep(0.5 - time_since_last_call)

        self._last_call_time = time.time()

        options = self.opts.copy()
        options.update(kwargs)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **options,
            )

            if response.choices and response.choices[0].message.content is not None:
                return response.choices[0].message.content
            return ""

        except OpenAIError as e:
            error_message = f"OpenAI API error: {str(e)}"
            return error_message
        except Exception as e:
            error_message = f"Unexpected error when calling OpenAI API: {str(e)}"
            return error_message

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using OpenAI's Chat Completion API.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters to override the default options

        Returns:
            The generated text response from the model

        Error Handling:
            Instead of raising exceptions, this method returns error messages as strings:
            - "OpenAI API error: ..." for OpenAI-specific errors, which may include:
              - Authentication errors (invalid API key)
              - Rate limit errors (too many requests)
              - Quota exceeded errors (billing issues)
              - Invalid request errors (bad parameters)
              - Server errors (OpenAI service issues)
            - "Unexpected error when calling OpenAI API: ..." for other errors

        """
        return super().generate(prompt, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously generate text using OpenAI's Chat Completion API.

        This method provides a non-blocking way to generate text from OpenAI models,
        making it suitable for use in async applications like web servers, GUI
        applications, or any context where you don't want to block the main thread.
        It uses OpenAI's native async client for optimal performance.

        The implementation includes rate limiting (minimum 500ms between calls)
        to help avoid OpenAI API rate limits. It uses a shared AsyncOpenAI client
        instance that is created once during initialization and reused for all
        calls, providing better resource management and performance.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters to override the default options, including:
                - temperature: Controls randomness (0.0-1.0)
                - max_tokens: Maximum number of tokens to generate
                - top_p: Controls diversity via nucleus sampling
                - frequency_penalty: Reduces repetition of token sequences
                - presence_penalty: Reduces repetition of topics

        Returns:
            The generated text response from the model

        Error Handling:
            Instead of raising exceptions, this method returns error messages as strings:
            - "OpenAI API error: ..." for OpenAI-specific errors, which may include:
              - Authentication errors (invalid API key)
              - Rate limit errors (too many requests)
              - Quota exceeded errors (billing issues)
              - Invalid request errors (bad parameters)
              - Server errors (OpenAI service issues)
            - "Unexpected error when calling OpenAI API: ..." for other errors

        Note:
            The shared AsyncOpenAI client instance is properly managed throughout
            the lifecycle of the OpenAIClient object, ensuring resources are
            cleaned up when the client is no longer needed.

        """
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        if time_since_last_call < 0.5:  # 500ms minimum between calls
            await asyncio.sleep(0.5 - time_since_last_call)

        self._last_call_time = time.time()

        options = self.opts.copy()
        options.update(kwargs)

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **options,
            )

            if response.choices and response.choices[0].message.content is not None:
                return response.choices[0].message.content
            return ""

        except OpenAIError as e:
            error_message = f"OpenAI API error: {str(e)}"
            return error_message
        except Exception as e:
            error_message = f"Unexpected error when calling OpenAI API: {str(e)}"
            return error_message

    async def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Asynchronously stream text generation from OpenAI's Chat Completion API.

        This method provides real-time streaming of tokens as they're generated
        by the model, rather than waiting for the complete response. This is
        particularly valuable for:

        1. Creating responsive UIs that show text generation in real-time
        2. Processing very long responses without waiting for completion
        3. Implementing early-stopping logic based on generated content
        4. Reducing perceived latency for end users

        The implementation uses OpenAI's native streaming support by setting
        the `stream=True` parameter. It uses the shared AsyncOpenAI client
        instance that is created once during initialization and reused for all
        calls, providing better resource management and performance.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters to override the default options, including:
                - temperature: Controls randomness (0.0-1.0)
                - max_tokens: Maximum number of tokens to generate
                - top_p: Controls diversity via nucleus sampling
                - frequency_penalty: Reduces repetition of token sequences
                - presence_penalty: Reduces repetition of topics

        Yields:
            Chunks of the generated text response from the model as they become
            available. Each chunk is a string containing a portion of the response.

        Error Handling:
            Instead of raising exceptions, this method yields error messages as strings:
            - "OpenAI API error: ..." for OpenAI-specific errors, which may include:
              - Authentication errors (invalid API key)
              - Rate limit errors (too many requests)
              - Quota exceeded errors (billing issues)
              - Invalid request errors (bad parameters)
              - Server errors (OpenAI service issues)
            - "Unexpected error when calling OpenAI API: ..." for other errors

        Note:
            The stream parameter is automatically set to True and will override
            any value provided in kwargs.

        """
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        if time_since_last_call < 0.5:  # 500ms minimum between calls
            await asyncio.sleep(0.5 - time_since_last_call)

        self._last_call_time = time.time()

        options = self.opts.copy()
        options.update(kwargs)

        options["stream"] = True

        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **options,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except OpenAIError as e:
            yield f"OpenAI API error: {str(e)}"
        except Exception as e:
            yield f"Unexpected error when calling OpenAI API: {str(e)}"

    async def acomplete_batch(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """Asynchronously generate text for multiple prompts in a single batch.

        This method optimizes the processing of multiple prompts by:
        1. Checking the cache for all prompts at once
        2. Acquiring the semaphore only once for the entire batch
        3. Using asyncio.gather to run multiple requests concurrently

        While OpenAI doesn't support true batched API requests in a single call,
        this implementation is still more efficient than processing prompts
        sequentially by reducing overhead and maximizing concurrency.

        Args:
            prompts: List of input texts to send to the model
            **kwargs: Additional parameters to override the default options

        Returns:
            List of generated text responses from the model
        """
        if not prompts:
            return []

        if self._use_cache:
            cache_keys = [self._get_cache_key(prompt, **kwargs) for prompt in prompts]
            cached_results = [self._cache.get(key, "") for key in cache_keys]
            if all(result is not None for result in cached_results):
                return cached_results  # Now guaranteed to be List[str] since we use "" as default

        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        if time_since_last_call < 0.5:  # 500ms minimum between calls
            await asyncio.sleep(0.5 - time_since_last_call)

        self._last_call_time = time.time()

        options = self.opts.copy()
        options.update(kwargs)

        if self._semaphore is not None:
            async with self._semaphore:

                async def process_prompt(prompt):
                    try:
                        response = await self.async_client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            **options,
                        )

                        if (
                            response.choices
                            and response.choices[0].message.content is not None
                        ):
                            return response.choices[0].message.content
                        return ""
                    except OpenAIError as e:
                        return f"OpenAI API error: {str(e)}"
                    except Exception as e:
                        return f"Unexpected error when calling OpenAI API: {str(e)}"

                results = await asyncio.gather(
                    *[process_prompt(prompt) for prompt in prompts]
                )
        else:

            async def process_prompt(prompt):
                try:
                    response = await self.async_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        **options,
                    )

                    if (
                        response.choices
                        and response.choices[0].message.content is not None
                    ):
                        return response.choices[0].message.content
                    return ""
                except OpenAIError as e:
                    return f"OpenAI API error: {str(e)}"
                except Exception as e:
                    return f"Unexpected error when calling OpenAI API: {str(e)}"

            results = await asyncio.gather(
                *[process_prompt(prompt) for prompt in prompts]
            )

        if self._use_cache:
            for i, prompt in enumerate(prompts):
                cache_key = self._get_cache_key(prompt, **kwargs)
                self._cache[cache_key] = results[i]

        return results

    async def aembed(self, text: str) -> List[float]:
        """Get embeddings for a text using OpenAI's Embeddings API.

        This method provides embeddings that can be used for semantic caching
        and similarity calculations. It uses OpenAI's embedding models to
        generate vector representations of text.

        Args:
            text: The text to embed

        Returns:
            List of embedding values
        """
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        if time_since_last_call < 0.5:  # 500ms minimum between calls
            await asyncio.sleep(0.5 - time_since_last_call)

        self._last_call_time = time.time()

        try:
            response = await self.async_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )

            embeddings = response.data[0].embedding
            self._embedding_cache[text] = embeddings
            return embeddings

        except OpenAIError as e:
            print(f"OpenAI API error when creating embeddings: {str(e)}")
            return []
        except Exception as e:
            print(f"Unexpected error when creating embeddings: {str(e)}")
            return []

    def embed(self, text: str) -> List[float]:
        """Get embeddings for a text using OpenAI's Embeddings API.

        This method provides a synchronous wrapper around aembed.

        Args:
            text: The text to embed

        Returns:
            List of embedding values
        """
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        if time_since_last_call < 0.5:  # 500ms minimum between calls
            time.sleep(0.5 - time_since_last_call)

        self._last_call_time = time.time()

        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )

            embeddings = response.data[0].embedding
            self._embedding_cache[text] = embeddings
            return embeddings

        except OpenAIError as e:
            print(f"OpenAI API error when creating embeddings: {str(e)}")
            return []
        except Exception as e:
            print(f"Unexpected error when creating embeddings: {str(e)}")
            return []

    async def aclose(self) -> None:
        """Asynchronously close the client and clean up resources.

        This method ensures that all resources used by the async client are
        properly released when the client is no longer needed. It should be
        called when you're done using the client to prevent resource leaks.

        Example usage:
            ```python
            client = OpenAIClient(api_key="your-api-key")
            try:
                result = await client.acomplete("Hello, world!")
                print(result)
            finally:
                await client.aclose()
            ```

        Or with an async context manager:
            ```python
            async with AsyncContextManager(OpenAIClient(api_key="your-api-key")) as client:
                result = await client.acomplete("Hello, world!")
                print(result)
            ```

        Returns:
            None

        """
        if hasattr(self.async_client, "close") and callable(self.async_client.close):
            await self.async_client.close()
        elif hasattr(self.async_client, "aclose") and callable(
            self.async_client.aclose
        ):
            await self.async_client.aclose()
