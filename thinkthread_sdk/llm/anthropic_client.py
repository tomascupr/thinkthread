"""Anthropic Claude implementation of the LLMClient interface.

This module provides a client for interacting with Anthropic's Claude models
through their API.
"""

from typing import Dict, Any, AsyncIterator
import time
import requests
import asyncio
import aiohttp

from .base import LLMClient


class AnthropicClient(LLMClient):
    """Anthropic implementation of LLMClient.

    This class provides an interface to Anthropic's API for generating text
    using Claude models through direct API calls.
    """

    HUMAN_PROMPT = "\n\nHuman: "
    AI_PROMPT = "\n\nAssistant: "

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-haiku-20240307",
        **opts: Dict[str, Any],
    ) -> None:
        """Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key
            model: Model name to use (default: "claude-2")
            **opts: Additional options to pass to the API (e.g., temperature, max_tokens)

        """
        super().__init__(model_name=model)
        self.api_key = api_key
        self.model = model
        self.opts = opts

        self.api_url = "https://api.anthropic.com/v1/messages"

        self._last_call_time: float = 0.0

    def _generate_uncached(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using Anthropic's API without using cache.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters to override the default options

        Returns:
            The generated text response from the model

        Error Handling:
            Instead of raising exceptions, this method returns error messages as strings:
            - "Anthropic API request error: ..." for HTTP request errors, which may include:
              - Authentication errors (invalid API key)
              - Rate limit errors (too many requests)
              - Quota exceeded errors (billing issues)
              - Invalid request errors (bad parameters)
              - Connection errors (network issues)
            - "Unexpected error when calling Anthropic API: ..." for other errors

        """
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        if time_since_last_call < 0.5:  # 500ms minimum between calls
            time.sleep(0.5 - time_since_last_call)

        self._last_call_time = time.time()

        options = self.opts.copy()
        options.update(kwargs)

        max_tokens = options.pop("max_tokens", 1000)

        temperature = options.pop("temperature", 1.0)

        try:
            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

            for key, value in options.items():
                payload[key] = value

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }

            response = requests.post(self.api_url, headers=headers, json=payload)

            response.raise_for_status()

            response_data = response.json()

            if "content" in response_data and len(response_data["content"]) > 0:
                for content_block in response_data["content"]:
                    if content_block.get("type") == "text":
                        return content_block.get("text", "")

            return ""

        except requests.exceptions.RequestException as e:
            error_message = f"Anthropic API request error: {str(e)}"
            return error_message
        except Exception as e:
            error_message = f"Unexpected error when calling Anthropic API: {str(e)}"
            return error_message

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using Anthropic's API through the official SDK.

        Uses the base class implementation which handles caching.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters to override the default options

        Returns:
            The generated text response from the model

        Error Handling:
            Instead of raising exceptions, this method returns error messages as strings.
            See _generate_uncached for details on specific error types.
        """
        return super().generate(prompt, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously generate text using Anthropic's API.

        This method provides a non-blocking way to generate text from Anthropic's
        Claude models, making it suitable for use in async applications like web
        servers, GUI applications, or any context where you don't want to block
        the main thread. It uses aiohttp for asynchronous HTTP requests.

        The implementation includes rate limiting (minimum 500ms between calls)
        to help avoid Anthropic API rate limits. It creates a new aiohttp ClientSession
        for each call, which is appropriate for serverless environments but may
        not be optimal for high-throughput applications.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters to override the default options, including:
                - temperature: Controls randomness (0.0-1.0)
                - max_tokens: Maximum number of tokens to generate
                - top_k: Limits sampling to the k most likely tokens
                - top_p: Controls diversity via nucleus sampling

        Returns:
            The generated text response from the model

        Error Handling:
            Instead of raising exceptions, this method returns error messages as strings:
            - "Anthropic API request error: ..." for aiohttp ClientError exceptions, which may include:
              - Authentication errors (invalid API key)
              - Rate limit errors (HTTP 429 Too Many Requests)
              - Quota exceeded errors (billing issues)
              - Invalid request errors (HTTP 400 Bad Request)
              - Connection errors (network issues, timeouts)
              - Server errors (HTTP 5xx errors)
            - "Unexpected error when calling Anthropic API: ..." for other errors

        Note:
            This implementation uses proper async context managers for the aiohttp
            ClientSession and response objects to ensure resources are properly
            cleaned up even in case of exceptions.

        """
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        if time_since_last_call < 0.5:  # 500ms minimum between calls
            await asyncio.sleep(0.5 - time_since_last_call)

        self._last_call_time = time.time()

        options = self.opts.copy()
        options.update(kwargs)

        max_tokens = options.pop("max_tokens", 1000)
        temperature = options.pop("temperature", 1.0)

        try:
            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

            for key, value in options.items():
                payload[key] = value

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, headers=headers, json=payload
                ) as response:
                    response.raise_for_status()
                    response_data = await response.json()

                    if "content" in response_data and len(response_data["content"]) > 0:
                        for content_block in response_data["content"]:
                            if content_block.get("type") == "text":
                                return content_block.get("text", "")

            return ""

        except aiohttp.ClientError as e:
            error_message = f"Anthropic API request error: {str(e)}"
            return error_message
        except Exception as e:
            error_message = f"Unexpected error when calling Anthropic API: {str(e)}"
            return error_message

    async def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Asynchronously stream text generation from Anthropic's API.

        This method simulates streaming by splitting the complete response into
        sentence-like chunks and yielding them with a small delay. While Anthropic's
        API does support native streaming, this implementation uses a simpler approach
        that works well for most use cases without requiring additional complexity.

        The simulated streaming is useful for:
        1. Providing a responsive user experience with progressive output
        2. Testing streaming UI components without complex streaming logic
        3. Demonstrating the benefits of streaming in educational contexts
        4. Allowing early processing of partial responses

        The implementation first gets the complete response using `acomplete`,
        then splits it by periods and yields each sentence-like chunk with a
        delay to simulate network latency and token-by-token generation.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters to override the default options, including:
                - temperature: Controls randomness (0.0-1.0)
                - max_tokens: Maximum number of tokens to generate
                - top_k: Limits sampling to the k most likely tokens
                - top_p: Controls diversity via nucleus sampling

        Yields:
            Chunks of the generated text response from the model, approximately
            sentence by sentence. If the response doesn't contain periods, the
            entire response is yielded as a single chunk.

        Error Handling:
            This method inherits error handling from the acomplete method:
            - If acomplete returns an error message, the entire error message is
              yielded as a single chunk
            - Error messages will begin with either "Anthropic API request error: ..."
              or "Unexpected error when calling Anthropic API: ..."
            - See acomplete method documentation for details on specific error types

        Note:
            The artificial delay (0.2s per chunk) can be adjusted to simulate
            different network conditions or model generation speeds.

        """
        full_response = await self.acomplete(prompt, **kwargs)

        chunks = [s.strip() + " " for s in full_response.split(".") if s.strip()]
        if not chunks:
            chunks = [full_response]

        for chunk in chunks:
            await asyncio.sleep(0.2)  # Simulate network delay
            yield chunk
