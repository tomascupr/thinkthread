"""Hugging Face implementation of the LLMClient interface.

This module provides a client for interacting with models hosted on the
Hugging Face Hub through their inference API.
"""

from typing import Any, AsyncIterator
import requests
import asyncio
import aiohttp

from .base import LLMClient


class HuggingFaceClient(LLMClient):
    """Hugging Face implementation of LLMClient.

    This class provides an interface to Hugging Face's text generation inference API
    for generating text using models hosted on the Hugging Face Hub.
    """

    def __init__(self, api_token: str, model: str = "gpt2", **opts: Any) -> None:
        """Initialize the Hugging Face client.

        Args:
            api_token: Hugging Face API token
            model: Model identifier to use (default: "gpt2")
            **opts: Additional options to pass to the API (e.g., temperature, max_new_tokens)

        """
        super().__init__(model_name=model)
        self.api_token = api_token
        self.model = model
        self.opts = opts

        self.base_url = f"https://api-inference.huggingface.co/models/{model}"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

    def _generate_uncached(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using Hugging Face's text generation inference API without using cache.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters to override the default options

        Returns:
            The generated text response from the model

        Error Handling:
            Instead of raising exceptions, this method returns error messages as strings:
            - "Hugging Face API error: {status_code} - {response_text}" for HTTP status errors
            - "Hugging Face API error: {error_message}" for API-reported errors
            - "Request error when calling Hugging Face API: ..." for request exceptions, which may include:
              - Authentication errors (invalid API token)
              - Connection errors (network issues)
              - Timeout errors (request took too long)
            - "Unexpected error when calling Hugging Face API: ..." for other errors

        """
        options = self.opts.copy()
        options.update(kwargs)

        payload = {"inputs": prompt, **options}

        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)

            if response.status_code != 200:
                return (
                    f"Hugging Face API error: {response.status_code} - {response.text}"
                )

            response_data = response.json()

            if isinstance(response_data, list) and len(response_data) > 0:
                if "generated_text" in response_data[0]:
                    return response_data[0]["generated_text"]
                else:
                    return str(response_data[0])
            elif isinstance(response_data, dict):
                if "generated_text" in response_data:
                    return response_data["generated_text"]
                elif "error" in response_data:
                    return f"Hugging Face API error: {response_data['error']}"
                else:
                    return str(response_data)
            else:
                return str(response_data)

        except requests.RequestException as e:
            error_message = f"Request error when calling Hugging Face API: {str(e)}"
            return error_message
        except Exception as e:
            error_message = f"Unexpected error when calling Hugging Face API: {str(e)}"
            return error_message

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using Hugging Face's text generation inference API.

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
        """Asynchronously generate text using Hugging Face's text generation inference API.

        This method provides a non-blocking way to generate text from Hugging Face
        models, making it suitable for use in async applications like web servers,
        GUI applications, or any context where you don't want to block the main thread.
        It uses aiohttp for asynchronous HTTP requests to the Hugging Face Inference API.

        The implementation creates a new aiohttp ClientSession for each call, which
        is appropriate for serverless environments but may not be optimal for
        high-throughput applications. It properly handles various response formats
        from the Hugging Face API, which can vary depending on the model being used.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters to override the default options, including:
                - temperature: Controls randomness (0.0-1.0)
                - max_new_tokens: Maximum number of tokens to generate
                - top_k: Limits sampling to the k most likely tokens
                - top_p: Controls diversity via nucleus sampling
                - repetition_penalty: Penalizes repeated tokens

        Returns:
            The generated text response from the model

        Error Handling:
            Instead of raising exceptions, this method returns error messages as strings:
            - "Hugging Face API error: {status_code} - {response_text}" for HTTP status errors
            - "Hugging Face API error: {error_message}" for API-reported errors
            - "Request error when calling Hugging Face API: ..." for aiohttp ClientError exceptions, which may include:
              - Authentication errors (invalid API token)
              - Connection errors (network issues)
              - Timeout errors (request took too long)
              - DNS resolution errors
              - SSL certificate errors
            - "Unexpected error when calling Hugging Face API: ..." for other errors

        Note:
            This implementation uses proper async context managers for the aiohttp
            ClientSession and response objects to ensure resources are properly
            cleaned up even in case of exceptions.

        """
        options = self.opts.copy()
        options.update(kwargs)

        payload = {"inputs": prompt, **options}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url, headers=self.headers, json=payload
                ) as response:
                    if response.status != 200:
                        return f"Hugging Face API error: {response.status} - {await response.text()}"

                    response_data = await response.json()

                    if isinstance(response_data, list) and len(response_data) > 0:
                        if "generated_text" in response_data[0]:
                            return response_data[0]["generated_text"]
                        else:
                            return str(response_data[0])
                    elif isinstance(response_data, dict):
                        if "generated_text" in response_data:
                            return response_data["generated_text"]
                        elif "error" in response_data:
                            return f"Hugging Face API error: {response_data['error']}"
                        else:
                            return str(response_data)
                    else:
                        return str(response_data)

        except aiohttp.ClientError as e:
            error_message = f"Request error when calling Hugging Face API: {str(e)}"
            return error_message
        except Exception as e:
            error_message = f"Unexpected error when calling Hugging Face API: {str(e)}"
            return error_message

    async def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Asynchronously stream text generation from Hugging Face's API.

        This method simulates streaming by splitting the complete response into
        small word groups and yielding them with a small delay. While some Hugging
        Face models support native streaming, this implementation uses a simpler
        approach that works consistently across all models without requiring
        model-specific streaming configurations.

        The simulated streaming is useful for:
        1. Providing a responsive user experience with progressive output
        2. Testing streaming UI components without complex streaming logic
        3. Demonstrating the benefits of streaming in educational contexts
        4. Allowing early processing of partial responses

        The implementation first gets the complete response using `acomplete`,
        then splits it into words and yields them in small groups (3 words at a time)
        with a delay to simulate network latency and token-by-token generation.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters to override the default options, including:
                - temperature: Controls randomness (0.0-1.0)
                - max_new_tokens: Maximum number of tokens to generate
                - top_k: Limits sampling to the k most likely tokens
                - top_p: Controls diversity via nucleus sampling
                - repetition_penalty: Penalizes repeated tokens

        Yields:
            Small chunks of the generated text response (3 words at a time),
            with spaces preserved between words and a trailing space after
            each chunk.

        Error Handling:
            This method inherits error handling from the acomplete method:
            - If acomplete returns an error message, the entire error message is
              yielded as a single chunk
            - Error messages will begin with either "Hugging Face API error: ...",
              "Request error when calling Hugging Face API: ...", or
              "Unexpected error when calling Hugging Face API: ..."
            - See acomplete method documentation for details on specific error types

        Note:
            The artificial delay (0.1s per chunk) can be adjusted to simulate
            different network conditions or model generation speeds.

        """
        full_response = await self.acomplete(prompt, **kwargs)

        words = full_response.split()

        for i in range(0, len(words), 3):  # Yield 3 words at a time
            chunk = " ".join(words[i : i + 3])
            await asyncio.sleep(0.1)  # Simulate network delay
            yield chunk + " "
