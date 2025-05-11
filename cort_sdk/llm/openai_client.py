from typing import Optional, Dict, Any
import time
import openai
from openai import OpenAIError

from .base import LLMClient


class OpenAIClient(LLMClient):
    """
    OpenAI implementation of LLMClient.
    
    This class provides an interface to OpenAI's API for generating text
    using models like GPT-4 or GPT-3.5-turbo through the Chat Completion endpoint.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        **opts
    ):
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model name to use (default: "gpt-4")
            **opts: Additional options to pass to the API (e.g., temperature, max_tokens)
        """
        super().__init__(model_name=model)
        self.api_key = api_key
        self.model = model
        self.opts = opts
        
        # Initialize the OpenAI client
        self.client = openai.OpenAI(api_key=api_key)
        
        self._last_call_time = 0

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI's Chat Completion API.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters to override the default options

        Returns:
            The generated text response from the model

        Raises:
            Exception: If there's an error communicating with the OpenAI API
        """
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        if time_since_last_call < 0.5:  # 500ms minimum between calls
            time.sleep(0.5 - time_since_last_call)
        
        self._last_call_time = time.time()
        
        options = self.opts.copy()
        options.update(kwargs)
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **options
            )
            
            return response.choices[0].message.content
            
        except OpenAIError as e:
            error_message = f"OpenAI API error: {str(e)}"
            return error_message
        except Exception as e:
            error_message = f"Unexpected error when calling OpenAI API: {str(e)}"
            return error_message
