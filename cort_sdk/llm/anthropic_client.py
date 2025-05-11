from typing import Optional, Dict, Any
import time
import json
import requests

from .base import LLMClient


class AnthropicClient(LLMClient):
    """
    Anthropic implementation of LLMClient.
    
    This class provides an interface to Anthropic's API for generating text
    using Claude models through direct API calls.
    """
    
    HUMAN_PROMPT = "\n\nHuman: "
    AI_PROMPT = "\n\nAssistant: "

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-haiku-20240307",
        **opts
    ):
        """
        Initialize the Anthropic client.

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
        
        self._last_call_time = 0

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Anthropic's API through the official SDK.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters to override the default options

        Returns:
            The generated text response from the model

        Raises:
            Exception: If there's an error communicating with the Anthropic API
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
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            for key, value in options.items():
                payload[key] = value
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload
            )
            
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
