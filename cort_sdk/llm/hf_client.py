from typing import Optional, Dict, Any, Union, List
import requests

from .base import LLMClient


class HuggingFaceClient(LLMClient):
    """
    Hugging Face implementation of LLMClient.
    
    This class provides an interface to Hugging Face's text generation inference API
    for generating text using models hosted on the Hugging Face Hub.
    """

    def __init__(
        self,
        api_token: str,
        model: str = "gpt2",
        **opts
    ):
        """
        Initialize the Hugging Face client.

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
            "Content-Type": "application/json"
        }

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Hugging Face's text generation inference API.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters to override the default options

        Returns:
            The generated text response from the model

        Raises:
            Exception: If there's an error communicating with the Hugging Face API
        """
        options = self.opts.copy()
        options.update(kwargs)
        
        payload = {
            "inputs": prompt,
            **options
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                return f"Hugging Face API error: {response.status_code} - {response.text}"
            
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
