from typing import Dict, List, Optional, Union
from cort_sdk.llm import LLMClient


class DummyLLMClient(LLMClient):
    """
    A dummy LLM client for testing purposes.
    
    This client can be configured to return predefined responses for specific prompts
    or to follow a sequence of responses regardless of the prompt.
    """

    def __init__(
        self,
        model_name: Optional[str] = "dummy-model",
        responses: Optional[Dict[str, str]] = None,
        sequence: Optional[List[str]] = None,
    ):
        """
        Initialize the DummyLLMClient.
        
        Args:
            model_name: Name of the dummy model
            responses: Dictionary mapping prompts to responses
            sequence: List of responses to return in sequence, regardless of prompt
        """
        super().__init__(model_name=model_name)
        self.responses = responses or {}
        self.sequence = sequence or []
        self.sequence_index = 0
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response based on the configured behavior.
        
        If a sequence is provided, returns the next response in the sequence.
        Otherwise, looks for an exact match in the responses dictionary.
        If no match is found, returns a default response.
        
        Args:
            prompt: The input text
            **kwargs: Additional parameters (ignored in dummy implementation)
            
        Returns:
            The generated response
        """
        if self.sequence and self.sequence_index < len(self.sequence):
            response = self.sequence[self.sequence_index]
            self.sequence_index += 1
            return response
            
        if prompt in self.responses:
            return self.responses[prompt]
            
        return f"Dummy response to: {prompt}"
