import pytest
from thinkthread_sdk.llm import OpenAIClient, AnthropicClient, HuggingFaceClient, DummyLLMClient


class TestLLMFeatureParity:
    """Test that all LLM clients implement the required features."""
    
    def test_client_interface(self):
        """Test that all clients implement the required interface."""
        openai_client = OpenAIClient(api_key="fake_key")
        anthropic_client = AnthropicClient(api_key="fake_key")
        hf_client = HuggingFaceClient(api_token="fake_token")
        dummy_client = DummyLLMClient()
        
        for client in [openai_client, anthropic_client, hf_client, dummy_client]:
            assert hasattr(client, "generate")
            assert hasattr(client, "acomplete")
            assert hasattr(client, "astream")
            assert hasattr(client, "enable_cache")
            assert hasattr(client, "aclose")
    
    def test_caching_interface(self):
        """Test that caching works across all clients."""
        for ClientClass in [OpenAIClient, AnthropicClient, HuggingFaceClient, DummyLLMClient]:
            if ClientClass == HuggingFaceClient:
                client = ClientClass(api_token="fake_token")
            elif ClientClass == DummyLLMClient:
                client = ClientClass()
            else:
                client = ClientClass(api_key="fake_key")
            
            client.enable_cache(True)
            assert client._use_cache is True
            
            client.enable_cache(False)
            assert client._use_cache is False
    
    def test_semantic_caching_interface(self):
        """Test that semantic caching interface is available."""
        for ClientClass in [OpenAIClient, AnthropicClient, HuggingFaceClient, DummyLLMClient]:
            if ClientClass == HuggingFaceClient:
                client = ClientClass(api_token="fake_token")
            elif ClientClass == DummyLLMClient:
                client = ClientClass()
            else:
                client = ClientClass(api_key="fake_key")
            
            client.enable_semantic_cache(True, similarity_threshold=0.8)
            assert client._use_semantic_cache is True
            assert client._semantic_similarity_threshold == 0.8
