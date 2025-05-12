import requests
from unittest.mock import patch, MagicMock

from cort_sdk.llm.hf_client import HuggingFaceClient


def test_hf_client_init():
    """Test that the HuggingFaceClient initializes correctly with default and custom values."""
    client = HuggingFaceClient(api_token="test_token")
    assert client.model == "gpt2"
    assert client.api_token == "test_token"
    assert client.base_url == "https://api-inference.huggingface.co/models/gpt2"
    assert client.headers["Authorization"] == "Bearer test_token"

    client = HuggingFaceClient(api_token="test_token", model="facebook/opt-350m")
    assert client.model == "facebook/opt-350m"
    assert (
        client.base_url
        == "https://api-inference.huggingface.co/models/facebook/opt-350m"
    )

    client = HuggingFaceClient(
        api_token="test_token", temperature=0.7, max_new_tokens=100
    )
    assert client.opts.get("temperature") == 0.7
    assert client.opts.get("max_new_tokens") == 100


@patch("requests.post")
def test_generate_success_list_response(mock_post):
    """Test successful text generation with list response format."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"generated_text": "Test response"}]
    mock_post.return_value = mock_response

    client = HuggingFaceClient(api_token="test_token")
    response = client.generate("Test prompt")

    assert response == "Test response"

    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == "https://api-inference.huggingface.co/models/gpt2"
    assert call_args[1]["headers"]["Authorization"] == "Bearer test_token"
    assert call_args[1]["json"]["inputs"] == "Test prompt"


@patch("requests.post")
def test_generate_success_dict_response(mock_post):
    """Test successful text generation with dict response format."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"generated_text": "Test response"}
    mock_post.return_value = mock_response

    client = HuggingFaceClient(api_token="test_token")
    response = client.generate("Test prompt")

    assert response == "Test response"


@patch("requests.post")
def test_generate_with_options(mock_post):
    """Test generation with custom options."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"generated_text": "Test response"}]
    mock_post.return_value = mock_response

    client = HuggingFaceClient(api_token="test_token", temperature=0.7)

    client.generate("Test prompt", temperature=0.2, max_new_tokens=50)

    call_args = mock_post.call_args
    assert call_args[1]["json"]["temperature"] == 0.2
    assert call_args[1]["json"]["max_new_tokens"] == 50


@patch("requests.post")
def test_generate_api_error(mock_post):
    """Test handling of API errors."""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad Request"
    mock_post.return_value = mock_response

    client = HuggingFaceClient(api_token="test_token")
    response = client.generate("Test prompt")

    assert "Hugging Face API error" in response


@patch("requests.post")
def test_generate_api_error_in_response(mock_post):
    """Test handling of API errors in the response body."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"error": "Model is currently loading"}
    mock_post.return_value = mock_response

    client = HuggingFaceClient(api_token="test_token")
    response = client.generate("Test prompt")

    assert "Hugging Face API error" in response


@patch("requests.post")
def test_generate_request_exception(mock_post):
    """Test handling of request exceptions."""
    mock_post.side_effect = requests.RequestException("Connection error")

    client = HuggingFaceClient(api_token="test_token")
    response = client.generate("Test prompt")

    assert "Request error when calling Hugging Face API" in response


def test_manual_example():
    """Manual test example (not automatically run).

    This test demonstrates how to use the HuggingFaceClient with a real API token.
    To run this test, uncomment the code and set your API token.
    """
