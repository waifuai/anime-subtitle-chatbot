"""
Test suite for the Anime Subtitle Chatbot.

This module contains comprehensive unit tests and integration tests for all
components of the anime chatbot system, including provider implementations,
provider selection logic, prediction functionality, and CLI interfaces.
Uses pytest fixtures and mocking to ensure reliable and isolated testing.
"""

import pytest
import os
import pathlib
import tempfile
from unittest.mock import MagicMock, patch, mock_open

# Import functions from the updated predict script and providers
from src.scripts.predict import (
    load_examples,
    predict,
)

from src.anime_chatbot.providers.gemini import GeminiProvider
from src.anime_chatbot.providers.openrouter import OpenRouterProvider
from src.anime_chatbot.provider_selector import (
    ProviderConfig,
    resolve_provider,
    resolve_model,
    generate_response,
)

# --- Test Helper Functions ---

def test_load_examples_success(tmp_path):
    """Tests loading examples successfully."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_file = data_dir / "input.txt"
    output_file = data_dir / "output.txt"

    input_content = "Hello?\nHow are you?\n"
    output_content = "Hi!\nI'm good.\n"
    input_file.write_text(input_content, encoding='utf-8')
    output_file.write_text(output_content, encoding='utf-8')

    examples = load_examples(str(input_file), str(output_file))

    assert len(examples) == 2
    assert examples[0] == ("Hello?", "Hi!")
    assert examples[1] == ("How are you?", "I'm good.")

def test_load_examples_file_not_found(tmp_path, capsys):
    """Tests loading examples when files are missing."""
    input_file = tmp_path / "nonexistent_input.txt"
    output_file = tmp_path / "nonexistent_output.txt"

    examples = load_examples(str(input_file), str(output_file))
    captured = capsys.readouterr()

    assert examples == []
    assert "Error: Example file not found" in captured.out

def test_load_examples_empty_lines(tmp_path):
    """Tests that empty lines are skipped when loading examples."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_file = data_dir / "input.txt"
    output_file = data_dir / "output.txt"

    input_content = "Line 1\n\nLine 3"
    output_content = "Resp 1\nResp 2\nResp 3" # Mismatched lines intentionally
    input_file.write_text(input_content, encoding='utf-8')
    output_file.write_text(output_content, encoding='utf-8')

    examples = load_examples(str(input_file), str(output_file))

    # Only pairs with non-empty input and output are kept
    assert len(examples) == 2
    assert examples[0] == ("Line 1", "Resp 1")
    assert examples[1] == ("Line 3", "Resp 3")


def test_load_examples_with_temp_files(tmp_path):
    """Tests loading examples using temporary files."""
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"

    input_content = "Hello?\nHow are you?\nWhat's your name?"
    output_content = "Hi there!\nI'm doing well.\nMy name is Bot."
    input_file.write_text(input_content, encoding='utf-8')
    output_file.write_text(output_content, encoding='utf-8')

    examples = load_examples(str(input_file), str(output_file))

    assert len(examples) == 3
    assert examples[0] == ("Hello?", "Hi there!")
    assert examples[1] == ("How are you?", "I'm doing well.")
    assert examples[2] == ("What's your name?", "My name is Bot.")

# --- Test Provider Selector Functionality ---

def test_resolve_provider_with_cli_provider():
    """Tests provider resolution with CLI provider specified."""
    provider = resolve_provider("gemini")
    assert provider == "gemini"

def test_resolve_provider_with_env_var(monkeypatch):
    """Tests provider resolution with environment variable."""
    monkeypatch.setenv("CHATBOT_PROVIDER", "openrouter")
    provider = resolve_provider(None)
    assert provider == "openrouter"

def test_resolve_provider_default():
    """Tests provider resolution with default fallback."""
    provider = resolve_provider(None)
    assert provider == "openrouter"

def test_resolve_provider_cli_priority(monkeypatch):
    """Tests that CLI provider takes priority over environment variable."""
    monkeypatch.setenv("CHATBOT_PROVIDER", "gemini")
    provider = resolve_provider("openrouter")
    assert provider == "openrouter"

def test_resolve_model_with_override():
    """Tests model resolution with override specified."""
    model = resolve_model("gemini", "custom-model")
    assert model == "custom-model"

def test_resolve_model_from_file(tmp_path, monkeypatch):
    """Tests model resolution from model file."""
    # Create a temporary model file
    model_file = tmp_path / ".model-gemini"
    model_file.write_text("custom-gemini-model")

    # Mock the home directory
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)

    model = resolve_model("gemini", None)
    assert model == "custom-gemini-model"

def test_resolve_model_default_gemini():
    """Tests default Gemini model resolution."""
    model = resolve_model("gemini", None)
    assert model == "gemini-2.5-pro"

def test_resolve_model_default_openrouter():
    """Tests default OpenRouter model resolution."""
    model = resolve_model("openrouter", None)
    assert model == OpenRouterProvider.DEFAULT_MODEL

def test_resolve_model_unknown_provider():
    """Tests model resolution for unknown provider."""
    model = resolve_model("unknown", None)
    assert model is None

# --- Test Provider API Interaction (using mocks) ---

@pytest.fixture
def mock_gemini_provider():
    """Fixture for a mocked GeminiProvider."""
    with patch('src.anime_chatbot.providers.gemini.GeminiProvider') as mock_provider_class, \
         patch('google.generativeai.configure'), \
         patch('google.generativeai.GenerativeModel') as mock_model_class:

        mock_provider_instance = MagicMock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.generate.return_value = "Mocked Gemini response"

        # Mock the model instance
        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance

        mock_response = MagicMock()
        mock_response.parts = [MagicMock()]
        mock_response.text = "Mocked Gemini response"
        mock_model_instance.generate_content.return_value = mock_response

        yield mock_provider_instance

@pytest.fixture
def mock_openrouter_provider():
    """Fixture for a mocked OpenRouterProvider."""
    with patch('src.anime_chatbot.providers.openrouter.OpenRouterProvider') as mock_provider_class:
        mock_provider_instance = MagicMock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.generate.return_value = "Mocked OpenRouter response"
        yield mock_provider_instance

def test_generate_response_gemini_provider():
    """Tests response generation using Gemini provider."""
    prompt = "Test prompt"
    examples = [("Ex1", "Resp1")]
    cfg = ProviderConfig(provider="gemini", model="models/gemini-1.5-pro")

    with patch('src.anime_chatbot.provider_selector.GeminiProvider') as mock_provider_class:
        mock_provider_instance = MagicMock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.generate.return_value = "Mocked Gemini response"

        response = generate_response(prompt, examples, cfg)

        assert response == "Mocked Gemini response"
        mock_provider_instance.generate.assert_called_once_with(prompt, examples, "models/gemini-1.5-pro")

def test_generate_response_openrouter_provider():
    """Tests response generation using OpenRouter provider."""
    prompt = "Test prompt"
    examples = [("Ex1", "Resp1")]
    cfg = ProviderConfig(provider="openrouter", model="deepseek/deepseek-chat")

    with patch('src.anime_chatbot.provider_selector.OpenRouterProvider') as mock_provider_class:
        mock_provider_instance = MagicMock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.generate.return_value = "Mocked OpenRouter response"

        response = generate_response(prompt, examples, cfg)

        assert response == "Mocked OpenRouter response"
        mock_provider_instance.generate.assert_called_once_with(prompt, examples, "deepseek/deepseek-chat")

def test_generate_response_unknown_provider(capsys):
    """Tests response generation with unknown provider."""
    prompt = "Test prompt"
    examples = []
    cfg = ProviderConfig(provider="unknown")

    response = generate_response(prompt, examples, cfg)
    captured = capsys.readouterr()

    assert response is None
    assert "Unknown provider: unknown" in captured.out

def test_generate_response_provider_failure():
    """Tests handling of provider generation failure."""
    prompt = "Test prompt"
    examples = []
    cfg = ProviderConfig(provider="gemini")

    with patch('src.anime_chatbot.provider_selector.GeminiProvider') as mock_provider_class:
        mock_provider_instance = MagicMock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.generate.return_value = None

        response = generate_response(prompt, examples, cfg)

        assert response is None

# --- Test Gemini Provider with API Mocks ---

@pytest.fixture
def mock_google_generativeai():
    """Fixture for mocking the entire google.generativeai module."""
    with patch('google.generativeai.configure') as mock_configure, \
         patch('google.generativeai.GenerativeModel') as mock_GenerativeModel:

        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        # Simulate a successful response with text
        mock_response.parts = [MagicMock()]  # Simulate having parts
        mock_response.text = "Generated anime response."
        mock_response.prompt_feedback = None  # No blocking
        mock_model_instance.generate_content.return_value = mock_response

        mock_GenerativeModel.return_value = mock_model_instance

        yield {
            "configure": mock_configure,
            "GenerativeModel": mock_GenerativeModel,
            "model_instance": mock_model_instance,
            "mock_response": mock_response
        }

def test_gemini_provider_success(mock_google_generativeai):
    """Tests successful response generation using mocked Gemini API."""
    # Mock the API key resolution
    with patch.object(GeminiProvider, '_resolve_api_key', return_value='fake_key'):
        provider = GeminiProvider()
        prompt = "User input prompt"
        examples = [("Ex Input 1", "Ex Output 1"), ("Ex Input 2", "Ex Output 2")]

        response_text = provider.generate(prompt, examples, GeminiProvider.DEFAULT_MODEL)

        # Assertions
        mock_google_generativeai["configure"].assert_called_once_with(api_key='fake_key')
        mock_google_generativeai["GenerativeModel"].assert_called_once_with(GeminiProvider.DEFAULT_MODEL)
        mock_google_generativeai["model_instance"].generate_content.assert_called_once()

        # Check prompt construction (basic check)
        call_args, _ = mock_google_generativeai["model_instance"].generate_content.call_args
        generated_prompt = call_args[0]
        assert "You are an anime chatbot." in generated_prompt
        assert "Input: Ex Input 1" in generated_prompt
        assert "Output: Ex Output 1" in generated_prompt
        assert "Input: Ex Input 2" in generated_prompt
        assert "Output: Ex Output 2" in generated_prompt
        assert f"Input: {prompt}" in generated_prompt
        assert generated_prompt.endswith("Output:")

        assert response_text == "Generated anime response."

def test_gemini_provider_blocked(mock_google_generativeai, capsys):
    """Tests handling of a blocked prompt response."""
    with patch.object(GeminiProvider, '_resolve_api_key', return_value='fake_key'):
        provider = GeminiProvider()
        prompt = "Problematic prompt"
        examples = []

        # Configure mock response for blocking
        mock_google_generativeai["mock_response"].parts = []  # No parts when blocked
        mock_google_generativeai["mock_response"].text = None  # No text when blocked
        mock_google_generativeai["mock_response"].prompt_feedback = MagicMock()
        mock_google_generativeai["mock_response"].prompt_feedback.block_reason = "SAFETY"
        mock_google_generativeai["model_instance"].generate_content.return_value = mock_google_generativeai["mock_response"]

        response_text = provider.generate(prompt, examples, GeminiProvider.DEFAULT_MODEL)
        captured = capsys.readouterr()

        assert "Warning: Prompt blocked. Reason: SAFETY" in captured.out
        assert response_text == "[Blocked: SAFETY]"

def test_gemini_provider_api_error(mock_google_generativeai, caplog):
    """Tests handling of an exception during API call."""
    with patch.object(GeminiProvider, '_resolve_api_key', return_value='fake_key'):
        provider = GeminiProvider()
        prompt = "User input"
        examples = []

        # Configure mock to raise an exception
        mock_google_generativeai["model_instance"].generate_content.side_effect = Exception("API connection failed")

        response_text = provider.generate(prompt, examples, GeminiProvider.DEFAULT_MODEL)

        assert response_text == "[API Error: API connection failed]"
        assert "Gemini API error: API connection failed" in caplog.text

def test_gemini_provider_no_api_key(capsys):
    """Tests handling when no API key is available."""
    with patch.object(GeminiProvider, '_resolve_api_key', return_value=None):
        provider = GeminiProvider()
        prompt = "User input"
        examples = []

        response_text = provider.generate(prompt, examples, GeminiProvider.DEFAULT_MODEL)
        captured = capsys.readouterr()

        assert response_text is None
        assert "Gemini API key is missing" in captured.out

# --- Test OpenRouter Provider with API Mocks ---

@pytest.fixture
def mock_requests_post():
    """Fixture for mocking requests.post used by OpenRouterProvider."""
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated OpenRouter response"}}]
        }
        mock_post.return_value = mock_response
        yield mock_post

def test_openrouter_provider_success(mock_requests_post):
    """Tests successful response generation using mocked OpenRouter API."""
    with patch.object(OpenRouterProvider, '_resolve_api_key', return_value='fake_key'):
        provider = OpenRouterProvider()
        prompt = "User input prompt"
        examples = [("Ex Input 1", "Ex Output 1")]

        response_text = provider.generate(prompt, examples, OpenRouterProvider.DEFAULT_MODEL)

        assert response_text == "Generated OpenRouter response"
        mock_requests_post.assert_called_once()

def test_openrouter_provider_no_api_key(capsys):
    """Tests handling when no OpenRouter API key is available."""
    with patch.object(OpenRouterProvider, '_resolve_api_key', return_value=None):
        provider = OpenRouterProvider()
        prompt = "User input"
        examples = []

        response_text = provider.generate(prompt, examples, OpenRouterProvider.DEFAULT_MODEL)
        captured = capsys.readouterr()

        assert response_text is None
        assert "OpenRouter API key is missing" in captured.out

def test_openrouter_provider_api_error(mock_requests_post):
    """Tests handling of API error response."""
    with patch.object(OpenRouterProvider, '_resolve_api_key', return_value='fake_key'):
        provider = OpenRouterProvider()
        prompt = "User input"
        examples = []

        # Configure mock for error response
        mock_requests_post.return_value.status_code = 401

        response_text = provider.generate(prompt, examples, OpenRouterProvider.DEFAULT_MODEL)

        assert response_text is None
        mock_requests_post.assert_called_once()

# --- Test CLI and Main Functionality ---

def test_predict_with_missing_data_dir(tmp_path, capsys):
    """Tests predict function with missing data directory."""
    missing_dir = tmp_path / "missing"

    predict(data_dir=str(missing_dir))

    captured = capsys.readouterr()
    assert "Data directory not found" in captured.out

def test_predict_with_valid_data_dir(tmp_path, mock_gemini_provider):
    """Tests predict function with valid data directory."""
    # Create data directory and files
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "input.txt").write_text("Hello?\n")
    (data_dir / "output.txt").write_text("Hi!\n")

    # This would normally run interactive mode, but we'll test the setup
    with patch('src.scripts.predict._run_interactive_mode') as mock_interactive:
        predict(data_dir=str(data_dir), provider="gemini")
        mock_interactive.assert_called_once()

def test_predict_batch_mode(tmp_path, mock_gemini_provider):
    """Tests batch mode functionality."""
    # Create input file
    input_file = tmp_path / "input.txt"
    input_file.write_text("Prompt 1\nPrompt 2\n")

    # Create data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "input.txt").write_text("Example?\n")
    (data_dir / "output.txt").write_text("Response!\n")

    # Mock the batch mode function
    with patch('src.scripts.predict._run_batch_mode') as mock_batch:
        predict(input_file=str(input_file), data_dir=str(data_dir), provider="gemini")
        mock_batch.assert_called_once()

# --- Test ProviderConfig ---

def test_provider_config_defaults():
    """Tests ProviderConfig with default values."""
    cfg = ProviderConfig()
    assert cfg.provider == "openrouter"
    assert cfg.model is None
    assert cfg.timeout == 60
    assert cfg.max_retries == 2
    assert cfg.retry_backoff_seconds == 1.0

def test_provider_config_custom_values():
    """Tests ProviderConfig with custom values."""
    cfg = ProviderConfig(
        provider="gemini",
        model="custom-model",
        timeout=120,
        max_retries=5,
        retry_backoff_seconds=2.0
    )
    assert cfg.provider == "gemini"
    assert cfg.model == "custom-model"
    assert cfg.timeout == 120
    assert cfg.max_retries == 5
    assert cfg.retry_backoff_seconds == 2.0

# --- Placeholder for potential future CLI tests ---
# def test_predict_cli_batch_mode(tmp_path, mock_gemini_client):
#     # Needs more setup: mock file I/O, mock Path.home, mock argparse
#     pass

# def test_predict_cli_interactive_mode(mock_gemini_client):
#     # Needs more setup: mock input(), mock Path.home
#     pass