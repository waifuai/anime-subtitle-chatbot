"""
OpenRouter provider for the Anime Subtitle Chatbot.

This module implements the OpenRouterProvider class, which handles authentication,
prompt building, and response generation using the OpenRouter API with proper
error handling, retry logic, and logging capabilities. Supports multiple models
through the OpenRouter unified API.
"""

from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import Optional, List, Tuple
import requests

logger = logging.getLogger(__name__)


class OpenRouterProvider:
    """
    Provider for OpenRouter API integration.

    This class handles authentication, prompt building, and response generation
    using the OpenRouter API with proper error handling, logging, and retries.
    """
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "openrouter/free"
    API_KEY_FILE = Path.home() / ".api-openrouter"

    def __init__(self, timeout: int = 60, max_retries: int = 2, retry_backoff_seconds: float = 1.0) -> None:
        """
        Initialize the OpenRouter provider.

        Args:
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts on failure.
            retry_backoff_seconds: Base delay between retries in seconds.
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    @staticmethod
    def _resolve_api_key() -> Optional[str]:
        """
        Resolve API key from environment variables or file.

        Priority order:
        1. OPENROUTER_API_KEY environment variable
        2. ~/.api-openrouter file

        Returns:
            The API key if found, None otherwise.
        """
        # Check environment variables first (more secure)
        env_key = os.getenv("OPENROUTER_API_KEY")
        if env_key and env_key.strip():
            api_key = env_key.strip()
            if len(api_key) < 10:  # Basic validation
                logger.warning("OpenRouter API key appears to be too short (< 10 characters)")
            return api_key

        # Fall back to file-based key
        try:
            key_path = OpenRouterProvider.API_KEY_FILE
            if key_path.is_file():
                api_key = key_path.read_text(encoding="utf-8").strip()
                if api_key:
                    if len(api_key) < 10:
                        logger.warning(f"OpenRouter API key in {key_path} appears to be too short (< 10 characters)")
                    logger.info(f"Using OpenRouter API key from {key_path}")
                    return api_key
                else:
                    logger.warning(f"OpenRouter API key file {key_path} is empty")
            else:
                logger.info(f"OpenRouter API key file {key_path} not found")
        except PermissionError as e:
            logger.error(f"Permission denied reading OpenRouter API key file: {e}")
        except UnicodeDecodeError as e:
            logger.error(f"Invalid encoding in OpenRouter API key file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error reading OpenRouter API key file: {e}")

        return None

    @staticmethod
    def _build_prompt(prompt_text: str, examples: List[Tuple[str, str]]) -> str:
        """
        Build a few-shot prompt for the OpenRouter API.

        Args:
            prompt_text: The user's input text to generate a response for.
            examples: List of (input, output) tuples for few-shot examples.

        Returns:
            Formatted prompt string with instructions and examples.
        """
        parts = [
            "You are an anime chatbot. Given an input dialogue line, generate a relevant response in the style of anime subtitles.",
            "Follow the format of the examples.",
            "\n---\n",
        ]

        for inp, outp in examples:
            if inp and outp:  # Skip empty examples
                parts.append(f"Input: {inp}")
                parts.append(f"Output: {outp}")
                parts.append("---")

        parts.append(f"Input: {prompt_text}")
        parts.append("Output:")
        return "\n".join(parts)

    def _post(self, payload: dict, headers: dict) -> Optional[requests.Response]:
        """
        Make HTTP POST request to OpenRouter API with retry logic.

        Args:
            payload: Request payload dictionary.
            headers: Request headers dictionary.

        Returns:
            Response object if successful, None if all retries failed.
        """
        attempt = 0
        while attempt <= self.max_retries:
            try:
                logger.debug(f"Making OpenRouter API request (attempt {attempt + 1}/{self.max_retries + 1})")
                resp = requests.post(self.API_URL, headers=headers, json=payload, timeout=self.timeout)
                logger.info(f"OpenRouter API request successful (status: {resp.status_code})")
                return resp
            except requests.exceptions.Timeout as e:
                logger.warning(f"OpenRouter API request timeout (attempt {attempt + 1}): {e}")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"OpenRouter API connection error (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"OpenRouter API request failed (attempt {attempt + 1}): {e}")

            attempt += 1
            if attempt <= self.max_retries:
                delay = self.retry_backoff_seconds * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying OpenRouter API request in {delay:.1f} seconds...")
                time.sleep(delay)

        logger.error(f"OpenRouter API request failed after {self.max_retries + 1} attempts")
        print(f"OpenRouter request failed after retries")
        return None

    def generate(self, prompt_text: str, examples: List[Tuple[str, str]], model: Optional[str]) -> Optional[str]:
        """
        Generate a response using the OpenRouter API.

        Args:
            prompt_text: The user's input text to generate a response for.
            examples: List of (input, output) tuples for few-shot examples.
            model: Model name to use, defaults to DEFAULT_MODEL if None.

        Returns:
            Generated response text, or None if generation failed.
        """
        # Input validation
        if not prompt_text or not prompt_text.strip():
            logger.warning("Empty prompt provided to OpenRouter provider")
            return None

        if len(prompt_text.strip()) > 10000:  # Reasonable limit
            logger.warning(f"Prompt too long ({len(prompt_text)} chars), truncating")
            prompt_text = prompt_text.strip()[:10000]

        # API key resolution
        api_key = self._resolve_api_key()
        if not api_key:
            error_msg = "OpenRouter API key is missing. Set OPENROUTER_API_KEY or create ~/.api-openrouter"
            logger.error(error_msg)
            print(error_msg)
            return None

        model_name = model or self.DEFAULT_MODEL
        logger.info(f"Generating response using OpenRouter model: {model_name}")

        # Prepare request
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": self._build_prompt(prompt_text, examples)}],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Make request
        resp = self._post(payload, headers)
        if not resp:
            return None

        # Handle HTTP errors
        if resp.status_code != 200:
            try:
                body = resp.text[:500]
            except Exception:
                body = "<unreadable>"

            if resp.status_code == 401:
                logger.error("OpenRouter API authentication failed - check API key")
                print("OpenRouter authentication failed. Please check your API key.")
            elif resp.status_code == 429:
                logger.warning("OpenRouter API rate limit exceeded")
                print("OpenRouter rate limit exceeded. Please try again later.")
            elif resp.status_code >= 500:
                logger.error(f"OpenRouter API server error: {resp.status_code}")
                print(f"OpenRouter server error: {resp.status_code}")
            else:
                logger.error(f"OpenRouter API error {resp.status_code}: {body}")
                print(f"OpenRouter error {resp.status_code}: {body}")
            return None

        # Parse response
        try:
            data = resp.json()
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"OpenRouter API returned invalid JSON: {e}")
            print(f"OpenRouter invalid JSON response: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing OpenRouter API response: {e}")
            print(f"Error parsing OpenRouter response: {e}")
            return None

        # Extract content
        choices = data.get("choices", [])
        if not choices:
            logger.error("OpenRouter API returned no choices in response")
            print("OpenRouter API returned no choices")
            return None

        content = (choices[0].get("message", {}).get("content") or "").strip()
        if not content:
            logger.warning("OpenRouter API returned empty content")
            return "[Error: Empty response from API]"

        logger.info("Successfully generated response from OpenRouter API")
        return content