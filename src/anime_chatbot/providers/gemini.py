from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional, List, Tuple
import google.generativeai as genai

logger = logging.getLogger(__name__)


class GeminiProvider:
    """
    Provider for Google Gemini API integration.

    This class handles authentication, prompt building, and response generation
    using the Google Gemini API with proper error handling and logging.
    """
    DEFAULT_MODEL = "models/gemini-2.5-pro"
    API_KEY_FILE = ".api-gemini"

    def __init__(self, timeout: int = 60) -> None:
        """
        Initialize the Gemini provider.

        Args:
            timeout: Request timeout in seconds (kept for interface parity).
        """
        self.timeout = timeout

    @staticmethod
    def _resolve_api_key() -> Optional[str]:
        """
        Resolve API key from environment variables or file.

        Priority order:
        1. GEMINI_API_KEY environment variable
        2. GOOGLE_API_KEY environment variable
        3. ~/.api-gemini file

        Returns:
            The API key if found, None otherwise.
        """
        # Check environment variables first (more secure)
        env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if env_key and env_key.strip():
            api_key = env_key.strip()
            if len(api_key) < 10:  # Basic validation
                logger.warning("API key appears to be too short (< 10 characters)")
            return api_key

        # Fall back to file-based key
        try:
            key_path = Path.home() / GeminiProvider.API_KEY_FILE
            if key_path.is_file():
                api_key = key_path.read_text(encoding="utf-8").strip()
                if api_key:
                    if len(api_key) < 10:
                        logger.warning(f"API key in {key_path} appears to be too short (< 10 characters)")
                    logger.info(f"Using API key from {key_path}")
                    return api_key
                else:
                    logger.warning(f"API key file {key_path} is empty")
            else:
                logger.info(f"API key file {key_path} not found")
        except PermissionError as e:
            logger.error(f"Permission denied reading API key file: {e}")
        except UnicodeDecodeError as e:
            logger.error(f"Invalid encoding in API key file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error reading API key file: {e}")

        return None

    @staticmethod
    def _build_prompt(prompt_text: str, examples: List[Tuple[str, str]]) -> str:
        """
        Build a few-shot prompt for the Gemini API.

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

    def generate(self, prompt_text: str, examples: List[Tuple[str, str]], model: Optional[str]) -> Optional[str]:
        """
        Generate a response using the Gemini API.

        Args:
            prompt_text: The user's input text to generate a response for.
            examples: List of (input, output) tuples for few-shot examples.
            model: Model name to use, defaults to DEFAULT_MODEL if None.

        Returns:
            Generated response text, or None if generation failed.
        """
        # Input validation
        if not prompt_text or not prompt_text.strip():
            logger.warning("Empty prompt provided to Gemini provider")
            return None

        if len(prompt_text.strip()) > 10000:  # Reasonable limit
            logger.warning(f"Prompt too long ({len(prompt_text)} chars), truncating")
            prompt_text = prompt_text.strip()[:10000]

        # API key resolution
        api_key = self._resolve_api_key()
        if not api_key:
            error_msg = "Gemini API key is missing. Set GEMINI_API_KEY/GOOGLE_API_KEY or create ~/.api-gemini"
            logger.error(error_msg)
            print(error_msg)
            return None

        model_name = model or self.DEFAULT_MODEL
        logger.info(f"Generating response using model: {model_name}")

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(self._build_prompt(prompt_text, examples))

            # Handle successful response
            if getattr(response, "parts", None):
                text = response.text.strip()
                if text:
                    logger.info("Successfully generated response from Gemini API")
                    return text
                else:
                    logger.warning("Gemini API returned empty response")
                    return "[Error: Empty response from API]"

            # Handle blocked content
            elif getattr(response, "prompt_feedback", None) and getattr(response.prompt_feedback, "block_reason", None):
                block_reason = response.prompt_feedback.block_reason
                logger.warning(f"Prompt blocked by Gemini API: {block_reason}")
                print(f"Warning: Prompt blocked. Reason: {block_reason}")
                return f"[Blocked: {block_reason}]"

            # Fallback attempt to extract text
            else:
                try:
                    text = response.text.strip()
                    if text:
                        logger.info("Successfully extracted text from fallback response")
                        return text
                    else:
                        logger.error("Could not extract meaningful text from Gemini response")
                        return "[Error: Could not parse response]"
                except AttributeError:
                    logger.error("Response object does not have expected text attribute")
                    return "[Error: Malformed response from API]"

        except genai.types.BlockedPromptException as e:
            logger.warning(f"Prompt blocked by Gemini API: {e}")
            return f"[Blocked: {str(e)}]"
        except Exception as e:
            # Handle any API-related exceptions
            logger.error(f"Gemini API error: {e}")
            return f"[API Error: {str(e)}]"
        except Exception as e:
            logger.error(f"Unexpected error calling Gemini API: {e}")
            print(f"An error occurred calling the Gemini API: {e}")
            return None