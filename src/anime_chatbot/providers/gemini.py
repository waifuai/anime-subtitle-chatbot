from __future__ import annotations
import os
from typing import Optional, List, Tuple
import google.generativeai as genai


class GeminiProvider:
    DEFAULT_MODEL = "models/gemini-2.5-pro"
    API_KEY_FILE = ".api-gemini"

    def __init__(self, timeout: int = 60):
        # google.generativeai uses its own internal timeouts; keep for parity
        self.timeout = timeout

    @staticmethod
    def _resolve_api_key() -> Optional[str]:
        env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()
        try:
            from pathlib import Path
            p = Path.home() / GeminiProvider.API_KEY_FILE
            if p.is_file():
                return p.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        return None

    @staticmethod
    def _build_prompt(prompt_text: str, examples: List[Tuple[str, str]]) -> str:
        parts = [
            "You are an anime chatbot. Given an input dialogue line, generate a relevant response in the style of anime subtitles.",
            "Follow the format of the examples.",
            "\n---\n",
        ]
        for inp, outp in examples:
            parts.append(f"Input: {inp}")
            parts.append(f"Output: {outp}")
            parts.append("---")
        parts.append(f"Input: {prompt_text}")
        parts.append("Output:")
        return "\n".join(parts)

    def generate(self, prompt_text: str, examples: List[Tuple[str, str]], model: Optional[str]) -> Optional[str]:
        api_key = self._resolve_api_key()
        if not api_key:
            print("Gemini API key is missing. Set GEMINI_API_KEY/GOOGLE_API_KEY or create ~/.api-gemini")
            return None
        model_name = model or self.DEFAULT_MODEL
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(self._build_prompt(prompt_text, examples))
            if getattr(response, "parts", None):
                return response.text.strip()
            elif getattr(response, "prompt_feedback", None) and getattr(response.prompt_feedback, "block_reason", None):
                print(f"Warning: Prompt blocked. Reason: {response.prompt_feedback.block_reason}")
                return f"[Blocked: {response.prompt_feedback.block_reason}]"
            else:
                try:
                    return response.text.strip()
                except Exception:
                    print("Warning: Could not extract text from Gemini response.")
                    return "[Error: Could not parse response]"
        except Exception as e:
            print(f"An error occurred calling the Gemini API: {e}")
            return None