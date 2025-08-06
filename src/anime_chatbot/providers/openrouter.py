from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Optional, List, Tuple
import requests


class OpenRouterProvider:
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "openrouter/horizon-beta"
    API_KEY_FILE = Path.home() / ".api-openrouter"

    def __init__(self, timeout: int = 60, max_retries: int = 2, retry_backoff_seconds: float = 1.0):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    @staticmethod
    def _resolve_api_key() -> Optional[str]:
        env_key = os.getenv("OPENROUTER_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()
        try:
            if OpenRouterProvider.API_KEY_FILE.is_file():
                return OpenRouterProvider.API_KEY_FILE.read_text(encoding="utf-8").strip()
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

    def _post(self, payload: dict, headers: dict) -> Optional[requests.Response]:
        attempt = 0
        while True:
            try:
                resp = requests.post(self.API_URL, headers=headers, json=payload, timeout=self.timeout)
                return resp
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    print(f"OpenRouter request failed after retries: {e}")
                    return None
                time.sleep(self.retry_backoff_seconds)

    def generate(self, prompt_text: str, examples: List[Tuple[str, str]], model: Optional[str]) -> Optional[str]:
        api_key = self._resolve_api_key()
        if not api_key:
            print("OpenRouter API key is missing. Set OPENROUTER_API_KEY or create ~/.api-openrouter")
            return None
        model_name = model or self.DEFAULT_MODEL
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": self._build_prompt(prompt_text, examples)}],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = self._post(payload, headers)
        if not resp:
            return None
        if resp.status_code != 200:
            try:
                body = resp.text[:500]
            except Exception:
                body = "<unreadable>"
            print(f"OpenRouter non-200: {resp.status_code} {body}")
            return None
        try:
            data = resp.json()
        except Exception as e:
            print(f"OpenRouter invalid JSON: {e}")
            return None
        choices = data.get("choices", [])
        if not choices:
            return None
        content = (choices[0].get("message", {}).get("content") or "").strip()
        if not content:
            return None
        return content