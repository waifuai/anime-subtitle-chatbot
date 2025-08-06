from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

from .providers.openrouter import OpenRouterProvider
from .providers.gemini import GeminiProvider

DEFAULT_PROVIDER = "openrouter"  # OpenRouter is default

MODEL_FILE_GEMINI = ".model-gemini"
MODEL_FILE_OPENROUTER = ".model-openrouter"


@dataclass
class ProviderConfig:
    provider: str = DEFAULT_PROVIDER
    model: Optional[str] = None
    timeout: int = 60
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0


def resolve_provider(cli_provider: Optional[str] = None) -> str:
    if cli_provider and cli_provider.strip():
        return cli_provider.strip().lower()
    env_provider = os.getenv("CHATBOT_PROVIDER")
    if env_provider and env_provider.strip():
        return env_provider.strip().lower()
    return DEFAULT_PROVIDER


def _read_model_file(filename: str) -> Optional[str]:
    try:
        import pathlib
        p = pathlib.Path.home() / filename
        if p.is_file():
            txt = p.read_text(encoding="utf-8").strip()
            if txt:
                return txt
    except Exception:
        pass
    return None


def resolve_model(provider: str, override: Optional[str] = None) -> Optional[str]:
    if override and override.strip():
        return override.strip()
    if provider == "openrouter":
        return _read_model_file(MODEL_FILE_OPENROUTER) or OpenRouterProvider.DEFAULT_MODEL
    if provider == "gemini":
        return _read_model_file(MODEL_FILE_GEMINI) or GeminiProvider.DEFAULT_MODEL
    return None


def generate_response(prompt_text: str,
                      examples: List[Tuple[str, str]],
                      cfg: ProviderConfig) -> Optional[str]:
    provider = resolve_provider(cfg.provider)
    model = resolve_model(provider, cfg.model)
    if provider == "openrouter":
        client = OpenRouterProvider(timeout=cfg.timeout,
                                    max_retries=cfg.max_retries,
                                    retry_backoff_seconds=cfg.retry_backoff_seconds)
        return client.generate(prompt_text, examples, model)
    elif provider == "gemini":
        client = GeminiProvider(timeout=cfg.timeout)
        return client.generate(prompt_text, examples, model)
    else:
        print(f"Unknown provider: {provider}")
        return None