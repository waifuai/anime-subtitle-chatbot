"""
Anime Chatbot package initialization.

This module serves as the main entry point for the anime_chatbot package,
providing convenient access to core functionality including provider selection,
response generation, and configuration management.
"""

# anime_chatbot package export surface
from .provider_selector import generate_response, resolve_provider, ProviderConfig  # re-export for convenience