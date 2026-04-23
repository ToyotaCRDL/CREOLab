"""
Configuration loader for reading settings from JSON files.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """
    Loads configuration from JSON files in the config directory.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize config loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir
        self._config_cache = {}
        self._active_provider = None
    
    def load_config(self, config_file: str) -> Optional[Dict[str, Any]]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_file: Name of the config file (e.g., "api_config.json")
            
        Returns:
            Dictionary with configuration data or None if not found
        """
        if config_file in self._config_cache:
            return self._config_cache[config_file]
        
        config_path = os.path.join(self.config_dir, config_file)
        
        if not os.path.exists(config_path):
            print(f"Warning: Config file not found: {config_path}")
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self._config_cache[config_file] = config_data
            return config_data
            
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            return None

    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.load_config("api_config.json") or {}
    
    def set_provider(self, provider: str):
        """Set the active LLM provider (called from CLI --provider)."""
        self._active_provider = provider

    def get_provider(self, default: str = "openai") -> str:
        """Get the active LLM provider name."""
        if self._active_provider:
            return self._active_provider
        config = self.get_api_config()
        return config.get("default_provider", default)

    def _get_active_provider_config(self) -> Dict[str, Any]:
        """Get the config section for the active provider."""
        config = self.get_api_config()
        provider = self.get_provider()
        return config.get(provider, {})

    def get_model(self, default: str = "gpt-5-2025-08-07") -> str:
        """Get the model name for the active provider."""
        return self._get_active_provider_config().get("model", default)

    def get_llm_settings(self) -> Dict[str, Any]:
        """Get LLM API settings (max_tokens, etc.) for the active provider."""
        prov_cfg = self._get_active_provider_config()
        return {
            "max_tokens": prov_cfg.get("max_tokens", 16000),
            "temperature": 1.0,
            "seed": None
        }

    def get_thinking_mode(self) -> Optional[str]:
        """Get thinking mode for the active provider (e.g. 'adaptive'), or None."""
        return self._get_active_provider_config().get("thinking")

    def get_max_images(self, default: int = None) -> Optional[int]:
        """Get max images per request for the active provider (None = unlimited)."""
        return self._get_active_provider_config().get("max_images", default)



# Global config loader instance
config_loader = ConfigLoader()
