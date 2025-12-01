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
    
    def load_config(self, config_file: str) -> Optional[Dict[str, Any]]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_file: Name of the config file (e.g., "api_config.json")
            
        Returns:
            Dictionary with configuration data or None if not found
        """
        # Check cache first
        if config_file in self._config_cache:
            return self._config_cache[config_file]
        
        # Construct full path
        config_path = os.path.join(self.config_dir, config_file)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Cache the result
        self._config_cache[config_file] = config_data
        return config_data

    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.load_config("api_config.json") or {}
    
    
    
    def get_openai_model(self, default: str = "gpt-5-2025-08-07") -> str:
        """Get the OpenAI model name from config."""
        config = self.get_api_config()
        return config.get("model", default)
    
    
    def get_openai_settings(self) -> Dict[str, Any]:
        """Get OpenAI API settings."""
        config = self.get_api_config()
        return {
            "max_tokens": config.get("max_tokens", 16000)
        }


# Global config loader instance
config_loader = ConfigLoader()
