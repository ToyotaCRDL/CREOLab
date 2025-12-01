"""
Prompt loader for reading prompt templates from JSON files.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class PromptLoader:
    """
    Loads prompt templates from JSON files.
    """
    
    def __init__(self, prompts_dir: str = "config/prompts"):
        """
        Initialize prompt loader.
        
        Args:
            prompts_dir: Directory containing prompt JSON files
        """
        self.prompts_dir = prompts_dir
        self._prompt_cache = {}
    
    def load_prompt_file(self, filename: str) -> Optional[Dict[str, str]]:
        """
        Load prompt data from a JSON file.
        
        Args:
            filename: Name of the prompt file (e.g., "common_captioning_prompts.json")
            
        Returns:
            Dictionary with 'description' and 'prompt' keys or None if not found
        """
        # Check cache first
        if filename in self._prompt_cache:
            return self._prompt_cache[filename]
        
        # Construct full path
        prompt_path = os.path.join(self.prompts_dir, filename)
        
        if not os.path.exists(prompt_path):
            print(f"Warning: Prompt file not found: {prompt_path}")
            return None
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
        
        # Check if it's the old format (has 'prompt' key) or new format (has nested prompt objects)
        if 'prompt' not in prompt_data:
            # New format - check if it has valid nested structures
            has_valid_structure = False
            for key, value in prompt_data.items():
                if isinstance(value, dict) and 'prompt' in value:
                    has_valid_structure = True
                    break
            
            if not has_valid_structure:
                print(f"Warning: Invalid prompt file structure: {prompt_path}")
                return None
        
        # Cache the result
        self._prompt_cache[filename] = prompt_data
        return prompt_data
            
    
    def get_prompt(self, filename: str, prompt_key: str) -> str:
        """
        Get prompt by filename and key.
        
        Args:
            filename: Prompt file name (e.g., "common_captioning_prompts.json")
            prompt_key: Prompt key within the file (e.g., "segment_analysis")
            
        Returns:
            Prompt text
            
        Raises:
            FileNotFoundError: If prompt file is not found
            ValueError: If prompt key is not found or invalid
        """
        data = self.load_prompt_file(filename)
        
        if prompt_key not in data:
            error_msg = f"CRITICAL ERROR: Prompt key '{prompt_key}' not found in {filename}"
            print(f"✗ {error_msg}")
            raise ValueError(error_msg)
        
        prompt_section = data[prompt_key]
        
        if isinstance(prompt_section, dict) and "prompt" in prompt_section:
            return prompt_section["prompt"]
        
        error_msg = f"CRITICAL ERROR: Invalid prompt structure for '{prompt_key}' in {filename}"
        print(f"✗ {error_msg}")
        raise ValueError(error_msg)


# Global prompt loader instance
prompt_loader = PromptLoader()
