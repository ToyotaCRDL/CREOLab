"""
OpenAI API client for language model integration.
"""

import os
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

from .base_client import BaseLLMClient

try:
    from ..utils.config_loader import config_loader
except ImportError:
    from utils.config_loader import config_loader

load_dotenv()


class OpenAIVisionClient(BaseLLMClient):
    """Client for interacting with OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize OpenAI client."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model or config_loader.get_model(default="gpt-5-2025-08-07")
    
    def analyze_text_only(self,
                         prompt: str,
                         max_tokens: Optional[int] = None) -> str:
        max_tokens = self._resolve_max_tokens(max_tokens)
        self._log_text_request(prompt, max_tokens)
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
            
            content = response.choices[0].message.content
            if content:
                print(f"    DEBUG: analyze_text_only response length: {len(content)}")
                print(f"    DEBUG: analyze_text_only response preview: {content[:100]}...")
            else:
                print(f"    DEBUG: analyze_text_only response is None or empty")
                print(f"    DEBUG: Full response object: {response}")
            
            return content
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            print(f"    DEBUG: Model: {self.model}")
            print(f"    DEBUG: Prompt length: {len(prompt)} characters")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"    DEBUG: API response: {e.response.text}")
            raise

    def analyze_frames(self,
                       frame_paths: List[str],
                       prompt: str,
                       max_tokens: Optional[int] = None) -> str:
        max_tokens = self._resolve_max_tokens(max_tokens)
        num_images = len([p for p in frame_paths if os.path.exists(p)])
        self._log_frames_request(prompt, num_images, max_tokens)
        
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                messages[0]["content"].append(self._create_image_message(frame_path))
            else:
                print(f"Warning: Frame not found: {frame_path}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            print(f"    DEBUG: Model: {self.model}")
            print(f"    DEBUG: Prompt length: {len(prompt)} characters")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"    DEBUG: API response: {e.response.text}")
            raise
