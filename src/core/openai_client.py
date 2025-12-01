"""
OpenAI API client for language model integration.
"""

import os
import base64
from typing import List, Dict, Optional
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

from src.utils.config_loader import config_loader

# Load environment variables from .env file
load_dotenv()


class OpenAIVisionClient:
    """Client for interacting with OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize OpenAI client."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model or config_loader.get_openai_model(default="gpt-5-2025-08-07")
        
        # Validate that only GPT-5 series models are used
        if not self.model.startswith("gpt-5"):
            raise ValueError(f"Only GPT-5 series models are supported. Got: {self.model}")
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _create_image_message(self, image_path: str) -> Dict:
        """Create image message for API request.
        
        Uses standard image format:
        - type: "image_url"
        - image_url.url: base64 encoded image data
        - image_url.detail: "high" for detailed analysis
        """
        base64_image = self._encode_image(image_path)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            }
        }
    
    def analyze_text_only(self,
                         prompt: str,
                         max_tokens: Optional[int] = None) -> str:
        """
        Analyze text-only prompt with language model (no images).
        
        Args:
            prompt: Text prompt for analysis
            max_tokens: Maximum tokens in response (None = use config)
            
        Returns:
            Generated text response
        """
        # Load settings from config if not provided
        settings = config_loader.get_openai_settings()
        if max_tokens is None:
            max_tokens = settings["max_tokens"]
        
        # Estimate prompt length and adjust max_tokens for long prompts
        prompt_length = len(prompt)
        estimated_prompt_tokens = prompt_length // 4  # Rough estimate: 4 chars per token
        
        print(f"    DEBUG: Prompt length: {prompt_length} characters (~{estimated_prompt_tokens} tokens)")
        
        # Use max_tokens from settings if not provided
        if max_tokens is None:
            max_tokens = settings["max_tokens"]
        
        print(f"    DEBUG: Using max_tokens: {max_tokens}")
        
        # Create text-only message
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # API call with updated parameters
        api_params = {
            "model": self.model,
            "messages": messages
        }
        
        # Seed parameter disabled for compatibility - no seed parameter sent to API
        
        # Use max_completion_tokens parameter for GPT-5 series
        api_params["max_completion_tokens"] = max_tokens
        print(f"    DEBUG: max_completion_tokens set to {max_tokens}")
        
        response = self.client.chat.completions.create(**api_params)
        
        content = response.choices[0].message.content
        if content:
            print(f"    DEBUG: analyze_text_only response length: {len(content)}")
            print(f"    DEBUG: analyze_text_only response preview: {content[:100]}...")
        else:
            print(f"    DEBUG: analyze_text_only response is None or empty")
            print(f"    DEBUG: Full response object: {response}")
        
        return content
        

    def analyze_frames(self,
                       frame_paths: List[str],
                       prompt: str,
                       max_tokens: Optional[int] = None) -> str:
        """
        Analyze multiple frames with vision model.
        
        Args:
            frame_paths: List of paths to frame images
            prompt: Text prompt for analysis
            max_tokens: Maximum tokens in response (None = use config)
            
        Returns:
            Generated text response
        """
        # Load settings from config if not provided
        settings = config_loader.get_openai_settings()
        if max_tokens is None:
            max_tokens = settings["max_tokens"]
        
        # Estimate prompt length and image token consumption
        prompt_length = len(prompt)
        estimated_prompt_tokens = prompt_length // 4  # Rough estimate: 4 chars per token
        
        # Estimate image token consumption (high detail images consume significant tokens)
        num_images = len([path for path in frame_paths if os.path.exists(path)])
        estimated_image_tokens = num_images * 765  # Estimated high detail image token cost
        total_estimated_tokens = estimated_prompt_tokens + estimated_image_tokens
        
        print(f"    DEBUG: Prompt length: {prompt_length} characters (~{estimated_prompt_tokens} tokens)")
        print(f"    DEBUG: Number of images: {num_images} (~{estimated_image_tokens} tokens)")
        print(f"    DEBUG: Total estimated input tokens: ~{total_estimated_tokens}")
        
        # Use max_tokens from settings if not provided
        if max_tokens is None:
            max_tokens = settings["max_tokens"]
        
        print(f"    DEBUG: Using max_tokens: {max_tokens}")
        
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Add all frame images to the message
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                image_message = self._create_image_message(frame_path)
                messages[0]["content"].append(image_message)
            else:
                print(f"Warning: Frame not found: {frame_path}")
        
        # API call with updated parameters
        api_params = {
            "model": self.model,
            "messages": messages
        }
        
        # Seed parameter disabled for compatibility - no seed parameter sent to API
        
        # Use max_completion_tokens parameter for GPT-5 series
        api_params["max_completion_tokens"] = max_tokens
        print(f"    DEBUG: max_completion_tokens set to {max_tokens}")
        
        response = self.client.chat.completions.create(**api_params)
        
        return response.choices[0].message.content
            
    
    def analyze_single_frame(self,
                           frame_path: str,
                           prompt: str,
                           max_tokens: Optional[int] = None) -> str:
        """
        Analyze a single frame with vision model.
        
        Args:
            frame_path: Path to frame image
            prompt: Text prompt for analysis
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text response
        """
        return self.analyze_frames([frame_path], prompt, max_tokens)
    
    def analyze_multiple_frames(self,
                              frame_paths: List[str],
                              prompt: str,
                              max_tokens: Optional[int] = None) -> str:
        """
        Alias for analyze_frames method for enhanced object detection.
        
        Args:
            frame_paths: List of paths to frame images
            prompt: Text prompt for analysis
            max_tokens: Maximum tokens in response (None = use config)
            
        Returns:
            Generated text response
        """
        return self.analyze_frames(frame_paths, prompt, max_tokens)
    
