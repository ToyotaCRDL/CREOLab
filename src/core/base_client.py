"""
Abstract base class for LLM clients.
Defines the common interface that all provider-specific clients must implement.
"""

import base64
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

try:
    from ..utils.config_loader import config_loader
except ImportError:
    from utils.config_loader import config_loader


class BaseLLMClient(ABC):
    """Abstract base class for language model clients."""

    model: str

    def _resolve_max_tokens(self, max_tokens: Optional[int]) -> int:
        """Return *max_tokens* if explicitly given, otherwise fall back to config."""
        if max_tokens is not None:
            return max_tokens
        return config_loader.get_llm_settings()["max_tokens"]

    @staticmethod
    def _log_text_request(prompt: str, max_tokens: int) -> None:
        """Print common DEBUG lines for a text-only request."""
        prompt_length = len(prompt)
        estimated_prompt_tokens = prompt_length // 4
        print(f"    DEBUG: Prompt length: {prompt_length} characters (~{estimated_prompt_tokens} tokens)")
        print(f"    DEBUG: Using max_tokens: {max_tokens}")

    @staticmethod
    def _log_frames_request(prompt: str, num_images: int, max_tokens: int) -> None:
        """Print common DEBUG lines for a frames request."""
        print(f"    DEBUG: Prompt length: {len(prompt)} characters")
        print(f"    DEBUG: Number of images: {num_images}")
        print(f"    DEBUG: Using max_tokens: {max_tokens}")

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Encode an image file to a base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _create_image_message(image_path: str) -> Dict:
        """Create an OpenAI-compatible image_url content block."""
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
            },
        }

    @abstractmethod
    def analyze_text_only(self,
                         prompt: str,
                         max_tokens: Optional[int] = None) -> str:
        """Analyze text-only prompt with language model (no images)."""
        ...

    @abstractmethod
    def analyze_frames(self,
                       frame_paths: List[str],
                       prompt: str,
                       max_tokens: Optional[int] = None) -> str:
        """Analyze multiple frames with vision model."""
        ...

    def analyze_single_frame(self,
                           frame_path: str,
                           prompt: str,
                           max_tokens: Optional[int] = None) -> str:
        """Analyze a single frame with vision model."""
        return self.analyze_frames([frame_path], prompt, max_tokens)

