"""
Google Gemini (Vertex AI) client for language model integration.
Requires environment variables set in .env:
  GOOGLE_GENAI_USE_VERTEXAI=True
  GOOGLE_CLOUD_PROJECT=<your-project-id>
  GOOGLE_CLOUD_LOCATION=<location>
"""

import os
import time
from typing import List, Optional, Callable, TypeVar
from dotenv import load_dotenv

from google import genai
from google.genai import types

from .base_client import BaseLLMClient

T = TypeVar("T")

try:
    from ..utils.config_loader import config_loader
except ImportError:
    from utils.config_loader import config_loader

load_dotenv()


class GeminiVisionClient(BaseLLMClient):
    """Client for interacting with Google Gemini API via Vertex AI."""

    MAX_RETRIES = 3
    RETRY_WAIT_SECONDS = 60

    def __init__(self, model: Optional[str] = None):
        """Initialize Gemini client.

        Args:
            model: Gemini model name (defaults to config setting)
        """
        self.client = genai.Client()
        self.model = model or config_loader.get_model(default="gemini-3.1-pro-preview")

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        """Return True if the exception represents a 429 quota/rate-limit error."""
        exc_str = str(exc)
        if "429" in exc_str:
            return True
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status == 429:
            return True
        cause = getattr(exc, "__cause__", None)
        if cause is not None and "429" in str(cause):
            return True
        return False

    def _call_with_retry(self, api_call: Callable[[], T], context_msg: str) -> T:
        """Execute *api_call* with automatic retry on 429 errors.

        Args:
            api_call: Zero-arg callable that performs the Gemini API request.
            context_msg: Short description used in log messages.
        """
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return api_call()
            except Exception as e:
                if self._is_quota_error(e) and attempt < self.MAX_RETRIES:
                    print(
                        f"    WARNING: Gemini API 429 quota error on {context_msg} "
                        f"(attempt {attempt}/{self.MAX_RETRIES}). "
                        f"Waiting {self.RETRY_WAIT_SECONDS}s before retry..."
                    )
                    time.sleep(self.RETRY_WAIT_SECONDS)
                    continue
                raise

    def analyze_text_only(self,
                         prompt: str,
                         max_tokens: Optional[int] = None) -> str:
        max_tokens = self._resolve_max_tokens(max_tokens)
        self._log_text_request(prompt, max_tokens)

        def _do_request() -> str:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                ),
            )
            content = response.text
            if content:
                print(f"    DEBUG: analyze_text_only response length: {len(content)}")
                print(f"    DEBUG: analyze_text_only response preview: {content[:100]}...")
            else:
                print(f"    DEBUG: analyze_text_only response is None or empty")
            return content

        try:
            return self._call_with_retry(_do_request, "analyze_text_only")
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            print(f"    DEBUG: Model: {self.model}")
            print(f"    DEBUG: Prompt length: {len(prompt)} characters")
            raise

    def analyze_frames(self,
                       frame_paths: List[str],
                       prompt: str,
                       max_tokens: Optional[int] = None) -> str:
        max_tokens = self._resolve_max_tokens(max_tokens)
        num_images = len([p for p in frame_paths if os.path.exists(p)])
        self._log_frames_request(prompt, num_images, max_tokens)

        parts: list = [prompt]
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                with open(frame_path, "rb") as f:
                    image_bytes = f.read()
                mime = "image/png" if frame_path.lower().endswith(".png") else "image/jpeg"
                parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))
            else:
                print(f"Warning: Frame not found: {frame_path}")

        def _do_request() -> str:
            response = self.client.models.generate_content(
                model=self.model,
                contents=parts,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text

        try:
            return self._call_with_retry(_do_request, "analyze_frames")
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            print(f"    DEBUG: Model: {self.model}")
            print(f"    DEBUG: Prompt length: {len(prompt)} characters")
            raise
