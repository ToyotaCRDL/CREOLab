"""
Anthropic Claude client for language model integration.
Requires environment variable set in .env:
  ANTHROPIC_API_KEY=<your-api-key>
"""

import json
import os
import base64
import re
import time
from typing import List, Optional, Callable, TypeVar
from dotenv import load_dotenv

import anthropic

from .base_client import BaseLLMClient

T = TypeVar("T")

try:
    from ..utils.config_loader import config_loader
except ImportError:
    from utils.config_loader import config_loader

load_dotenv()


class AnthropicVisionClient(BaseLLMClient):
    """Client for interacting with Anthropic Claude Messages API."""

    MAX_RETRIES = 3
    RETRY_WAIT_SECONDS = 60
    OVERLOAD_WAIT_SECONDS = 600

    def __init__(self, model: Optional[str] = None):
        """Initialize Anthropic client.

        Args:
            model: Claude model name (defaults to config setting)
        """
        self.client = anthropic.Anthropic()
        self.model = model or config_loader.get_model(default="claude-sonnet-4-6")

    @staticmethod
    def _sanitize_response(text: str) -> str:
        """Extract the final JSON or text content from a Claude response.

        Claude sometimes exhibits a "self-correction" pattern: it outputs an
        initial JSON answer then adds commentary like "Wait, let me
        re-analyze more carefully." followed by a revised JSON.  When
        downstream parsers see both in a single string they fail with
        "Extra data".

        Strategy:
          1. If fenced code blocks exist, return the last one.
          2. Otherwise, locate all top-level bare JSON objects / arrays
             via ``json.JSONDecoder.raw_decode`` and return the last one.
          3. Fall back to the original text (plain-text responses like
             procedure steps are not affected).
        """
        if not text:
            return text

        blocks = re.findall(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if blocks:
            return blocks[-1].strip()

        decoder = json.JSONDecoder()
        last_json: Optional[str] = None
        i = 0
        while i < len(text):
            if text[i] in ("{", "["):
                try:
                    _obj, end = decoder.raw_decode(text, i)
                    last_json = text[i:end]
                    i = end
                    continue
                except json.JSONDecodeError:
                    pass
            i += 1

        if last_json is not None:
            return last_json.strip()

        return text

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Return True if the exception represents a 429 rate-limit error."""
        if isinstance(exc, anthropic.RateLimitError):
            return True
        exc_str = str(exc)
        if "429" in exc_str:
            return True
        status = getattr(exc, "status_code", None)
        if status == 429:
            return True
        return False

    @staticmethod
    def _is_overloaded_error(exc: Exception) -> bool:
        """Return True if the exception represents a 529 overloaded error."""
        if isinstance(exc, anthropic.APIStatusError) and getattr(exc, "status_code", None) == 529:
            return True
        if "529" in str(exc) or "overloaded" in str(exc).lower():
            return True
        return False

    def _build_api_params(self, max_tokens: int, messages: list) -> dict:
        """Build parameters dict for messages.create(), adding thinking
        if the active provider config specifies a thinking mode."""
        params: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        thinking_mode = config_loader.get_thinking_mode()
        if thinking_mode:
            params["thinking"] = {"type": thinking_mode}
        return params

    @staticmethod
    def _extract_text(message) -> Optional[str]:
        """Return the text from the last 'text' content block.

        When adaptive thinking is enabled the response contains both
        ``thinking`` and ``text`` blocks.  We always want the final text."""
        return next(
            (block.text for block in reversed(message.content)
             if block.type == "text"),
            None,
        )

    def _call_with_retry(self, api_call: Callable[[], T], context_msg: str) -> T:
        """Execute *api_call* with automatic retry on 429 / 529 errors.

        Args:
            api_call: Zero-arg callable that performs the API request.
            context_msg: Short description used in log messages.
        """
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return api_call()
            except Exception as e:
                retryable = attempt < self.MAX_RETRIES
                if self._is_overloaded_error(e) and retryable:
                    print(
                        f"    WARNING: Anthropic API 529 overloaded on {context_msg} "
                        f"(attempt {attempt}/{self.MAX_RETRIES}). "
                        f"Waiting {self.OVERLOAD_WAIT_SECONDS}s before retry..."
                    )
                    time.sleep(self.OVERLOAD_WAIT_SECONDS)
                    continue
                if self._is_rate_limit_error(e) and retryable:
                    print(
                        f"    WARNING: Anthropic API 429 rate-limit error on {context_msg} "
                        f"(attempt {attempt}/{self.MAX_RETRIES}). "
                        f"Waiting {self.RETRY_WAIT_SECONDS}s before retry..."
                    )
                    time.sleep(self.RETRY_WAIT_SECONDS)
                    continue
                raise

    def _call_api(self, api_params: dict, context_msg: str) -> str:
        """API call with 429 retry (inner) and truncation retry (outer)."""
        for trunc_attempt in range(1, self.MAX_RETRIES + 1):
            def _do():
                return self.client.messages.create(**api_params)

            message = self._call_with_retry(_do, context_msg)

            if message.stop_reason == "max_tokens" and trunc_attempt < self.MAX_RETRIES:
                print(f"    WARNING: Response truncated (stop_reason=max_tokens). "
                      f"Retrying ({trunc_attempt}/{self.MAX_RETRIES})...")
                continue

            if message.stop_reason == "max_tokens":
                print("    WARNING: Response still truncated after retries.")

            return self._extract_text(message)

    def analyze_text_only(self,
                         prompt: str,
                         max_tokens: Optional[int] = None) -> str:
        max_tokens = self._resolve_max_tokens(max_tokens)
        self._log_text_request(prompt, max_tokens)

        api_params = self._build_api_params(
            max_tokens, [{"role": "user", "content": prompt}]
        )

        try:
            content = self._call_api(api_params, "analyze_text_only")
            return self._sanitize_response(content)
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
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

        content_blocks = []
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                with open(frame_path, "rb") as f:
                    image_data = base64.standard_b64encode(f.read()).decode("utf-8")
                mime = "image/png" if frame_path.lower().endswith(".png") else "image/jpeg"
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": image_data,
                    },
                })
            else:
                print(f"Warning: Frame not found: {frame_path}")

        content_blocks.append({"type": "text", "text": prompt})

        api_params = self._build_api_params(
            max_tokens, [{"role": "user", "content": content_blocks}]
        )

        try:
            content = self._call_api(api_params, "analyze_frames")
            return self._sanitize_response(content)
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            print(f"    DEBUG: Model: {self.model}")
            print(f"    DEBUG: Prompt length: {len(prompt)} characters")
            raise
