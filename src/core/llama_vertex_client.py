"""
Llama on Vertex AI (MaaS) client via the OpenAI-compatible endpoint.

Authentication uses Application Default Credentials (ADC).
Requires:
  - GOOGLE_CLOUD_PROJECT env var (or gcloud default project)
  - ADC configured (gcloud auth application-default login)
"""

import json
import os
import re
import time
from typing import List, Optional

from openai import OpenAI
from google.auth import default as google_auth_default
import google.auth.transport.requests
from dotenv import load_dotenv

from .base_client import BaseLLMClient

try:
    from ..utils.config_loader import config_loader
except ImportError:
    from utils.config_loader import config_loader

load_dotenv()

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class LlamaVertexClient(BaseLLMClient):
    """Client for Llama models hosted on Vertex AI MaaS (OpenAI-compatible endpoint)."""

    MAX_RETRIES = 3
    RETRY_WAIT_SECONDS = 60
    SYSTEM_MESSAGE = (
        "You are a precise assistant. "
        "You may reason and analyze freely, but you MUST wrap your final answer "
        "inside <answer> and </answer> tags. "
        "Only the content inside <answer>...</answer> will be used."
    )

    def __init__(self, model: Optional[str] = None):
        provider_cfg = config_loader._get_active_provider_config()
        self.location = provider_cfg.get("location", "us-east5")
        self.project = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project:
            raise ValueError("GOOGLE_CLOUD_PROJECT env var is not set")

        self.model = model or config_loader.get_model(
            default="meta/llama-4-maverick-17b-128e-instruct-maas"
        )

        self._cred, _ = google_auth_default(scopes=_SCOPES)
        self._auth_request = google.auth.transport.requests.Request()

        base_url = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project}/locations/{self.location}/endpoints/openapi"
        )
        self._refresh_credentials()
        self.client = OpenAI(base_url=base_url, api_key=self._cred.token)

    def _refresh_credentials(self):
        """Refresh the ADC token and update the client api_key."""
        self._cred.refresh(self._auth_request)
        if hasattr(self, "client"):
            self.client.api_key = self._cred.token

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        exc_str = str(exc)
        if "429" in exc_str or "503" in exc_str:
            return True
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status in (429, 503):
            return True
        return False

    def _call_with_retry(self, api_call, context_msg: str):
        for attempt in range(1, self.MAX_RETRIES + 1):
            self._refresh_credentials()
            try:
                return api_call()
            except Exception as e:
                if self._is_retryable(e) and attempt < self.MAX_RETRIES:
                    print(
                        f"    WARNING: Vertex AI API error on {context_msg} "
                        f"(attempt {attempt}/{self.MAX_RETRIES}). "
                        f"Waiting {self.RETRY_WAIT_SECONDS}s before retry..."
                    )
                    time.sleep(self.RETRY_WAIT_SECONDS)
                    continue
                raise

    @staticmethod
    def _strip_final_score(text: str) -> str:
        """Remove ``"final_score"`` entries from raw JSON text.

        During our experiments, Llama models were observed to emit
        arithmetic expressions (e.g. ``"final_score": 10 - 2 - 1``) as
        the value of ``final_score``, which are not valid JSON and cause
        ``json.loads`` to fail.  This function strips the offending
        entries so that the rest of the JSON can be parsed successfully.

        Note: regardless of whether the original ``final_score`` was
        valid JSON or not, ``_recalculate_final_score`` always
        recomputes it from ``initial_score`` and ``deductions``, so the
        LLM-provided value is never used.  This function exists solely
        to unblock JSON parsing when the value is malformed.
        """
        cleaned = re.sub(r'\s*"final_score"\s*:[^\n}]*[,\n]?\s*', "", text)
        cleaned = re.sub(r",(\s*})", r"\1", cleaned)
        return cleaned

    @staticmethod
    def _recalculate_final_score(parsed: dict) -> dict:
        """Recompute ``final_score`` as ``initial_score - sum(deductions)``.

        Always overwrites any ``final_score`` value the LLM may have
        returned, ensuring a consistent and trustworthy score.
        """
        if "initial_score" not in parsed or "deductions" not in parsed:
            return parsed
        score = parsed["initial_score"]
        for d in parsed.get("deductions", []):
            score -= d.get("points_deducted", 0)
        parsed["final_score"] = max(0, score)
        return parsed

    @staticmethod
    def _sanitize_response(text: str) -> str:
        """Extract the first complete JSON object from a response that may
        contain duplicated or extra data (common with Llama models)."""
        if not text:
            return text

        answer_match = re.search(
            r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL
        )
        if answer_match:
            text = answer_match.group(1)

        stripped = re.sub(r"^```(?:json)?\s*", "", text.strip())
        stripped = re.sub(r"\s*```\s*$", "", stripped)

        def _try_parse_and_finalize(candidate: str) -> Optional[str]:
            for attempt_text in (candidate, LlamaVertexClient._strip_final_score(candidate)):
                try:
                    parsed = json.loads(attempt_text)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "id" in parsed[0]:
                    return json.dumps({"objects": parsed}, ensure_ascii=False)
                if isinstance(parsed, dict):
                    parsed = LlamaVertexClient._recalculate_final_score(parsed)
                    return json.dumps(parsed, ensure_ascii=False, indent=2)
                return attempt_text
            return None

        result = _try_parse_and_finalize(stripped)
        if result is not None:
            return result

        start = stripped.find("{")
        if start == -1:
            return text
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                if in_string:
                    escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start:i + 1]
                    result = _try_parse_and_finalize(candidate)
                    if result is not None:
                        return result
                    break

        return text

    def analyze_text_only(self,
                         prompt: str,
                         max_tokens: Optional[int] = None) -> str:
        max_tokens = self._resolve_max_tokens(max_tokens)
        self._log_text_request(prompt, max_tokens)

        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]

        def _do_request() -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        try:
            raw_content = self._call_with_retry(_do_request, "analyze_text_only")
            if raw_content:
                content = self._sanitize_response(raw_content)
                print(f"    DEBUG: analyze_text_only response length: {len(content)}")
                print(f"    DEBUG: analyze_text_only response preview: {content[:100]}...")
            else:
                content = raw_content
                print(f"    DEBUG: analyze_text_only response is None or empty")
            return content
        except Exception as e:
            print(f"Error calling Vertex AI Llama API: {e}")
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

        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                messages[1]["content"].append(self._create_image_message(frame_path))
            else:
                print(f"Warning: Frame not found: {frame_path}")

        def _do_request() -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        try:
            raw_content = self._call_with_retry(_do_request, "analyze_frames")
            if raw_content:
                content = self._sanitize_response(raw_content)
                print(f"    DEBUG: analyze_frames response length: {len(content)}")
                print(f"    DEBUG: analyze_frames response preview: {content[:200]}...")
            else:
                content = raw_content
                print(f"    DEBUG: analyze_frames response is None or empty")
            return content
        except Exception as e:
            print(f"Error calling Vertex AI Llama API: {e}")
            print(f"    DEBUG: Model: {self.model}")
            print(f"    DEBUG: Prompt length: {len(prompt)} characters")
            raise
