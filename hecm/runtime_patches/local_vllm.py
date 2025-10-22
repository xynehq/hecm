# hecm/runtime_patches/local_vllm.py
"""
Runtime patch to make cursor_agent_tools.factory.create_agent return a
LocalVLLMAgent for models starting with "archit11/". This allows you to
run local vLLM instances that expose an OpenAI-compatible HTTP API (chat/completions).

Features:
- Reads LOCAL_VLLM_URL env var (defaults to http://localhost:8005/v1)
- Async chat with basic retry/backoff for 5xx/connection errors
- Multiple response-shape extraction heuristics (OpenAI-style and variants)
- Minimal implementations of BaseAgent abstract methods so it can be instantiated
- Idempotent patching (won't override original twice)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

import cursor_agent_tools.factory as factory
from cursor_agent_tools.base import BaseAgent
from cursor_agent_tools.logger import get_logger

logger = get_logger(__name__)

# Environment-configurable base URL for local vLLM
DEFAULT_LOCAL_VLLM_URL = "http://localhost:8005/v1"
LOCAL_VLLM_URL = os.getenv("LOCAL_VLLM_URL", DEFAULT_LOCAL_VLLM_URL).rstrip("/")


class LocalVLLMAgent(BaseAgent):
    """
    Minimal Local vLLM agent implementing BaseAgent interface enough for tests.
    Communicates with an OpenAI-compatible local server (chat/completions).
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        timeout: int = 180,
        temperature: float = 0.0,
        default_tool_timeout: int = 300,
        permission_callback: Optional[Any] = None,
        permission_options: Optional[Any] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        **kwargs: Any,
    ):
        # Keep attributes other parts of code may expect
        self.model = model
        self.base_url = (base_url or LOCAL_VLLM_URL).rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.default_tool_timeout = default_tool_timeout
        self.permission_callback = permission_callback
        self.permission_options = permission_options
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._tools: List[Any] = []
        self._extra = kwargs

        logger.info(f"LocalVLLMAgent init model={self.model} base_url={self.base_url}")

    # ----------------------
    # Chat API (async)
    # ----------------------
    async def chat(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Send a chat request to the local vLLM and return a normalized dict:
        { "message": str, "raw": <original-json>, "tool_calls": [], "thinking": None }
        """
        import httpx  # local import to avoid mandatory dependency if not used elsewhere

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", 512),
        }

        # Allow caller to override path if their vLLM uses a different route
        endpoint_override = kwargs.get("endpoint_path")
        if endpoint_override:
            url = f"{self.base_url.rstrip('/')}/{endpoint_override.lstrip('/')}"
        else:
            # Default to OpenAI-style chat/completions
            url = f"{self.base_url}/chat/completions"

        attempt = 0
        last_exc = None
        while attempt < self.max_retries:
            attempt += 1
            try:
                timeout = httpx.Timeout(self.timeout)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, json=payload)
                    text = resp.text
                    status = resp.status_code

                    if status >= 500:
                        # server error; retry
                        logger.warning(f"LocalVLLM server error {status}, attempt {attempt}/{self.max_retries}")
                        await asyncio.sleep(self.backoff_factor * attempt)
                        continue

                    if status >= 400:
                        # client error — probably unrecoverable; return structured error
                        logger.error(f"LocalVLLM returned {status}: {text}")
                        return {
                            "message": f"Error: local vLLM returned {status}: {text}",
                            "tool_calls": [],
                            "thinking": None,
                            "raw": {"status_code": status, "body": text},
                        }

                    # success
                    j = resp.json()
                    assistant_msg = self._extract_assistant_text(j)
                    return {"message": assistant_msg, "raw": j, "tool_calls": [], "thinking": None}

            except Exception as exc:
                last_exc = exc
                logger.warning(f"LocalVLLM request failed (attempt {attempt}): {exc}")
                await asyncio.sleep(self.backoff_factor * attempt)
                continue

        # exhausted retries
        logger.error(f"LocalVLLM request failed after {self.max_retries} attempts: {last_exc}")
        return {
            "message": f"Error: local vLLM request failed after {self.max_retries} attempts: {last_exc}",
            "tool_calls": [],
            "thinking": None,
            "raw": None,
        }

    def chat_sync(self, prompt: str, **kwargs) -> Dict[str, Any]:
        "Synchronous wrapper for code that expects a blocking call."
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in this thread
            loop = None

        if loop and loop.is_running():
            # Running inside an event loop — run in a new task and wait
            return asyncio.run(self.chat(prompt, **kwargs))
        else:
            return asyncio.get_event_loop().run_until_complete(self.chat(prompt, **kwargs))

    # ----------------------
    # Helper: extract assistant text from various shapes
    # ----------------------
    @staticmethod
    def _extract_assistant_text(resp_json: Any) -> str:
        """
        Try multiple common response shapes:
         - OpenAI chat: choices[0].message.content
         - OpenAI legacy/text: choices[0].text
         - other APIs: result, output, data, etc.
        Return a string (may be JSON-stringified fallback).
        """
        try:
            # OpenAI / OpenAI-compatible chat
            choices = resp_json.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                if isinstance(first, dict):
                    # modern chat shape
                    msg = first.get("message")
                    if isinstance(msg, dict) and "content" in msg:
                        # content can be string or dict
                        content = msg.get("content")
                        return content if isinstance(content, str) else json.dumps(content)
                    # legacy text shape
                    if "text" in first:
                        return first["text"]
                    # sometimes 'delta' streaming fragments
                    if "delta" in first and isinstance(first["delta"], dict):
                        return first["delta"].get("content", json.dumps(first["delta"]))

            # other common shapes
            if "result" in resp_json:
                r = resp_json["result"]
                if isinstance(r, list) and r:
                    return r[0].get("content") or json.dumps(r[0])
                if isinstance(r, dict):
                    return r.get("content") or json.dumps(r)

            if "output" in resp_json:
                out = resp_json["output"]
                if isinstance(out, str):
                    return out
                if isinstance(out, list) and out:
                    return out[0]
                if isinstance(out, dict) and "text" in out:
                    return out["text"]

            if "data" in resp_json and isinstance(resp_json["data"], dict):
                # some servers embed textual result under data
                for k in ("text", "content", "message"):
                    if k in resp_json["data"]:
                        return resp_json["data"][k]

        except Exception:
            # fail silently to fallback
            pass

        # Fallback: pretty JSON (trimmed)
        try:
            s = json.dumps(resp_json)
            return s[:4000]  # trim to avoid huge strings
        except Exception:
            return str(resp_json)[:4000]

    # ----------------------
    # Minimal BaseAgent abstract methods (safe stubs)
    # ----------------------
    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        For tests, we just return an empty list. Implement real tool execution if required.
        """
        logger.debug("LocalVLLMAgent._execute_tool_calls called; returning empty results")
        return []

    def _generate_system_prompt(self, *args, **kwargs) -> str:
        """Return a simple system prompt compatible with pipelines that call it."""
        return "You are a local vLLM used for testing. Keep responses concise."

    def _prepare_tools(self, tool_configs: Optional[List[Dict[str, Any]]] = None) -> None:
        """No tools prepared by default; keep a copy if provided."""
        self._tools = [] if tool_configs is None else list(tool_configs or [])

    def get_structured_output(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw output to a small structured dict expected by some code paths."""
        text = raw_output.get("message") if isinstance(raw_output, dict) else str(raw_output)
        return {"text": text, "raw": raw_output}

    async def query_image(self, image_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """Image queries are not supported in this minimal runtime patch."""
        return {"error": "LocalVLLMAgent: query_image not implemented in runtime patch."}

    def close(self) -> None:
        """No persistent connections to close in this minimal agent."""
        return None


# ----------------------
# Factory patching
# ----------------------
# Replace the patched_create_agent function in hecm/runtime_patches/local_vllm.py with this:

def patched_create_agent(*args, **kwargs):
    """
    If model starts with archit11/, return LocalVLLMAgent.
    Accepts either positional model or model kwarg.
    This version pops handled keys out of kwargs to avoid passing duplicates.
    """
    model = kwargs.pop("model", None) or (args[0] if args else None)
    if isinstance(model, str) and model.startswith("archit11/"):
        logger.info(f"[patched factory] Using LocalVLLMAgent for model: {model}")

        # Compute base_url first (pop to avoid duplicates)
        base_url = kwargs.pop("base_url", None) or kwargs.pop("host", None) or os.getenv("LOCAL_VLLM_URL", LOCAL_VLLM_URL)

        # Pop commonly passed-in args so they are not also present in kwargs
        temperature = kwargs.pop("temperature", 0.0)
        timeout = kwargs.pop("timeout", 180)
        default_tool_timeout = kwargs.pop("default_tool_timeout", 300)
        permission_callback = kwargs.pop("permission_callback", None)
        permission_options = kwargs.pop("permissions", None)
        max_retries = kwargs.pop("max_retries", 3)
        backoff_factor = kwargs.pop("backoff_factor", 0.5)

        # Any remaining kwargs are safe to forward
        ctor_kwargs = dict(kwargs)  # shallow copy of leftovers

        return LocalVLLMAgent(
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            default_tool_timeout=default_tool_timeout,
            permission_callback=permission_callback,
            permission_options=permission_options,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            **ctor_kwargs,
        )

    # default behavior
    return factory._orig_create_agent(*args, **kwargs)



# Install the patch idempotently
if not hasattr(factory, "_orig_create_agent"):
    factory._orig_create_agent = factory.create_agent
    factory.create_agent = patched_create_agent
    logger.info("Patched cursor_agent_tools.factory.create_agent -> LocalVLLMAgent")
else:
    logger.debug("LocalVLLM patch already installed (skipping)")

# Expose create_agent for convenience (importing this module will do the patch)
from cursor_agent_tools.factory import create_agent  # noqa: E402,F401
