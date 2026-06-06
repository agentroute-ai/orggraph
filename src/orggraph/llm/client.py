"""LLM client wrapper that handles Ollama vs vLLM quirks transparently.

Both backends speak OpenAI-compatible HTTP, but:
  - Ollama's /v1 shim ignores `response_format` and silently truncates
    context to 2048 tokens. We pass `extra_body.options` (Ollama-native
    fields) for num_ctx, num_predict, format=json.
  - vLLM honors OpenAI's `response_format=json_object`.

Detection is by URL: port 11434 = Ollama, anything else = vLLM.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from orggraph.llm.parsing import extract_json


def detect_backend(base_url: str) -> str:
    """Return 'ollama' or 'vllm' based on the URL."""
    if ":11434" in base_url:
        return "ollama"
    return "vllm"


class LLMClient:
    """OpenAI-compatible chat completion that returns parsed JSON."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 600.0):
        self.base_url = base_url
        self.backend = detect_backend(base_url)
        self.timeout = timeout
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def json_chat(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 4096,
    ) -> dict[str, Any] | None:
        """Call the chat endpoint with JSON-mode output. Returns parsed dict or None."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
            "timeout": self.timeout,
            "max_tokens": max_tokens,
        }

        if self.backend == "ollama":
            kwargs["extra_body"] = {
                "options": {
                    "num_ctx": 16384,
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "seed": seed,
                    "format": "json",
                }
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = self._client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
        except Exception:  # noqa: BLE001 — caller wants None, not exceptions
            return None

        if not raw:
            return None
        return extract_json(raw)
