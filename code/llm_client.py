"""Minimal OpenAI-compatible chat-completions client.

Uses only the standard library (urllib) so the submission has zero
third-party runtime dependencies. Works against any OpenAI-compatible
`/chat/completions` endpoint, including for vision-capable models (image
content blocks with a base64 data URL) — that covers both Featherless AI
and Gemini's OpenAI-compatibility layer
(https://generativelanguage.googleapis.com/v1beta/openai), which is why
this repo can point the vision step at Gemini and the text step at
Featherless with the same client class.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from config import ProviderConfig


class LLMCallError(Exception):
    pass


@dataclass
class ChatResult:
    content: str
    input_tokens: int
    output_tokens: int
    model: str


_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')


class RateLimiter:
    """Sliding-window client-side throttle so we stay under a provider's
    free-tier requests-per-minute cap instead of discovering it via 429s.
    """

    def __init__(self, requests_per_minute: Optional[int]):
        self.requests_per_minute = requests_per_minute
        self._lock = threading.Lock()
        self._calls: deque[float] = deque()

    def wait(self) -> None:
        if not self.requests_per_minute:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
                if len(self._calls) < self.requests_per_minute:
                    self._calls.append(now)
                    return
                sleep_for = 60.0 - (now - self._calls[0]) + 0.05
            time.sleep(max(sleep_for, 0.05))


class OpenAICompatibleClient:
    def __init__(self, config: ProviderConfig, timeout: float = 60.0, max_retries: int = 5):
        self.config = config
        self.timeout = timeout
        self.max_retries = max_retries
        self._rate_limiter = RateLimiter(config.requests_per_minute)

    def _post(self, path: str, payload: dict) -> dict:
        if not self.config.is_configured:
            raise LLMCallError(
                f"No API key configured for provider '{self.config.provider}'. "
                "Add it to a local .env file (see .env.example) or export it "
                "in your shell before running any step that needs it."
            )
        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
                # Some providers (e.g. Featherless, behind Cloudflare) reject
                # the default "Python-urllib/x.y" user agent as a bot
                # signature (HTTP 403, Cloudflare error code 1010).
                "User-Agent": "Mozilla/5.0 (compatible; buy-or-wait-hackathon/1.0)",
            },
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._rate_limiter.wait()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    delay_match = _RETRY_DELAY_RE.search(body)
                    delay = float(delay_match.group(1)) + 2 if delay_match else min(2 ** attempt, 10)
                    time.sleep(min(delay, 90))
                    last_error = e
                    continue
                raise LLMCallError(f"{self.config.provider} HTTP {e.code}: {body}") from e
            except urllib.error.URLError as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 10))
                    continue
                raise LLMCallError(f"{self.config.provider} request failed: {e}") from e
        raise LLMCallError(f"{self.config.provider} request failed after retries: {last_error}")

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format_json: bool = False,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}

        raw = self._post("chat/completions", payload)
        choice = raw["choices"][0]
        content = choice["message"]["content"]
        usage = raw.get("usage", {}) or {}
        return ChatResult(
            content=content,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            model=raw.get("model", self.config.model),
        )

    def chat_text(self, system_prompt: str, user_prompt: str, **kwargs) -> ChatResult:
        return self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )

    def chat_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str | Path,
        **kwargs,
    ) -> ChatResult:
        image_bytes = Path(image_path).read_bytes()
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        return self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            **kwargs,
        )
