"""Environment-driven configuration. Never hardcode secrets or endpoints
that require a key — everything here is read from the environment so the
same code runs for any contributor without editing source.

Both the vision (blank-amount image extraction) and text (message signal
parsing) steps default to Featherless AI, via its OpenAI-compatible
chat/completions endpoint (see llm_client.py). A Gemini vision path is
still available as an override, but Featherless has no daily quota (only
a per-account concurrency-unit limit, handled below), so it's the
reliable default for a time-boxed hackathon run.

    FEATHERLESS_API_KEY          - your Featherless AI API key
    FEATHERLESS_BASE_URL         - default: https://api.featherless.ai/v1
    FEATHERLESS_TEXT_MODEL       - default: Qwen/Qwen2.5-7B-Instruct
    FEATHERLESS_MODEL            - generic fallback for FEATHERLESS_TEXT_MODEL
    FEATHERLESS_VISION_MODEL     - default: Qwen/Qwen3-VL-30B-A3B-Instruct
    FEATHERLESS_MAX_CONCURRENCY  - default: 1 (a 4-unit plan running a
                                    4-unit-per-request model can only do 1
                                    request at a time — check your plan's
                                    concurrency-unit limit vs. the model's
                                    per-request cost before raising this)
    FEATHERLESS_VISION_MAX_CONCURRENCY - default: 1, independent of the
                                    text concurrency above
    FEATHERLESS_MAX_RPM          - default: unset (no client-side throttle)

Optional Gemini override for the vision step (its free tier caps at a
handful of requests per minute AND ~20/day, so it stalls easily):
    GEMINI_API_KEY          - your Gemini API key (aistudio.google.com/apikey)
    GEMINI_BASE_URL         - default: https://generativelanguage.googleapis.com/v1beta/openai
    GEMINI_VISION_MODEL     - default: gemini-3.6-flash
    GEMINI_MAX_CONCURRENCY  - default: 1
    GEMINI_MAX_RPM          - default: 4
    VISION_PROVIDER=gemini  - set this to actually route vision to Gemini
                              instead of the Featherless default

Optional Groq override for the vision step (fast inference, no daily cap
seen in practice on the free tier — a good fallback when Gemini's daily
quota is exhausted):
    GROQ_API_KEY            - your Groq API key (console.groq.com/keys)
    GROQ_BASE_URL           - default: https://api.groq.com/openai/v1
    GROQ_VISION_MODEL       - default: meta-llama/llama-4-scout-17b-16e-instruct
    GROQ_MAX_CONCURRENCY    - default: 2
    GROQ_MAX_RPM            - default: unset (no client-side throttle)
    VISION_PROVIDER=groq    - set this to actually route vision to Groq

Set these in a local `.env` file (see `.env.example`) or your shell —
never commit real values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


# Load a .env next to the repo root if present, without overriding
# variables already set in the real environment.
_load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    # tolerate a stray trailing quote from a malformed .env line
    cleaned = value.strip().strip('"').strip("'")
    try:
        return max(1, int(cleaned))
    except ValueError:
        return default


def _parse_optional_int(value: str | None) -> Optional[int]:
    if value is None:
        return None
    cleaned = value.strip().strip('"').strip("'")
    if cleaned == "":
        return None
    try:
        return max(1, int(cleaned))
    except ValueError:
        return None


@dataclass(frozen=True)
class ProviderConfig:
    provider: str  # short label used in usage tracking, e.g. "gemini" / "featherless"
    api_key: str | None
    base_url: str
    model: str
    max_concurrency: int = 1
    requests_per_minute: Optional[int] = None  # None = no client-side throttle

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class AppConfig:
    vision: ProviderConfig
    text: ProviderConfig


def load_app_config() -> AppConfig:
    featherless_base_url = os.environ.get("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")
    featherless_rpm = _parse_optional_int(os.environ.get("FEATHERLESS_MAX_RPM"))

    vision_provider = os.environ.get("VISION_PROVIDER", "featherless").lower()
    if vision_provider == "gemini":
        vision = ProviderConfig(
            provider="gemini",
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url=os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
            model=os.environ.get("GEMINI_VISION_MODEL", "gemini-3.6-flash"),
            max_concurrency=_parse_int(os.environ.get("GEMINI_MAX_CONCURRENCY"), 1),
            requests_per_minute=_parse_optional_int(os.environ.get("GEMINI_MAX_RPM")) or 4,
        )
    elif vision_provider == "groq":
        vision = ProviderConfig(
            provider="groq",
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            model=os.environ.get("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
            max_concurrency=_parse_int(os.environ.get("GROQ_MAX_CONCURRENCY"), 2),
            requests_per_minute=_parse_optional_int(os.environ.get("GROQ_MAX_RPM")),
        )
    else:
        vision = ProviderConfig(
            provider="featherless",
            api_key=os.environ.get("FEATHERLESS_API_KEY"),
            base_url=featherless_base_url,
            model=os.environ.get("FEATHERLESS_VISION_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
            max_concurrency=_parse_int(os.environ.get("FEATHERLESS_VISION_MAX_CONCURRENCY"), 1),
            requests_per_minute=featherless_rpm,
        )

    text_model = (
        os.environ.get("FEATHERLESS_TEXT_MODEL")
        or os.environ.get("FEATHERLESS_MODEL")
        or "Qwen/Qwen2.5-7B-Instruct"
    )
    text = ProviderConfig(
        provider="featherless",
        api_key=os.environ.get("FEATHERLESS_API_KEY"),
        base_url=featherless_base_url,
        model=text_model,
        max_concurrency=_parse_int(os.environ.get("FEATHERLESS_MAX_CONCURRENCY"), 1),
        requests_per_minute=featherless_rpm,
    )
    return AppConfig(vision=vision, text=text)
