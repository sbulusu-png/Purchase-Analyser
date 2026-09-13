"""Tiny JSON-file cache so repeated runs don't re-spend LLM tokens.

Keyed by a caller-chosen string (e.g. "vision:event_253" or
"message:message_014"). Safe to delete at any time to force fresh calls.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional


class JsonCache:
    """Thread-safe: extraction runs multiple LLM calls concurrently
    (see FEATHERLESS_MAX_CONCURRENCY), and all of them read/write this
    cache, so every access goes through a lock.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                self._data: dict[str, Any] = json.load(f)
        else:
            self._data = {}

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._flush()

    def _flush(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False, sort_keys=True)
