"""Accumulates LLM call/token usage so evaluation/usage_report.md (Phase 7)
can be generated straight from a real run instead of estimated by hand.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class UsageRecord:
    provider: str
    model: str
    call_type: str  # "vision_extract" | "message_parse" | "explanation" | ...
    request_key: str  # e.g. event_id or message_id, for traceability
    input_tokens: int
    output_tokens: int
    cached: bool = False


@dataclass
class UsageTracker:
    records: list[UsageRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(
        self,
        provider: str,
        model: str,
        call_type: str,
        request_key: str,
        input_tokens: int,
        output_tokens: int,
        cached: bool = False,
    ) -> None:
        with self._lock:
            self.records.append(
                UsageRecord(
                    provider=provider,
                    model=model,
                    call_type=call_type,
                    request_key=request_key,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached=cached,
                )
            )

    def totals_by_model(self) -> dict[str, dict[str, int]]:
        totals: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        for r in self.records:
            if r.cached:
                continue
            t = totals[r.model]
            t["calls"] += 1
            t["input_tokens"] += r.input_tokens
            t["output_tokens"] += r.output_tokens
        return dict(totals)

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)
