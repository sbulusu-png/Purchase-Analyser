"""Direct unit tests for MessageSignalStore's conflict resolution when two
messages target the same event_id (resolve.py). AGENTS.md Sec 6.3 requires
"newer records from the same source" to win among otherwise-equal explicit
signals (amend/cancel/delay); this dataset happens to have zero real
occurrences of this conflict (verified by inspection), so there is no
sample/full-dataset run that would ever exercise this path -- these
synthetic cases are the only coverage it gets. Plain asserts, no test
framework, consistent with the project's zero-dependency policy: run with
`python3 code/test_resolve.py`.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from resolve import MessageSignalStore


def _dataset(messages: list[tuple[str, str]]) -> SimpleNamespace:
    return SimpleNamespace(messages=[SimpleNamespace(message_id=mid, sent_at=sent_at) for mid, sent_at in messages])


def _store(cache: dict, dataset: SimpleNamespace) -> MessageSignalStore:
    tmp_dir = Path(tempfile.mkdtemp())
    (tmp_dir / "message_signals.json").write_text(json.dumps(cache), encoding="utf-8")
    return MessageSignalStore(tmp_dir, dataset)


def test_newer_message_wins_when_cache_order_matches_time_order():
    cache = {
        "message_01": {"intent": "amend_event", "target_event_id": "event_500", "new_amount": 100},
        "message_02": {"intent": "cancel_event", "target_event_id": "event_500", "new_amount": None},
    }
    dataset = _dataset([("message_01", "2025-01-01T00:00:00Z"), ("message_02", "2025-06-01T00:00:00Z")])
    store = _store(cache, dataset)
    assert store.signal_for("event_500")["intent"] == "cancel_event"


def test_newer_message_wins_even_when_cache_order_is_reversed():
    # Same facts as above, but the older message appears LAST in the cache
    # dict -- proves the tiebreak is sent_at, not "last one iterated".
    cache = {
        "message_02": {"intent": "cancel_event", "target_event_id": "event_500", "new_amount": None},
        "message_01": {"intent": "amend_event", "target_event_id": "event_500", "new_amount": 100},
    }
    dataset = _dataset([("message_01", "2025-01-01T00:00:00Z"), ("message_02", "2025-06-01T00:00:00Z")])
    store = _store(cache, dataset)
    assert store.signal_for("event_500")["intent"] == "cancel_event"


def test_older_message_id_but_newer_sent_at_still_wins():
    # Mirrors the real dataset quirk: message_02 (lower id) is sent AFTER
    # message_01 (higher id would otherwise win under old "last in cache"
    # logic if it happened to be inserted second).
    cache = {
        "message_01": {"intent": "cancel_event", "target_event_id": "event_777", "new_amount": None},
        "message_02": {"intent": "amend_event", "target_event_id": "event_777", "new_amount": 250},
    }
    dataset = _dataset([("message_02", "2019-08-31T09:30:00Z"), ("message_01", "2025-07-29T09:30:00Z")])
    store = _store(cache, dataset)
    assert store.signal_for("event_777")["intent"] == "cancel_event"


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} message-conflict-resolution tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
