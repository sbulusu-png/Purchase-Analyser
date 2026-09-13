"""Buy or Wait? — pipeline entry point.

Current status: Phase 1 (data models/loaders/currency) and Phase 2
(multimodal extraction: blank-amount image reading + message signal
parsing) are implemented. Phases 3-7 (simulation engine, plan ranking,
explanation generation, output assembly, usage report) are not wired up
yet, so this does NOT produce output.csv yet.

Usage:
    python3 code/main.py                 # dry run: report scope only, no LLM calls, no keys needed
    python3 code/main.py --run           # actually call the LLM providers for extraction
    python3 code/main.py --run --limit-messages 10 --limit-images 2   # cheap smoke test

Requires no third-party packages. For --run, set GEMINI_API_KEY (used for
the vision/blank-amount step) and FEATHERLESS_API_KEY (used for the
message-parsing step) in a local `.env` file — see `.env.example`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Extracted note/summary text from the LLMs can contain non-ASCII currency
# symbols (e.g. Rupee sign) that the default Windows console codepage
# (cp1252) cannot encode; replace rather than crash on print.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config import load_app_config
from extraction import Phase2Extractor
from llm_client import OpenAICompatibleClient
from loaders import Dataset
from usage_tracker import UsageTracker

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = REPO_ROOT / "code" / ".cache"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="Actually call Featherless AI instead of a dry run.")
    parser.add_argument("--limit-messages", type=int, default=None, help="Only process the first N messages.")
    parser.add_argument("--limit-images", type=int, default=None, help="Only process the first N blank-amount events.")
    args = parser.parse_args()

    print(f"Loading dataset from {DATASET_DIR} ...")
    dataset = Dataset.load(DATASET_DIR)
    blank_events = [e for e in dataset.events if e.is_blank_amount]
    print(
        f"Loaded: {len(dataset.requests)} requests, {len(dataset.sample_requests)} sample requests, "
        f"{len(dataset.profiles)} profiles, {len(dataset.events)} events "
        f"({len(blank_events)} with a blank amount), {len(dataset.payment_options)} payment options, "
        f"{len(dataset.messages)} messages, {len(dataset.images)} images, "
        f"{len(dataset.exchange_rates)} exchange rate rows."
    )

    missing_images = [e for e in blank_events if e.event_id not in dataset.image_by_event]
    if missing_images:
        print(f"WARNING: {len(missing_images)} blank-amount events have no linked image: "
              f"{[e.event_id for e in missing_images]}")

    config = load_app_config()

    if not args.run:
        print("\nDry run (no LLM calls). Would process:")
        print(f"  - {len(blank_events)} blank-amount events via vision extraction")
        print(f"  - {len(dataset.messages)} messages via text signal extraction")
        print(f"  vision: {config.vision.provider} / {config.vision.model} "
              f"(configured: {config.vision.is_configured})")
        print(f"  text:   {config.text.provider} / {config.text.model} "
              f"(configured: {config.text.is_configured})")
        print("\nRun again with --run once both providers' API keys are set to actually extract.")
        return 0

    if not config.vision.is_configured or not config.text.is_configured:
        print(
            f"ERROR: --run requires an API key for the vision provider ({config.vision.provider}) "
            f"and the text provider ({config.text.provider}) to both be set (in .env or the environment).",
            file=sys.stderr,
        )
        return 1

    vision_client = OpenAICompatibleClient(config.vision)
    text_client = OpenAICompatibleClient(config.text)
    tracker = UsageTracker()
    extractor = Phase2Extractor(dataset, vision_client, text_client, config, tracker, CACHE_DIR)

    events_to_run = blank_events if args.limit_images is None else blank_events[: args.limit_images]
    print(f"\nRunning vision extraction on {len(events_to_run)} blank-amount events "
          f"(concurrency={config.vision.max_concurrency}, rpm cap={config.vision.requests_per_minute}) ...")
    vision_errors = 0
    for event, result, error in extractor.run_vision_phase(events_to_run):
        if error is not None:
            vision_errors += 1
            print(f"  [ERROR] {event.event_id}: {error}")
            continue
        status = "OK" if result.amount is not None and result.confidence >= 0.5 else "LOW-CONFIDENCE/NULL"
        print(f"  [{status}] {event.event_id} -> {result.amount} {result.currency} "
              f"(confidence={result.confidence:.2f}) {result.note}")
        if result.amount is None:
            vision_errors += 1

    messages_to_run = dataset.messages if args.limit_messages is None else dataset.messages[: args.limit_messages]
    print(f"\nRunning message signal extraction on {len(messages_to_run)} messages "
          f"(concurrency={config.text.max_concurrency}) ...")
    message_errors = 0
    for message, signal, error in extractor.run_message_phase(messages_to_run):
        if error is not None:
            message_errors += 1
            print(f"  [ERROR] {message.message_id}: {error}")
            continue
        print(f"  [{signal.intent:14s}] {message.message_id} target={signal.target_event_id} "
              f"amount={signal.new_amount} date={signal.new_date} conf={signal.confidence:.2f}")

    usage_path = CACHE_DIR / "usage_log.json"
    tracker.save_json(usage_path)
    totals = tracker.totals_by_model()
    print(f"\nToken usage this run (excludes cache hits), saved to {usage_path}:")
    for model, t in totals.items():
        print(f"  {model}: {t['calls']} calls, {t['input_tokens']} input tokens, {t['output_tokens']} output tokens")

    if vision_errors or message_errors:
        print(f"\nCompleted with {vision_errors} vision issues and {message_errors} message errors.")
        return 1

    print("\nPhase 2 extraction complete. Results cached in code/.cache/ for reuse by later phases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
