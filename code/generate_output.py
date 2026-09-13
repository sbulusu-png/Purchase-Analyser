"""Phase 6 — final output.csv assembly.

Runs the full decision pipeline (Phase 3 forecast -> Phase 4 plan
selection -> Phase 5 explanation) over every row in dataset/requests.csv
(the 250 evaluation requests -- sample_requests.csv is public/solved
context, not something to predict, per README.md) and writes the
required columns to output.csv at the repo root.

Usage:
    python3 code/generate_output.py             # deterministic explanations only, no LLM calls
    python3 code/generate_output.py --explain    # also apply the LLM explanation polish (cached)

Deterministic by default (AGENTS.md: "keep behavior deterministic where
possible") since explain.py's LLM step is optional and every number it
produces must already match the deterministic sentence it started from
(see explain.py) -- so --explain changes wording, never the graded
numeric/categorical fields.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from config import load_app_config
from decide import decide
from explain import ExplanationCache
from llm_client import OpenAICompatibleClient
from loaders import Dataset
from resolve import MessageSignalStore, VisionAmountStore
from usage_tracker import UsageTracker

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = REPO_ROOT / "code" / ".cache"
OUTPUT_PATH = REPO_ROOT / "output.csv"

HEADER = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--explain", action="store_true", help="Apply the cached/LLM explanation polish.")
    args = parser.parse_args()

    print(f"Loading dataset from {DATASET_DIR} ...")
    dataset = Dataset.load(DATASET_DIR)
    vision_store = VisionAmountStore(CACHE_DIR)
    message_store = MessageSignalStore(CACHE_DIR)

    explain_client = None
    explain_tracker = None
    explain_cache = None
    if args.explain:
        config = load_app_config()
        explain_cache = ExplanationCache(CACHE_DIR / "explanations.json")
        if not config.text.is_configured:
            print(
                "NOTE: FEATHERLESS_API_KEY is not set -- requests already in "
                "code/.cache/explanations.json will still use their cached LLM wording; "
                "any request without a cache entry falls back to the deterministic "
                "explanation instead of failing (see explain.py's grounding fallback).",
                file=sys.stderr,
            )
        explain_client = OpenAICompatibleClient(config.text)
        explain_tracker = UsageTracker()

    print(f"Deciding {len(dataset.requests)} requests ...")
    rows = []
    errors = 0
    for req in dataset.requests:
        try:
            row = decide(dataset, req, vision_store, message_store, explain_client, explain_tracker, explain_cache)
            rows.append(row)
        except Exception as e:  # noqa: BLE001 - surfaced per-row, doesn't abort the run
            errors += 1
            print(f"  ERROR {req.request_id}: {e}", file=sys.stderr)

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for row in rows:
            writer.writerow(row.to_csv_row())

    print(f"\nWrote {len(rows)} rows to {OUTPUT_PATH} ({errors} errors).")
    if explain_tracker:
        totals = explain_tracker.totals_by_model()
        for model, t in totals.items():
            print(f"  explanation usage: {model}: {t['calls']} calls, {t['input_tokens']} in, {t['output_tokens']} out")

    if len(rows) != len(dataset.requests):
        print("WARNING: row count does not match dataset/requests.csv.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
