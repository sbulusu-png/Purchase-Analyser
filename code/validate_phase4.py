"""Validates the full Phase 4 decision (decide.py) against
dataset/sample_requests.csv's reference output columns."""

from __future__ import annotations

import csv
from pathlib import Path

from decide import decide
from loaders import Dataset
from resolve import MessageSignalStore, VisionAmountStore

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = REPO_ROOT / "code" / ".cache"


def main() -> int:
    dataset = Dataset.load(DATASET_DIR)
    vision_store = VisionAmountStore(CACHE_DIR)
    message_store = MessageSignalStore(CACHE_DIR, dataset)

    with (DATASET_DIR / "sample_requests.csv").open(encoding="utf-8-sig") as f:
        raw_by_id = {r["request_id"]: r for r in csv.DictReader(f)}

    status_match = 0
    method_match = 0
    plan_match = 0
    date_match = 0
    total = 0

    for req in dataset.sample_requests:
        total += 1
        ref = raw_by_id[req.request_id]
        row = decide(dataset, req, vision_store, message_store)

        s_ok = row.affordability_status == ref["affordability_status"]
        m_ok = row.recommended_payment_method == ref["recommended_payment_method"]
        p_ok = row.payment_plan == ref["payment_plan"]
        d_ok = row.earliest_date_for_full_payment == ref["earliest_date_for_full_payment"]
        status_match += s_ok
        method_match += m_ok
        plan_match += p_ok
        date_match += d_ok

        flag = "OK" if (s_ok and m_ok) else "DIFF"
        print(f"{req.request_id:12} status: {row.affordability_status:20} ref: {ref['affordability_status']:20} "
              f"method: {row.recommended_payment_method:15} ref: {ref['recommended_payment_method']:15} {flag}")
        if not p_ok:
            print(f"    our_plan: {row.payment_plan}")
            print(f"    ref_plan: {ref['payment_plan']}")
        if row.spending_changes_needed != ref["spending_changes_needed"]:
            print(f"    our_changes: {row.spending_changes_needed}  ref_changes: {ref['spending_changes_needed']}")

    print(f"\nstatus match: {status_match}/{total}, method match: {method_match}/{total}, "
          f"plan match: {plan_match}/{total}, date match: {date_match}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
