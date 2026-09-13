"""Validates the Phase 3 simulator against dataset/sample_requests.csv,
which has hand-worked/reference output columns. Compares our computed
amount_safe_to_pay and earliest_date_for_full_payment (the two values the
simulator alone is responsible for, independent of plan/method selection
which is Phase 4) against the sample's values and reports the diff.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from loaders import Dataset
from resolve import MessageSignalStore, VisionAmountStore, resolve_user_events
from simulate import amount_safe_to_pay, build_daily_forecast, earliest_date_for_full_payment

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = REPO_ROOT / "code" / ".cache"


def main() -> int:
    dataset = Dataset.load(DATASET_DIR)
    vision_store = VisionAmountStore(CACHE_DIR)
    message_store = MessageSignalStore(CACHE_DIR)

    total = 0
    amount_close = 0
    date_exact = 0
    date_both_empty = 0
    max_abs_err = 0.0
    rows = []

    for req in dataset.sample_requests:
        total += 1
        profile = dataset.profile_by_user[req.user_id]
        request_date = date.fromisoformat(req.request_date)
        forecast_end = request_date.fromordinal(request_date.toordinal() + 90)

        forward_events, warnings = resolve_user_events(
            dataset, req.user_id, profile, request_date, forecast_end,
            vision_store, message_store,
        )
        forecast = build_daily_forecast(profile.current_available_balance, forward_events, request_date)

        computed_safe = amount_safe_to_pay(forecast, profile.minimum_balance_to_keep, req.requested_amount)
        computed_date = earliest_date_for_full_payment(forecast, profile.minimum_balance_to_keep, req.requested_amount)

        # sample_requests.csv rows carry the reference output columns via
        # the raw CSV (Request.from_row doesn't parse them), so re-read directly.
        rows.append((req, computed_safe, computed_date, warnings))

    # Re-read raw sample rows to get the actual reference output columns.
    with (DATASET_DIR / "sample_requests.csv").open(encoding="utf-8-sig") as f:
        raw_by_id = {r["request_id"]: r for r in csv.DictReader(f)}

    print(f"{'request_id':12} {'req_amt':>12} {'our_safe':>12} {'ref_safe':>12} {'diff':>10}  "
          f"{'our_date':10} {'ref_date':10}  match")
    for req, computed_safe, computed_date, warnings in rows:
        ref = raw_by_id[req.request_id]
        ref_safe = float(ref["amount_safe_to_pay"])
        ref_date = ref["earliest_date_for_full_payment"] or None
        our_date_str = computed_date.isoformat() if computed_date else ""
        diff = computed_safe - ref_safe
        max_abs_err = max(max_abs_err, abs(diff))
        amt_ok = abs(diff) <= max(1.0, 0.01 * ref_safe)
        if amt_ok:
            amount_close += 1
        date_ok = (our_date_str == (ref_date or ""))
        if date_ok:
            date_exact += 1
            if not ref_date:
                date_both_empty += 1
        flag = "OK" if amt_ok and date_ok else ("AMT" if not amt_ok else "DATE")
        print(f"{req.request_id:12} {req.requested_amount:12.2f} {computed_safe:12.2f} {ref_safe:12.2f} "
              f"{diff:10.2f}  {our_date_str:10} {ref_date or '':10}  {flag}")
        if warnings:
            for w in warnings:
                print(f"    WARNING: {w}")

    print(f"\n{amount_close}/{total} amounts within tolerance, {date_exact}/{total} dates exact "
          f"(of which {date_both_empty} both empty), max abs amount error = {max_abs_err:.2f}")

    total_abs_err = 0.0
    total_rel_err = 0.0
    rel_count = 0
    for req, computed_safe, _, _ in rows:
        ref = raw_by_id[req.request_id]
        ref_safe = float(ref["amount_safe_to_pay"])
        diff = abs(computed_safe - ref_safe)
        total_abs_err += diff
        if ref_safe > 0:
            total_rel_err += diff / ref_safe
            rel_count += 1
    print(f"sum abs amount error = {total_abs_err:.2f}, mean abs error = {total_abs_err/total:.2f}, "
          f"mean relative error (where ref>0) = {total_rel_err/rel_count:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
