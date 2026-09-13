"""Direct unit tests for Candidate.rank_key's 6-step tie-break order
(problem_statement.md "Choosing Between Safe Plans"). Each test isolates
one rule by holding every earlier-ranked field equal between two
candidates and checking only that rule breaks the tie -- the sample and
structural checks (validate_phase3.py/validate_phase4.py) exercise this
indirectly, but had never confirmed each rule individually fires in the
right direction. Plain asserts, no test framework, consistent with the
project's zero-dependency policy: run with `python3 code/test_plans.py`.
"""

from __future__ import annotations

from datetime import date

from plans import Candidate


def _cand(
    completes=True, changes=0, total=1000.0, start=date(2025, 1, 1), payments=1, option_id=-1
) -> Candidate:
    return Candidate(
        method="full_payment",
        payments=[(start, total)] * payments,
        total_paid=total,
        completes_by_deadline=completes,
        payment_option_id=option_id,
        spending_changes=["x"] * changes,  # rank_key only checks truthiness/len semantics via bool
    )


def _best(*candidates: Candidate) -> Candidate:
    return min(candidates, key=Candidate.rank_key)


def test_rule1_completes_by_deadline_wins_regardless_of_everything_else():
    on_time = _cand(completes=True, total=999999, payments=99)
    late_but_cheaper = _cand(completes=False, total=1, payments=1)
    assert _best(on_time, late_but_cheaper) is on_time


def test_rule2_no_spending_changes_preferred_when_deadline_tied():
    clean = _cand(changes=0, total=999999)
    with_changes = _cand(changes=1, total=1)
    assert _best(clean, with_changes) is clean


def test_rule3_minimize_total_paid_when_deadline_and_changes_tied():
    cheaper = _cand(total=100, start=date(2025, 6, 1), payments=5)
    pricier = _cand(total=200, start=date(2025, 1, 1), payments=1)
    assert _best(cheaper, pricier) is cheaper


def test_rule4_start_earlier_when_above_tied():
    earlier = _cand(total=100, start=date(2025, 1, 1), payments=5)
    later = _cand(total=100, start=date(2025, 2, 1), payments=1)
    assert _best(earlier, later) is earlier


def test_rule5_fewer_payments_when_above_tied():
    fewer = _cand(total=100, start=date(2025, 1, 1), payments=1, option_id=99)
    more = _cand(total=100, start=date(2025, 1, 1), payments=3, option_id=1)
    assert _best(fewer, more) is fewer


def test_rule6_lowest_payment_option_id_as_final_tiebreak():
    lower_id = _cand(total=100, start=date(2025, 1, 1), payments=2, option_id=5)
    higher_id = _cand(total=100, start=date(2025, 1, 1), payments=2, option_id=42)
    assert _best(lower_id, higher_id) is lower_id


def test_rules_apply_in_strict_priority_order_not_just_pairwise():
    # Worse on rule 1 must lose even if it wins every later rule outright.
    misses_deadline_but_wins_everything_else = _cand(
        completes=False, changes=0, total=1, start=date(2025, 1, 1), payments=1, option_id=0
    )
    meets_deadline_but_loses_everything_else = _cand(
        completes=True, changes=1, total=999999, start=date(2025, 12, 31), payments=50, option_id=999
    )
    assert _best(misses_deadline_but_wins_everything_else, meets_deadline_but_loses_everything_else) is (
        meets_deadline_but_loses_everything_else
    )


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} ranking tie-break tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
