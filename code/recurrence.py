"""Recurrence detection and projection, used by resolve.py to extend both
income and expense series beyond their last known occurrence for the
90-day forecast. See resolve.py's module docstring for how this is wired
in and the empirical comparison against alternatives that were tried and
rejected (no projection at all; projecting only profile-listed
categories).

Design decisions (documented here so they're auditable):

- Grouped by (user, category). In this dataset a category maps to a single
  logical recurring bill or spending pattern per user (rent, utilities,
  groceries, transport, dining, subscriptions, ...), so category is a
  reliable recurrence key without needing per-vendor/description matching.
- A category is "recurring" only if it has at least `min_occurrences`
  (default 2 -- the minimum a period can technically be computed from)
  known dated occurrences of the *same direction* (debit or credit; see
  resolve.py, which calls this once per direction) with a reasonably
  consistent gap between them (coefficient of variation of the gaps below
  50%). The variance check is what actually guards against false positives
  like two unrelated one-off purchases a few weeks apart -- requiring 3+
  occurrences instead of a variance check was also tried and scored worse
  in the empirical comparison (see resolve.py).
- The period is the median gap (in days) between consecutive occurrences.
  Projection starts strictly after the single latest known occurrence for
  that category (whether that occurrence is historical or an explicit
  future record already in financial_events.csv), so an explicit future
  record (e.g. a scheduled bill) is never double-counted alongside a
  synthetic projection for the same cycle.
- Projected amounts default to the mean of the last few real occurrences.
  "max" and "min" are also supported: "max" was tried for expenses (a more
  literal reading of "forecast essential variable spending conservatively")
  and "min" was tried for volatile income (e.g. gig-platform payouts,
  reasoning that overestimating available cash is the riskier failure
  mode) but both scored worse in aggregate against the reference samples
  than plain "mean" in both directions, so "mean" is the default for both.
- Income (e.g. category == "salary") is otherwise projected the same way
  as any other category here; resolve.py additionally lets an
  employer-sourced message override the projected amount from a stated
  effective date, since a historical estimate can't know about a
  confirmed future raise, and suppresses projection entirely when the
  latest occurrence's own description signals the series won't continue
  (e.g. "Final employer payroll").
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional


@dataclass
class ResolvedOccurrence:
    event_date: date
    amount: float  # in home currency, positive magnitude
    category: str
    flexibility: str
    minimum_allowed_amount: Optional[float]
    event_id: Optional[str]  # None for a synthetic/projected occurrence
    description: str = ""
    # The real financial_events.csv row this occurrence is or is modeled
    # after -- equal to event_id for a real occurrence; for a projected
    # one, the most recent real occurrence in the same series. Lets a
    # spending change cite a real, valid event_id (as the output format
    # requires) for a recurring commitment even when the specific future
    # instance being changed is itself synthetic. See plans.py.
    template_event_id: Optional[str] = None


def project_recurring(
    occurrences_by_category: dict[str, list[ResolvedOccurrence]],
    forecast_end: date,
    min_occurrences: int = 2,
    conservative: str = "mean",
) -> list[ResolvedOccurrence]:
    """For each category with enough known occurrences at a consistent
    enough gap, project synthetic future occurrences (at the detected
    period and a conservative amount) strictly after the latest known
    occurrence, up to forecast_end. Direction-agnostic: the caller decides
    whether a category's occurrences are income or expense.
    """
    projected: list[ResolvedOccurrence] = []
    for category, occs in occurrences_by_category.items():
        if len(occs) < min_occurrences:
            continue
        occs_sorted = sorted(occs, key=lambda o: o.event_date)
        dates = [o.event_date for o in occs_sorted]
        gaps = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
        gaps = [g for g in gaps if g > 0]
        if not gaps:
            continue
        period_days = round(statistics.median(gaps))
        if period_days <= 0:
            continue
        if len(gaps) >= 2:
            gap_mean = statistics.mean(gaps)
            gap_stdev = statistics.pstdev(gaps)
            if gap_mean > 0 and (gap_stdev / gap_mean) > 0.5:
                continue  # too irregular to treat as a predictable recurring series

        recent = occs_sorted[-min(6, len(occs_sorted)):]
        last = occs_sorted[-1]
        if conservative == "max":
            conservative_amount = max(o.amount for o in recent)
        elif conservative == "min":
            conservative_amount = min(o.amount for o in recent)
        elif conservative == "last":
            conservative_amount = last.amount
        else:
            conservative_amount = statistics.mean(o.amount for o in recent)

        next_date = last.event_date + timedelta(days=period_days)
        while next_date <= forecast_end:
            projected.append(
                ResolvedOccurrence(
                    event_date=next_date,
                    amount=conservative_amount,
                    category=category,
                    flexibility=last.flexibility,
                    minimum_allowed_amount=last.minimum_allowed_amount,
                    event_id=None,
                    template_event_id=last.template_event_id or last.event_id,
                )
            )
            next_date = next_date + timedelta(days=period_days)
    return projected
