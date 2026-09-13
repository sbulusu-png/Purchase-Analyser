"""Phase 4 — candidate payment plans, the 6-step tie-break ranking, and the
spending_changes_needed rescue when no plan is safe on the baseline forecast.

Candidate generation (problem_statement.md "Choosing Between Safe Plans"):

- full_payment: pay the full amount today. Eligible only when "full_payment"
  is in payment_methods_user_will_consider AND today's amount_safe_to_pay
  covers the full requested_amount.
- wait: pay the full amount on earliest_date_for_full_payment. Eligible
  only when "full_payment" is accepted AND that date exists and is after
  request_date (if it's request_date itself, full_payment already covers
  it) AND is on or before desired_completion_date.
- partial_payment: pay amount_safe_to_pay today, the remainder on
  earliest_date_for_full_payment. Eligible only when the request allows
  it, "partial_payment" is accepted, 0 < amount_safe_to_pay <
  requested_amount, and that second date is on or before
  desired_completion_date. Safe by construction: amount_safe_to_pay and
  earliest_date_for_full_payment are each independently defined (see
  simulate.py) so that paying amount_safe_to_pay today never breaches the
  floor before the second payment, and the second payment amount is
  exactly what a single full payment on that date was already verified
  safe for -- is_schedule_safe double-checks this instead of trusting it
  blindly.
- installments: one candidate per request_payment_options.csv row with
  payment_method == "installments" for this request. Eligible only when
  "installments" is accepted, max_installment_months isn't exceeded, and
  the exact supplied schedule (is_schedule_safe) never breaches the floor.

Ranking (six steps, applied in order, first difference wins):
1. completes the request by desired_completion_date
2. requires no spending changes
3. minimizes total amount paid
4. starts earlier
5. uses fewer payments
6. lowest payment_option_id (installments only; other methods sort first)

If nothing above is safe, `rescue_with_spending_changes` greedily stops or
reduces up to 3 flexible, non-protected, user-permitted events (highest
relief first) until some candidate becomes safe, preferring the fewest
changes. Stopping and reducing the same event are mutually exclusive by
construction: each event contributes at most one candidate action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from loaders import Dataset
from resolve import ForwardEvent
from schemas import FinancialProfile, PaymentOption, Request
from simulate import (
    DailyForecast,
    amount_safe_to_pay as compute_amount_safe_to_pay,
    build_daily_forecast,
    earliest_date_for_full_payment as compute_earliest_date,
    is_schedule_safe,
)

STOPPABLE_FLEXIBILITY = {"stoppable", "reducible_or_stoppable"}
REDUCIBLE_FLEXIBILITY = {"reducible", "reducible_or_stoppable"}
MAX_SPENDING_CHANGES = 3


@dataclass
class SpendingChange:
    kind: str  # "stop" | "reduce_to"
    event_id: str
    new_amount: Optional[float] = None  # only for reduce_to
    relief: float = 0.0  # cash freed up, home currency

    def to_str(self) -> str:
        if self.kind == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{_fmt_amount(self.new_amount)}"


@dataclass
class Candidate:
    method: str  # full_payment | partial_payment | installments | wait
    payments: list[tuple[date, float]]  # chronological
    total_paid: float
    completes_by_deadline: bool
    payment_option_id: int  # -1 for non-installment methods (sorts first)
    spending_changes: list[SpendingChange] = field(default_factory=list)

    @property
    def start_date(self) -> date:
        return self.payments[0][0]

    @property
    def payments_count(self) -> int:
        return len(self.payments)

    def rank_key(self) -> tuple:
        return (
            0 if self.completes_by_deadline else 1,
            0 if not self.spending_changes else 1,
            round(self.total_paid, 2),
            self.start_date,
            self.payments_count,
            self.payment_option_id,
        )


def _fmt_amount(amount: float) -> str:
    if amount == round(amount):
        return str(int(round(amount)))
    return f"{amount:g}"


def _installment_schedule(option: PaymentOption) -> list[tuple[date, float]]:
    freq = option.payment_frequency_days or 0
    start = date.fromisoformat(option.first_payment_date)
    return [(start + timedelta(days=freq * i), option.payment_amount) for i in range(option.number_of_payments)]


def _option_id_num(option_id: str) -> int:
    return int(option_id.rsplit("_", 1)[-1])


def build_candidates(
    dataset: Dataset,
    profile: FinancialProfile,
    request: Request,
    forecast: DailyForecast,
    request_date: date,
    desired_completion_date: date,
) -> list[Candidate]:
    accepted = set(profile.payment_methods_user_will_consider)
    requested = request.requested_amount
    safe_today = compute_amount_safe_to_pay(forecast, profile.minimum_balance_to_keep, requested)
    earliest = compute_earliest_date(forecast, profile.minimum_balance_to_keep, requested)

    candidates: list[Candidate] = []

    if "full_payment" in accepted and safe_today >= requested - 0.005:
        candidates.append(
            Candidate(
                method="full_payment",
                payments=[(request_date, requested)],
                total_paid=requested,
                completes_by_deadline=request_date <= desired_completion_date,
                payment_option_id=-1,
            )
        )

    if "full_payment" in accepted and earliest is not None and earliest > request_date:
        candidates.append(
            Candidate(
                method="wait",
                payments=[(earliest, requested)],
                total_paid=requested,
                completes_by_deadline=earliest <= desired_completion_date,
                payment_option_id=-1,
            )
        )

    if (
        request.allows_partial_payment
        and "partial_payment" in accepted
        and 0 < safe_today < requested - 0.005
        and earliest is not None
        and earliest <= desired_completion_date
    ):
        remainder = requested - safe_today
        schedule = [(request_date, safe_today), (earliest, remainder)]
        if is_schedule_safe(forecast, profile.minimum_balance_to_keep, schedule):
            candidates.append(
                Candidate(
                    method="partial_payment",
                    payments=schedule,
                    total_paid=requested,
                    completes_by_deadline=True,
                    payment_option_id=-1,
                )
            )

    if "installments" in accepted and profile.max_installment_months:
        for option in dataset.payment_options_by_request.get(request.request_id, []):
            if option.payment_method != "installments":
                continue
            span_days = (option.number_of_payments - 1) * (option.payment_frequency_days or 0)
            span_months = round(span_days / 30.0)
            if span_months > profile.max_installment_months:
                continue
            schedule = _installment_schedule(option)
            if not is_schedule_safe(forecast, profile.minimum_balance_to_keep, schedule):
                continue
            candidates.append(
                Candidate(
                    method="installments",
                    payments=schedule,
                    total_paid=option.total_payable_amount,
                    completes_by_deadline=schedule[-1][0] <= desired_completion_date,
                    payment_option_id=_option_id_num(option.payment_option_id),
                )
            )

    return candidates


def _candidate_events(forward_events: list[ForwardEvent], profile: FinancialProfile) -> list[SpendingChange]:
    """Every valid stop/reduce action available in the forecast window,
    sorted by relief (largest first).

    Grouped by `reference_event_id` rather than per-occurrence: most of the
    forecast window is made of *projected* recurring debits (event_id is
    None -- there's no single financial_events.csv row for "next month's
    groceries"), so a change here means "stop/reduce this recurring
    series for the rest of the window", costed as the total relief across
    every occurrence of it, and cited by the real event_id the series is
    modeled after (reference_event_id) since the output format requires
    a real id. A one-off explicit debit with no series (reference_event_id
    set but appearing only once) is handled the same way, trivially.
    """
    protect = set(profile.expense_categories_to_protect)
    willing_stop = set(profile.expense_categories_user_is_willing_to_stop)
    willing_reduce = set(profile.expense_categories_user_is_willing_to_reduce)

    groups: dict[str, list[ForwardEvent]] = {}
    for e in forward_events:
        if e.reference_event_id is None or e.delta >= 0 or e.category in protect:
            continue
        groups.setdefault(e.reference_event_id, []).append(e)

    changes: list[SpendingChange] = []
    for ref_id, events in groups.items():
        category = events[0].category
        flexibility = events[0].flexibility
        min_allowed = events[0].minimum_allowed_amount
        can_stop = flexibility in STOPPABLE_FLEXIBILITY and category in willing_stop
        can_reduce = flexibility in REDUCIBLE_FLEXIBILITY and category in willing_reduce and min_allowed is not None

        if can_stop:
            relief = sum(-e.delta for e in events)
            changes.append(SpendingChange(kind="stop", event_id=ref_id, relief=relief))
        elif can_reduce:
            relief = sum(max(0.0, (-e.delta) - min_allowed) for e in events)
            if relief > 0:
                changes.append(SpendingChange(kind="reduce_to", event_id=ref_id, new_amount=min_allowed, relief=relief))
    changes.sort(key=lambda c: c.relief, reverse=True)
    return changes


def _apply_changes(forward_events: list[ForwardEvent], changes: list[SpendingChange]) -> list[ForwardEvent]:
    by_ref = {c.event_id: c for c in changes}
    adjusted = []
    for e in forward_events:
        change = by_ref.get(e.reference_event_id) if e.reference_event_id else None
        if change is None:
            adjusted.append(e)
        elif change.kind == "stop":
            continue
        else:
            capped_amount = min(-e.delta, change.new_amount)
            adjusted.append(
                ForwardEvent(
                    event_date=e.event_date, delta=-capped_amount, category=e.category,
                    flexibility=e.flexibility, minimum_allowed_amount=e.minimum_allowed_amount,
                    event_id=e.event_id, description=e.description, reference_event_id=e.reference_event_id,
                )
            )
    return adjusted


def rescue_with_spending_changes(
    dataset: Dataset,
    profile: FinancialProfile,
    request: Request,
    forward_events: list[ForwardEvent],
    starting_balance: float,
    request_date: date,
    desired_completion_date: date,
) -> Optional[tuple[list[Candidate], list[SpendingChange], DailyForecast]]:
    """Tries the highest-relief flexible events first, one at a time up to
    MAX_SPENDING_CHANGES, stopping as soon as some candidate becomes safe
    (fewest changes preferred, per the ranking rules). Returns
    (candidates, changes_applied, forecast) or None if nothing helps.
    """
    available = _candidate_events(forward_events, profile)
    for n in range(1, min(MAX_SPENDING_CHANGES, len(available)) + 1):
        trial = available[:n]
        adjusted_events = _apply_changes(forward_events, trial)
        forecast = build_daily_forecast(starting_balance, adjusted_events, request_date)
        candidates = build_candidates(dataset, profile, request, forecast, request_date, desired_completion_date)
        if candidates:
            for c in candidates:
                c.spending_changes = trial
            return candidates, trial, forecast
    return None
