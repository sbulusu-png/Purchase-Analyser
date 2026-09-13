"""The deterministic 90-day balance simulator (Phase 3).

Pure arithmetic, no LLM involved. Builds a day-by-day balance array over
the forecast window from `resolve.ForwardEvent`s, then answers the two
core "90-Day Safety Check" questions from problem_statement.md:

- amount_safe_to_pay: the most payable today without the projected balance
  ever dropping below minimum_balance_to_keep at any point in the window.
- earliest_date_for_full_payment: the first day on which paying the full
  requested_amount keeps every day (before and after) at or above the
  minimum.

Key property this relies on: a one-time payment on day d is a constant
downward shift applied to every day's balance from d onward and no others.
That means the *only* thing that matters for "is a payment on day d safe"
is (a) whether every day strictly before d was already safe on its own
(the payment can't fix a pre-existing shortfall — it only ever subtracts),
and (b) whether the minimum balance from d onward, after subtracting the
payment, still clears the floor. Both are simple min-array lookups once
the daily balances are computed, which is why this file iterates once
per calendar day (~91 iterations per request) rather than needing a more
elaborate search.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from resolve import ForwardEvent


@dataclass
class DailyForecast:
    request_date: date
    forecast_end: date
    daily_balance: list[float]  # daily_balance[i] = balance at request_date + i days, after that day's events

    def day_index(self, d: date) -> int:
        return (d - self.request_date).days


def build_daily_forecast(
    starting_balance: float,
    forward_events: list[ForwardEvent],
    request_date: date,
    forecast_days: int = 90,
) -> DailyForecast:
    forecast_end = request_date + timedelta(days=forecast_days)
    n_days = forecast_days + 1  # inclusive of request_date and forecast_end
    deltas = [0.0] * n_days
    for e in forward_events:
        idx = (e.event_date - request_date).days
        if 0 <= idx < n_days:
            deltas[idx] += e.delta

    daily_balance = [0.0] * n_days
    running = starting_balance
    for i in range(n_days):
        running += deltas[i]
        daily_balance[i] = running

    return DailyForecast(request_date=request_date, forecast_end=forecast_end, daily_balance=daily_balance)


def _prefix_min_exclusive(daily_balance: list[float]) -> list[Optional[float]]:
    """prefix[i] = min(daily_balance[0:i]), or None for i == 0 (no prior days)."""
    prefix: list[Optional[float]] = [None] * len(daily_balance)
    running_min = None
    for i in range(len(daily_balance)):
        prefix[i] = running_min
        running_min = daily_balance[i] if running_min is None else min(running_min, daily_balance[i])
    return prefix


def _suffix_min_inclusive(daily_balance: list[float]) -> list[float]:
    """suffix[i] = min(daily_balance[i:])."""
    n = len(daily_balance)
    suffix = [0.0] * n
    running_min = daily_balance[-1]
    for i in range(n - 1, -1, -1):
        running_min = min(running_min, daily_balance[i])
        suffix[i] = running_min
    return suffix


def amount_safe_to_pay(forecast: DailyForecast, minimum_balance_to_keep: float, requested_amount: float) -> float:
    min_balance = min(forecast.daily_balance)
    headroom = min_balance - minimum_balance_to_keep
    return max(0.0, min(requested_amount, headroom))


def earliest_date_for_full_payment(
    forecast: DailyForecast, minimum_balance_to_keep: float, requested_amount: float
) -> Optional[date]:
    prefix = _prefix_min_exclusive(forecast.daily_balance)
    suffix = _suffix_min_inclusive(forecast.daily_balance)
    for i in range(len(forecast.daily_balance)):
        prefix_ok = prefix[i] is None or prefix[i] >= minimum_balance_to_keep
        suffix_ok = (suffix[i] - requested_amount) >= minimum_balance_to_keep
        if prefix_ok and suffix_ok:
            return forecast.request_date + timedelta(days=i)
    return None


def is_schedule_safe(
    forecast: DailyForecast,
    minimum_balance_to_keep: float,
    payments: list[tuple[date, float]],
) -> bool:
    """Checks a multi-payment schedule (partial payment or an installment
    plan) against the same daily balances: apply each payment on its date
    (and every day after, since it's a one-time debit persisting forward)
    and confirm the floor is never breached at any point in the window.
    """
    n = len(forecast.daily_balance)
    extra = [0.0] * n
    for pay_date, amount in payments:
        idx = (pay_date - forecast.request_date).days
        if idx < 0:
            return False  # a payment scheduled before request_date is not a valid plan
        if idx < n:
            extra[idx] += amount
    cumulative_extra = 0.0
    for i in range(n):
        cumulative_extra += extra[i]
        if forecast.daily_balance[i] - cumulative_extra < minimum_balance_to_keep:
            return False
    return True
