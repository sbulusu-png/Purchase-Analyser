"""Phase 4/5 top-level entry point: turns one Request into the full set of
output.csv fields.

amount_safe_to_pay and earliest_date_for_full_payment are always the
*baseline* Phase 3 values (before any spending changes), per
problem_statement.md: "the first date the full amount passes the safety
check without optional spending changes." Everything else --
affordability_status, recommended_payment_method, payment_plan,
spending_changes_needed -- comes from whichever candidate plan wins the
ranking, which may have needed spending changes to become safe at all.

decision_explanation is always grounded (see explain.py): the
deterministic sentence built straight from these facts, optionally
reworded by an LLM that is independently checked against the same facts
and discarded if it doesn't stay grounded. Pass an `explain_client` to
enable the LLM rewording; omit it (the default) to stay fully
deterministic, e.g. for fast repeated validation runs.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from explain import ExplanationCache, deterministic_explanation, polish_with_llm
from llm_client import OpenAICompatibleClient
from loaders import Dataset
from plans import Candidate, build_candidates, rescue_with_spending_changes
from resolve import MessageSignalStore, VisionAmountStore, resolve_user_events
from schemas import OutputRow, Request
from simulate import amount_safe_to_pay as compute_amount_safe_to_pay
from simulate import build_daily_forecast
from simulate import earliest_date_for_full_payment as compute_earliest_date
from usage_tracker import UsageTracker


def _format_plan(payments: list[tuple[date, float]]) -> str:
    return "|".join(f"{d.isoformat()}:{_fmt(a)}" for d, a in payments)


def _fmt(amount: float) -> str:
    if abs(amount - round(amount)) < 1e-6:
        return str(int(round(amount)))
    return f"{amount:.2f}".rstrip("0").rstrip(".")


def decide(
    dataset: Dataset,
    request: Request,
    vision_store: VisionAmountStore,
    message_store: MessageSignalStore,
    explain_client: Optional[OpenAICompatibleClient] = None,
    explain_tracker: Optional[UsageTracker] = None,
    explain_cache: Optional[ExplanationCache] = None,
) -> OutputRow:
    profile = dataset.profile_by_user[request.user_id]
    request_date = date.fromisoformat(request.request_date)
    desired_completion_date = date.fromisoformat(request.desired_completion_date)
    forecast_end = request_date + timedelta(days=90)

    forward_events, _warnings = resolve_user_events(
        dataset, request.user_id, profile, request_date, forecast_end, vision_store, message_store
    )
    baseline_forecast = build_daily_forecast(profile.current_available_balance, forward_events, request_date)

    baseline_safe = compute_amount_safe_to_pay(baseline_forecast, profile.minimum_balance_to_keep, request.requested_amount)
    baseline_earliest = compute_earliest_date(baseline_forecast, profile.minimum_balance_to_keep, request.requested_amount)

    candidates = build_candidates(dataset, profile, request, baseline_forecast, request_date, desired_completion_date)

    if not candidates:
        rescue = rescue_with_spending_changes(
            dataset, profile, request, forward_events, profile.current_available_balance,
            request_date, desired_completion_date,
        )
        if rescue is not None:
            candidates, _changes, _forecast = rescue

    chosen: Candidate | None = min(candidates, key=Candidate.rank_key) if candidates else None

    if chosen is None:
        status = "not_affordable"
        method = "not_recommended"
        payment_plan = "none"
        spending_changes_str = "none"
    else:
        method = chosen.method
        payment_plan = _format_plan(chosen.payments)
        spending_changes_str = (
            "|".join(c.to_str() for c in chosen.spending_changes) if chosen.spending_changes else "none"
        )
        if chosen.spending_changes:
            status = "affordable_with_plan"
        elif method == "full_payment":
            status = "affordable_now"
        elif method == "wait":
            status = "affordable_later"
        else:
            status = "affordable_with_plan"

    grounded = deterministic_explanation(dataset, request, profile, method, chosen)
    explanation = grounded
    if explain_client is not None:
        cached = explain_cache.get(request.request_id) if explain_cache else None
        if cached is not None:
            explanation = cached
        else:
            explanation = polish_with_llm(explain_client, explain_tracker or UsageTracker(), request, grounded, request.request_id)
            if explain_cache:
                explain_cache.set(request.request_id, explanation)

    return OutputRow(
        request_id=request.request_id,
        amount_safe_to_pay=round(baseline_safe, 2),
        affordability_status=status,
        recommended_payment_method=method,
        payment_plan=payment_plan,
        earliest_date_for_full_payment=baseline_earliest.isoformat() if baseline_earliest else "",
        spending_changes_needed=spending_changes_str,
        decision_explanation=explanation,
    )
