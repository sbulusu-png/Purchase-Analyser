"""Per-user event resolution: turns raw financial_events.csv rows (plus
Phase 2's vision/message extractions) into a clean, currency-converted,
signed cash-flow timeline ready for the 90-day simulator.

Inclusion rules (from problem_statement.md / AGENTS.md §6.3), applied
uniformly by (status, direction, event_type) rather than per-category, so
every special case the spec calls out (pending bonuses, commissions,
refunds, lottery proceeds, investment gains, unrealized investment value)
falls out of the same small rule set instead of needing its own branch:

- status "cancelled" or "failed" -> excluded (never happened / voided).
- status "unrealized", or event_type "investment_valuation", or
  direction "non_cash" -> excluded (a value snapshot, not a cash movement).
- status "settled" or "scheduled" -> included at its cash date, regardless
  of direction (both are confirmed: it already happened, or it's a
  confirmed future commitment such as "next confirmed salary").
- status "pending": debit -> included ("reserve pending debits"); credit
  -> excluded ("do not count pending credits ... until they settle").

The cash date is settlement_date when present, else event_date.

Message-signal integration on a *specific existing event* (from Phase 2's
message_signals.json) is only applied when a signal's target_event_id is a
real event_id belonging to this user's own events:
- amend_event: overrides amount and/or date on the matched event.
- cancel_event: excludes the matched event entirely.
- delay_event: shifts the matched event's date to new_date.
- confirm_event / anything else: no change.
If more than one message targets the same event, the one with the latest
sent_at wins (see MessageSignalStore) -- not file/cache order, since
message_id order in this dataset doesn't reliably track chronological
sent_at order.

Recurring income and expenses ARE projected forward beyond the last known
occurrence of each (user, category, direction) series -- see
recurrence.project_recurring for the detection rule (>=2 occurrences, a
consistent gap, mean-of-recent-occurrences amount). Debit and credit
occurrences of the same category are tracked and projected separately, so
a one-off purchase+refund pair can't be misread as a recurring bill (this
was a real bug, caught by validate_phase3.py, before the split). This
design was reached empirically: cross-checking against
dataset/sample_requests.csv's 25 reference rows, symmetric (income AND
expense) projection across ALL categories with a mean-based estimate cut
the mean error against those references by roughly 80% compared to not
projecting at all, or to projecting only a profile-restricted subset of
categories -- see git history / log.txt for the earlier, worse-performing
attempts and why they were rejected. It is not a perfect match to the
reference values (some requests are still off by a meaningful margin,
most plausibly because a message describes a fact this resolver doesn't
yet act on, or because 90 days spans a genuinely irregular life event) but
it is the best-performing approach found by trying several plausible
readings of "forecast recurring income and expenses" against the one
source of reference values available.

Employer-sourced messages (source_type == "employer") that state a
concrete new pay rate and effective date (e.g. "monthly salary increased
to X, effective <date>") override the *amount* of the generic salary
projection from that date forward, since the generic projection only
knows the historical average and would otherwise miss a confirmed raise
or cut entirely. Two more employer-message cases -- a full stop
("employment/seasonal contract has ended", no continuing amount given)
and a same-day correction ("remaining confirmed salary is X", no future
effective date given) -- are also detected and applied, since neither
carries the (new_amount, new_date) pair the above override needs, and
neither is visible in financial_events.csv itself when the event rows
were never updated to reflect it. See the salary-discontinuation block
below for details.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from currency import convert, convert_nearest
from loaders import Dataset
from recurrence import ResolvedOccurrence, project_recurring
from schemas import FinancialEvent, FinancialProfile

EXCLUDED_STATUSES = {"cancelled", "failed", "unrealized"}
CONFIRMED_STATUSES = {"settled", "scheduled"}
_EVENT_ID_RE = re.compile(r"^event_\d+$")


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


@dataclass
class ForwardEvent:
    """One signed cash-flow entry in the user's home currency, on or after
    request_date. `event_id` is None for a synthetic recurrence projection;
    `reference_event_id` is always a real financial_events.csv id when one
    exists for this occurrence's series (see ResolvedOccurrence.template_event_id),
    letting Phase 4 cite a real id for a spending change even on a
    projected future instance.
    """

    event_date: date
    delta: float  # positive = credit, negative = debit, home currency
    category: str
    flexibility: str
    minimum_allowed_amount: Optional[float]
    event_id: Optional[str]
    description: str
    reference_event_id: Optional[str] = None


def _parse_sent_at(sent_at: str) -> str:
    # ISO 8601 timestamps ("...Z" or with an explicit offset) sort
    # correctly as plain strings once "Z" is normalized to "+00:00", so a
    # lexical comparison is enough here without needing datetime parsing
    # (and its version-dependent "Z" support) at all.
    return sent_at.replace("Z", "+00:00")


class MessageSignalStore:
    """Loads code/.cache/message_signals.json once and answers, per event,
    whether a valid amend/cancel/delay signal targets it -- and separately,
    per message, what its raw signal was (used for employer-sourced salary
    change notices, which usually have no target_event_id at all since
    there's rarely an existing ledger row for a future pay rate).

    When multiple messages target the same event, the one with the latest
    `sent_at` wins, per problem_statement.md / AGENTS.md Sec 6.3's conflict
    order ("explicit cancellation, settlement, or amendment first; then
    newer records from the same source"): all of amend/cancel/delay are
    equally "explicit" signals, so the tiebreak between two of them is
    recency, not cache/file ordering. This matters in practice: message_id
    order in this dataset does not reliably track chronological sent_at
    order (e.g. message_02 is sent before message_01), so picking "the
    last one seen" instead of "the newest one" can silently apply a
    stale, superseded instruction.
    """

    def __init__(self, cache_dir: str | Path, dataset: Optional[Dataset] = None):
        path = Path(cache_dir) / "message_signals.json"
        self._by_message_id: dict[str, dict] = {}
        self._by_event_id: dict[str, dict] = {}
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            self._by_message_id = json.load(f)
        sent_at_by_message_id = {m.message_id: m.sent_at for m in dataset.messages} if dataset else {}
        best_sent_at: dict[str, str] = {}
        for message_id, sig in self._by_message_id.items():
            target = sig.get("target_event_id")
            if not target or not _EVENT_ID_RE.match(str(target)):
                continue
            sent_at = sent_at_by_message_id.get(message_id, "")
            if target not in self._by_event_id or _parse_sent_at(sent_at) >= _parse_sent_at(best_sent_at[target]):
                self._by_event_id[target] = sig
                best_sent_at[target] = sent_at

    def signal_for(self, event_id: str) -> Optional[dict]:
        return self._by_event_id.get(event_id)

    def signal_for_message(self, message_id: str) -> Optional[dict]:
        return self._by_message_id.get(message_id)


class VisionAmountStore:
    def __init__(self, cache_dir: str | Path):
        path = Path(cache_dir) / "vision_extractions.json"
        with path.open("r", encoding="utf-8") as f:
            self._data = json.load(f)

    def amount_for(self, event_id: str) -> Optional[float]:
        entry = self._data.get(event_id)
        if entry is None:
            return None
        return entry.get("amount")


def _cash_date(event: FinancialEvent) -> date:
    raw = event.settlement_date or event.event_date
    return _parse_date(raw)


def _is_included(event: FinancialEvent) -> bool:
    if event.event_type == "investment_valuation":
        return False
    if event.direction == "non_cash":
        return False
    if event.status in EXCLUDED_STATUSES:
        return False
    if event.status in CONFIRMED_STATUSES:
        return True
    if event.status == "pending":
        return event.direction == "debit"
    return False


def resolve_user_events(
    dataset: Dataset,
    user_id: str,
    profile: FinancialProfile,
    request_date: date,
    forecast_end: date,
    vision_store: VisionAmountStore,
    message_store: MessageSignalStore,
) -> tuple[list[ForwardEvent], list[str]]:
    """Returns (forward_events, warnings). forward_events covers
    [request_date, forecast_end] inclusive: the explicit events in that
    window (see module docstring for inclusion rules) plus synthetic
    recurring income/expense projections and employer-message pay-rate
    overrides (see module docstring).
    """
    warnings: list[str] = []
    home_currency = profile.home_currency
    raw_events = dataset.events_by_user.get(user_id, [])

    debit_occurrences_by_category: dict[str, list[ResolvedOccurrence]] = {}
    credit_occurrences_by_category: dict[str, list[ResolvedOccurrence]] = {}
    forward_events: list[ForwardEvent] = []

    for event in raw_events:
        if not _is_included(event):
            continue

        amount = event.amount
        if amount is None:
            amount = vision_store.amount_for(event.event_id)
            if amount is None:
                warnings.append(f"{event.event_id}: blank amount with no resolved value; skipped")
                continue

        event_date = _cash_date(event)
        currency = event.currency

        signal = message_store.signal_for(event.event_id)
        if signal:
            intent = signal.get("intent")
            if intent == "cancel_event":
                continue
            if intent == "amend_event":
                if signal.get("new_amount") is not None:
                    amount = float(signal["new_amount"])
                    currency = signal.get("new_currency") or currency
                if signal.get("new_date"):
                    try:
                        event_date = _parse_date(signal["new_date"])
                    except ValueError:
                        pass
            elif intent == "delay_event" and signal.get("new_date"):
                try:
                    event_date = _parse_date(signal["new_date"])
                except ValueError:
                    pass

        home_amount = convert(dataset, amount, currency, home_currency, event_date.isoformat())

        # Debit and credit occurrences of a category are tracked separately
        # for recurrence detection: mixing them (e.g. a refund sharing a
        # category with an unrelated one-off purchase) can manufacture a
        # false "recurring" pattern out of two unrelated one-time events.
        occ = ResolvedOccurrence(
            event_date=event_date,
            amount=home_amount,
            category=event.category,
            flexibility=event.flexibility,
            minimum_allowed_amount=event.minimum_allowed_amount,
            event_id=event.event_id,
            description=event.description,
            template_event_id=event.event_id,
        )
        bucket = debit_occurrences_by_category if event.direction == "debit" else credit_occurrences_by_category
        bucket.setdefault(event.category, []).append(occ)

        if request_date <= event_date <= forecast_end:
            signed = home_amount if event.direction == "credit" else -home_amount
            forward_events.append(
                ForwardEvent(
                    event_date=event_date,
                    delta=signed,
                    category=event.category,
                    flexibility=event.flexibility,
                    minimum_allowed_amount=event.minimum_allowed_amount,
                    event_id=event.event_id,
                    description=event.description,
                    reference_event_id=event.event_id,
                )
            )

    for occ in project_recurring(debit_occurrences_by_category, forecast_end, min_occurrences=2, conservative="mean"):
        if occ.event_date < request_date:
            continue
        forward_events.append(
            ForwardEvent(
                event_date=occ.event_date, delta=-occ.amount, category=occ.category,
                flexibility=occ.flexibility, minimum_allowed_amount=occ.minimum_allowed_amount,
                event_id=None, description=f"projected recurring {occ.category}",
                reference_event_id=occ.template_event_id,
            )
        )
    # "salary" is projected like any other recurring category UNLESS the
    # single most recent occurrence's own description signals the series
    # won't continue -- this dataset uses a deliberate, consistent
    # vocabulary for that ("Final employer payroll", "Previous employer
    # payroll", "Payroll before leave"), found by hand-checking
    # dataset/financial_events.csv's full set of salary descriptions after
    # discovering that a clean 5-month monthly pattern is exactly what a
    # *terminated* income stream looks like right up until it stops (a
    # user whose last salary row read "Final employer payroll" was
    # incorrectly projected 3 more months of pay by blind recurrence,
    # while request_01/03/13/etc. rely on exactly this projection to reach
    # a sane forecast for still-employed users with no explicit "next"
    # salary row). Excluding salary outright scored far worse in aggregate
    # against dataset/sample_requests.csv than this targeted check.
    # A broader attempt to also exclude "unreliable" gig/freelance-style
    # income (platform payouts, commissions, project/retainer payments) by
    # the same keyword approach was tried and reverted: it fixed request_10
    # (an explicit message warns that user's gig payout "isn't withdrawable
    # until completed") but broke request_09 (freelance project income the
    # reference DOES count) and request_11 (a mixed "Base salary" +
    # "Monthly sales commission" series in the same category, where the
    # most recent entry happening to be the commission excluded the whole
    # series, including the legitimate salary component). Distinguishing
    # these cases properly needs a signal Phase 2's message extraction
    # doesn't currently produce (something like "income reliability"),
    # not a description keyword -- left as a known limitation rather than
    # a fix that trades one wrong answer for two others.
    discontinuity_markers = ("final", "previous employer", "before leave")
    salary_occs = credit_occurrences_by_category.get("salary", [])
    description_discontinued = bool(salary_occs) and any(
        marker in max(salary_occs, key=lambda o: o.event_date).description.lower()
        for marker in discontinuity_markers
    )

    # The description-based check above only catches discontinuation when
    # financial_events.csv's own last salary row was updated to say so. This
    # dataset also has two employer-message templates (English and
    # Indonesian) that announce the *same* fact -- "seasonal contract has
    # ended" / "employment has ended", with no continuing income confirmed
    # -- for users whose event rows were never updated at all, which the
    # description check silently misses. A second, distinct template
    # ("one household income source has ended; the remaining confirmed
    # monthly salary is X") isn't a full stop -- it gives a concrete
    # ongoing amount, just with no explicit future effective date, so the
    # employer-raise loop below (which requires a new_date) was skipping it
    # too. Both were found by hand-checking every employer message
    # containing "ended"/"berakhir" against message_signals.json: every
    # match with new_amount == None is a full stop, every match with a
    # new_amount is a same-day amount correction -- confirmed against the
    # actual 250-request dataset, not assumed.
    message_salary_discontinued = False
    message_salary_override: Optional[tuple[float, str]] = None
    for message in dataset.messages_by_user.get(user_id, []):
        if message.source_type != "employer":
            continue
        signal = message_store.signal_for_message(message.message_id)
        if not signal or signal.get("intent") != "new_fact":
            continue
        text = message.message_text.lower()
        if "ended" not in text and "berakhir" not in text:
            continue
        if signal.get("new_amount") is None:
            message_salary_discontinued = True
        elif not signal.get("new_date"):
            message_salary_override = (float(signal["new_amount"]), signal.get("new_currency") or home_currency)

    salary_discontinued = description_discontinued or message_salary_discontinued
    other_credit_categories = (
        {c: v for c, v in credit_occurrences_by_category.items() if c != "salary"}
        if (salary_discontinued or message_salary_override)
        else credit_occurrences_by_category
    )
    for occ in project_recurring(other_credit_categories, forecast_end, min_occurrences=2, conservative="mean"):
        if occ.event_date < request_date:
            continue
        forward_events.append(
            ForwardEvent(
                event_date=occ.event_date, delta=occ.amount, category=occ.category,
                flexibility=occ.flexibility, minimum_allowed_amount=occ.minimum_allowed_amount,
                event_id=None, description=f"projected recurring {occ.category}",
            )
        )

    # A message-confirmed salary correction with no future effective date
    # (see above) takes effect immediately, not on a future date -- so
    # project it at the user's own salary cadence from their last known
    # occurrence, the same way the generic projection above would have,
    # just at the corrected amount instead of the historical mean.
    if message_salary_override and not salary_discontinued:
        override_amount, override_currency = message_salary_override
        salary_dates = sorted(o.event_date for o in salary_occs)
        gaps = [(salary_dates[i] - salary_dates[i - 1]).days for i in range(1, len(salary_dates))]
        gaps = [g for g in gaps if g > 0]
        period_days = round(statistics.median(gaps)) if gaps else 30
        last = max(salary_occs, key=lambda o: o.event_date)
        next_date = last.event_date + timedelta(days=period_days)
        while next_date <= forecast_end:
            if next_date >= request_date:
                home_amount = convert_nearest(
                    dataset, override_amount, override_currency, home_currency, next_date.isoformat()
                )
                forward_events.append(
                    ForwardEvent(
                        event_date=next_date, delta=home_amount, category="salary",
                        flexibility=last.flexibility, minimum_allowed_amount=last.minimum_allowed_amount,
                        event_id=None, description="projected recurring salary (corrected per message)",
                        reference_event_id=last.template_event_id or last.event_id,
                    )
                )
            next_date += timedelta(days=period_days)

    # An employer-sourced message stating a concrete new pay rate and
    # effective date (e.g. "monthly salary increased to X, effective
    # <date>") creates a new recurring salary series from that date
    # forward, at the cadence of the user's own salary history (default
    # monthly if there isn't one) -- this is the one case where projecting
    # future salary is warranted, because the message is a concrete,
    # dated confirmation rather than an extrapolation from a pattern that
    # could just as easily have ended.
    for message in dataset.messages_by_user.get(user_id, []):
        if message.source_type != "employer":
            continue
        signal = message_store.signal_for_message(message.message_id)
        if not signal or signal.get("new_amount") is None or not signal.get("new_date"):
            continue
        try:
            effective_date = _parse_date(signal["new_date"])
        except ValueError:
            continue
        if effective_date > forecast_end:
            continue
        signal_currency = signal.get("new_currency") or home_currency
        salary_dates = sorted(o.event_date for o in credit_occurrences_by_category.get("salary", []))
        gaps = [(salary_dates[i] - salary_dates[i - 1]).days for i in range(1, len(salary_dates))]
        gaps = [g for g in gaps if g > 0]
        period_days = round(statistics.median(gaps)) if gaps else 30

        existing_salary_dates = {fe.event_date for fe in forward_events if fe.category == "salary"}
        occurrence_date = effective_date
        while occurrence_date <= forecast_end:
            if occurrence_date >= request_date and occurrence_date not in existing_salary_dates:
                home_amount = convert_nearest(
                    dataset, float(signal["new_amount"]), signal_currency, home_currency, occurrence_date.isoformat()
                )
                forward_events.append(
                    ForwardEvent(
                        event_date=occurrence_date, delta=home_amount, category="salary",
                        flexibility="fixed", minimum_allowed_amount=None, event_id=None,
                        description=f"projected salary change ({message.message_id})",
                    )
                )
            occurrence_date += timedelta(days=period_days)

    forward_events.sort(key=lambda e: e.event_date)
    return forward_events, warnings
