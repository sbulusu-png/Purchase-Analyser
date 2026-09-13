"""Currency conversion using the fixed, dated rates in exchange_rates.csv.

The spec is explicit: "For a foreign-currency cash event, use the row for
its settlement date and the stated from_currency to_currency direction."
That means the required (date, from, to) row is guaranteed to exist when a
conversion is actually needed. We do an exact lookup and raise rather than
inventing an inverse or a nearby-date rate — a missing rate means the
caller asked for a conversion the data does not support, which should be
investigated, not silently guessed.
"""

from __future__ import annotations

from loaders import Dataset


class MissingExchangeRateError(Exception):
    pass


def get_rate(dataset: Dataset, rate_date: str, from_currency: str, to_currency: str) -> float:
    if from_currency == to_currency:
        return 1.0
    key = (rate_date, from_currency, to_currency)
    rate = dataset.rate_by_key.get(key)
    if rate is None:
        raise MissingExchangeRateError(
            f"No exchange rate for {from_currency}->{to_currency} on {rate_date}"
        )
    return rate


def convert(
    dataset: Dataset,
    amount: float,
    from_currency: str,
    to_currency: str,
    rate_date: str,
) -> float:
    rate = get_rate(dataset, rate_date, from_currency, to_currency)
    return amount * rate


def get_nearest_rate(dataset: Dataset, rate_date: str, from_currency: str, to_currency: str) -> float:
    """Best-effort lookup for a *synthetic* date (a recurrence projection),
    where no exchange_rates.csv row can exist since the date was never
    real. Exact lookup is tried first; failing that, the closest dated
    rate for the same currency pair is used (preferring one on/before
    rate_date, else the earliest available) rather than raising -- unlike
    get_rate, which stays strict for real events per the spec.
    """
    if from_currency == to_currency:
        return 1.0
    exact = dataset.rate_by_key.get((rate_date, from_currency, to_currency))
    if exact is not None:
        return exact
    candidates = [r for r in dataset.exchange_rates if r.from_currency == from_currency and r.to_currency == to_currency]
    if not candidates:
        raise MissingExchangeRateError(f"No exchange rate for {from_currency}->{to_currency} on any date")
    on_or_before = [r for r in candidates if r.rate_date <= rate_date]
    chosen = max(on_or_before, key=lambda r: r.rate_date) if on_or_before else min(candidates, key=lambda r: r.rate_date)
    return chosen.rate


def convert_nearest(
    dataset: Dataset,
    amount: float,
    from_currency: str,
    to_currency: str,
    rate_date: str,
) -> float:
    return amount * get_nearest_rate(dataset, rate_date, from_currency, to_currency)
