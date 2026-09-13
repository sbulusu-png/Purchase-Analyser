"""CSV loading for the dataset/ folder, with convenience indices for joins."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from schemas import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageRecord,
    Message,
    PaymentOption,
    Request,
)


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@dataclass
class Dataset:
    dataset_dir: Path

    requests: list[Request] = field(default_factory=list)
    sample_requests: list[Request] = field(default_factory=list)
    profiles: list[FinancialProfile] = field(default_factory=list)
    events: list[FinancialEvent] = field(default_factory=list)
    payment_options: list[PaymentOption] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    images: list[ImageRecord] = field(default_factory=list)
    exchange_rates: list[ExchangeRate] = field(default_factory=list)

    profile_by_user: dict[str, FinancialProfile] = field(default_factory=dict)
    events_by_user: dict[str, list[FinancialEvent]] = field(default_factory=lambda: defaultdict(list))
    event_by_id: dict[str, FinancialEvent] = field(default_factory=dict)
    payment_options_by_request: dict[str, list[PaymentOption]] = field(
        default_factory=lambda: defaultdict(list)
    )
    messages_by_request: dict[str, list[Message]] = field(default_factory=lambda: defaultdict(list))
    messages_by_event: dict[str, list[Message]] = field(default_factory=lambda: defaultdict(list))
    messages_by_user: dict[str, list[Message]] = field(default_factory=lambda: defaultdict(list))
    image_by_event: dict[str, ImageRecord] = field(default_factory=dict)
    images_by_request: dict[str, list[ImageRecord]] = field(default_factory=lambda: defaultdict(list))
    rate_by_key: dict[tuple[str, str, str], float] = field(default_factory=dict)

    @classmethod
    def load(cls, dataset_dir: str | Path) -> "Dataset":
        d = Path(dataset_dir)
        ds = cls(dataset_dir=d)

        ds.requests = [Request.from_row(r) for r in _read_rows(d / "requests.csv")]
        ds.sample_requests = [
            Request.from_row(r) for r in _read_rows(d / "sample_requests.csv")
        ]
        ds.profiles = [FinancialProfile.from_row(r) for r in _read_rows(d / "financial_profiles.csv")]
        ds.events = [FinancialEvent.from_row(r) for r in _read_rows(d / "financial_events.csv")]
        ds.payment_options = [
            PaymentOption.from_row(r) for r in _read_rows(d / "request_payment_options.csv")
        ]
        ds.messages = [Message.from_row(r) for r in _read_rows(d / "messages.csv")]
        ds.images = [ImageRecord.from_row(r) for r in _read_rows(d / "images.csv")]
        ds.exchange_rates = [
            ExchangeRate.from_row(r) for r in _read_rows(d / "exchange_rates.csv")
        ]

        ds._build_indices()
        return ds

    def _build_indices(self) -> None:
        for p in self.profiles:
            self.profile_by_user[p.user_id] = p

        for e in self.events:
            self.events_by_user[e.user_id].append(e)
            self.event_by_id[e.event_id] = e

        for po in self.payment_options:
            self.payment_options_by_request[po.request_id].append(po)

        for m in self.messages:
            self.messages_by_user[m.user_id].append(m)
            if m.request_id:
                self.messages_by_request[m.request_id].append(m)
            if m.related_event_id:
                self.messages_by_event[m.related_event_id].append(m)

        for img in self.images:
            if img.related_event_id:
                self.image_by_event[img.related_event_id] = img
            if img.request_id:
                self.images_by_request[img.request_id].append(img)

        for r in self.exchange_rates:
            self.rate_by_key[(r.rate_date, r.from_currency, r.to_currency)] = r.rate

    def image_path(self, image_id: str) -> Path:
        return self.dataset_dir / "media" / "images" / f"{image_id}.png"

    def all_evaluation_and_sample_requests(self) -> list[Request]:
        return [*self.requests, *self.sample_requests]
