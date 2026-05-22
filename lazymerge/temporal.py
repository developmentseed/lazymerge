from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta

import numpy as np

_EPOCH = date(2000, 1, 1)


class TemporalGrouper(ABC):
    @abstractmethod
    def group_key(self, datetime_str: str) -> str:
        """Map an RFC 3339 datetime string to a sortable group label."""

    @abstractmethod
    def datetime_filter(self, group_key: str) -> tuple[str, str]:
        """Return (start, end) datetime range for SQL filtering. End is exclusive."""

    @abstractmethod
    def to_datetime64(self, group_key: str) -> np.datetime64:
        """Produce a datetime64[D] coordinate value for the group."""

    def _parse_date(self, datetime_str: str) -> date:
        return datetime.fromisoformat(datetime_str).date()


class _DayGrouper(TemporalGrouper):
    def group_key(self, datetime_str: str) -> str:
        return self._parse_date(datetime_str).isoformat()

    def datetime_filter(self, group_key: str) -> tuple[str, str]:
        d = date.fromisoformat(group_key)
        start = f"{d.isoformat()}T00:00:00Z"
        end = f"{(d + timedelta(days=1)).isoformat()}T00:00:00Z"
        return start, end

    def to_datetime64(self, group_key: str) -> np.datetime64:
        return np.datetime64(group_key, "D")


class _WeekGrouper(TemporalGrouper):
    def group_key(self, datetime_str: str) -> str:
        d = self._parse_date(datetime_str)
        iso_year, iso_week, _ = d.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"

    def datetime_filter(self, group_key: str) -> tuple[str, str]:
        monday = self._monday_of(group_key)
        start = f"{monday.isoformat()}T00:00:00Z"
        end = f"{(monday + timedelta(days=7)).isoformat()}T00:00:00Z"
        return start, end

    def to_datetime64(self, group_key: str) -> np.datetime64:
        return np.datetime64(self._monday_of(group_key).isoformat(), "D")

    def _monday_of(self, group_key: str) -> date:
        year_str, week_str = group_key.split("-W")
        return date.fromisocalendar(int(year_str), int(week_str), 1)


class _MonthGrouper(TemporalGrouper):
    def group_key(self, datetime_str: str) -> str:
        d = self._parse_date(datetime_str)
        return f"{d.year}-{d.month:02d}"

    def datetime_filter(self, group_key: str) -> tuple[str, str]:
        year, month = (int(x) for x in group_key.split("-"))
        start_date = date(year, month, 1)
        end_date = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
        return f"{start_date.isoformat()}T00:00:00Z", f"{end_date.isoformat()}T00:00:00Z"

    def to_datetime64(self, group_key: str) -> np.datetime64:
        return np.datetime64(f"{group_key}-01", "D")


class _YearGrouper(TemporalGrouper):
    def group_key(self, datetime_str: str) -> str:
        return str(self._parse_date(datetime_str).year)

    def datetime_filter(self, group_key: str) -> tuple[str, str]:
        year = int(group_key)
        return f"{year}-01-01T00:00:00Z", f"{year + 1}-01-01T00:00:00Z"

    def to_datetime64(self, group_key: str) -> np.datetime64:
        return np.datetime64(f"{group_key}-01-01", "D")


class _FixedDayGrouper(TemporalGrouper):
    def __init__(self, n: int) -> None:
        self._n = n

    def group_key(self, datetime_str: str) -> str:
        d = self._parse_date(datetime_str)
        bucket = (d - _EPOCH).days // self._n
        return f"{bucket:06d}"

    def datetime_filter(self, group_key: str) -> tuple[str, str]:
        bucket = int(group_key)
        start_date = _EPOCH + timedelta(days=bucket * self._n)
        end_date = start_date + timedelta(days=self._n)
        return f"{start_date.isoformat()}T00:00:00Z", f"{end_date.isoformat()}T00:00:00Z"

    def to_datetime64(self, group_key: str) -> np.datetime64:
        bucket = int(group_key)
        d = _EPOCH + timedelta(days=bucket * self._n)
        return np.datetime64(d.isoformat(), "D")


def grouper_from_period(period: str) -> TemporalGrouper:
    """Parse an ISO 8601 duration string and return the appropriate grouper.

    Supported formats: P1D, P1W, P1M, P1Y, PnD (n >= 2), PnW (converted to n*7 day fixed window).
    """
    match = re.fullmatch(r"P(\d+)([DWMY])", period)
    if match is None:
        raise ValueError(f"Unsupported temporal grouping period: {period!r}")

    n = int(match.group(1))
    unit = match.group(2)

    if unit == "D" and n == 1:
        return _DayGrouper()
    if unit == "D" and n >= 2:
        return _FixedDayGrouper(n)
    if unit == "W" and n == 1:
        return _WeekGrouper()
    if unit == "W" and n >= 2:
        return _FixedDayGrouper(n * 7)
    if unit == "M" and n == 1:
        return _MonthGrouper()
    if unit == "Y" and n == 1:
        return _YearGrouper()

    raise ValueError(f"Unsupported temporal grouping period: {period!r}")
