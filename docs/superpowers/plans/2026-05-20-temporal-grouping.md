# Temporal Grouping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `temporal_grouping` parameter to `merge()` that groups sources into temporal buckets and produces an output array with a time dimension.

**Architecture:** New `temporal.py` module with grouper classes (strategy pattern) handles bucketing RFC 3339 datetimes into periods. `sources.py` gets a `query_temporal_groups()` function to discover distinct time groups via DataFusion. `merge.py` prepends a time dimension to the target array and passes temporal SQL filters per-chunk to `_merge_block()`.

**Tech Stack:** Python 3.12, numpy (datetime64), DataFusion SQL, cubed (lazy arrays), zarr, pytest

---

### Task 1: Create `temporal.py` — grouper classes and factory

**Files:**
- Create: `lazymerge/temporal.py`
- Test: `tests/test_temporal.py`

- [ ] **Step 1: Write failing tests for `_DayGrouper`**

Create `tests/test_temporal.py`:

```python
import numpy as np
import pytest

from lazymerge.temporal import grouper_from_period


class TestDayGrouper:
    def setup_method(self):
        self.grouper = grouper_from_period("P1D")

    def test_group_key(self):
        assert self.grouper.group_key("2024-06-15T10:30:00Z") == "2024-06-15"

    def test_group_key_strips_time(self):
        assert self.grouper.group_key("2024-01-01T23:59:59Z") == "2024-01-01"

    def test_datetime_filter(self):
        start, end = self.grouper.datetime_filter("2024-06-15")
        assert start == "2024-06-15T00:00:00Z"
        assert end == "2024-06-16T00:00:00Z"

    def test_to_datetime64(self):
        result = self.grouper.to_datetime64("2024-06-15")
        assert result == np.datetime64("2024-06-15", "D")

    def test_keys_sort_chronologically(self):
        keys = [
            self.grouper.group_key("2024-12-31T00:00:00Z"),
            self.grouper.group_key("2024-01-01T00:00:00Z"),
            self.grouper.group_key("2024-06-15T00:00:00Z"),
        ]
        assert sorted(keys) == ["2024-01-01", "2024-06-15", "2024-12-31"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_temporal.py -v`
Expected: `ModuleNotFoundError: No module named 'lazymerge.temporal'`

- [ ] **Step 3: Write failing tests for `_WeekGrouper`**

Append to `tests/test_temporal.py`:

```python
class TestWeekGrouper:
    def setup_method(self):
        self.grouper = grouper_from_period("P1W")

    def test_group_key_monday(self):
        # 2024-06-17 is a Monday
        assert self.grouper.group_key("2024-06-17T10:00:00Z") == "2024-W25"

    def test_group_key_sunday_same_week(self):
        # 2024-06-23 is a Sunday — same ISO week as Monday 2024-06-17
        assert self.grouper.group_key("2024-06-23T10:00:00Z") == "2024-W25"

    def test_group_key_year_boundary(self):
        # 2024-12-30 is a Monday in ISO week 1 of 2025
        assert self.grouper.group_key("2024-12-30T10:00:00Z") == "2025-W01"

    def test_datetime_filter(self):
        start, end = self.grouper.datetime_filter("2024-W25")
        assert start == "2024-06-17T00:00:00Z"
        assert end == "2024-06-24T00:00:00Z"

    def test_to_datetime64(self):
        result = self.grouper.to_datetime64("2024-W25")
        # Monday of ISO week 25 in 2024 is June 17
        assert result == np.datetime64("2024-06-17", "D")
```

- [ ] **Step 4: Write failing tests for `_MonthGrouper`**

Append to `tests/test_temporal.py`:

```python
class TestMonthGrouper:
    def setup_method(self):
        self.grouper = grouper_from_period("P1M")

    def test_group_key(self):
        assert self.grouper.group_key("2024-06-15T10:00:00Z") == "2024-06"

    def test_group_key_first_day(self):
        assert self.grouper.group_key("2024-01-01T00:00:00Z") == "2024-01"

    def test_datetime_filter(self):
        start, end = self.grouper.datetime_filter("2024-06")
        assert start == "2024-06-01T00:00:00Z"
        assert end == "2024-07-01T00:00:00Z"

    def test_datetime_filter_december(self):
        start, end = self.grouper.datetime_filter("2024-12")
        assert start == "2024-12-01T00:00:00Z"
        assert end == "2025-01-01T00:00:00Z"

    def test_to_datetime64(self):
        result = self.grouper.to_datetime64("2024-06")
        assert result == np.datetime64("2024-06-01", "D")
```

- [ ] **Step 5: Write failing tests for `_YearGrouper`**

Append to `tests/test_temporal.py`:

```python
class TestYearGrouper:
    def setup_method(self):
        self.grouper = grouper_from_period("P1Y")

    def test_group_key(self):
        assert self.grouper.group_key("2024-06-15T10:00:00Z") == "2024"

    def test_datetime_filter(self):
        start, end = self.grouper.datetime_filter("2024")
        assert start == "2024-01-01T00:00:00Z"
        assert end == "2025-01-01T00:00:00Z"

    def test_to_datetime64(self):
        result = self.grouper.to_datetime64("2024")
        assert result == np.datetime64("2024-01-01", "D")
```

- [ ] **Step 6: Write failing tests for `_FixedDayGrouper`**

Append to `tests/test_temporal.py`:

```python
class TestFixedDayGrouper:
    def setup_method(self):
        self.grouper = grouper_from_period("P16D")

    def test_group_key_epoch_start(self):
        # 2000-01-01 is the epoch — should be bucket 000000
        key = self.grouper.group_key("2000-01-01T00:00:00Z")
        assert key == "000000"

    def test_group_key_same_bucket(self):
        # Days 0-15 from epoch are bucket 0
        key1 = self.grouper.group_key("2000-01-01T00:00:00Z")
        key2 = self.grouper.group_key("2000-01-16T00:00:00Z")
        assert key1 == key2

    def test_group_key_next_bucket(self):
        key = self.grouper.group_key("2000-01-17T00:00:00Z")
        assert key == "000001"

    def test_keys_sort_chronologically(self):
        keys = [
            self.grouper.group_key("2024-06-15T00:00:00Z"),
            self.grouper.group_key("2000-01-01T00:00:00Z"),
            self.grouper.group_key("2024-01-01T00:00:00Z"),
        ]
        assert keys == sorted(keys) or sorted(keys) == sorted(keys)
        # More precisely: they should sort in temporal order
        assert sorted(keys) == [
            self.grouper.group_key("2000-01-01T00:00:00Z"),
            self.grouper.group_key("2024-01-01T00:00:00Z"),
            self.grouper.group_key("2024-06-15T00:00:00Z"),
        ]

    def test_datetime_filter(self):
        start, end = self.grouper.datetime_filter("000001")
        assert start == "2000-01-17T00:00:00Z"
        assert end == "2000-02-02T00:00:00Z"

    def test_to_datetime64(self):
        result = self.grouper.to_datetime64("000001")
        assert result == np.datetime64("2000-01-17", "D")
```

- [ ] **Step 7: Write failing tests for `grouper_from_period` factory**

Append to `tests/test_temporal.py`:

```python
class TestGrouperFromPeriod:
    def test_p1d(self):
        g = grouper_from_period("P1D")
        assert g.group_key("2024-01-01T00:00:00Z") == "2024-01-01"

    def test_p1w(self):
        g = grouper_from_period("P1W")
        assert g.group_key("2024-06-17T00:00:00Z") == "2024-W25"

    def test_p1m(self):
        g = grouper_from_period("P1M")
        assert g.group_key("2024-06-15T00:00:00Z") == "2024-06"

    def test_p1y(self):
        g = grouper_from_period("P1Y")
        assert g.group_key("2024-06-15T00:00:00Z") == "2024"

    def test_p16d(self):
        g = grouper_from_period("P16D")
        assert g.group_key("2000-01-17T00:00:00Z") == "000001"

    def test_p2w_converts_to_fixed_14(self):
        g = grouper_from_period("P2W")
        # P2W = 14-day fixed window
        key1 = g.group_key("2000-01-01T00:00:00Z")
        key2 = g.group_key("2000-01-14T00:00:00Z")
        assert key1 == key2

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            grouper_from_period("invalid")

    def test_unsupported_duration(self):
        with pytest.raises(ValueError):
            grouper_from_period("PT1H")
```

- [ ] **Step 8: Run tests to confirm all fail**

Run: `python -m pytest tests/test_temporal.py -v`
Expected: All tests fail with `ModuleNotFoundError`

- [ ] **Step 9: Implement `lazymerge/temporal.py`**

Create `lazymerge/temporal.py`:

```python
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
        return datetime.fromisoformat(datetime_str.replace("Z", "+00:00")).date()


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
        if month == 12:
            end_date = date(year + 1, 1, 1)
        else:
            end_date = date(year, month + 1, 1)
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
```

- [ ] **Step 10: Run tests to verify they pass**

Run: `python -m pytest tests/test_temporal.py -v`
Expected: All tests PASS

- [ ] **Step 11: Commit**

```bash
git add lazymerge/temporal.py tests/test_temporal.py
git commit -m "feat: add temporal grouper classes and factory"
```

---

### Task 2: Add `query_temporal_groups()` to `sources.py`

**Files:**
- Modify: `lazymerge/sources.py`
- Test: `tests/test_temporal.py` (append)

- [ ] **Step 1: Write failing test for `query_temporal_groups`**

Append to `tests/test_temporal.py`:

```python
from unittest.mock import patch, MagicMock


class TestQueryTemporalGroups:
    def test_groups_datetimes_by_period(self):
        """query_temporal_groups should bucket distinct datetimes and return sorted group keys."""
        from lazymerge.temporal import grouper_from_period
        from lazymerge.sources import query_temporal_groups

        grouper = grouper_from_period("P1M")

        # Mock the DataFusion query to return distinct datetime values
        mock_batch = MagicMock()
        mock_batch.num_rows = 4
        mock_col = MagicMock()
        mock_col.__getitem__ = lambda self, i: MagicMock(
            as_py=lambda: [
                "2024-06-15T10:00:00Z",
                "2024-06-20T10:00:00Z",
                "2024-07-01T10:00:00Z",
                "2024-08-10T10:00:00Z",
            ][i]
        )
        mock_batch.column.return_value = mock_col

        with patch("lazymerge.sources.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = [
                "2024-06-15T10:00:00Z",
                "2024-06-20T10:00:00Z",
                "2024-07-01T10:00:00Z",
                "2024-08-10T10:00:00Z",
            ]
            groups = query_temporal_groups(
                store=MagicMock(),
                bbox_4326=(-180, -90, 180, 90),
                grouper=grouper,
            )

        assert groups == ["2024-06", "2024-07", "2024-08"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_temporal.py::TestQueryTemporalGroups -v`
Expected: FAIL with `ImportError: cannot import name 'query_temporal_groups'`

- [ ] **Step 3: Implement `query_temporal_groups` in `sources.py`**

Add the following import at the top of `lazymerge/sources.py` (after the existing `from typing import cast` line):

```python
from typing import Any, cast
```

(Change `from typing import cast` to `from typing import Any, cast` — `Any` is already used but imported implicitly via the function signature.)

Then add this function at the end of `lazymerge/sources.py`:

```python
def query_temporal_groups(
    store: Any,
    bbox_4326: tuple[float, float, float, float],
    grouper: Any,
    sql_filter: str | None = None,
) -> list[str]:
    """Query distinct datetime values from /meta and bucket them into temporal groups.

    Args:
        store: An obstore-compatible object store, Icechunk Session, or zarr Store.
        bbox_4326: Query bounding box in EPSG:4326 (xmin, ymin, xmax, ymax).
        grouper: A TemporalGrouper instance used to bucket datetime strings.
        sql_filter: Optional SQL expression appended as an AND clause.

    Returns:
        Sorted list of unique group keys.
    """
    import asyncio
    from datafusion import SessionContext
    from geodatafusion import register_all
    from zarr_datafusion_search import ZarrTable

    async def _query() -> list[str]:
        try:
            from icechunk import IcechunkStore
            if isinstance(store, IcechunkStore):
                zarr_table = await ZarrTable.from_icechunk(
                    session=store.session, group_path="/meta"
                )
            else:
                zarr_table = await ZarrTable.from_obstore(store, "/meta")
        except ImportError:
            zarr_table = await ZarrTable.from_obstore(store, "/meta")

        ctx = SessionContext()
        register_all(ctx)
        ctx.register_table("meta", zarr_table)

        xmin, ymin, xmax, ymax = bbox_4326
        query = (
            "SELECT DISTINCT datetime "
            "FROM meta "
            "WHERE ST_Intersects(bbox, ST_GeomFromText("
            f"'POLYGON(({xmin} {ymin}, {xmax} {ymin}, {xmax} {ymax}, {xmin} {ymax}, {xmin} {ymin}))'"
            "))"
        )
        if sql_filter is not None:
            query += f" AND ({sql_filter})"

        df = ctx.sql(query)
        batches = df.collect()

        datetimes: list[str] = []
        for batch in batches:
            for i in range(batch.num_rows):
                datetimes.append(str(batch.column("datetime")[i].as_py()))
        return datetimes

    datetimes = asyncio.run(_query())
    keys = sorted(set(grouper.group_key(dt) for dt in datetimes))
    return keys
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_temporal.py::TestQueryTemporalGroups -v`
Expected: PASS

- [ ] **Step 5: Run all existing tests to check for regressions**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add lazymerge/sources.py tests/test_temporal.py
git commit -m "feat: add query_temporal_groups for temporal bucketing via DataFusion"
```

---

### Task 3: Add `temporal_grouping` parameter to `merge()` and `_merge_block()`

**Files:**
- Modify: `lazymerge/merge.py:93-348`
- Test: `tests/test_merge.py` (append)

- [ ] **Step 1: Write failing test — temporal_grouping requires datafusion**

Append to `tests/test_merge.py`:

```python
def test_merge_temporal_grouping_requires_datafusion():
    """temporal_grouping without datafusion=True should raise ValueError."""
    store, root = _make_same_crs_store()
    index = scan_store(root)

    with pytest.raises(ValueError, match="temporal_grouping requires datafusion=True"):
        merge(
            store=store,
            crs="EPSG:32618",
            bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
            resolution=10.0,
            source_index=index,
            temporal_grouping="P1D",
        )
```

Add `import pytest` at the top of `tests/test_merge.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_merge.py::test_merge_temporal_grouping_requires_datafusion -v`
Expected: FAIL with `TypeError: merge() got an unexpected keyword argument 'temporal_grouping'`

- [ ] **Step 3: Write failing test — temporal_grouping produces time dimension**

Append to `tests/test_merge.py`:

```python
def test_merge_temporal_grouping_creates_time_dimension():
    """temporal_grouping should produce a 3D (time, y, x) output."""
    store, root = _make_same_crs_store()

    # Mock query_temporal_groups to return 2 monthly groups
    # Mock query_datafusion_sources to return sources filtered by time
    from lazymerge.sources import SourceEntry, ScanIndex

    entries = []
    for name, x_origin in [("source_a", 500000.0), ("source_b", 501000.0)]:
        entries.append(SourceEntry(
            path=name,
            spatial_attrs=SpatialAttrs(
                dimensions=["y", "x"],
                transform=(10.0, 0.0, x_origin, 0.0, -10.0, 6000000.0),
                bbox=(x_origin, 5999000.0, x_origin + 1000.0, 6000000.0),
                shape=(100, 100),
            ),
            proj_attrs=ProjAttrs(code="EPSG:32618"),
            chunk_shape=(50, 50),
        ))

    with patch("lazymerge.merge.query_temporal_groups", return_value=["2024-06", "2024-07"]), \
         patch("lazymerge.merge.query_datafusion_sources", return_value=entries):
        result_arr, _, _ = merge(
            store=store,
            crs="EPSG:32618",
            bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
            resolution=10.0,
            chunk_size=(50, 50),
            datafusion=True,
            temporal_grouping="P1M",
        )

    data = result_arr.compute()
    # 2 time steps, 100 rows, 200 cols
    assert data.shape == (2, 100, 200)
```

- [ ] **Step 4: Write failing test — temporal_grouping + bands produces 4D**

Append to `tests/test_merge.py`:

```python
def test_merge_temporal_grouping_with_bands():
    """temporal_grouping + bands should produce a 4D (time, band, y, x) output."""
    store, root = _make_multiband_store()

    from lazymerge.sources import SourceEntry

    entries = []
    for name, x_origin in [("scene_a", 500000.0), ("scene_b", 501000.0)]:
        entries.append(SourceEntry(
            path=name,
            spatial_attrs=SpatialAttrs(
                dimensions=["y", "x"],
                transform=(10.0, 0.0, x_origin, 0.0, -10.0, 6000000.0),
                bbox=(x_origin, 5999000.0, x_origin + 1000.0, 6000000.0),
                shape=(100, 100),
            ),
            proj_attrs=ProjAttrs(code="EPSG:32618"),
            chunk_shape=(50, 50),
        ))

    with patch("lazymerge.merge.query_temporal_groups", return_value=["2024-06", "2024-07"]), \
         patch("lazymerge.merge.query_datafusion_sources", return_value=entries):
        result_arr, _, _ = merge(
            store=store,
            crs="EPSG:32618",
            bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
            resolution=10.0,
            chunk_size=(50, 50),
            datafusion=True,
            temporal_grouping="P1M",
            bands=["red", "green"],
        )

    data = result_arr.compute()
    # 2 time steps, 2 bands, 100 rows, 200 cols
    assert data.shape == (2, 2, 100, 200)
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `python -m pytest tests/test_merge.py::test_merge_temporal_grouping_requires_datafusion tests/test_merge.py::test_merge_temporal_grouping_creates_time_dimension tests/test_merge.py::test_merge_temporal_grouping_with_bands -v`
Expected: All three FAIL

- [ ] **Step 6: Implement temporal_grouping in `merge()` and `_merge_block()`**

Modify `lazymerge/merge.py`:

**Add imports** at the top (after existing imports):

```python
from lazymerge.sources import ScanIndex, SourceEntry, select_overview, query_datafusion_sources, query_temporal_groups
from lazymerge.temporal import TemporalGrouper, grouper_from_period
```

(Replace the existing `from lazymerge.sources import ...` line to add `query_temporal_groups`.)

**Update `_merge_block` signature** (line 93) to accept temporal grouping kwargs:

```python
def _merge_block(
    block: np.ndarray,
    block_id: tuple[int, ...],
    source_index: ScanIndex | None,
    target_spatial: SpatialAttrs,
    target_proj: ProjAttrs,
    store: Any,
    chunk_size: tuple[int, int],
    resampling: str,
    bands: list[str] | None = None,
    datafusion: bool = False,
    sortby: str | None = None,
    nodata: float | int | None = None,
    sql_filter: str | None = None,
    temporal_grouper: TemporalGrouper | None = None,
    time_groups: list[str] | None = None,
) -> np.ndarray:
```

**Update block_id unpacking** in `_merge_block` (replace lines 108-119):

```python
    target_crs = target_proj.code
    if target_crs is None:
        raise ValueError("target_proj.code must not be None")

    has_time = temporal_grouper is not None and time_groups is not None
    multi_band = bands is not None and len(bands) > 1

    # Unpack block_id based on which dimensions are present.
    # Dimension order: [time], [band], row, col
    idx = list(block_id)
    time_idx: int | None = None
    if has_time:
        time_idx = idx.pop(0)
    if multi_band:
        band_idx = idx.pop(0)
        band = bands[band_idx]
    else:
        band = bands[0] if bands else None
    row_idx, col_idx = idx
```

**Update actual_shape computation** (replace lines 131-135):

```python
    # Actual chunk shape (may be smaller at edges) — always the last 2 dims
    actual_shape: tuple[int, int] = (block.shape[-2], block.shape[-1])
```

**Update `_result` helper** (replace lines 140-141):

```python
    def _result(arr: np.ndarray) -> np.ndarray:
        # Prepend singleton dimensions for time and/or band as needed
        if has_time and multi_band:
            return arr[np.newaxis, np.newaxis, :, :]
        if has_time or multi_band:
            return arr[np.newaxis, :, :]
        return arr
```

**Update the temporal SQL filter** in the datafusion branch (replace lines 143-150):

```python
    # Build effective SQL filter combining user filter and temporal filter
    effective_sql_filter = sql_filter
    if has_time and time_idx is not None:
        group_key = time_groups[time_idx]
        t_start, t_end = temporal_grouper.datetime_filter(group_key)
        temporal_clause = f"\"datetime\" >= '{t_start}' AND \"datetime\" < '{t_end}'"
        if effective_sql_filter is not None:
            effective_sql_filter = f"({effective_sql_filter}) AND ({temporal_clause})"
        else:
            effective_sql_filter = temporal_clause

    # Pass 1: find intersecting sources
    if datafusion:
        bbox_4326 = _reproject_bbox_to_4326(cb, target_crs)
        sources = query_datafusion_sources(store, bbox_4326, sortby=sortby, sql_filter=effective_sql_filter)
    elif source_index is not None:
        sources = source_index.find_intersecting_sources(cb, target_crs)
    else:
        return _result(output)
```

**Update `merge()` signature** (replace lines 291-304):

```python
def merge(
    store: Any,
    crs: str,
    bbox: tuple[float, float, float, float],
    resolution: float,
    chunk_size: tuple[int, int] = (512, 512),
    source_index: ScanIndex | None = None,
    resampling: str = "nearest",
    bands: list[str] | str | None = None,
    datafusion: bool = False,
    sortby: str | None = None,
    nodata: float | int | None = None,
    sql_filter: str | None = None,
    dtype: str = "float32",
    temporal_grouping: str | None = None,
) -> tuple[cubed.Array, SpatialAttrs, ProjAttrs]:
```

**Add validation and temporal setup** in `merge()` (after `bands_list` normalisation, before `create_target`):

```python
    # Validate temporal_grouping
    temporal_grouper: TemporalGrouper | None = None
    time_groups: list[str] | None = None
    if temporal_grouping is not None:
        if not datafusion:
            raise ValueError("temporal_grouping requires datafusion=True")
        temporal_grouper = grouper_from_period(temporal_grouping)
        bbox_4326 = _reproject_bbox_to_4326(bbox, crs)
        time_groups = query_temporal_groups(
            store, bbox_4326, temporal_grouper, sql_filter=sql_filter,
        )
```

**Update target array construction** in `merge()` (replace the block from `create_target` through the multi_band section):

```python
    target, target_spatial, target_proj = create_target(
        crs=crs, bbox=bbox, resolution=resolution,
        chunk_size=chunk_size, dtype=dtype,
    )

    multi_band = bands_list is not None and len(bands_list) > 1
    has_time = time_groups is not None and len(time_groups) > 0

    # Build the full shape with optional leading dimensions: [time], [band], H, W
    if has_time and multi_band:
        n_times = len(time_groups)
        n_bands = len(bands_list)
        target = cubed.full(
            shape=(n_times, n_bands, *target.shape),
            fill_value=float("nan"),
            dtype=target.dtype,
            chunks=(1, 1, *target.chunksize),
        )
    elif has_time:
        n_times = len(time_groups)
        target = cubed.full(
            shape=(n_times, *target.shape),
            fill_value=float("nan"),
            dtype=target.dtype,
            chunks=(1, *target.chunksize),
        )
    elif multi_band:
        n_bands = len(bands_list)
        target = cubed.full(
            shape=(n_bands, *target.shape),
            fill_value=float("nan"),
            dtype=target.dtype,
            chunks=(1, *target.chunksize),
        )
```

**Update `map_blocks` call** to pass temporal kwargs:

```python
    result = cubed.map_blocks(
        _merge_block,
        target,
        dtype=target.dtype,
        chunks=target.chunks,
        source_index=source_index,
        target_spatial=target_spatial,
        target_proj=target_proj,
        store=store,
        chunk_size=chunk_size,
        resampling=resampling,
        bands=bands_list,
        datafusion=datafusion,
        sortby=sortby,
        nodata=nodata,
        sql_filter=sql_filter,
        temporal_grouper=temporal_grouper,
        time_groups=time_groups,
    )

    return result, target_spatial, target_proj
```

- [ ] **Step 7: Run temporal grouping tests**

Run: `python -m pytest tests/test_merge.py::test_merge_temporal_grouping_requires_datafusion tests/test_merge.py::test_merge_temporal_grouping_creates_time_dimension tests/test_merge.py::test_merge_temporal_grouping_with_bands -v`
Expected: All three PASS

- [ ] **Step 8: Run full test suite for regressions**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 9: Commit**

```bash
git add lazymerge/merge.py tests/test_merge.py
git commit -m "feat: add temporal_grouping parameter to merge for time-dimensioned output"
```

---

### Task 4: Export public API and final verification

**Files:**
- Modify: `lazymerge/__init__.py`

- [ ] **Step 1: Add temporal exports to `__init__.py`**

Add to `lazymerge/__init__.py`:

```python
from lazymerge.temporal import (
    TemporalGrouper as TemporalGrouper,
    grouper_from_period as grouper_from_period,
)
```

Also add `query_temporal_groups` to the sources import:

```python
from lazymerge.sources import (
    ScanIndex as ScanIndex,
    SourceEntry as SourceEntry,
    query_datafusion_sources as query_datafusion_sources,
    query_temporal_groups as query_temporal_groups,
    scan_store as scan_store,
    select_overview as select_overview,
)
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add lazymerge/__init__.py
git commit -m "feat: export TemporalGrouper and grouper_from_period from public API"
```
