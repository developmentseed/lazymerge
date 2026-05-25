# Temporal Grouping for lazymerge

## Overview

Add a `temporal_grouping` parameter to `merge()` that creates a time dimension in the output array by grouping source scenes into temporal buckets (calendar day, week, month, year, or fixed n-day windows). Mirrors the temporal grouping concept from [lazycogs](https://developmentseed.org/lazycogs/guides/temporal-grouping/).

When `temporal_grouping` is not passed, behavior is unchanged.

## Constraints

- `temporal_grouping` requires `datafusion=True` — it queries the `/meta` table's `datetime` column.
- The datetime column is always named `"datetime"` — no configurability needed.
- ISO 8601 duration string format: `P1D`, `P1W`, `P1M`, `P1Y`, `P16D`, `PnD`.

## Architecture

### New module: `lazymerge/temporal.py`

Port the lazycogs grouper pattern with these components:

**Abstract base:** `TemporalGrouper` with three methods:
- `group_key(datetime_str: str) -> str` — maps an RFC 3339 datetime string to a sortable group label.
- `datetime_filter(group_key: str) -> tuple[str, str]` — returns `(start, end)` datetime range for SQL filtering of a group.
- `to_datetime64(group_key: str) -> np.datetime64` — produces a `datetime64[D]` coordinate value for the group.

**Concrete implementations:**
- `_DayGrouper` — calendar day (`P1D`)
- `_WeekGrouper` — ISO week anchored on Monday (`P1W`)
- `_MonthGrouper` — calendar month (`P1M`)
- `_YearGrouper` — calendar year (`P1Y`)
- `_FixedDayGrouper(n)` — n-day windows aligned to epoch 2000-01-01 (`PnD` where n >= 2)

**Factory:** `grouper_from_period(period: str) -> TemporalGrouper` — parses ISO 8601 duration strings, returns the appropriate grouper. Raises `ValueError` for unsupported formats.

### Changes to `lazymerge/sources.py`

Add a new function `query_temporal_groups()`:
- Queries `SELECT DISTINCT datetime FROM meta WHERE ST_Intersects(...) AND (sql_filter)`.
- Applies the grouper to bucket results into ordered, deduplicated group keys.
- Returns a list of group keys (sorted).

### Changes to `lazymerge/merge.py`

**`merge()` function:**
1. Accept new parameter `temporal_grouping: str | None = None`.
2. Validate: if `temporal_grouping` is set but `datafusion` is `False`, raise `ValueError`.
3. When `temporal_grouping` is set:
   - Create a grouper via `grouper_from_period(temporal_grouping)`.
   - Run `query_temporal_groups()` against the full bbox (reprojected to 4326) with the sql_filter to get the list of time groups.
   - Prepend a time dimension to the target array, sized `n_times`.
4. Pass the grouper, time group keys, and temporal grouping flag through to `_merge_block` via `map_blocks` kwargs.

**`_merge_block()` function:**
1. When temporal grouping is active, extract the time index from `block_id` and look up the group key.
2. Use `grouper.datetime_filter(group_key)` to build a temporal SQL filter clause (e.g., `"datetime" >= '...' AND "datetime" < '...'`).
3. Combine with existing `sql_filter` when calling `query_datafusion_sources()`.

**Dimension stacking order:**
- No time, no bands: `(H, W)` — block_id is `(row, col)`
- Bands only: `(n_bands, H, W)` — block_id is `(band, row, col)`
- Time only: `(n_times, H, W)` — block_id is `(time, row, col)`
- Time + bands: `(n_times, n_bands, H, W)` — block_id is `(time, band, row, col)`

### Changes to `lazymerge/__init__.py`

Export `grouper_from_period` and `TemporalGrouper` from the public API so callers can reconstruct time coordinates from group keys using `grouper.to_datetime64(key)`.

## What does not change

- `target.py`, `warp.py`, `conventions.py`, `explain.py` — no modifications needed.
- `ScanIndex`-based (non-datafusion) merge paths — unaffected.
- Return type of `merge()` stays `tuple[cubed.Array, SpatialAttrs, ProjAttrs]`.

## Testing

- Unit tests for each grouper (key generation, datetime filter ranges, datetime64 conversion).
- Unit test for `grouper_from_period` factory (valid periods + invalid input).
- Integration test: `merge()` with `temporal_grouping="P1D"` produces array with time dimension of expected size.
- Integration test: time + bands produces 4D array.
- Test that `temporal_grouping` without `datafusion=True` raises `ValueError`.
