# lazymerge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure Zarr + cubed array mosaic library that uses Zarr spatial conventions for CRS-aware chunk indexing and pyproj+numpy for reprojection.

**Architecture:** Five modules (conventions, sources, target, warp, merge) with a demo script. The conventions layer reads/writes `spatial:` and `proj:` metadata on Zarr v3 nodes. The sources module builds a spatial index. The warp module is a standalone pyproj+numpy reprojection function. The merge module uses `cubed.map_blocks` to orchestrate lazy mosaic computation. Since `cubed.Array` has no `.attrs`, convention metadata is carried in a `MergeResult` namedtuple alongside the cubed array.

**Tech Stack:** zarr (v3), cubed, pyproj, numpy, pytest

---

## File Structure

```
lazymerge/
  __init__.py
  conventions.py    — SpatialAttrs, ProjAttrs dataclasses + read/write/chunk_bbox functions
  sources.py        — SourceEntry, SourceIndex protocol, ScanIndex implementation
  target.py         — create_target(), to_zarr()
  warp.py           — warp_chunk() standalone reprojection
  merge.py          — merge() cubed map_blocks orchestrator

tests/
  __init__.py
  test_conventions.py
  test_sources.py
  test_target.py
  test_warp.py
  test_merge.py

demo.py             — end-to-end synthetic data demo
pyproject.toml      — dependencies
```

---

### Task 1: Project Setup

**Files:**
- Modify: `pyproject.toml`
- Create: `lazymerge/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Update pyproject.toml with dependencies**

```toml
[project]
name = "lazymerge"
version = "0.1.0"
description = "Pure Zarr + cubed array mosaic library"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "zarr>=3.0",
    "cubed",
    "pyproj",
    "numpy",
]

[project.optional-dependencies]
dev = [
    "pytest",
]
```

- [ ] **Step 2: Create package init**

```python
# lazymerge/__init__.py
```

Empty file — just makes it a package.

- [ ] **Step 3: Create tests init**

```python
# tests/__init__.py
```

- [ ] **Step 4: Install the project**

Run: `uv sync --extra dev`
Expected: Installs zarr, cubed, pyproj, numpy, pytest

- [ ] **Step 5: Verify imports work**

Run: `uv run python -c "import zarr; import cubed; import pyproj; import numpy; print('ok')"`
Expected: `ok`

---

### Task 2: Conventions — SpatialAttrs and ProjAttrs Dataclasses

**Files:**
- Create: `lazymerge/conventions.py`
- Create: `tests/test_conventions.py`

- [ ] **Step 1: Write failing tests for dataclass construction**

```python
# tests/test_conventions.py
from lazymerge.conventions import SpatialAttrs, ProjAttrs


def test_spatial_attrs_defaults():
    sa = SpatialAttrs(
        dimensions=["y", "x"],
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0),
        bbox=(500000.0, 5990000.0, 510000.0, 6000000.0),
        shape=(1000, 1000),
    )
    assert sa.dimensions == ["y", "x"]
    assert sa.registration == "pixel"


def test_proj_attrs_code():
    pa = ProjAttrs(code="EPSG:32618")
    assert pa.code == "EPSG:32618"
    assert pa.wkt2 is None
    assert pa.projjson is None


def test_proj_attrs_requires_at_least_one():
    import pytest
    with pytest.raises(ValueError):
        ProjAttrs()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_conventions.py -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement dataclasses**

```python
# lazymerge/conventions.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpatialAttrs:
    dimensions: list[str]
    transform: tuple[float, ...]
    bbox: tuple[float, ...]
    shape: tuple[int, ...]
    registration: str = "pixel"


@dataclass(frozen=True)
class ProjAttrs:
    code: str | None = None
    wkt2: str | None = None
    projjson: dict | None = None

    def __post_init__(self):
        if self.code is None and self.wkt2 is None and self.projjson is None:
            raise ValueError("ProjAttrs requires at least one of: code, wkt2, projjson")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_conventions.py -v`
Expected: 3 passed

---

### Task 3: Conventions — Read/Write Functions

**Files:**
- Modify: `lazymerge/conventions.py`
- Modify: `tests/test_conventions.py`

- [ ] **Step 1: Write failing tests for write and read**

Append to `tests/test_conventions.py`:

```python
import zarr
import numpy as np
from lazymerge.conventions import read_spatial, read_proj, write_spatial, write_proj


def test_write_and_read_spatial():
    store = zarr.storage.MemoryStore()
    arr = zarr.create_array(store, name="test", shape=(100, 100), dtype="f4", chunks=(50, 50))
    sa = SpatialAttrs(
        dimensions=["y", "x"],
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0),
        bbox=(500000.0, 5999000.0, 501000.0, 6000000.0),
        shape=(100, 100),
    )
    write_spatial(arr, sa)
    result = read_spatial(arr)
    assert result.dimensions == sa.dimensions
    assert result.transform == sa.transform
    assert result.bbox == sa.bbox
    assert result.shape == sa.shape
    assert result.registration == "pixel"


def test_write_and_read_proj():
    store = zarr.storage.MemoryStore()
    arr = zarr.create_array(store, name="test2", shape=(100, 100), dtype="f4", chunks=(50, 50))
    pa = ProjAttrs(code="EPSG:32618")
    write_proj(arr, pa)
    result = read_proj(arr)
    assert result.code == "EPSG:32618"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_conventions.py::test_write_and_read_spatial tests/test_conventions.py::test_write_and_read_proj -v`
Expected: FAIL — `ImportError` for the new functions

- [ ] **Step 3: Implement read/write functions**

Append to `lazymerge/conventions.py`:

```python
import zarr


def write_spatial(node: zarr.Group | zarr.Array, attrs: SpatialAttrs) -> None:
    node.attrs["spatial:dimensions"] = attrs.dimensions
    node.attrs["spatial:transform"] = list(attrs.transform)
    node.attrs["spatial:bbox"] = list(attrs.bbox)
    node.attrs["spatial:shape"] = list(attrs.shape)
    node.attrs["spatial:registration"] = attrs.registration


def write_proj(node: zarr.Group | zarr.Array, attrs: ProjAttrs) -> None:
    if attrs.code is not None:
        node.attrs["proj:code"] = attrs.code
    if attrs.wkt2 is not None:
        node.attrs["proj:wkt2"] = attrs.wkt2
    if attrs.projjson is not None:
        node.attrs["proj:projjson"] = attrs.projjson


def read_spatial(node: zarr.Group | zarr.Array) -> SpatialAttrs:
    a = node.attrs
    return SpatialAttrs(
        dimensions=list(a["spatial:dimensions"]),
        transform=tuple(a["spatial:transform"]),
        bbox=tuple(a["spatial:bbox"]),
        shape=tuple(a["spatial:shape"]),
        registration=a.get("spatial:registration", "pixel"),
    )


def read_proj(node: zarr.Group | zarr.Array) -> ProjAttrs:
    a = node.attrs
    return ProjAttrs(
        code=a.get("proj:code"),
        wkt2=a.get("proj:wkt2"),
        projjson=a.get("proj:projjson"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_conventions.py -v`
Expected: 5 passed

---

### Task 4: Conventions — chunk_bbox

**Files:**
- Modify: `lazymerge/conventions.py`
- Modify: `tests/test_conventions.py`

- [ ] **Step 1: Write failing test for chunk_bbox**

Append to `tests/test_conventions.py`:

```python
from lazymerge.conventions import chunk_bbox


def test_chunk_bbox_origin():
    """Chunk (0, 0) should return the top-left portion of the array bbox."""
    sa = SpatialAttrs(
        dimensions=["y", "x"],
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0),
        bbox=(500000.0, 5990000.0, 510000.0, 6000000.0),
        shape=(1000, 1000),
    )
    # Chunk (0, 0) with chunk_shape (256, 256):
    # x: 500000 + 0*256*10 = 500000  to  500000 + 256*10 = 502560
    # y: 6000000 + 0*256*(-10) = 6000000  to  6000000 + 256*(-10) = 5997440
    # bbox = (xmin, ymin, xmax, ymax) = (500000, 5997440, 502560, 6000000)
    result = chunk_bbox(sa, chunk_coords=(0, 0), chunk_shape=(256, 256))
    assert result == (500000.0, 5997440.0, 502560.0, 6000000.0)


def test_chunk_bbox_offset():
    """Chunk (1, 2) should be offset by 1 row and 2 columns of chunks."""
    sa = SpatialAttrs(
        dimensions=["y", "x"],
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0),
        bbox=(500000.0, 5990000.0, 510000.0, 6000000.0),
        shape=(1000, 1000),
    )
    # Chunk (1, 2) with chunk_shape (256, 256):
    # col_offset = 2 * 256 = 512, row_offset = 1 * 256 = 256
    # x: 500000 + 512*10 = 505120  to  505120 + 256*10 = 507680
    # y: 6000000 + 256*(-10) = 5997440  to  5997440 + 256*(-10) = 5994880
    # bbox = (505120, 5994880, 507680, 5997440)
    result = chunk_bbox(sa, chunk_coords=(1, 2), chunk_shape=(256, 256))
    assert result == (505120.0, 5994880.0, 507680.0, 5997440.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_conventions.py::test_chunk_bbox_origin tests/test_conventions.py::test_chunk_bbox_offset -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement chunk_bbox**

Append to `lazymerge/conventions.py`:

```python
def chunk_bbox(
    spatial_attrs: SpatialAttrs,
    chunk_coords: tuple[int, ...],
    chunk_shape: tuple[int, ...],
) -> tuple[float, float, float, float]:
    a, b, c, d, e, f = spatial_attrs.transform
    row_idx, col_idx = chunk_coords
    rows, cols = chunk_shape

    row_offset = row_idx * rows
    col_offset = col_idx * cols

    # Four corners of the chunk in pixel space, transformed to coordinates.
    # For a north-up image (b=0, d=0):
    #   x = a * col + c
    #   y = e * row + f
    x0 = a * col_offset + b * row_offset + c
    y0 = d * col_offset + e * row_offset + f
    x1 = a * (col_offset + cols) + b * (row_offset + rows) + c
    y1 = d * (col_offset + cols) + e * (row_offset + rows) + f

    xmin = min(x0, x1)
    xmax = max(x0, x1)
    ymin = min(y0, y1)
    ymax = max(y0, y1)

    return (xmin, ymin, xmax, ymax)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_conventions.py -v`
Expected: 7 passed

---

### Task 5: Warp Module

**Files:**
- Create: `lazymerge/warp.py`
- Create: `tests/test_warp.py`

- [ ] **Step 1: Write failing tests for warp_chunk**

```python
# tests/test_warp.py
import numpy as np
from lazymerge.warp import warp_chunk


def test_warp_same_crs_same_grid():
    """When source and target have the same CRS and grid, output equals input."""
    source = np.arange(16, dtype="float32").reshape(4, 4)
    transform = (1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    result = warp_chunk(
        source_data=source,
        source_transform=transform,
        source_crs="EPSG:32618",
        target_transform=transform,
        target_crs="EPSG:32618",
        target_shape=(4, 4),
    )
    np.testing.assert_array_equal(result, source)


def test_warp_same_crs_offset_grid():
    """Target grid is offset by 2 pixels in x — should shift the data."""
    source = np.arange(16, dtype="float32").reshape(4, 4)
    source_transform = (1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    # Target starts at x=2 instead of x=0
    target_transform = (1.0, 0.0, 2.0, 0.0, -1.0, 4.0)
    result = warp_chunk(
        source_data=source,
        source_transform=source_transform,
        source_crs="EPSG:32618",
        target_transform=target_transform,
        target_crs="EPSG:32618",
        target_shape=(4, 4),
    )
    # Target pixel (0,0) maps to source pixel (0,2), etc.
    # Columns 0-1 of target = columns 2-3 of source
    # Columns 2-3 of target = out of bounds = NaN
    expected = np.full((4, 4), np.nan, dtype="float32")
    expected[:, 0:2] = source[:, 2:4]
    np.testing.assert_array_equal(result, expected)


def test_warp_out_of_bounds_all_nan():
    """Target grid doesn't overlap source at all — all NaN."""
    source = np.ones((4, 4), dtype="float32")
    source_transform = (1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    # Target is far away
    target_transform = (1.0, 0.0, 1000.0, 0.0, -1.0, 1004.0)
    result = warp_chunk(
        source_data=source,
        source_transform=source_transform,
        source_crs="EPSG:32618",
        target_transform=target_transform,
        target_crs="EPSG:32618",
        target_shape=(4, 4),
    )
    assert np.all(np.isnan(result))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_warp.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement warp_chunk**

```python
# lazymerge/warp.py
from __future__ import annotations

import numpy as np
from pyproj import Transformer


def warp_chunk(
    source_data: np.ndarray,
    source_transform: tuple[float, ...],
    source_crs: str,
    target_transform: tuple[float, ...],
    target_crs: str,
    target_shape: tuple[int, int],
    resampling: str = "nearest",
) -> np.ndarray:
    t_a, t_b, t_c, t_d, t_e, t_f = target_transform
    s_a, s_b, s_c, s_d, s_e, s_f = source_transform

    rows, cols = target_shape
    row_idx, col_idx = np.mgrid[0:rows, 0:cols]

    # Target pixel centers (offset by 0.5 for pixel-center convention)
    target_x = t_a * (col_idx + 0.5) + t_b * (row_idx + 0.5) + t_c
    target_y = t_d * (col_idx + 0.5) + t_e * (row_idx + 0.5) + t_f

    # Reproject target coordinates to source CRS
    if target_crs != source_crs:
        transformer = Transformer.from_crs(target_crs, source_crs, always_xy=True)
        source_x, source_y = transformer.transform(target_x, target_y)
    else:
        source_x, source_y = target_x, target_y

    # Inverse of source affine: pixel coords from spatial coords
    # For affine [a, b, c, d, e, f]:
    #   x = a*col + b*row + c
    #   y = d*col + e*row + f
    # Inverse (assuming no rotation, b=0, d=0):
    #   col = (x - c) / a
    #   row = (y - f) / e
    # General inverse for 2x2:
    det = s_a * s_e - s_b * s_d
    src_col = (s_e * (source_x - s_c) - s_b * (source_y - s_f)) / det
    src_row = (-s_d * (source_x - s_c) + s_a * (source_y - s_f)) / det

    # Subtract 0.5 to go from pixel-center coords to array indices
    src_col = src_col - 0.5
    src_row = src_row - 0.5

    src_h, src_w = source_data.shape

    if resampling == "nearest":
        src_col_i = np.round(src_col).astype(int)
        src_row_i = np.round(src_row).astype(int)

        valid = (src_col_i >= 0) & (src_col_i < src_w) & (src_row_i >= 0) & (src_row_i < src_h)

        output = np.full(target_shape, np.nan, dtype=source_data.dtype)
        output[valid] = source_data[src_row_i[valid], src_col_i[valid]]
        return output
    else:
        raise ValueError(f"Unsupported resampling method: {resampling}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_warp.py -v`
Expected: 3 passed

---

### Task 6: Sources — SourceEntry and ScanIndex

**Files:**
- Create: `lazymerge/sources.py`
- Create: `tests/test_sources.py`

- [ ] **Step 1: Write failing tests for scan_store and find_intersecting_sources**

```python
# tests/test_sources.py
import numpy as np
import zarr
from lazymerge.conventions import SpatialAttrs, ProjAttrs, write_spatial, write_proj
from lazymerge.sources import scan_store


def _make_source_store():
    """Create a store with two source arrays in the same CRS but different extents."""
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")

    # Source A: covers x=[500000, 501000], y=[5999000, 6000000] in EPSG:32618
    a = root.create_array("source_a", shape=(100, 100), dtype="f4", chunks=(50, 50))
    a[:] = 1.0
    write_spatial(a, SpatialAttrs(
        dimensions=["y", "x"],
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0),
        bbox=(500000.0, 5999000.0, 501000.0, 6000000.0),
        shape=(100, 100),
    ))
    write_proj(a, ProjAttrs(code="EPSG:32618"))

    # Source B: covers x=[501000, 502000], y=[5999000, 6000000] in EPSG:32618
    b = root.create_array("source_b", shape=(100, 100), dtype="f4", chunks=(50, 50))
    b[:] = 2.0
    write_spatial(b, SpatialAttrs(
        dimensions=["y", "x"],
        transform=(10.0, 0.0, 501000.0, 0.0, -10.0, 6000000.0),
        bbox=(501000.0, 5999000.0, 502000.0, 6000000.0),
        shape=(100, 100),
    ))
    write_proj(b, ProjAttrs(code="EPSG:32618"))

    return store, root


def test_scan_store_finds_all_sources():
    store, root = _make_source_store()
    index = scan_store(root)
    assert len(index.entries) == 2
    paths = {e.path for e in index.entries}
    assert paths == {"source_a", "source_b"}


def test_find_intersecting_sources_full_overlap():
    """A target bbox covering both sources should return both."""
    store, root = _make_source_store()
    index = scan_store(root)
    results = index.find_intersecting_sources(
        target_bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
        target_crs="EPSG:32618",
    )
    assert len(results) == 2


def test_find_intersecting_sources_partial():
    """A target bbox covering only source_a's extent should return only source_a."""
    store, root = _make_source_store()
    index = scan_store(root)
    results = index.find_intersecting_sources(
        target_bbox=(500000.0, 5999000.0, 500500.0, 6000000.0),
        target_crs="EPSG:32618",
    )
    assert len(results) == 1
    assert results[0].path == "source_a"


def test_find_intersecting_sources_none():
    """A target bbox far away should return nothing."""
    store, root = _make_source_store()
    index = scan_store(root)
    results = index.find_intersecting_sources(
        target_bbox=(0.0, 0.0, 1.0, 1.0),
        target_crs="EPSG:32618",
    )
    assert len(results) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sources.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement SourceEntry and ScanIndex**

```python
# lazymerge/sources.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import zarr
from pyproj import Transformer

from lazymerge.conventions import (
    ProjAttrs,
    SpatialAttrs,
    chunk_bbox,
    read_proj,
    read_spatial,
)


@dataclass(frozen=True)
class SourceEntry:
    path: str
    spatial_attrs: SpatialAttrs
    proj_attrs: ProjAttrs
    chunk_shape: tuple[int, ...]


def _bboxes_intersect(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _reproject_bbox(
    bbox: tuple[float, float, float, float],
    src_crs: str,
    dst_crs: str,
) -> tuple[float, float, float, float]:
    if src_crs == dst_crs:
        return bbox
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xs = [bbox[0], bbox[0], bbox[2], bbox[2]]
    ys = [bbox[1], bbox[3], bbox[1], bbox[3]]
    tx, ty = transformer.transform(xs, ys)
    return (min(tx), min(ty), max(tx), max(ty))


class ScanIndex:
    def __init__(self, entries: list[SourceEntry]):
        self.entries = entries
        self._reprojected_bboxes: dict[tuple[str, str], tuple[float, float, float, float]] = {}

    def _get_bbox_in_crs(self, entry: SourceEntry, target_crs: str) -> tuple[float, float, float, float]:
        cache_key = (entry.path, target_crs)
        if cache_key not in self._reprojected_bboxes:
            src_crs = entry.proj_attrs.code or "EPSG:4326"
            self._reprojected_bboxes[cache_key] = _reproject_bbox(
                entry.spatial_attrs.bbox, src_crs, target_crs
            )
        return self._reprojected_bboxes[cache_key]

    def find_intersecting_sources(
        self,
        target_bbox: tuple[float, float, float, float],
        target_crs: str,
    ) -> list[SourceEntry]:
        results = []
        for entry in self.entries:
            src_bbox = self._get_bbox_in_crs(entry, target_crs)
            if _bboxes_intersect(src_bbox, target_bbox):
                results.append(entry)
        return results

    def find_intersecting_chunks(
        self,
        source: SourceEntry,
        target_chunk_bbox: tuple[float, float, float, float],
        target_crs: str,
    ) -> list[tuple[str, tuple[int, int]]]:
        src_crs = source.proj_attrs.code or "EPSG:4326"
        src_bbox = _reproject_bbox(target_chunk_bbox, target_crs, src_crs)

        sa = source.spatial_attrs
        n_row_chunks = -(-sa.shape[0] // source.chunk_shape[0])  # ceiling division
        n_col_chunks = -(-sa.shape[1] // source.chunk_shape[1])

        results = []
        for ri in range(n_row_chunks):
            for ci in range(n_col_chunks):
                cb = chunk_bbox(sa, (ri, ci), source.chunk_shape)
                if _bboxes_intersect(cb, src_bbox):
                    results.append((source.path, (ri, ci)))
        return results


def scan_store(root: zarr.Group) -> ScanIndex:
    entries = []
    for name, item in root.members():
        if isinstance(item, zarr.Array):
            try:
                sa = read_spatial(item)
                pa = read_proj(item)
            except KeyError:
                continue
            chunk_shape = item.chunks
            entries.append(SourceEntry(
                path=name,
                spatial_attrs=sa,
                proj_attrs=pa,
                chunk_shape=tuple(chunk_shape),
            ))
    return ScanIndex(entries)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sources.py -v`
Expected: 4 passed

---

### Task 7: Sources — find_intersecting_chunks

**Files:**
- Modify: `tests/test_sources.py`

- [ ] **Step 1: Write failing tests for find_intersecting_chunks**

Append to `tests/test_sources.py`:

```python
from lazymerge.conventions import chunk_bbox


def test_find_intersecting_chunks_subset():
    """Only chunks overlapping the target chunk bbox should be returned."""
    store, root = _make_source_store()
    index = scan_store(root)
    source_a = [e for e in index.entries if e.path == "source_a"][0]

    # Target chunk covers only the left half of source_a: x=[500000, 500500]
    results = index.find_intersecting_chunks(
        source=source_a,
        target_chunk_bbox=(500000.0, 5999000.0, 500500.0, 6000000.0),
        target_crs="EPSG:32618",
    )
    # source_a has 100x100 pixels with 50x50 chunks = 2x2 chunk grid
    # Left half (x < 500500) = col chunks 0 only
    paths_and_coords = [(p, c) for p, c in results]
    col_indices = {c[1] for _, c in paths_and_coords}
    assert col_indices == {0}
    assert len(results) > 0


def test_find_intersecting_chunks_no_overlap():
    """Target chunk bbox outside source should return empty list."""
    store, root = _make_source_store()
    index = scan_store(root)
    source_a = [e for e in index.entries if e.path == "source_a"][0]

    results = index.find_intersecting_chunks(
        source=source_a,
        target_chunk_bbox=(600000.0, 5999000.0, 601000.0, 6000000.0),
        target_crs="EPSG:32618",
    )
    assert len(results) == 0
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_sources.py -v`
Expected: 6 passed (these should pass immediately since find_intersecting_chunks was implemented in Task 6)

---

### Task 8: Target — create_target

**Files:**
- Create: `lazymerge/target.py`
- Create: `tests/test_target.py`

- [ ] **Step 1: Write failing tests for create_target**

```python
# tests/test_target.py
import math
import cubed
from lazymerge.target import create_target


def test_create_target_shape():
    """Shape should be computed from bbox and resolution."""
    arr, spatial, proj = create_target(
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 501000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(256, 256),
    )
    # width = (501000 - 500000) / 10 = 100
    # height = (6000000 - 5999000) / 10 = 100
    assert arr.shape == (100, 100)


def test_create_target_chunks():
    arr, spatial, proj = create_target(
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 501000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(50, 50),
    )
    assert arr.chunksize == (50, 50)


def test_create_target_spatial_attrs():
    arr, spatial, proj = create_target(
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 501000.0, 6000000.0),
        resolution=10.0,
    )
    assert spatial.dimensions == ["y", "x"]
    assert spatial.transform == (10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0)
    assert spatial.bbox == (500000.0, 5999000.0, 501000.0, 6000000.0)
    assert spatial.shape == (100, 100)


def test_create_target_proj_attrs():
    arr, spatial, proj = create_target(
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 501000.0, 6000000.0),
        resolution=10.0,
    )
    assert proj.code == "EPSG:32618"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_target.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement create_target**

Since `cubed.Array` has no `.attrs`, `create_target` returns a tuple of `(cubed.Array, SpatialAttrs, ProjAttrs)`.

```python
# lazymerge/target.py
from __future__ import annotations

import math

import cubed
import numpy as np
import zarr

from lazymerge.conventions import ProjAttrs, SpatialAttrs, write_proj, write_spatial


def create_target(
    crs: str,
    bbox: tuple[float, float, float, float],
    resolution: float,
    chunk_size: tuple[int, int] = (512, 512),
    dtype: str = "float32",
    fill_value: float = float("nan"),
) -> tuple[cubed.Array, SpatialAttrs, ProjAttrs]:
    xmin, ymin, xmax, ymax = bbox
    width = math.ceil((xmax - xmin) / resolution)
    height = math.ceil((ymax - ymin) / resolution)

    arr = cubed.full(
        shape=(height, width),
        fill_value=fill_value,
        dtype=dtype,
        chunks=chunk_size,
    )

    spatial = SpatialAttrs(
        dimensions=["y", "x"],
        transform=(resolution, 0.0, xmin, 0.0, -resolution, ymax),
        bbox=bbox,
        shape=(height, width),
    )
    proj = ProjAttrs(code=crs)

    return arr, spatial, proj


def to_zarr(
    array: cubed.Array,
    spatial: SpatialAttrs,
    proj: ProjAttrs,
    store: str | zarr.storage.Store,
    path: str | None = None,
) -> None:
    result = array.compute()
    root = zarr.open_group(store, mode="w")
    arr_path = path or "data"
    zarr_arr = root.create_array(arr_path, data=result, chunks=array.chunksize)
    write_spatial(zarr_arr, spatial)
    write_proj(zarr_arr, proj)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_target.py -v`
Expected: 4 passed

---

### Task 9: Target — to_zarr

**Files:**
- Modify: `tests/test_target.py`

- [ ] **Step 1: Write failing test for to_zarr**

Append to `tests/test_target.py`:

```python
import numpy as np
import zarr
from lazymerge.target import to_zarr
from lazymerge.conventions import read_spatial, read_proj


def test_to_zarr_writes_conventions():
    arr, spatial, proj = create_target(
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 501000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(50, 50),
    )
    store = zarr.storage.MemoryStore()
    to_zarr(arr, spatial, proj, store, path="output")

    root = zarr.open_group(store, mode="r")
    out = root["output"]
    assert out.shape == (100, 100)

    result_spatial = read_spatial(out)
    assert result_spatial.transform == spatial.transform
    assert result_spatial.bbox == spatial.bbox

    result_proj = read_proj(out)
    assert result_proj.code == "EPSG:32618"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_target.py -v`
Expected: 5 passed (should pass immediately since to_zarr was implemented in Task 8)

---

### Task 10: Merge — Block Function and merge()

**Files:**
- Create: `lazymerge/merge.py`
- Create: `tests/test_merge.py`

- [ ] **Step 1: Write failing test for merge with same-CRS sources**

```python
# tests/test_merge.py
import numpy as np
import zarr
from lazymerge.conventions import SpatialAttrs, ProjAttrs, write_spatial, write_proj
from lazymerge.sources import scan_store
from lazymerge.target import create_target
from lazymerge.merge import merge


def _make_two_source_store():
    """Two adjacent source arrays in EPSG:32618, each 100x100 at 10m resolution."""
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")

    # Source A: x=[500000, 501000], y=[5999000, 6000000], filled with 1.0
    a = root.create_array("source_a", shape=(100, 100), dtype="f4", chunks=(50, 50))
    a[:] = 1.0
    write_spatial(a, SpatialAttrs(
        dimensions=["y", "x"],
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0),
        bbox=(500000.0, 5999000.0, 501000.0, 6000000.0),
        shape=(100, 100),
    ))
    write_proj(a, ProjAttrs(code="EPSG:32618"))

    # Source B: x=[501000, 502000], y=[5999000, 6000000], filled with 2.0
    b = root.create_array("source_b", shape=(100, 100), dtype="f4", chunks=(50, 50))
    b[:] = 2.0
    write_spatial(b, SpatialAttrs(
        dimensions=["y", "x"],
        transform=(10.0, 0.0, 501000.0, 0.0, -10.0, 6000000.0),
        bbox=(501000.0, 5999000.0, 502000.0, 6000000.0),
        shape=(100, 100),
    ))
    write_proj(b, ProjAttrs(code="EPSG:32618"))

    return store, root


def test_merge_two_adjacent_sources():
    """Merging two adjacent same-CRS sources into a target spanning both."""
    store, root = _make_two_source_store()
    index = scan_store(root)

    # Target spans both sources: x=[500000, 502000]
    target, spatial, proj = create_target(
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(50, 50),
    )

    result_arr, result_spatial, result_proj = merge(
        source_index=index,
        target=target,
        target_spatial=spatial,
        target_proj=proj,
        store=store,
    )

    data = result_arr.compute()
    assert data.shape == (100, 200)
    # Left half (cols 0-99) should be 1.0 from source_a
    np.testing.assert_array_equal(data[:, :100], 1.0)
    # Right half (cols 100-199) should be 2.0 from source_b
    np.testing.assert_array_equal(data[:, 100:], 2.0)


def test_merge_no_sources():
    """Target that doesn't overlap any source should be all NaN."""
    store, root = _make_two_source_store()
    index = scan_store(root)

    target, spatial, proj = create_target(
        crs="EPSG:32618",
        bbox=(600000.0, 5999000.0, 601000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(50, 50),
    )

    result_arr, result_spatial, result_proj = merge(
        source_index=index,
        target=target,
        target_spatial=spatial,
        target_proj=proj,
        store=store,
    )

    data = result_arr.compute()
    assert np.all(np.isnan(data))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_merge.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement merge**

```python
# lazymerge/merge.py
from __future__ import annotations

import numpy as np
import zarr
import cubed

from lazymerge.conventions import SpatialAttrs, ProjAttrs, chunk_bbox, read_spatial, read_proj
from lazymerge.sources import ScanIndex
from lazymerge.warp import warp_chunk


def _merge_block(
    block: np.ndarray,
    block_id: tuple[int, ...],
    source_index: ScanIndex,
    target_spatial: SpatialAttrs,
    target_proj: ProjAttrs,
    store: zarr.storage.Store,
    chunk_size: tuple[int, int],
    resampling: str,
    aggregation: str,
) -> np.ndarray:
    target_crs = target_proj.code

    # Compute this chunk's spatial bbox
    row_idx, col_idx = block_id
    cb = chunk_bbox(target_spatial, (row_idx, col_idx), chunk_size)

    # Compute chunk-local transform (shift origin to this chunk's pixel offset)
    a, b, c, d, e, f = target_spatial.transform
    row_offset = row_idx * chunk_size[0]
    col_offset = col_idx * chunk_size[1]
    chunk_c = c + a * col_offset + b * row_offset
    chunk_f = f + d * col_offset + e * row_offset
    chunk_transform = (a, b, chunk_c, d, e, chunk_f)

    # Actual chunk shape (may be smaller at edges)
    actual_shape = block.shape

    output = np.full(actual_shape, np.nan, dtype=block.dtype)

    # Pass 1: find intersecting sources
    sources = source_index.find_intersecting_sources(cb, target_crs)
    if not sources:
        return output

    for source_entry in sources:
        # Pass 2: find intersecting source chunks
        src_chunks = source_index.find_intersecting_chunks(source_entry, cb, target_crs)
        if not src_chunks:
            continue

        src_crs = source_entry.proj_attrs.code
        root = zarr.open_group(store, mode="r")
        src_array = root[source_entry.path]

        for _, (src_row, src_col) in src_chunks:
            # Read source chunk data
            sr_start = src_row * source_entry.chunk_shape[0]
            sr_end = min(sr_start + source_entry.chunk_shape[0], source_entry.spatial_attrs.shape[0])
            sc_start = src_col * source_entry.chunk_shape[1]
            sc_end = min(sc_start + source_entry.chunk_shape[1], source_entry.spatial_attrs.shape[1])
            src_data = src_array[sr_start:sr_end, sc_start:sc_end]

            # Compute source chunk transform (shift origin to source chunk pixel offset)
            sa, sb, sc, sd, se, sf = source_entry.spatial_attrs.transform
            src_chunk_c = sc + sa * sc_start + sb * sr_start
            src_chunk_f = sf + sd * sc_start + se * sr_start
            src_chunk_transform = (sa, sb, src_chunk_c, sd, se, src_chunk_f)

            warped = warp_chunk(
                source_data=np.asarray(src_data),
                source_transform=src_chunk_transform,
                source_crs=src_crs,
                target_transform=chunk_transform,
                target_crs=target_crs,
                target_shape=actual_shape,
                resampling=resampling,
            )

            if aggregation == "first":
                mask = np.isnan(output) & ~np.isnan(warped)
                output[mask] = warped[mask]
            elif aggregation == "last":
                mask = ~np.isnan(warped)
                output[mask] = warped[mask]

    return output


def merge(
    source_index: ScanIndex,
    target: cubed.Array,
    target_spatial: SpatialAttrs,
    target_proj: ProjAttrs,
    store: zarr.storage.Store,
    resampling: str = "nearest",
    aggregation: str = "first",
) -> tuple[cubed.Array, SpatialAttrs, ProjAttrs]:
    chunk_size = target.chunksize

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
        aggregation=aggregation,
    )

    return result, target_spatial, target_proj
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_merge.py -v`
Expected: 2 passed

---

### Task 11: End-to-End Demo Script

**Files:**
- Create: `demo.py`

- [ ] **Step 1: Write the demo script**

```python
# demo.py
"""End-to-end demo of lazymerge with synthetic data."""
import numpy as np
import zarr

from lazymerge.conventions import SpatialAttrs, ProjAttrs, write_spatial, write_proj, read_spatial, read_proj
from lazymerge.sources import scan_store
from lazymerge.target import create_target, to_zarr
from lazymerge.merge import merge


def make_synthetic_store() -> tuple[zarr.storage.Store, zarr.Group]:
    """Create 4 synthetic source arrays across 2 UTM zones with overlapping extents."""
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")

    sources = [
        # UTM 18N sources
        {
            "name": "utm18n_tile_a",
            "crs": "EPSG:32618",
            "bbox": (500000.0, 5990000.0, 505000.0, 5995000.0),
            "resolution": 10.0,
            "fill": 100.0,
        },
        {
            "name": "utm18n_tile_b",
            "crs": "EPSG:32618",
            "bbox": (503000.0, 5990000.0, 508000.0, 5995000.0),
            "resolution": 10.0,
            "fill": 200.0,
        },
        # UTM 18N sources at different y extent
        {
            "name": "utm18n_tile_c",
            "crs": "EPSG:32618",
            "bbox": (500000.0, 5995000.0, 505000.0, 6000000.0),
            "resolution": 10.0,
            "fill": 300.0,
        },
        {
            "name": "utm18n_tile_d",
            "crs": "EPSG:32618",
            "bbox": (503000.0, 5995000.0, 508000.0, 6000000.0),
            "resolution": 10.0,
            "fill": 400.0,
        },
    ]

    for src in sources:
        xmin, ymin, xmax, ymax = src["bbox"]
        res = src["resolution"]
        width = int((xmax - xmin) / res)
        height = int((ymax - ymin) / res)

        arr = root.create_array(src["name"], shape=(height, width), dtype="f4", chunks=(256, 256))
        arr[:] = src["fill"]

        write_spatial(arr, SpatialAttrs(
            dimensions=["y", "x"],
            transform=(res, 0.0, xmin, 0.0, -res, ymax),
            bbox=(xmin, ymin, xmax, ymax),
            shape=(height, width),
        ))
        write_proj(arr, ProjAttrs(code=src["crs"]))

    return store, root


def main():
    print("=== lazymerge demo ===\n")

    # Step 1: Create synthetic sources
    print("1. Creating synthetic source arrays...")
    store, root = make_synthetic_store()
    for name, item in root.members():
        if isinstance(item, zarr.Array):
            print(f"   {name}: shape={item.shape}, chunks={item.chunks}")

    # Step 2: Build source index
    print("\n2. Scanning store for source arrays...")
    index = scan_store(root)
    print(f"   Found {len(index.entries)} source arrays")
    for entry in index.entries:
        print(f"   - {entry.path}: bbox={entry.spatial_attrs.bbox}, crs={entry.proj_attrs.code}")

    # Step 3: Create target array spanning all sources
    print("\n3. Creating target array...")
    target, spatial, proj = create_target(
        crs="EPSG:32618",
        bbox=(500000.0, 5990000.0, 508000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(256, 256),
    )
    print(f"   Target shape: {target.shape}")
    print(f"   Target chunks: {target.chunksize}")
    print(f"   Target CRS: {proj.code}")

    # Step 4: Merge
    print("\n4. Creating lazy merge...")
    result_arr, result_spatial, result_proj = merge(
        source_index=index,
        target=target,
        target_spatial=spatial,
        target_proj=proj,
        store=store,
    )
    print(f"   Result shape: {result_arr.shape}")

    # Step 5: Compute and verify
    print("\n5. Computing result...")
    data = result_arr.compute()
    print(f"   Computed shape: {data.shape}")
    print(f"   Non-NaN pixels: {np.count_nonzero(~np.isnan(data))}")
    print(f"   Unique values (excl NaN): {np.unique(data[~np.isnan(data)])}")

    # Step 6: Materialize to Zarr
    print("\n6. Materializing to Zarr store...")
    output_store = zarr.storage.MemoryStore()
    to_zarr(result_arr, result_spatial, result_proj, output_store, path="mosaic")

    # Step 7: Read back and verify
    print("\n7. Verifying output...")
    out_root = zarr.open_group(output_store, mode="r")
    out_arr = out_root["mosaic"]
    out_spatial = read_spatial(out_arr)
    out_proj = read_proj(out_arr)
    print(f"   Output shape: {out_arr.shape}")
    print(f"   Output CRS: {out_proj.code}")
    print(f"   Output bbox: {out_spatial.bbox}")
    print(f"   Output transform: {out_spatial.transform}")

    # Spot-check: pixel at (0, 0) is top-left of target = (500000, 6000000)
    # This should be in tile_c (fill=300) which covers x=[500000, 505000] y=[5995000, 6000000]
    val_top_left = data[0, 0]
    print(f"\n   Spot checks:")
    print(f"   Pixel (0, 0) [top-left, should be 300.0]: {val_top_left}")

    # Pixel at (999, 799) is bottom-right area near (508000, 5990000)
    # This should be in tile_b (fill=200) which covers x=[503000, 508000] y=[5990000, 5995000]
    val_bottom_right = data[999, 799]
    print(f"   Pixel (999, 799) [bottom-right, should be 200.0]: {val_bottom_right}")

    assert val_top_left == 300.0, f"Expected 300.0, got {val_top_left}"
    assert val_bottom_right == 200.0, f"Expected 200.0, got {val_bottom_right}"

    print("\n=== Demo complete! All checks passed. ===")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the demo**

Run: `uv run python demo.py`
Expected: All steps complete, spot checks pass, "Demo complete!" message.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass.

---

### Task 12: Package Init Exports

**Files:**
- Modify: `lazymerge/__init__.py`

- [ ] **Step 1: Add public API exports**

```python
# lazymerge/__init__.py
from lazymerge.conventions import SpatialAttrs, ProjAttrs, read_spatial, read_proj, write_spatial, write_proj, chunk_bbox
from lazymerge.sources import SourceEntry, ScanIndex, scan_store
from lazymerge.target import create_target, to_zarr
from lazymerge.warp import warp_chunk
from lazymerge.merge import merge
```

- [ ] **Step 2: Verify imports work**

Run: `uv run python -c "from lazymerge import merge, create_target, scan_store, warp_chunk; print('ok')"`
Expected: `ok`
