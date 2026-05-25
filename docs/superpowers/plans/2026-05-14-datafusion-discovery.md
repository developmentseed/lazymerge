# DataFusion Discovery + Overview Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add zarr-datafusion-search metadata as an alternative source discovery backend, and automatic overview selection via the zarr multiscales convention.

**Architecture:** `conventions.py` gains `OverviewLevel` and `read_multiscales()`. `sources.py` gains `select_overview()` and a `backend` parameter on `scan_store()`. `merge.py` gains a `band` parameter and overview-aware array resolution in `_merge_block`. All existing tests continue passing; new tests cover each new unit.

**Tech Stack:** zarr (v3), zarr-cm, zarr-datafusion-search (optional), cubed, pyproj, numpy, pytest

---

## File Structure

```
lazymerge/
  conventions.py  — ADD OverviewLevel dataclass, read_multiscales()
  sources.py      — ADD select_overview(), _scan_datafusion(), update scan_store() signature
  merge.py        — ADD band param, overview selection + band navigation in _merge_block
  __init__.py     — ADD exports for OverviewLevel, read_multiscales, select_overview

tests/
  test_conventions.py  — ADD tests for OverviewLevel, read_multiscales
  test_sources.py      — ADD tests for select_overview, scan_store with backend="datafusion"
  test_merge.py        — ADD tests for band param and overview selection in merge
```

---

### Task 1: OverviewLevel dataclass and read_multiscales

**Files:**
- Modify: `lazymerge/conventions.py`
- Test: `tests/test_conventions.py`

- [ ] **Step 1: Write failing test for OverviewLevel construction**

```python
# In tests/test_conventions.py — add at end of file

from lazymerge.conventions import OverviewLevel


def test_overview_level_construction():
    ol = OverviewLevel(path="1", scale=(2.0, 2.0), resolution=20.0)
    assert ol.path == "1"
    assert ol.scale == (2.0, 2.0)
    assert ol.resolution == 20.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_conventions.py::test_overview_level_construction -v`
Expected: FAIL with `ImportError: cannot import name 'OverviewLevel'`

- [ ] **Step 3: Implement OverviewLevel dataclass**

Add to `lazymerge/conventions.py` after `ProjAttrs`:

```python
@dataclass(frozen=True)
class OverviewLevel:
    path: str
    scale: tuple[float, float]
    resolution: float
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_conventions.py::test_overview_level_construction -v`
Expected: PASS

- [ ] **Step 5: Write failing test for read_multiscales with multiscales attr**

```python
# In tests/test_conventions.py

from lazymerge.conventions import read_multiscales


def test_read_multiscales_with_overviews():
    """A group with multiscales attr should return overview levels ordered finest to coarsest."""
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")
    band_group = root.create_group("red")

    # Create base array with spatial attrs (10m resolution)
    base = band_group.create_array("0", shape=(1000, 1000), dtype="f4", chunks=(256, 256))
    write_spatial(
        base,
        SpatialAttrs(
            dimensions=["y", "x"],
            transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0),
            bbox=(500000.0, 5990000.0, 510000.0, 6000000.0),
            shape=(1000, 1000),
        ),
    )

    # Create overview arrays
    band_group.create_array("1", shape=(500, 500), dtype="f4", chunks=(256, 256))
    band_group.create_array("2", shape=(250, 250), dtype="f4", chunks=(256, 256))

    # Set multiscales attr on the band group
    band_group.attrs["multiscales"] = {
        "layout": [
            {"asset": "0", "transform": {"scale": [1.0, 1.0], "translation": [0.0, 0.0]}},
            {
                "asset": "1",
                "derived_from": "0",
                "transform": {"scale": [2.0, 2.0], "translation": [0.5, 0.5]},
            },
            {
                "asset": "2",
                "derived_from": "1",
                "transform": {"scale": [2.0, 2.0], "translation": [0.5, 0.5]},
            },
        ]
    }

    overviews = read_multiscales(band_group)
    assert overviews is not None
    assert len(overviews) == 2  # excludes the base level

    # Level 1: 2x scale from base → 20m resolution
    assert overviews[0].path == "1"
    assert overviews[0].scale == (2.0, 2.0)
    assert overviews[0].resolution == 20.0

    # Level 2: 2x scale from level 1 → 4x cumulative → 40m resolution
    assert overviews[1].path == "2"
    assert overviews[1].scale == (4.0, 4.0)
    assert overviews[1].resolution == 40.0
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_conventions.py::test_read_multiscales_with_overviews -v`
Expected: FAIL with `ImportError: cannot import name 'read_multiscales'`

- [ ] **Step 7: Implement read_multiscales**

Add to `lazymerge/conventions.py` after `chunk_bbox`:

```python
def read_multiscales(group: zarr.Group) -> list[OverviewLevel] | None:
    """Parse the zarr multiscales convention from a group.

    Returns overview levels (excluding the base level) ordered finest to
    coarsest, or None if no multiscales attribute exists.

    Resolution is computed from the base level's spatial:transform and
    the cumulative scale factors.
    """
    attrs = dict(group.attrs)
    if "multiscales" not in attrs:
        return None

    layout = attrs["multiscales"]["layout"]
    if len(layout) <= 1:
        return None

    # Read base resolution from the first layout entry's array
    base_path = layout[0]["asset"]
    base_array = group[base_path]
    base_spatial = read_spatial(base_array)
    native_res = abs(base_spatial.transform[0])

    overviews: list[OverviewLevel] = []
    cumulative_scale_y = 1.0
    cumulative_scale_x = 1.0

    for level in layout[1:]:
        scale = level["transform"]["scale"]
        cumulative_scale_y *= scale[0]
        cumulative_scale_x *= scale[1]
        resolution = native_res * cumulative_scale_x
        overviews.append(
            OverviewLevel(
                path=level["asset"],
                scale=(cumulative_scale_y, cumulative_scale_x),
                resolution=resolution,
            )
        )

    return overviews
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_conventions.py::test_read_multiscales_with_overviews -v`
Expected: PASS

- [ ] **Step 9: Write failing test for read_multiscales without attr**

```python
# In tests/test_conventions.py

def test_read_multiscales_no_attr():
    """A group without multiscales attr should return None."""
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")
    band_group = root.create_group("red")
    band_group.create_array("data", shape=(100, 100), dtype="f4", chunks=(50, 50))

    result = read_multiscales(band_group)
    assert result is None
```

- [ ] **Step 10: Run test to verify it passes**

Run: `pytest tests/test_conventions.py::test_read_multiscales_no_attr -v`
Expected: PASS (already handled by the `if "multiscales" not in attrs` check)

- [ ] **Step 11: Run full conventions test suite**

Run: `pytest tests/test_conventions.py -v`
Expected: All tests PASS

- [ ] **Step 12: Commit**

```bash
git add lazymerge/conventions.py tests/test_conventions.py
git commit -m "feat: add OverviewLevel dataclass and read_multiscales"
```

---

### Task 2: select_overview function

**Files:**
- Modify: `lazymerge/sources.py`
- Test: `tests/test_sources.py`

- [ ] **Step 1: Write failing test — returns None when no overviews**

```python
# In tests/test_sources.py — add at end of file

from lazymerge.sources import select_overview
from lazymerge.conventions import OverviewLevel


def test_select_overview_no_overviews():
    result = select_overview(overviews=[], target_res=20.0, native_res=10.0)
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sources.py::test_select_overview_no_overviews -v`
Expected: FAIL with `ImportError: cannot import name 'select_overview'`

- [ ] **Step 3: Write failing test — returns None when target res <= native**

```python
# In tests/test_sources.py

def test_select_overview_target_finer_than_native():
    overviews = [
        OverviewLevel(path="1", scale=(2.0, 2.0), resolution=20.0),
    ]
    result = select_overview(overviews=overviews, target_res=5.0, native_res=10.0)
    assert result is None
```

- [ ] **Step 4: Write failing test — selects coarsest level still <= target**

```python
# In tests/test_sources.py

def test_select_overview_picks_coarsest_within_target():
    overviews = [
        OverviewLevel(path="1", scale=(2.0, 2.0), resolution=20.0),
        OverviewLevel(path="2", scale=(4.0, 4.0), resolution=40.0),
        OverviewLevel(path="3", scale=(8.0, 8.0), resolution=80.0),
    ]
    # Target is 50m — should pick level 2 (40m), not level 3 (80m)
    result = select_overview(overviews=overviews, target_res=50.0, native_res=10.0)
    assert result is not None
    assert result.path == "2"
    assert result.resolution == 40.0
```

- [ ] **Step 5: Write failing test — target between native and finest overview**

```python
# In tests/test_sources.py

def test_select_overview_target_between_native_and_finest():
    overviews = [
        OverviewLevel(path="1", scale=(2.0, 2.0), resolution=20.0),
    ]
    # Target is 15m, native is 10m, finest overview is 20m — fall back to full res
    result = select_overview(overviews=overviews, target_res=15.0, native_res=10.0)
    assert result is None
```

- [ ] **Step 6: Implement select_overview**

Add to `lazymerge/sources.py` after the imports:

```python
from lazymerge.conventions import OverviewLevel
```

Then after `_reproject_bbox`:

```python
def select_overview(
    overviews: list[OverviewLevel],
    target_res: float,
    native_res: float,
) -> OverviewLevel | None:
    """Choose the coarsest overview whose resolution is <= target_res.

    Picks the finest source data that avoids upsampling: the selected
    overview's pixel size is no larger than the output pixel size, so each
    output pixel samples at least as much original detail as it represents.

    Args:
        overviews: Overview levels ordered finest to coarsest.
        target_res: Target pixel size in the source's native CRS units.
        native_res: Full-resolution pixel size.

    Returns:
        An OverviewLevel, or None to use full resolution.
    """
    if not overviews:
        return None
    if target_res <= native_res:
        return None

    selected: OverviewLevel | None = None
    for overview in overviews:
        if overview.resolution <= target_res:
            selected = overview
        else:
            break
    return selected
```

- [ ] **Step 7: Run all select_overview tests**

Run: `pytest tests/test_sources.py -k "select_overview" -v`
Expected: All 4 tests PASS

- [ ] **Step 8: Run full sources test suite**

Run: `pytest tests/test_sources.py -v`
Expected: All tests PASS

- [ ] **Step 9: Commit**

```bash
git add lazymerge/sources.py tests/test_sources.py
git commit -m "feat: add select_overview for multiscale level selection"
```

---

### Task 3: DataFusion backend for scan_store

**Files:**
- Modify: `lazymerge/sources.py`
- Modify: `pyproject.toml`
- Test: `tests/test_sources.py`

- [ ] **Step 1: Add zarr-datafusion-search as optional dependency**

In `pyproject.toml`, add after the `dev` optional dependencies:

```toml
[project.optional-dependencies]
dev = [
    "pytest",
    "ruff",
    "mypy",
]
datafusion = [
    "zarr-datafusion-search",
]
```

- [ ] **Step 2: Write failing test — scan_store with backend="conventions" (existing behavior preserved)**

```python
# In tests/test_sources.py

def test_scan_store_backend_conventions_default():
    """backend='conventions' should behave identically to the original scan_store."""
    store, root = _make_source_store()
    index = scan_store(root, backend="conventions")
    assert len(index.entries) == 2
    paths = {e.path for e in index.entries}
    assert paths == {"source_a", "source_b"}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_sources.py::test_scan_store_backend_conventions_default -v`
Expected: FAIL with `TypeError: scan_store() got an unexpected keyword argument 'backend'`

- [ ] **Step 4: Refactor scan_store to accept backend parameter**

In `lazymerge/sources.py`, replace the existing `scan_store` function:

```python
def _scan_conventions(root: zarr.Group) -> list[SourceEntry]:
    entries: list[SourceEntry] = []
    for name, item in root.members():
        if isinstance(item, zarr.Array):
            try:
                sa = read_spatial(item)
                pa = read_proj(item)
            except KeyError:
                continue
            entries.append(SourceEntry(
                path=name,
                spatial_attrs=sa,
                proj_attrs=pa,
                chunk_shape=tuple(item.chunks),
            ))
    return entries


def _scan_datafusion(root: zarr.Group) -> list[SourceEntry]:
    try:
        from zarr_datafusion_search import ZarrTable
    except ImportError as e:
        raise ImportError(
            "zarr-datafusion-search is required for backend='datafusion'. "
            "Install it with: pip install zarr-datafusion-search"
        ) from e

    from datafusion import SessionContext

    meta = root["meta"]

    ids = meta["id"][:]
    bboxes = meta["bbox"][:]
    epsgs = meta["proj:epsg"][:]
    t0 = meta["transform_0"][:]
    t1 = meta["transform_1"][:]
    t2 = meta["transform_2"][:]
    t3 = meta["transform_3"][:]
    t4 = meta["transform_4"][:]
    t5 = meta["transform_5"][:]
    shape_x = meta["shape_x"][:]
    shape_y = meta["shape_y"][:]

    entries: list[SourceEntry] = []
    for i in range(len(ids)):
        transform = (float(t0[i]), float(t1[i]), float(t2[i]),
                      float(t3[i]), float(t4[i]), float(t5[i]))
        sx, sy = int(shape_x[i]), int(shape_y[i])
        sa = SpatialAttrs(
            dimensions=["y", "x"],
            transform=transform,
            bbox=(
                float(transform[2]),
                float(transform[5] + transform[4] * sy),
                float(transform[2] + transform[0] * sx),
                float(transform[5]),
            ),
            shape=(sy, sx),
        )
        pa = ProjAttrs(code=f"EPSG:{int(epsgs[i])}")
        entries.append(SourceEntry(
            path=str(ids[i]),
            spatial_attrs=sa,
            proj_attrs=pa,
            chunk_shape=(0, 0),  # placeholder — resolved lazily
        ))
    return entries


def scan_store(
    root: zarr.Group,
    backend: str = "conventions",
) -> ScanIndex:
    if backend == "conventions":
        entries = _scan_conventions(root)
    elif backend == "datafusion":
        entries = _scan_datafusion(root)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'conventions' or 'datafusion'.")
    return ScanIndex(entries)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_sources.py::test_scan_store_backend_conventions_default -v`
Expected: PASS

- [ ] **Step 6: Write failing test — scan_store with backend="datafusion"**

```python
# In tests/test_sources.py

import numpy as np


def _make_datafusion_store():
    """Create a store with /meta group containing DataFusion-style columnar arrays."""
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")
    meta = root.create_group("meta")

    # Two sources: scene_a in EPSG:32618, scene_b in EPSG:32617
    ids = np.array(["scene_a", "scene_b"])
    meta.create_array("id", data=ids)
    meta.create_array("proj:epsg", data=np.array([32618, 32617], dtype="int64"))
    meta.create_array("transform_0", data=np.array([10.0, 10.0]))
    meta.create_array("transform_1", data=np.array([0.0, 0.0]))
    meta.create_array("transform_2", data=np.array([500000.0, 892673.0]))
    meta.create_array("transform_3", data=np.array([0.0, 0.0]))
    meta.create_array("transform_4", data=np.array([-10.0, -10.0]))
    meta.create_array("transform_5", data=np.array([6000000.0, 6016818.0]))
    meta.create_array("shape_x", data=np.array([100, 108], dtype="int64"))
    meta.create_array("shape_y", data=np.array([100, 108], dtype="int64"))

    # Create placeholder bbox array (not used directly — bbox derived from transform)
    meta.create_array("bbox", data=np.array([b"", b""], dtype=object))

    return store, root


def test_scan_store_datafusion_backend():
    """backend='datafusion' should read /meta arrays and build SourceEntry objects."""
    store, root = _make_datafusion_store()
    index = scan_store(root, backend="datafusion")
    assert len(index.entries) == 2
    paths = {e.path for e in index.entries}
    assert paths == {"scene_a", "scene_b"}

    scene_a = [e for e in index.entries if e.path == "scene_a"][0]
    assert scene_a.proj_attrs.code == "EPSG:32618"
    assert scene_a.spatial_attrs.transform == (10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0)
    assert scene_a.spatial_attrs.shape == (100, 100)
    # chunk_shape is a placeholder
    assert scene_a.chunk_shape == (0, 0)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_sources.py::test_scan_store_datafusion_backend -v`
Expected: PASS

- [ ] **Step 8: Write failing test — invalid backend raises ValueError**

```python
# In tests/test_sources.py

import pytest


def test_scan_store_invalid_backend():
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")
    with pytest.raises(ValueError, match="Unknown backend"):
        scan_store(root, backend="invalid")
```

- [ ] **Step 9: Run test to verify it passes**

Run: `pytest tests/test_sources.py::test_scan_store_invalid_backend -v`
Expected: PASS

- [ ] **Step 10: Run full sources test suite**

Run: `pytest tests/test_sources.py -v`
Expected: All tests PASS (existing tests still work because `backend` defaults to `"conventions"`)

- [ ] **Step 11: Commit**

```bash
git add lazymerge/sources.py tests/test_sources.py pyproject.toml
git commit -m "feat: add datafusion backend for scan_store"
```

---

### Task 4: band parameter and overview selection in merge

**Files:**
- Modify: `lazymerge/merge.py`
- Test: `tests/test_merge.py`

- [ ] **Step 1: Write failing test — merge with band=None preserves existing behavior**

```python
# In tests/test_merge.py — add at end

def test_merge_band_none_preserves_behavior():
    """Passing band=None explicitly should work identically to omitting it."""
    store, root = _make_same_crs_store()
    index = scan_store(root)

    target, spatial, proj = create_target(
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(50, 50),
    )

    result_arr, _, _ = merge(
        source_index=index,
        target=target,
        target_spatial=spatial,
        target_proj=proj,
        store=store,
        band=None,
    )

    data = result_arr.compute()
    assert data.shape == (100, 200)
    np.testing.assert_array_equal(data[:, :100], 1.0)
    np.testing.assert_array_equal(data[:, 100:], 2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_merge.py::test_merge_band_none_preserves_behavior -v`
Expected: FAIL with `TypeError: merge() got an unexpected keyword argument 'band'`

- [ ] **Step 3: Add band parameter to merge() signature**

In `lazymerge/merge.py`, update the `merge` function signature:

```python
def merge(
    source_index: ScanIndex,
    target: cubed.Array,
    target_spatial: SpatialAttrs,
    target_proj: ProjAttrs,
    store: Store,
    resampling: str = "nearest",
    band: str | None = None,
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
        band=band,
    )

    return result, target_spatial, target_proj
```

And update `_merge_block` to accept (but not yet use) the `band` parameter:

```python
def _merge_block(
    block: np.ndarray,
    block_id: tuple[int, ...],
    source_index: ScanIndex,
    target_spatial: SpatialAttrs,
    target_proj: ProjAttrs,
    store: Store,
    chunk_size: tuple[int, int],
    resampling: str,
    band: str | None = None,
) -> np.ndarray:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_merge.py::test_merge_band_none_preserves_behavior -v`
Expected: PASS

- [ ] **Step 5: Run full merge test suite to check no regressions**

Run: `pytest tests/test_merge.py -v`
Expected: All tests PASS

- [ ] **Step 6: Write failing test — merge with band parameter navigates into band group and selects overview**

```python
# In tests/test_merge.py

from lazymerge.conventions import write_spatial, write_proj, SpatialAttrs, ProjAttrs


def _make_multiscale_store():
    """Create a store with grouped sources containing band groups and multiscale overviews.

    Structure:
      scene_a/
        red/
          multiscales attr
          0/  (100x100 at 10m, filled with 1.0)
          1/  (50x50 at 20m, filled with 10.0)
      scene_b/
        red/
          multiscales attr
          0/  (100x100 at 10m, filled with 2.0)
          1/  (50x50 at 20m, filled with 20.0)
    """
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")

    for name, base_fill, ovr_fill, x_origin in [
        ("scene_a", 1.0, 10.0, 500000.0),
        ("scene_b", 2.0, 20.0, 501000.0),
    ]:
        scene = root.create_group(name)
        band = scene.create_group("red")

        # Base array (10m resolution)
        base = band.create_array("0", shape=(100, 100), dtype="f4", chunks=(50, 50))
        base[:] = base_fill
        write_spatial(
            base,
            SpatialAttrs(
                dimensions=["y", "x"],
                transform=(10.0, 0.0, x_origin, 0.0, -10.0, 6000000.0),
                bbox=(x_origin, 5999000.0, x_origin + 1000.0, 6000000.0),
                shape=(100, 100),
            ),
        )
        write_proj(base, ProjAttrs(code="EPSG:32618"))

        # Overview array (20m resolution)
        ovr = band.create_array("1", shape=(50, 50), dtype="f4", chunks=(50, 50))
        ovr[:] = ovr_fill
        write_spatial(
            ovr,
            SpatialAttrs(
                dimensions=["y", "x"],
                transform=(20.0, 0.0, x_origin, 0.0, -20.0, 6000000.0),
                bbox=(x_origin, 5999000.0, x_origin + 1000.0, 6000000.0),
                shape=(50, 50),
            ),
        )
        write_proj(ovr, ProjAttrs(code="EPSG:32618"))

        # multiscales attr on band group
        band.attrs["multiscales"] = {
            "layout": [
                {"asset": "0", "transform": {"scale": [1.0, 1.0], "translation": [0.0, 0.0]}},
                {
                    "asset": "1",
                    "derived_from": "0",
                    "transform": {"scale": [2.0, 2.0], "translation": [0.5, 0.5]},
                },
            ]
        }

    return store, root


def test_merge_with_band_uses_base_resolution():
    """When target res matches base (10m), merge should read from base arrays (level 0)."""
    store, root = _make_multiscale_store()

    # Manually build SourceEntries pointing to the scene groups (like datafusion would)
    from lazymerge.sources import SourceEntry, ScanIndex

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
    index = ScanIndex(entries)

    target, spatial, proj = create_target(
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(50, 50),
    )

    result_arr, _, _ = merge(
        source_index=index,
        target=target,
        target_spatial=spatial,
        target_proj=proj,
        store=store,
        band="red",
    )

    data = result_arr.compute()
    assert data.shape == (100, 200)
    # Should read from base (level 0): 1.0 and 2.0, NOT overview values 10.0 / 20.0
    np.testing.assert_array_equal(data[:, :100], 1.0)
    np.testing.assert_array_equal(data[:, 100:], 2.0)


def test_merge_with_band_selects_overview():
    """When target res is coarser than base, merge should select the appropriate overview."""
    store, root = _make_multiscale_store()

    from lazymerge.sources import SourceEntry, ScanIndex

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
    index = ScanIndex(entries)

    # Target at 20m resolution — should trigger overview selection (level 1)
    target, spatial, proj = create_target(
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
        resolution=20.0,
        chunk_size=(25, 25),
    )

    result_arr, _, _ = merge(
        source_index=index,
        target=target,
        target_spatial=spatial,
        target_proj=proj,
        store=store,
        band="red",
    )

    data = result_arr.compute()
    assert data.shape == (50, 100)
    # Should read from overview (level 1): 10.0 and 20.0, NOT base values 1.0 / 2.0
    np.testing.assert_array_equal(data[:, :50], 10.0)
    np.testing.assert_array_equal(data[:, 50:], 20.0)
```

- [ ] **Step 7: Run test to verify it fails**

Run: `pytest tests/test_merge.py::test_merge_with_band_uses_base_resolution -v`
Expected: FAIL (band navigation not yet implemented in `_merge_block`)

- [ ] **Step 8: Implement band navigation and overview selection in _merge_block**

Update `_merge_block` in `lazymerge/merge.py`. Add the import at the top:

```python
from lazymerge.conventions import SpatialAttrs, ProjAttrs, chunk_bbox, read_multiscales, read_spatial, read_proj, OverviewLevel
from lazymerge.sources import ScanIndex, select_overview
```

Replace the source reading section of `_merge_block` (the inner loop over `sources`) with band-aware logic. The full updated `_merge_block`:

```python
def _merge_block(
    block: np.ndarray,
    block_id: tuple[int, ...],
    source_index: ScanIndex,
    target_spatial: SpatialAttrs,
    target_proj: ProjAttrs,
    store: Store,
    chunk_size: tuple[int, int],
    resampling: str,
    band: str | None = None,
) -> np.ndarray:
    target_crs = target_proj.code
    if target_crs is None:
        raise ValueError("target_proj.code must not be None")

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
    actual_shape: tuple[int, int] = (block.shape[0], block.shape[1])

    output = np.full(actual_shape, np.nan, dtype=block.dtype)

    # Pass 1: find intersecting sources
    sources = source_index.find_intersecting_sources(cb, target_crs)
    if not sources:
        return output

    target_res = abs(target_spatial.transform[0])
    unfilled = int(np.count_nonzero(np.isnan(output)))

    for source_entry in sources:
        if unfilled == 0:
            break

        root = zarr.open_group(store, mode="r")

        if band is not None:
            # Navigate into source_group/band and select overview
            band_group = root[source_entry.path][band]
            overviews = read_multiscales(band_group)

            if overviews is not None:
                # Read base array's native resolution
                base_path = dict(band_group.attrs)["multiscales"]["layout"][0]["asset"]
                base_array = band_group[base_path]
                base_spatial = read_spatial(base_array)
                native_res = abs(base_spatial.transform[0])

                # Estimate target res in source CRS if CRS differs
                src_crs = source_entry.proj_attrs.code or "EPSG:4326"
                if src_crs != target_crs:
                    from pyproj import Transformer
                    transformer = Transformer.from_crs(target_crs, src_crs, always_xy=True)
                    cx = (cb[0] + cb[2]) / 2
                    cy = (cb[1] + cb[3]) / 2
                    x0, y0 = transformer.transform(cx, cy)
                    x1, y1 = transformer.transform(cx + target_res, cy)
                    src_target_res = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                else:
                    src_target_res = target_res

                selected = select_overview(overviews, src_target_res, native_res)
                if selected is not None:
                    src_array = band_group[selected.path]
                else:
                    src_array = base_array
            else:
                # No multiscales — use first available array in band group
                src_array = None
                for child_name, child in band_group.members():
                    if isinstance(child, zarr.Array):
                        src_array = child
                        break
                if src_array is None:
                    continue

            if not isinstance(src_array, zarr.Array):
                continue

            # Read spatial/proj from the selected array
            src_spatial = read_spatial(src_array)
            src_proj = read_proj(src_array)
            src_crs = src_proj.code or "EPSG:4326"
            src_chunk_shape = tuple(src_array.chunks)

            # Build a temporary SourceEntry for chunk-level filtering
            from lazymerge.sources import SourceEntry as _SE
            resolved_entry = _SE(
                path=source_entry.path,
                spatial_attrs=src_spatial,
                proj_attrs=src_proj,
                chunk_shape=src_chunk_shape,
            )

            # Pass 2: find intersecting source chunks
            src_chunks = source_index.find_intersecting_chunks(resolved_entry, cb, target_crs)
            if not src_chunks:
                continue

            for _, (src_row, src_col) in src_chunks:
                if unfilled == 0:
                    break

                sr_start = src_row * src_chunk_shape[0]
                sr_end = min(sr_start + src_chunk_shape[0], src_spatial.shape[0])
                sc_start = src_col * src_chunk_shape[1]
                sc_end = min(sc_start + src_chunk_shape[1], src_spatial.shape[1])
                src_data = src_array[sr_start:sr_end, sc_start:sc_end]

                sa_t = src_spatial.transform
                src_chunk_c = sa_t[2] + sa_t[0] * sc_start + sa_t[1] * sr_start
                src_chunk_f = sa_t[5] + sa_t[3] * sc_start + sa_t[4] * sr_start
                src_chunk_transform = (sa_t[0], sa_t[1], src_chunk_c, sa_t[3], sa_t[4], src_chunk_f)

                warped = warp_chunk(
                    source_data=np.asarray(src_data),
                    source_transform=src_chunk_transform,
                    source_crs=src_crs,
                    target_transform=chunk_transform,
                    target_crs=target_crs,
                    target_shape=actual_shape,
                    resampling=resampling,
                )

                mask = np.isnan(output) & ~np.isnan(warped)
                output[mask] = warped[mask]
                unfilled -= int(np.count_nonzero(mask))
        else:
            # Original behavior: source_entry.path points directly to an array
            src_array = root[source_entry.path]
            if not isinstance(src_array, zarr.Array):
                continue

            src_crs = source_entry.proj_attrs.code or "EPSG:4326"

            # Pass 2: find intersecting source chunks
            src_chunks = source_index.find_intersecting_chunks(source_entry, cb, target_crs)
            if not src_chunks:
                continue

            for _, (src_row, src_col) in src_chunks:
                if unfilled == 0:
                    break

                sr_start = src_row * source_entry.chunk_shape[0]
                sr_end = min(sr_start + source_entry.chunk_shape[0], source_entry.spatial_attrs.shape[0])
                sc_start = src_col * source_entry.chunk_shape[1]
                sc_end = min(sc_start + source_entry.chunk_shape[1], source_entry.spatial_attrs.shape[1])
                src_data = src_array[sr_start:sr_end, sc_start:sc_end]

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

                mask = np.isnan(output) & ~np.isnan(warped)
                output[mask] = warped[mask]
                unfilled -= int(np.count_nonzero(mask))

    return output
```

- [ ] **Step 9: Run band tests**

Run: `pytest tests/test_merge.py::test_merge_with_band_uses_base_resolution tests/test_merge.py::test_merge_with_band_selects_overview -v`
Expected: Both PASS

- [ ] **Step 10: Run full merge test suite**

Run: `pytest tests/test_merge.py -v`
Expected: All tests PASS (existing tests unaffected since `band` defaults to `None`)

- [ ] **Step 11: Commit**

```bash
git add lazymerge/merge.py tests/test_merge.py
git commit -m "feat: add band param and overview selection to merge"
```

---

### Task 5: Update exports and run full test suite

**Files:**
- Modify: `lazymerge/__init__.py`

- [ ] **Step 1: Update __init__.py exports**

Add the new public names to `lazymerge/__init__.py`:

```python
from lazymerge.conventions import (
    OverviewLevel as OverviewLevel,
    ProjAttrs as ProjAttrs,
    SpatialAttrs as SpatialAttrs,
    chunk_bbox as chunk_bbox,
    read_multiscales as read_multiscales,
    read_proj as read_proj,
    read_spatial as read_spatial,
    write_proj as write_proj,
    write_spatial as write_spatial,
)
from lazymerge.merge import merge as merge
from lazymerge.sources import (
    ScanIndex as ScanIndex,
    SourceEntry as SourceEntry,
    scan_store as scan_store,
    select_overview as select_overview,
)
from lazymerge.target import create_target as create_target, to_zarr as to_zarr
from lazymerge.warp import warp_chunk as warp_chunk
```

- [ ] **Step 2: Verify imports work**

Run: `python -c "from lazymerge import OverviewLevel, read_multiscales, select_overview; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add lazymerge/__init__.py
git commit -m "feat: export OverviewLevel, read_multiscales, select_overview"
```
