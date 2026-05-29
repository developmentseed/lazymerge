from __future__ import annotations

import threading
from concurrent.futures import Future

import numpy as np
import pytest
import zarr

from lazymerge.conventions import ProjAttrs, SpatialAttrs, write_proj, write_spatial
from lazymerge.merge import (
    _virtualize_futures,
    ensure_virtualized,
    merge,
)
from lazymerge.sources import ScanIndex, SourceEntry


@pytest.fixture(autouse=True)
def _clear_futures():
    """Clear virtualize futures after each test."""
    yield
    _virtualize_futures.clear()


def _make_entry(path: str = "test_source") -> SourceEntry:
    return SourceEntry(
        path=path,
        spatial_attrs=SpatialAttrs(
            dimensions=["y", "x"],
            transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0),
            bbox=(500000.0, 5999000.0, 501000.0, 6000000.0),
            shape=(100, 100),
        ),
        proj_attrs=ProjAttrs(code="EPSG:32618"),
        chunk_shape=(50, 50),
    )


# ---------- ensure_virtualized tests ----------


def test_ensure_virtualized_calls_callback_once():
    """5 threads call ensure_virtualized with the same source_id; callback runs exactly once."""
    call_count = 0
    lock = threading.Lock()

    def callback(_source_id, _entry, _store, _bands):
        nonlocal call_count
        with lock:
            call_count += 1

    entry = _make_entry()
    threads = []
    for _ in range(5):
        t = threading.Thread(
            target=ensure_virtualized,
            args=("same_source", callback, entry, None, ["red"]),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert call_count == 1


def test_ensure_virtualized_different_sources_run_independently():
    """3 different source_ids each trigger the callback exactly once."""
    called_ids: list[str] = []
    lock = threading.Lock()

    def callback(source_id, _entry, _store, _bands):
        with lock:
            called_ids.append(source_id)

    entry = _make_entry()
    for sid in ["src_a", "src_b", "src_c"]:
        ensure_virtualized(sid, callback, entry, None, ["red"])

    assert sorted(called_ids) == ["src_a", "src_b", "src_c"]


def test_ensure_virtualized_propagates_exception():
    """When the callback raises, ensure_virtualized propagates the exception."""

    def callback(_source_id, _entry, _store, _bands):
        raise RuntimeError("virtualize failed")

    entry = _make_entry()
    with pytest.raises(RuntimeError, match="virtualize failed"):
        ensure_virtualized("bad_source", callback, entry, None, ["red"])


# ---------- merge integration tests ----------


def _make_virtualize_store():
    """Create a zarr MemoryStore with two scenes for virtualize tests.

    Structure:
      scene_a/red/0  (100x100 at 10m, filled with 1.0)
      scene_b/red/0  (100x100 at 10m, filled with 2.0)
    """
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")

    for name, fill, x_origin in [
        ("scene_a", 1.0, 500000.0),
        ("scene_b", 2.0, 501000.0),
    ]:
        scene = root.create_group(name)
        band_group = scene.create_group("red")
        arr = band_group.create_array("0", shape=(100, 100), dtype="f4", chunks=(50, 50))
        arr[:] = fill
        write_spatial(
            arr,
            SpatialAttrs(
                dimensions=["y", "x"],
                transform=(10.0, 0.0, x_origin, 0.0, -10.0, 6000000.0),
                bbox=(x_origin, 5999000.0, x_origin + 1000.0, 6000000.0),
                shape=(100, 100),
            ),
        )
        write_proj(arr, ProjAttrs(code="EPSG:32618"))

    return store


def _make_virtualize_index():
    entries = []
    for name, x_origin in [("scene_a", 500000.0), ("scene_b", 501000.0)]:
        entries.append(
            SourceEntry(
                path=name,
                spatial_attrs=SpatialAttrs(
                    dimensions=["y", "x"],
                    transform=(10.0, 0.0, x_origin, 0.0, -10.0, 6000000.0),
                    bbox=(x_origin, 5999000.0, x_origin + 1000.0, 6000000.0),
                    shape=(100, 100),
                ),
                proj_attrs=ProjAttrs(code="EPSG:32618"),
                chunk_shape=(50, 50),
                metadata={"some": "data"},
            )
        )
    return ScanIndex(entries)


def test_merge_calls_virtualize_callback():
    """merge() with virtualize= calls the callback for each source."""
    store = _make_virtualize_store()
    index = _make_virtualize_index()

    called_source_ids: list[str] = []
    lock = threading.Lock()

    def mock_callback(source_id, _entry, _st, _bands):
        with lock:
            called_source_ids.append(source_id)

    result_arr, _, _, _ = merge(
        store=store,
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(50, 50),
        source_index=index,
        bands="red",
        virtualize=mock_callback,
    )
    _ = result_arr.compute()

    assert "scene_a" in called_source_ids
    assert "scene_b" in called_source_ids


def test_merge_without_virtualize_unchanged():
    """merge() without virtualize produces correct data (1.0 left, 2.0 right)."""
    store = _make_virtualize_store()
    index = _make_virtualize_index()

    result_arr, _, _, _ = merge(
        store=store,
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(50, 50),
        source_index=index,
        bands="red",
    )
    data = result_arr.compute()

    assert data.shape == (100, 200)
    np.testing.assert_array_equal(data[:, :100], 1.0)
    np.testing.assert_array_equal(data[:, 100:], 2.0)


def test_futures_cleared_after_merge():
    """_virtualize_futures is empty after merge() returns."""
    store = _make_virtualize_store()
    index = _make_virtualize_index()

    def noop_callback(_source_id, _entry, _st, _bands):
        pass

    # Pre-populate futures to verify merge clears them
    sentinel: Future[None] = Future()
    sentinel.set_result(None)
    _virtualize_futures["stale_key"] = sentinel

    _result_arr, _, _, _ = merge(
        store=store,
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(50, 50),
        source_index=index,
        bands="red",
        virtualize=noop_callback,
    )

    assert "stale_key" not in _virtualize_futures
