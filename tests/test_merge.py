from unittest.mock import patch

import numpy as np
import zarr

from lazymerge.conventions import ProjAttrs, SpatialAttrs, write_proj, write_spatial
from lazymerge.merge import merge
from lazymerge.sources import scan_store
from lazymerge.target import create_target
from lazymerge.warp import warp_chunk as real_warp_chunk


def _make_same_crs_store():
    """Two adjacent source arrays in EPSG:32618, each 100x100 at 10m resolution."""
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")

    # Source A: x=[500000, 501000], y=[5999000, 6000000], filled with 1.0
    a = root.create_array("source_a", shape=(100, 100), dtype="f4", chunks=(50, 50))
    a[:] = 1.0
    write_spatial(
        a,
        SpatialAttrs(
            dimensions=["y", "x"],
            transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0),
            bbox=(500000.0, 5999000.0, 501000.0, 6000000.0),
            shape=(100, 100),
        ),
    )
    write_proj(a, ProjAttrs(code="EPSG:32618"))

    # Source B: x=[501000, 502000], y=[5999000, 6000000], filled with 2.0
    b = root.create_array("source_b", shape=(100, 100), dtype="f4", chunks=(50, 50))
    b[:] = 2.0
    write_spatial(
        b,
        SpatialAttrs(
            dimensions=["y", "x"],
            transform=(10.0, 0.0, 501000.0, 0.0, -10.0, 6000000.0),
            bbox=(501000.0, 5999000.0, 502000.0, 6000000.0),
            shape=(100, 100),
        ),
    )
    write_proj(b, ProjAttrs(code="EPSG:32618"))

    return store, root


def _make_mixed_crs_store():
    """Two source arrays covering adjacent areas: source_a in EPSG:32618, source_b in EPSG:32617.

    source_a covers x=[500000, 501000], y=[5999000, 6000000] in UTM 18N.
    source_b covers the same geographic area as x=[501000, 502000], y=[5999000, 6000000]
    in UTM 18N, but is stored in UTM 17N coordinates.
    """
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")

    # Source A in UTM 18N: filled with 1.0
    a = root.create_array("source_a", shape=(100, 100), dtype="f4", chunks=(50, 50))
    a[:] = 1.0
    write_spatial(
        a,
        SpatialAttrs(
            dimensions=["y", "x"],
            transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0),
            bbox=(500000.0, 5999000.0, 501000.0, 6000000.0),
            shape=(100, 100),
        ),
    )
    write_proj(a, ProjAttrs(code="EPSG:32618"))

    # Source B in UTM 17N: same geographic area as UTM 18N x=[501000,502000] y=[5999000,6000000]
    # UTM 17N coords: x=[892673, 893756], y=[6015735, 6016818]
    b_width = int((893756.0 - 892673.0) / 10.0)
    b_height = int((6016818.0 - 6015735.0) / 10.0)
    b = root.create_array("source_b", shape=(b_height, b_width), dtype="f4", chunks=(50, 50))
    b[:] = 2.0
    write_spatial(
        b,
        SpatialAttrs(
            dimensions=["y", "x"],
            transform=(10.0, 0.0, 892673.0, 0.0, -10.0, 6016818.0),
            bbox=(892673.0, 6015735.0, 893756.0, 6016818.0),
            shape=(b_height, b_width),
        ),
    )
    write_proj(b, ProjAttrs(code="EPSG:32617"))

    return store, root


def test_merge_two_adjacent_sources_same_crs():
    """Merging two adjacent same-CRS sources into a target spanning both."""
    store, root = _make_same_crs_store()
    index = scan_store(root)

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


def test_merge_mixed_crs():
    """Merging sources from different CRS into a single target."""
    store, root = _make_mixed_crs_store()
    index = scan_store(root)

    # Target in UTM 18N spanning both sources
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
    )

    data = result_arr.compute()
    assert data.shape == (100, 200)

    # Check well inside source_a territory (same CRS, no reprojection ambiguity)
    np.testing.assert_array_equal(data[:, :80], 1.0)

    # Check well inside source_b territory (reprojected from UTM 17N)
    # Use a generous interior margin to avoid reprojection boundary effects
    interior_b = data[20:80, 120:180]
    assert np.all(interior_b == 2.0), (
        f"Expected all 2.0 in source_b interior, got unique values: {np.unique(interior_b)}"
    )

    # Verify both values are present in the full array
    unique_vals = np.unique(data[~np.isnan(data)])
    assert 1.0 in unique_vals
    assert 2.0 in unique_vals


def _make_overlapping_store():
    """Two source arrays that fully overlap the same extent, filled with different values."""
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")

    for name, fill in [("source_a", 1.0), ("source_b", 2.0)]:
        arr = root.create_array(name, shape=(100, 100), dtype="f4", chunks=(100, 100))
        arr[:] = fill
        write_spatial(
            arr,
            SpatialAttrs(
                dimensions=["y", "x"],
                transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0),
                bbox=(500000.0, 5999000.0, 501000.0, 6000000.0),
                shape=(100, 100),
            ),
        )
        write_proj(arr, ProjAttrs(code="EPSG:32618"))

    return store, root


def test_merge_early_stop():
    """With fully overlapping sources, the second source should be skipped
    once the first fills every pixel."""
    store, root = _make_overlapping_store()
    index = scan_store(root)

    target, spatial, proj = create_target(
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 501000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(100, 100),
    )

    warp_call_count = 0
    original_warp = real_warp_chunk

    def counting_warp(*args, **kwargs):
        nonlocal warp_call_count
        warp_call_count += 1
        return original_warp(*args, **kwargs)

    with patch("lazymerge.merge.warp_chunk", side_effect=counting_warp):
        result_arr, _, _ = merge(
            source_index=index,
            target=target,
            target_spatial=spatial,
            target_proj=proj,
            store=store,
        )
        data = result_arr.compute()

    # Only the first source's chunk should have been warped; second skipped
    assert warp_call_count == 1
    # All pixels filled by whichever source came first
    unique = np.unique(data)
    assert len(unique) == 1
    assert unique[0] in (1.0, 2.0)


def test_merge_no_sources():
    """Target that doesn't overlap any source should be all NaN."""
    store, root = _make_same_crs_store()
    index = scan_store(root)

    target, spatial, proj = create_target(
        crs="EPSG:32618",
        bbox=(600000.0, 5999000.0, 601000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(50, 50),
    )

    result_arr, _, _ = merge(
        source_index=index,
        target=target,
        target_spatial=spatial,
        target_proj=proj,
        store=store,
    )

    data = result_arr.compute()
    assert np.all(np.isnan(data))
