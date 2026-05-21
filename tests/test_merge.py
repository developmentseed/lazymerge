from unittest.mock import patch

import numpy as np
import pytest
import zarr

from lazymerge.conventions import ProjAttrs, SpatialAttrs, write_proj, write_spatial
from lazymerge.merge import merge
from lazymerge.sources import scan_store
from lazymerge.warp import warp_source_region as real_warp_source_region


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

    result_arr, result_spatial, result_proj, _ = merge(
        store=store,
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(50, 50),
        source_index=index,
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

    result_arr, _, _, _ = merge(
        store=store,
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(50, 50),
        source_index=index,
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

    warp_call_count = 0
    original_warp = real_warp_source_region

    def counting_warp(*args, **kwargs):
        nonlocal warp_call_count
        warp_call_count += 1
        return original_warp(*args, **kwargs)

    with patch("lazymerge.merge.warp_source_region", side_effect=counting_warp):
        result_arr, _, _, _ = merge(
            store=store,
            crs="EPSG:32618",
            bbox=(500000.0, 5999000.0, 501000.0, 6000000.0),
            resolution=10.0,
            chunk_size=(100, 100),
            source_index=index,
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

    result_arr, _, _, _ = merge(
        store=store,
        crs="EPSG:32618",
        bbox=(600000.0, 5999000.0, 601000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(50, 50),
        source_index=index,
    )

    data = result_arr.compute()
    assert np.all(np.isnan(data))


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
    result_arr, _, _, _ = merge(
        store=store,
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
        resolution=20.0,
        chunk_size=(25, 25),
        source_index=index,
        bands="red",
    )

    data = result_arr.compute()
    assert data.shape == (50, 100)
    # Should read from overview (level 1): 10.0 and 20.0, NOT base values 1.0 / 2.0
    np.testing.assert_array_equal(data[:, :50], 10.0)
    np.testing.assert_array_equal(data[:, 50:], 20.0)


def test_merge_bands_none_preserves_behavior():
    """Passing bands=None explicitly should work identically to omitting it."""
    store, root = _make_same_crs_store()
    index = scan_store(root)

    result_arr, _, _, _ = merge(
        store=store,
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(50, 50),
        source_index=index,
        bands=None,
    )

    data = result_arr.compute()
    assert data.shape == (100, 200)
    np.testing.assert_array_equal(data[:, :100], 1.0)
    np.testing.assert_array_equal(data[:, 100:], 2.0)


def _make_multiband_store():
    """Create a store with two scenes, each having 'red' and 'green' band groups.

    Structure:
      scene_a/
        red/0   (100x100 at 10m, filled with 1.0)
        green/0 (100x100 at 10m, filled with 10.0)
      scene_b/
        red/0   (100x100 at 10m, filled with 2.0)
        green/0 (100x100 at 10m, filled with 20.0)
    """
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")

    for name, x_origin, fills in [
        ("scene_a", 500000.0, {"red": 1.0, "green": 10.0}),
        ("scene_b", 501000.0, {"red": 2.0, "green": 20.0}),
    ]:
        scene = root.create_group(name)
        for band_name, fill in fills.items():
            band_group = scene.create_group(band_name)
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

    return store, root


def _make_multiband_index():
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
    return ScanIndex(entries)


def test_merge_multi_band():
    """Multiple bands produce a 3D (band, y, x) output."""
    store, root = _make_multiband_store()
    index = _make_multiband_index()

    result_arr, _, _, _ = merge(
        store=store,
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(50, 50),
        source_index=index,
        bands=["red", "green"],
    )

    data = result_arr.compute()
    assert data.shape == (2, 100, 200)
    # Band 0 = red: scene_a=1.0 (left), scene_b=2.0 (right)
    np.testing.assert_array_equal(data[0, :, :100], 1.0)
    np.testing.assert_array_equal(data[0, :, 100:], 2.0)
    # Band 1 = green: scene_a=10.0 (left), scene_b=20.0 (right)
    np.testing.assert_array_equal(data[1, :, :100], 10.0)
    np.testing.assert_array_equal(data[1, :, 100:], 20.0)


def test_merge_single_band_list_stays_2d():
    """A single-element bands list produces 2D output (no band dimension)."""
    store, root = _make_multiband_store()
    index = _make_multiband_index()

    result_arr, _, _, _ = merge(
        store=store,
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(50, 50),
        source_index=index,
        bands=["red"],
    )

    data = result_arr.compute()
    assert data.shape == (100, 200)
    np.testing.assert_array_equal(data[:, :100], 1.0)
    np.testing.assert_array_equal(data[:, 100:], 2.0)


def test_merge_single_band_string_stays_2d():
    """A string bands value produces 2D output (backward compat)."""
    store, root = _make_multiband_store()
    index = _make_multiband_index()

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


def test_merge_temporal_grouping_creates_time_dimension():
    """temporal_grouping should produce a 3D (time, y, x) output."""
    store, root = _make_same_crs_store()

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
        result_arr, _, _, time_coords = merge(
            store=store,
            crs="EPSG:32618",
            bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
            resolution=10.0,
            chunk_size=(50, 50),
            datafusion=True,
            temporal_grouping="P1M",
        )
        data = result_arr.compute()

    assert data.shape == (2, 100, 200)
    assert time_coords is not None
    np.testing.assert_array_equal(time_coords, [
        np.datetime64("2024-06-01", "D"),
        np.datetime64("2024-07-01", "D"),
    ])


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
        result_arr, _, _, _ = merge(
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

    assert data.shape == (2, 2, 100, 200)
