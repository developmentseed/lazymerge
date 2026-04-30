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
