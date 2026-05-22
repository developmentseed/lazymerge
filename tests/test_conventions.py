import pytest
import zarr

from lazymerge.conventions import (
    OverviewLevel,
    ProjAttrs,
    SpatialAttrs,
    chunk_bbox,
    read_multiscales,
    read_proj,
    read_spatial,
    write_proj,
    write_spatial,
)


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
    with pytest.raises(ValueError, match="at least one of"):
        ProjAttrs()


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


def test_chunk_bbox_origin():
    """Chunk (0, 0) should return the top-left portion of the array bbox."""
    sa = SpatialAttrs(
        dimensions=["y", "x"],
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0),
        bbox=(500000.0, 5990000.0, 510000.0, 6000000.0),
        shape=(1000, 1000),
    )
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
    result = chunk_bbox(sa, chunk_coords=(1, 2), chunk_shape=(256, 256))
    assert result == (505120.0, 5994880.0, 507680.0, 5997440.0)


def test_overview_level_construction():
    ol = OverviewLevel(path="1", scale=(2.0, 2.0), resolution=20.0)
    assert ol.path == "1"
    assert ol.scale == (2.0, 2.0)
    assert ol.resolution == 20.0


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
        ],
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


def test_read_multiscales_no_attr():
    """A group without multiscales attr should return None."""
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")
    band_group = root.create_group("red")
    band_group.create_array("data", shape=(100, 100), dtype="f4", chunks=(50, 50))

    result = read_multiscales(band_group)
    assert result is None
