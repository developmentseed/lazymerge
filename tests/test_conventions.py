import zarr

from lazymerge.conventions import (
    ProjAttrs,
    SpatialAttrs,
    chunk_bbox,
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
    import pytest

    with pytest.raises(ValueError):
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
