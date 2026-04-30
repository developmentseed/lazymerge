import zarr

from lazymerge.conventions import read_proj, read_spatial
from lazymerge.target import create_target, to_zarr


def test_create_target_shape():
    """Shape should be computed from bbox and resolution."""
    arr, spatial, proj = create_target(
        crs="EPSG:32618",
        bbox=(500000.0, 5999000.0, 501000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(256, 256),
    )
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
