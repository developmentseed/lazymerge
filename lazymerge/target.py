from __future__ import annotations

import math

import cubed
import zarr
from zarr.abc.store import Store

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
    store: str | Store,
    path: str | None = None,
) -> None:
    result = array.compute()
    root = zarr.open_group(store, mode="w")
    arr_path = path or "data"
    zarr_arr = root.create_array(arr_path, data=result, chunks=array.chunksize)
    write_spatial(zarr_arr, spatial)
    write_proj(zarr_arr, proj)
