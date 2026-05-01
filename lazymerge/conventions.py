from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import zarr
from zarr_cm import geo_proj, spatial


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
    projjson: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.code is None and self.wkt2 is None and self.projjson is None:
            raise ValueError("ProjAttrs requires at least one of: code, wkt2, projjson")


def write_spatial(node: zarr.Group | zarr.Array, attrs: SpatialAttrs) -> None:  # type: ignore[type-arg]
    data = spatial.create(
        dimensions=attrs.dimensions,
        bbox=list(attrs.bbox),
        transform=list(attrs.transform),
        shape=list(attrs.shape),
        registration=attrs.registration,
    )
    existing = dict(node.attrs)
    updated = spatial.insert(existing, data, overwrite=True)
    node.attrs.update(updated)


def write_proj(node: zarr.Group | zarr.Array, attrs: ProjAttrs) -> None:  # type: ignore[type-arg]
    data = geo_proj.create(
        code=attrs.code,
        wkt2=attrs.wkt2,
        projjson=attrs.projjson,
    )
    existing = dict(node.attrs)
    updated = geo_proj.insert(existing, data, overwrite=True)
    node.attrs.update(updated)


def read_spatial(node: zarr.Group | zarr.Array) -> SpatialAttrs:  # type: ignore[type-arg]
    _, data = spatial.extract(dict(node.attrs))
    return SpatialAttrs(
        dimensions=data["spatial:dimensions"],
        transform=tuple(data["spatial:transform"]),
        bbox=tuple(data["spatial:bbox"]),
        shape=tuple(data["spatial:shape"]),
        registration=data.get("spatial:registration", "pixel"),
    )


def read_proj(node: zarr.Group | zarr.Array) -> ProjAttrs:  # type: ignore[type-arg]
    _, data = geo_proj.extract(dict(node.attrs))
    return ProjAttrs(
        code=data.get("proj:code"),
        wkt2=data.get("proj:wkt2"),
        projjson=data.get("proj:projjson"),
    )


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

    x0 = a * col_offset + b * row_offset + c
    y0 = d * col_offset + e * row_offset + f
    x1 = a * (col_offset + cols) + b * (row_offset + rows) + c
    y1 = d * (col_offset + cols) + e * (row_offset + rows) + f

    xmin = min(x0, x1)
    xmax = max(x0, x1)
    ymin = min(y0, y1)
    ymax = max(y0, y1)

    return (xmin, ymin, xmax, ymax)
