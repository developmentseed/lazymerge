from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import zarr


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
    node.attrs["spatial:dimensions"] = attrs.dimensions
    node.attrs["spatial:transform"] = list(attrs.transform)
    node.attrs["spatial:bbox"] = list(attrs.bbox)
    node.attrs["spatial:shape"] = list(attrs.shape)
    node.attrs["spatial:registration"] = attrs.registration


def write_proj(node: zarr.Group | zarr.Array, attrs: ProjAttrs) -> None:  # type: ignore[type-arg]
    if attrs.code is not None:
        node.attrs["proj:code"] = attrs.code
    if attrs.wkt2 is not None:
        node.attrs["proj:wkt2"] = attrs.wkt2
    if attrs.projjson is not None:
        node.attrs["proj:projjson"] = attrs.projjson


def read_spatial(node: zarr.Group | zarr.Array) -> SpatialAttrs:  # type: ignore[type-arg]
    a = node.attrs
    return SpatialAttrs(
        dimensions=cast(list[str], a["spatial:dimensions"]),
        transform=tuple(cast(list[float], a["spatial:transform"])),
        bbox=tuple(cast(list[float], a["spatial:bbox"])),
        shape=tuple(cast(list[int], a["spatial:shape"])),
        registration=cast(str, a.get("spatial:registration", "pixel")),
    )


def read_proj(node: zarr.Group | zarr.Array) -> ProjAttrs:  # type: ignore[type-arg]
    a = node.attrs
    return ProjAttrs(
        code=cast(str | None, a.get("proj:code")),
        wkt2=cast(str | None, a.get("proj:wkt2")),
        projjson=cast(dict[str, Any] | None, a.get("proj:projjson")),
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
