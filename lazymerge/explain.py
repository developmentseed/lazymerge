"""Dry-run read estimator: explains which source regions would be read."""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import zarr
from pyproj import Transformer

from lazymerge.conventions import (
    ProjAttrs,
    SpatialAttrs,
    chunk_bbox,
    read_multiscales,
)
from lazymerge.merge import (
    _read_proj_or_derive,
    _read_spatial_or_derive,
    _reproject_bbox_to_4326,
    _resolve_array,
)
from lazymerge.sources import (
    ScanIndex,
    SourceEntry,
    query_datafusion_sources,
    select_overview,
)
from lazymerge.warp import _target_to_source_pixels


@dataclass
class SourceRead:
    """Read details for one source within one target chunk."""

    source_path: str
    source_crs: str
    overview_path: str
    native_resolution: float | None = None
    overview_resolution: float | None = None
    region_row_start: int = 0
    region_row_end: int = 0
    region_col_start: int = 0
    region_col_end: int = 0
    region_height: int = field(init=False)
    region_width: int = field(init=False)

    def __post_init__(self) -> None:
        self.region_height = self.region_row_end - self.region_row_start
        self.region_width = self.region_col_end - self.region_col_start


@dataclass
class ChunkRead:
    """All reads required for one target chunk."""

    chunk_row: int
    chunk_col: int
    chunk_height: int
    chunk_width: int
    chunk_bbox: tuple[float, float, float, float]
    source_reads: list[SourceRead]
    n_source_reads: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_source_reads = len(self.source_reads)


_MULTI_THRESHOLD = 2


@dataclass
class ExplainPlan:
    """Complete dry-run read plan for a lazymerge merge operation."""

    target_crs: str
    target_resolution: float
    target_shape: tuple[int, int]
    chunk_size: tuple[int, int]
    band: str | None
    resampling: str
    nodata: float | int | None
    chunk_reads: list[ChunkRead]

    @property
    def total_chunk_reads(self) -> int:
        return len(self.chunk_reads)

    @property
    def total_source_reads(self) -> int:
        return sum(c.n_source_reads for c in self.chunk_reads)

    @property
    def empty_chunk_count(self) -> int:
        return sum(1 for c in self.chunk_reads if c.n_source_reads == 0)

    @property
    def _n_tiles(self) -> tuple[int, int]:
        return (
            max(1, -(-self.target_shape[1] // self.chunk_size[1])),
            max(1, -(-self.target_shape[0] // self.chunk_size[0])),
        )

    def __repr__(self) -> str:
        n_x, n_y = self._n_tiles
        return (
            f"ExplainPlan: {n_x * n_y} spatial chunk(s) "
            f"= {self.total_chunk_reads} chunk read(s)\n"
            f"  Grid: {self.target_shape[1]}x{self.target_shape[0]} px | "
            f"Source reads: {self.total_source_reads}"
        )

    def summary(self) -> str:
        n_x, n_y = self._n_tiles
        counts = Counter(c.n_source_reads for c in self.chunk_reads)
        total = len(self.chunk_reads) or 1
        zero = counts.get(0, 0)
        one = counts.get(1, 0)
        two_plus = sum(v for k, v in counts.items() if k >= _MULTI_THRESHOLD)

        def pct(n: int) -> str:
            return f"({100 * n / total:.1f}%)"

        # Overview level distribution
        ovr_counts: Counter[str] = Counter()
        all_reads = [r for c in self.chunk_reads for r in c.source_reads]
        for r in all_reads:
            ovr_counts[r.overview_path] += 1

        # Read window stats
        widths = [r.region_width for r in all_reads]
        heights = [r.region_height for r in all_reads]
        avg_w = sum(widths) / len(widths) if widths else 0
        avg_h = sum(heights) / len(heights) if heights else 0

        lines = [
            "=== ExplainPlan ===",
            f"CRS:        {self.target_crs}  |  Resolution: {self.target_resolution} units/px"
            f"  |  Grid: {self.target_shape[1]} x {self.target_shape[0]} px",
            f"Band:       {self.band or '(none)'}",
            f"Resampling: {self.resampling}  |  Nodata: {self.nodata}",
            f"Chunks:     {self.chunk_size[1]} x {self.chunk_size[0]} px"
            f" -> {n_x}x{n_y} spatial tiles",
            "",
            f"Total chunk reads:      {self.total_chunk_reads}"
            f" ({n_x}x{n_y} spatial tiles)",
            f"Total source reads:     {self.total_source_reads}",
            f"Chunks with 0 sources:  {zero:>4} {pct(zero)}",
            f"Chunks with 1 source:   {one:>4} {pct(one)}",
            f"Chunks with 2+ sources: {two_plus:>4} {pct(two_plus)}",
            f"Max sources per chunk:  "
            f"{max((c.n_source_reads for c in self.chunk_reads), default=0)}",
        ]

        if all_reads:
            lines.append(
                f"Overview levels:        "
                f"{'  '.join(f'{k}: {v}' for k, v in sorted(ovr_counts.items()))}"
            )
            lines.append(f"Avg read window:        {avg_w:.0f} x {avg_h:.0f} px")

        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        from pandas import DataFrame

        rows = []
        for chunk in self.chunk_reads:
            base = {
                "chunk_row": chunk.chunk_row,
                "chunk_col": chunk.chunk_col,
                "chunk_height": chunk.chunk_height,
                "chunk_width": chunk.chunk_width,
                "n_source_reads": chunk.n_source_reads,
            }
            if chunk.source_reads:
                rows.extend(
                    {
                        **base,
                        "source_path": sr.source_path,
                        "source_crs": sr.source_crs,
                        "overview_path": sr.overview_path,
                        "native_resolution": sr.native_resolution,
                        "overview_resolution": sr.overview_resolution,
                        "region_row_start": sr.region_row_start,
                        "region_row_end": sr.region_row_end,
                        "region_col_start": sr.region_col_start,
                        "region_col_end": sr.region_col_end,
                        "region_height": sr.region_height,
                        "region_width": sr.region_width,
                    }
                    for sr in chunk.source_reads
                )
            else:
                rows.append(
                    {
                        **base,
                        "source_path": None,
                        "source_crs": None,
                        "overview_path": None,
                        "native_resolution": None,
                        "overview_resolution": None,
                        "region_row_start": None,
                        "region_row_end": None,
                        "region_col_start": None,
                        "region_col_end": None,
                        "region_height": None,
                        "region_width": None,
                    }
                )
        return DataFrame(rows)


def explain(
    store: Any,
    crs: str,
    bbox: tuple[float, float, float, float],
    resolution: float,
    chunk_size: tuple[int, int] = (512, 512),
    source_index: ScanIndex | None = None,
    resampling: str = "nearest",
    band: str | None = None,
    datafusion: bool = False,
    sortby: str | None = None,
    nodata: float | int | None = None,
) -> ExplainPlan:
    """Dry-run a merge: report which source regions would be read per chunk.

    Runs the same source-finding, overview-selection, and coordinate-mapping
    logic as ``merge``, but never reads pixel data.
    """
    import math

    target_crs = crs
    xmin, ymin, xmax, ymax = bbox
    target_w = math.ceil((xmax - xmin) / resolution)
    target_h = math.ceil((ymax - ymin) / resolution)
    target_spatial = SpatialAttrs(
        dimensions=["y", "x"],
        transform=(resolution, 0.0, xmin, 0.0, -resolution, ymax),
        bbox=bbox,
        shape=(target_h, target_w),
    )
    target_proj = ProjAttrs(code=crs)
    n_row_chunks = max(1, -(-target_h // chunk_size[0]))
    n_col_chunks = max(1, -(-target_w // chunk_size[1]))

    # Resolve zarr store once
    zarr_store: Any = store
    if hasattr(store, "prefix") and not isinstance(store, zarr.abc.store.Store):
        zarr_store = store.prefix

    chunk_reads: list[ChunkRead] = []

    for row_idx in range(n_row_chunks):
        for col_idx in range(n_col_chunks):
            cb = chunk_bbox(target_spatial, (row_idx, col_idx), chunk_size)

            # Actual chunk shape (may be smaller at edges)
            actual_h = min(chunk_size[0], target_h - row_idx * chunk_size[0])
            actual_w = min(chunk_size[1], target_w - col_idx * chunk_size[1])
            actual_shape = (actual_h, actual_w)

            # Chunk-local transform
            a, b, c, d, e, f = target_spatial.transform
            row_offset = row_idx * chunk_size[0]
            col_offset = col_idx * chunk_size[1]
            chunk_c = c + a * col_offset + b * row_offset
            chunk_f = f + d * col_offset + e * row_offset
            chunk_transform = (a, b, chunk_c, d, e, chunk_f)

            # Find intersecting sources.
            # query_datafusion_sources uses asyncio.run() internally, which
            # fails inside a running event loop (e.g. Jupyter).  Run it in a
            # thread to sidestep the nested-loop restriction.
            if datafusion:
                bbox_4326 = _reproject_bbox_to_4326(cb, target_crs)
                try:
                    asyncio.get_running_loop()
                    with concurrent.futures.ThreadPoolExecutor(1) as pool:
                        sources = pool.submit(
                            query_datafusion_sources, store, bbox_4326, sortby=sortby
                        ).result()
                except RuntimeError:
                    sources = query_datafusion_sources(store, bbox_4326, sortby=sortby)
            elif source_index is not None:
                sources = source_index.find_intersecting_sources(cb, target_crs)
            else:
                sources = []

            source_reads: list[SourceRead] = []

            for source_entry in sources:
                src_crs = source_entry.proj_attrs.code or "EPSG:4326"

                selected_path = "0"
                native_res: float | None = None
                overview_res: float | None = None

                if band is not None:
                    root = zarr.open_group(zarr_store, mode="r")
                    try:
                        source_group = root[source_entry.path]
                    except KeyError:
                        continue
                    if not isinstance(source_group, zarr.Group):
                        continue
                    try:
                        band_group = source_group[band]
                    except KeyError:
                        continue
                    if not isinstance(band_group, zarr.Group):
                        continue

                    band_attrs = dict(band_group.attrs)
                    source_attrs = dict(source_group.attrs)
                    if "multiscales" in band_attrs:
                        array_root = band_group
                    elif "multiscales" in source_attrs:
                        array_root = source_group
                    else:
                        array_root = band_group

                    base_array = _resolve_array(array_root, "0")
                    if base_array is not None:
                        base_spatial = _read_spatial_or_derive(
                            base_array, source_entry.spatial_attrs, scale_factor=1.0
                        )
                        native_res = abs(base_spatial.transform[0])

                    overviews = read_multiscales(array_root, native_res=native_res)

                    if overviews is not None and native_res is not None:
                        target_res = abs(target_spatial.transform[0])
                        if target_crs != src_crs:
                            transformer = Transformer.from_crs(
                                target_crs, src_crs, always_xy=True
                            )
                            cx = (cb[0] + cb[2]) / 2
                            cy = (cb[1] + cb[3]) / 2
                            x0, y0 = transformer.transform(cx, cy)
                            x1, y1 = transformer.transform(cx + target_res, cy)
                            src_target_res = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                        else:
                            src_target_res = target_res

                        overview = select_overview(overviews, src_target_res, native_res)
                        if overview is not None:
                            selected_path = overview.path
                            overview_res = overview.resolution

                    src_array = _resolve_array(array_root, selected_path)
                    if src_array is None or not isinstance(src_array, zarr.Array):
                        continue

                    ovr_scale = 1.0
                    if overviews is not None and selected_path != "0":
                        for ovr in overviews:
                            if ovr.path == selected_path:
                                ovr_scale = ovr.scale[1]
                                break
                    resolved_spatial = _read_spatial_or_derive(
                        src_array, source_entry.spatial_attrs, scale_factor=ovr_scale
                    )
                    resolved_proj = _read_proj_or_derive(
                        src_array, source_entry.proj_attrs
                    )
                    resolved_path = f"{source_entry.path}/{band}/{selected_path}"
                else:
                    root = zarr.open_group(zarr_store, mode="r")
                    try:
                        src_array = root[source_entry.path]
                    except KeyError:
                        continue
                    if not isinstance(src_array, zarr.Array):
                        continue
                    resolved_spatial = source_entry.spatial_attrs
                    resolved_proj = source_entry.proj_attrs
                    resolved_path = source_entry.path

                # Compute target->source pixel mapping to find the read region
                src_row_f, src_col_f = _target_to_source_pixels(
                    chunk_transform, target_crs, actual_shape,
                    resolved_spatial.transform, src_crs,
                )

                src_h = resolved_spatial.shape[0]
                src_w = resolved_spatial.shape[1]
                r_min = max(int(np.floor(np.nanmin(src_row_f))), 0)
                r_max = min(int(np.ceil(np.nanmax(src_row_f))) + 1, src_h)
                c_min = max(int(np.floor(np.nanmin(src_col_f))), 0)
                c_max = min(int(np.ceil(np.nanmax(src_col_f))) + 1, src_w)

                if r_min >= r_max or c_min >= c_max:
                    continue

                if overview_res is None and native_res is not None:
                    overview_res = native_res

                source_reads.append(
                    SourceRead(
                        source_path=resolved_path,
                        source_crs=src_crs,
                        overview_path=selected_path,
                        native_resolution=native_res,
                        overview_resolution=overview_res,
                        region_row_start=r_min,
                        region_row_end=r_max,
                        region_col_start=c_min,
                        region_col_end=c_max,
                    )
                )

            chunk_reads.append(
                ChunkRead(
                    chunk_row=row_idx,
                    chunk_col=col_idx,
                    chunk_height=actual_h,
                    chunk_width=actual_w,
                    chunk_bbox=cb,
                    source_reads=source_reads,
                )
            )

    return ExplainPlan(
        target_crs=target_crs,
        target_resolution=abs(target_spatial.transform[0]),
        target_shape=(target_h, target_w),
        chunk_size=chunk_size,
        band=band,
        resampling=resampling,
        nodata=nodata,
        chunk_reads=chunk_reads,
    )
