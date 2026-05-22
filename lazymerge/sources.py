from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import nest_asyncio
import zarr
from datafusion import SessionContext
from geodatafusion import register_all
from pyproj import Transformer
from zarr_datafusion_search import ZarrTable

from lazymerge.conventions import (
    OverviewLevel,
    ProjAttrs,
    SpatialAttrs,
    read_proj,
    read_spatial,
)

T = TypeVar("T")


def _run_async(coro: Any) -> Any:
    """Run an async coroutine, handling both fresh and running event loops (e.g. Jupyter)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    nest_asyncio.apply(loop)
    return loop.run_until_complete(coro)


Bbox = tuple[float, float, float, float]


@dataclass(frozen=True)
class SourceEntry:
    path: str
    spatial_attrs: SpatialAttrs
    proj_attrs: ProjAttrs
    chunk_shape: tuple[int, ...]


def _bboxes_intersect(a: Bbox, b: Bbox) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _reproject_bbox(bbox: Bbox, src_crs: str, dst_crs: str) -> Bbox:
    if src_crs == dst_crs:
        return bbox
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xs = [bbox[0], bbox[0], bbox[2], bbox[2]]
    ys = [bbox[1], bbox[3], bbox[1], bbox[3]]
    tx, ty = transformer.transform(xs, ys)
    return (min(tx), min(ty), max(tx), max(ty))


def select_overview(
    overviews: list[OverviewLevel],
    target_res: float,
    native_res: float,
) -> OverviewLevel | None:
    """Choose the coarsest overview whose resolution is <= target_res.

    Picks the finest source data that avoids upsampling: the selected
    overview's pixel size is no larger than the output pixel size, so each
    output pixel samples at least as much original detail as it represents.

    Args:
        overviews: Overview levels ordered finest to coarsest.
        target_res: Target pixel size in the source's native CRS units.
        native_res: Full-resolution pixel size.

    Returns:
        An OverviewLevel, or None to use full resolution.

    """
    if not overviews:
        return None
    if target_res <= native_res:
        return None

    selected: OverviewLevel | None = None
    for overview in overviews:
        if overview.resolution <= target_res:
            selected = overview
        else:
            break
    return selected


class ScanIndex:
    def __init__(self, entries: list[SourceEntry]) -> None:
        self.entries = entries
        self._reprojected_bboxes: dict[tuple[str, str], Bbox] = {}

    def _get_bbox_in_crs(self, entry: SourceEntry, target_crs: str) -> Bbox:
        cache_key = (entry.path, target_crs)
        if cache_key not in self._reprojected_bboxes:
            src_crs = entry.proj_attrs.code or "EPSG:4326"
            self._reprojected_bboxes[cache_key] = _reproject_bbox(
                cast("Bbox", entry.spatial_attrs.bbox),
                src_crs,
                target_crs,
            )
        return self._reprojected_bboxes[cache_key]

    def find_intersecting_sources(
        self,
        target_bbox: Bbox,
        target_crs: str,
    ) -> list[SourceEntry]:
        results = []
        for entry in self.entries:
            src_bbox = self._get_bbox_in_crs(entry, target_crs)
            if _bboxes_intersect(src_bbox, target_bbox):
                results.append(entry)
        return results


def scan_store(root: zarr.Group) -> ScanIndex:
    entries: list[SourceEntry] = []
    for name, item in root.members():
        if isinstance(item, zarr.Array):
            try:
                sa = read_spatial(item)
                pa = read_proj(item)
            except KeyError:
                continue
            entries.append(
                SourceEntry(
                    path=name,
                    spatial_attrs=sa,
                    proj_attrs=pa,
                    chunk_shape=tuple(item.chunks),
                )
            )
    return ScanIndex(entries)


def query_datafusion_sources(
    store: Any,
    bbox_4326: tuple[float, float, float, float],
    sortby: str | None = None,
    sql_filter: str | None = None,
) -> list[SourceEntry]:
    r"""Query the /meta group via DataFusion for sources intersecting bbox_4326.

    Issues a spatial SQL query against the /meta columnar arrays. Only sources
    whose bbox (stored in EPSG:4326) intersects the query bbox are returned.

    Args:
        store: An obstore-compatible object store (e.g. LocalStore, S3Store),
            an Icechunk Session, or a zarr Store backed by Icechunk.
        bbox_4326: Query bounding box in EPSG:4326 (xmin, ymin, xmax, ymax).
        sortby: Optional column name to sort results by (e.g. a datetime field).
            Controls the order in which sources are composited — earlier entries
            take priority for filling NaN pixels.
        sql_filter: Optional SQL expression appended as an AND clause to the
            spatial intersection query.  For example:
            ``'"eo:cloud_cover" < 20'`` or ``'"datetime" > \\'2024-01-01\\''``.

    Returns:
        List of SourceEntry objects for matching sources. chunk_shape is set
        to a placeholder (0, 0) since it is resolved from the actual array.

    """

    async def _query() -> list[SourceEntry]:
        # Detect Icechunk stores and use the appropriate constructor
        try:
            from icechunk import IcechunkStore  # noqa: PLC0415

            if isinstance(store, IcechunkStore):
                zarr_table = await ZarrTable.from_icechunk(
                    session=store.session,
                    group_path="/meta",
                )
            else:
                zarr_table = await ZarrTable.from_obstore(store, "/meta")
        except ImportError:
            zarr_table = await ZarrTable.from_obstore(store, "/meta")

        ctx = SessionContext()
        register_all(ctx)
        ctx.register_table("meta", zarr_table)

        xmin, ymin, xmax, ymax = bbox_4326
        query = (
            'SELECT id, "proj:epsg", '
            "transform_0, transform_1, transform_2, "
            "transform_3, transform_4, transform_5, "
            "shape_x, shape_y "
            "FROM meta "
            "WHERE ST_Intersects(bbox, ST_GeomFromText("
            f"'POLYGON(({xmin} {ymin}, {xmax} {ymin}, "
            f"{xmax} {ymax}, {xmin} {ymax}, {xmin} {ymin}))'"
            "))"
        )
        if sql_filter is not None:
            query += f" AND ({sql_filter})"
        if sortby is not None:
            query += f' ORDER BY "{sortby}"'

        df = ctx.sql(query)
        batches = df.collect()

        entries: list[SourceEntry] = []
        for batch in batches:
            for i in range(batch.num_rows):
                transform = (
                    float(batch.column("transform_0")[i].as_py()),
                    float(batch.column("transform_1")[i].as_py()),
                    float(batch.column("transform_2")[i].as_py()),
                    float(batch.column("transform_3")[i].as_py()),
                    float(batch.column("transform_4")[i].as_py()),
                    float(batch.column("transform_5")[i].as_py()),
                )
                sx = int(batch.column("shape_x")[i].as_py())
                sy = int(batch.column("shape_y")[i].as_py())
                epsg = int(batch.column("proj:epsg")[i].as_py())
                sa = SpatialAttrs(
                    dimensions=["y", "x"],
                    transform=transform,
                    bbox=(
                        float(transform[2]),
                        float(transform[5] + transform[4] * sy),
                        float(transform[2] + transform[0] * sx),
                        float(transform[5]),
                    ),
                    shape=(sy, sx),
                )
                pa = ProjAttrs(code=f"EPSG:{epsg}")
                entries.append(
                    SourceEntry(
                        path=str(batch.column("id")[i].as_py()),
                        spatial_attrs=sa,
                        proj_attrs=pa,
                        chunk_shape=(0, 0),
                    )
                )
        return entries

    result: list[SourceEntry] = _run_async(_query())
    return result


def query_temporal_groups(
    store: Any,
    bbox_4326: tuple[float, float, float, float],
    grouper: Any,
    sql_filter: str | None = None,
) -> list[str]:
    """Query distinct datetime values from /meta and bucket them into temporal groups.

    Args:
        store: An obstore-compatible object store, Icechunk Session, or zarr Store.
        bbox_4326: Query bounding box in EPSG:4326 (xmin, ymin, xmax, ymax).
        grouper: A TemporalGrouper instance used to bucket datetime strings.
        sql_filter: Optional SQL expression appended as an AND clause.

    Returns:
        Sorted list of unique group keys.

    """

    async def _query() -> list[str]:
        try:
            from icechunk import IcechunkStore  # noqa: PLC0415

            if isinstance(store, IcechunkStore):
                zarr_table = await ZarrTable.from_icechunk(
                    session=store.session,
                    group_path="/meta",
                )
            else:
                zarr_table = await ZarrTable.from_obstore(store, "/meta")
        except ImportError:
            zarr_table = await ZarrTable.from_obstore(store, "/meta")

        ctx = SessionContext()
        register_all(ctx)
        ctx.register_table("meta", zarr_table)

        xmin, ymin, xmax, ymax = bbox_4326
        query = (
            "SELECT DISTINCT datetime "
            "FROM meta "
            "WHERE ST_Intersects(bbox, ST_GeomFromText("
            f"'POLYGON(({xmin} {ymin}, {xmax} {ymin}, "
            f"{xmax} {ymax}, {xmin} {ymax}, {xmin} {ymin}))'"
            "))"
        )
        if sql_filter is not None:
            query += f" AND ({sql_filter})"

        df = ctx.sql(query)
        batches = df.collect()

        datetimes: list[str] = [
            str(batch.column("datetime")[i].as_py())
            for batch in batches
            for i in range(batch.num_rows)
        ]
        return datetimes

    datetimes = _run_async(_query())
    return sorted({grouper.group_key(dt) for dt in datetimes})
