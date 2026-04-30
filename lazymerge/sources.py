from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import zarr
from pyproj import Transformer

from lazymerge.conventions import (
    ProjAttrs,
    SpatialAttrs,
    chunk_bbox,
    read_proj,
    read_spatial,
)

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


class ScanIndex:
    def __init__(self, entries: list[SourceEntry]) -> None:
        self.entries = entries
        self._reprojected_bboxes: dict[tuple[str, str], Bbox] = {}

    def _get_bbox_in_crs(self, entry: SourceEntry, target_crs: str) -> Bbox:
        cache_key = (entry.path, target_crs)
        if cache_key not in self._reprojected_bboxes:
            src_crs = entry.proj_attrs.code or "EPSG:4326"
            self._reprojected_bboxes[cache_key] = _reproject_bbox(
                cast(Bbox, entry.spatial_attrs.bbox), src_crs, target_crs
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

    def find_intersecting_chunks(
        self,
        source: SourceEntry,
        target_chunk_bbox: Bbox,
        target_crs: str,
    ) -> list[tuple[str, tuple[int, int]]]:
        src_crs = source.proj_attrs.code or "EPSG:4326"
        src_bbox = _reproject_bbox(target_chunk_bbox, target_crs, src_crs)

        sa = source.spatial_attrs
        n_row_chunks = -(-sa.shape[0] // source.chunk_shape[0])  # ceiling division
        n_col_chunks = -(-sa.shape[1] // source.chunk_shape[1])

        results: list[tuple[str, tuple[int, int]]] = []
        for ri in range(n_row_chunks):
            for ci in range(n_col_chunks):
                cb = chunk_bbox(sa, (ri, ci), source.chunk_shape)
                if _bboxes_intersect(cb, src_bbox):
                    results.append((source.path, (ri, ci)))
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
            entries.append(SourceEntry(
                path=name,
                spatial_attrs=sa,
                proj_attrs=pa,
                chunk_shape=tuple(item.chunks),
            ))
    return ScanIndex(entries)
