from __future__ import annotations

import numpy as np
import zarr
import cubed
from zarr.abc.store import Store

from lazymerge.conventions import SpatialAttrs, ProjAttrs, chunk_bbox
from lazymerge.sources import ScanIndex
from lazymerge.warp import warp_chunk


def _merge_block(
    block: np.ndarray,
    block_id: tuple[int, ...],
    source_index: ScanIndex,
    target_spatial: SpatialAttrs,
    target_proj: ProjAttrs,
    store: Store,
    chunk_size: tuple[int, int],
    resampling: str,
    aggregation: str,
) -> np.ndarray:
    target_crs = target_proj.code
    if target_crs is None:
        raise ValueError("target_proj.code must not be None")

    # Compute this chunk's spatial bbox
    row_idx, col_idx = block_id
    cb = chunk_bbox(target_spatial, (row_idx, col_idx), chunk_size)

    # Compute chunk-local transform (shift origin to this chunk's pixel offset)
    a, b, c, d, e, f = target_spatial.transform
    row_offset = row_idx * chunk_size[0]
    col_offset = col_idx * chunk_size[1]
    chunk_c = c + a * col_offset + b * row_offset
    chunk_f = f + d * col_offset + e * row_offset
    chunk_transform = (a, b, chunk_c, d, e, chunk_f)

    # Actual chunk shape (may be smaller at edges)
    actual_shape: tuple[int, int] = (block.shape[0], block.shape[1])

    output = np.full(actual_shape, np.nan, dtype=block.dtype)

    # Pass 1: find intersecting sources
    sources = source_index.find_intersecting_sources(cb, target_crs)
    if not sources:
        return output

    for source_entry in sources:
        # Pass 2: find intersecting source chunks
        src_chunks = source_index.find_intersecting_chunks(source_entry, cb, target_crs)
        if not src_chunks:
            continue

        src_crs = source_entry.proj_attrs.code or "EPSG:4326"
        root = zarr.open_group(store, mode="r")
        src_array = root[source_entry.path]
        if not isinstance(src_array, zarr.Array):
            continue

        for _, (src_row, src_col) in src_chunks:
            # Read source chunk data
            sr_start = src_row * source_entry.chunk_shape[0]
            sr_end = min(sr_start + source_entry.chunk_shape[0], source_entry.spatial_attrs.shape[0])
            sc_start = src_col * source_entry.chunk_shape[1]
            sc_end = min(sc_start + source_entry.chunk_shape[1], source_entry.spatial_attrs.shape[1])
            src_data = src_array[sr_start:sr_end, sc_start:sc_end]

            # Compute source chunk transform (shift origin to source chunk pixel offset)
            sa, sb, sc, sd, se, sf = source_entry.spatial_attrs.transform
            src_chunk_c = sc + sa * sc_start + sb * sr_start
            src_chunk_f = sf + sd * sc_start + se * sr_start
            src_chunk_transform = (sa, sb, src_chunk_c, sd, se, src_chunk_f)

            warped = warp_chunk(
                source_data=np.asarray(src_data),
                source_transform=src_chunk_transform,
                source_crs=src_crs,
                target_transform=chunk_transform,
                target_crs=target_crs,
                target_shape=actual_shape,
                resampling=resampling,
            )

            if aggregation == "first":
                mask = np.isnan(output) & ~np.isnan(warped)
                output[mask] = warped[mask]
            elif aggregation == "last":
                mask = ~np.isnan(warped)
                output[mask] = warped[mask]

    return output


def merge(
    source_index: ScanIndex,
    target: cubed.Array,
    target_spatial: SpatialAttrs,
    target_proj: ProjAttrs,
    store: Store,
    resampling: str = "nearest",
    aggregation: str = "first",
) -> tuple[cubed.Array, SpatialAttrs, ProjAttrs]:
    chunk_size = target.chunksize

    result = cubed.map_blocks(
        _merge_block,
        target,
        dtype=target.dtype,
        chunks=target.chunks,
        source_index=source_index,
        target_spatial=target_spatial,
        target_proj=target_proj,
        store=store,
        chunk_size=chunk_size,
        resampling=resampling,
        aggregation=aggregation,
    )

    return result, target_spatial, target_proj
