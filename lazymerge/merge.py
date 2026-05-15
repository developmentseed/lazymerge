from __future__ import annotations

from typing import Any

import numpy as np
import zarr
import cubed

from pyproj import Transformer

from lazymerge.conventions import SpatialAttrs, ProjAttrs, chunk_bbox, read_multiscales, read_spatial, read_proj
from lazymerge.sources import ScanIndex, SourceEntry, select_overview, query_datafusion_sources, find_intersecting_chunks
from lazymerge.warp import warp_chunk


def _reproject_bbox_to_4326(
    bbox: tuple[float, float, float, float],
    src_crs: str,
) -> tuple[float, float, float, float]:
    """Reproject a bbox to EPSG:4326 for DataFusion queries."""
    if src_crs == "EPSG:4326":
        return bbox
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    xs = [bbox[0], bbox[0], bbox[2], bbox[2]]
    ys = [bbox[1], bbox[3], bbox[1], bbox[3]]
    tx, ty = transformer.transform(xs, ys)
    return (min(tx), min(ty), max(tx), max(ty))


def _merge_block(
    block: np.ndarray,
    block_id: tuple[int, ...],
    source_index: ScanIndex | None,
    target_spatial: SpatialAttrs,
    target_proj: ProjAttrs,
    store: Any,
    chunk_size: tuple[int, int],
    resampling: str,
    band: str | None = None,
    datafusion: bool = False,
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
    if datafusion:
        bbox_4326 = _reproject_bbox_to_4326(cb, target_crs)
        sources = query_datafusion_sources(store, bbox_4326)
    elif source_index is not None:
        sources = source_index.find_intersecting_sources(cb, target_crs)
    else:
        return output

    if not sources:
        return output

    # Resolve a zarr-compatible store for reading arrays.
    # obstore LocalStore isn't zarr-compatible, so use its path instead.
    zarr_store: Any = store
    if hasattr(store, "prefix") and not isinstance(store, zarr.abc.store.Store):
        zarr_store = store.prefix

    unfilled = int(np.count_nonzero(np.isnan(output)))

    for source_entry in sources:
        if unfilled == 0:
            break

        src_crs = source_entry.proj_attrs.code or "EPSG:4326"
        root = zarr.open_group(zarr_store, mode="r")

        if band is not None:
            # Navigate into source_group/band to resolve the array
            source_group = root[source_entry.path]
            if not isinstance(source_group, zarr.Group):
                continue
            band_group = source_group[band]
            if not isinstance(band_group, zarr.Group):
                continue

            overviews = read_multiscales(band_group)
            selected_path: str | None = None

            if overviews is not None:
                # Determine native resolution from base array
                base_spatial = read_spatial(band_group["0"])
                native_res = abs(base_spatial.transform[0])

                # Estimate target resolution in source CRS
                target_res = abs(target_spatial.transform[0])
                if target_crs != src_crs:
                    transformer = Transformer.from_crs(target_crs, src_crs, always_xy=True)
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

            if selected_path is None:
                # Use base array (first child or "0")
                selected_path = "0"

            src_array = band_group[selected_path]
            if not isinstance(src_array, zarr.Array):
                continue

            # Build resolved SourceEntry with the selected array's attrs
            resolved_spatial = read_spatial(src_array)
            resolved_proj = read_proj(src_array)
            resolved_chunk_shape = tuple(src_array.chunks)
            resolved_entry = SourceEntry(
                path=f"{source_entry.path}/{band}/{selected_path}",
                spatial_attrs=resolved_spatial,
                proj_attrs=resolved_proj,
                chunk_shape=resolved_chunk_shape,
            )
        else:
            src_array = root[source_entry.path]
            if not isinstance(src_array, zarr.Array):
                continue
            resolved_entry = source_entry

        # Pass 2: find intersecting source chunks
        src_chunks = find_intersecting_chunks(resolved_entry, cb, target_crs)
        if not src_chunks:
            continue

        for _, (src_row, src_col) in src_chunks:
            if unfilled == 0:
                break

            # Read source chunk data
            sr_start = src_row * resolved_entry.chunk_shape[0]
            sr_end = min(sr_start + resolved_entry.chunk_shape[0], resolved_entry.spatial_attrs.shape[0])
            sc_start = src_col * resolved_entry.chunk_shape[1]
            sc_end = min(sc_start + resolved_entry.chunk_shape[1], resolved_entry.spatial_attrs.shape[1])
            src_data = src_array[sr_start:sr_end, sc_start:sc_end]

            # Compute source chunk transform (shift origin to source chunk pixel offset)
            sa, sb, sc, sd, se, sf = resolved_entry.spatial_attrs.transform
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

            mask = np.isnan(output) & ~np.isnan(warped)
            output[mask] = warped[mask]
            unfilled -= int(np.count_nonzero(mask))

    return output


def merge(
    source_index: ScanIndex | None,
    target: cubed.Array,
    target_spatial: SpatialAttrs,
    target_proj: ProjAttrs,
    store: Any,
    resampling: str = "nearest",
    band: str | None = None,
    datafusion: bool = False,
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
        band=band,
        datafusion=datafusion,
    )

    return result, target_spatial, target_proj
