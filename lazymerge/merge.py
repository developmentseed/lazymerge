from __future__ import annotations

from typing import Any

import numpy as np
import zarr
import cubed

from pyproj import Transformer

from lazymerge.conventions import SpatialAttrs, ProjAttrs, chunk_bbox, read_multiscales, read_spatial, read_proj
from lazymerge.sources import ScanIndex, SourceEntry, select_overview, query_datafusion_sources
from lazymerge.warp import _target_to_source_pixels, warp_source_region


def _read_spatial_or_derive(
    node: zarr.Array,
    source_spatial: SpatialAttrs,
    scale_factor: float = 1.0,
) -> SpatialAttrs:
    """Read spatial attrs from node, falling back to source_spatial if missing.

    When VirtualiZarr arrays lack convention attrs, derive them from the
    source entry's spatial attrs, scaling the transform for the overview
    level and using the array's actual shape.
    """
    try:
        return read_spatial(node)
    except KeyError:
        pass
    a, b, c, d, e, f = source_spatial.transform
    return SpatialAttrs(
        dimensions=source_spatial.dimensions,
        transform=(a * scale_factor, b, c, d, e * scale_factor, f),
        bbox=source_spatial.bbox,
        shape=tuple(node.shape),
        registration=source_spatial.registration,
    )


def _read_proj_or_derive(
    node: zarr.Array,
    source_proj: ProjAttrs,
) -> ProjAttrs:
    """Read proj attrs from node, falling back to source_proj if missing."""
    try:
        return read_proj(node)
    except KeyError:
        return source_proj


def _resolve_array(group: zarr.Group, path: str) -> zarr.Array | None:
    """Navigate to an array, handling VirtualiZarr's group-wrapping pattern.

    VirtualiZarr wraps arrays in a group of the same name, so level "0"
    may be a group containing array "0" rather than a direct array.
    """
    try:
        node = group[path]
    except KeyError:
        return None
    if isinstance(node, zarr.Array):
        return node
    if isinstance(node, zarr.Group):
        # VirtualiZarr pattern: group "0" contains array "0"
        try:
            child = node[path]
            if isinstance(child, zarr.Array):
                return child
        except KeyError:
            pass
        # Fall back to first array child
        for _, member in node.members():
            if isinstance(member, zarr.Array):
                return member
    return None


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
    sortby: str | None = None,
    nodata: float | int | None = None,
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
        sources = query_datafusion_sources(store, bbox_4326, sortby=sortby)
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

            # Find the group with the multiscales convention attribute.
            # Per convention, asset paths are relative to this group.
            band_attrs = dict(band_group.attrs)
            source_attrs = dict(source_group.attrs)
            if "multiscales" in band_attrs:
                array_root = band_group
            elif "multiscales" in source_attrs:
                array_root = source_group
            else:
                array_root = band_group

            # Resolve base array and compute native resolution
            base_array = _resolve_array(array_root, "0")
            native_res: float | None = None
            if base_array is not None:
                base_spatial = _read_spatial_or_derive(
                    base_array, source_entry.spatial_attrs, scale_factor=1.0
                )
                native_res = abs(base_spatial.transform[0])

            # Parse multiscales layout with known native_res
            overviews = read_multiscales(array_root, native_res=native_res)
            selected_path: str | None = None

            if overviews is not None and native_res is not None:
                # Estimate target resolution in source CRS
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

            if selected_path is None:
                selected_path = "0"

            src_array = _resolve_array(array_root, selected_path)
            if src_array is None or not isinstance(src_array, zarr.Array):
                continue

            # Build resolved SourceEntry with the selected array's attrs
            # Compute scale factor for the selected overview level
            ovr_scale = 1.0
            if overviews is not None and selected_path != "0":
                for ovr in overviews:
                    if ovr.path == selected_path:
                        ovr_scale = ovr.scale[1]
                        break
            resolved_spatial = _read_spatial_or_derive(
                src_array, source_entry.spatial_attrs, scale_factor=ovr_scale
            )
            resolved_proj = _read_proj_or_derive(src_array, source_entry.proj_attrs)
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

        # Compute target→source pixel mapping once for this source
        src_row_f, src_col_f = _target_to_source_pixels(
            chunk_transform, target_crs, actual_shape,
            resolved_entry.spatial_attrs.transform, src_crs,
        )

        # Determine the bounding pixel range needed from the source
        src_h, src_w = resolved_entry.spatial_attrs.shape[0], resolved_entry.spatial_attrs.shape[1]
        r_min = max(int(np.floor(np.nanmin(src_row_f))), 0)
        r_max = min(int(np.ceil(np.nanmax(src_row_f))) + 1, src_h)
        c_min = max(int(np.floor(np.nanmin(src_col_f))), 0)
        c_max = min(int(np.ceil(np.nanmax(src_col_f))) + 1, src_w)

        if r_min >= r_max or c_min >= c_max:
            continue

        # Read one contiguous region from the source array
        src_data = np.asarray(src_array[r_min:r_max, c_min:c_max])

        # Shift coordinates to be relative to the read region
        warped = warp_source_region(
            src_data,
            src_row_f - r_min,
            src_col_f - c_min,
            src_data.shape,
            actual_shape,
            resampling=resampling,
            nodata=nodata,
        )

        mask = np.isnan(output) & ~np.isnan(warped)
        output[mask] = warped[mask]
        unfilled -= int(np.count_nonzero(mask))

    return output


def merge(
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
    dtype: str = "float32",
) -> tuple[cubed.Array, SpatialAttrs, ProjAttrs]:
    from lazymerge.target import create_target

    target, target_spatial, target_proj = create_target(
        crs=crs, bbox=bbox, resolution=resolution,
        chunk_size=chunk_size, dtype=dtype,
    )

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
        sortby=sortby,
        nodata=nodata,
    )

    return result, target_spatial, target_proj
