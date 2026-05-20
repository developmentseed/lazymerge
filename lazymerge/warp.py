from __future__ import annotations

import numpy as np
from pyproj import Transformer


def _target_to_source_pixels(
    target_transform: tuple[float, ...],
    target_crs: str,
    target_shape: tuple[int, int],
    source_transform: tuple[float, ...],
    source_crs: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Map every target pixel to fractional source pixel coordinates.

    Returns (src_row, src_col) arrays with shape *target_shape*.
    """
    t_a, t_b, t_c, t_d, t_e, t_f = target_transform
    s_a, s_b, s_c, s_d, s_e, s_f = source_transform

    rows, cols = target_shape
    row_idx, col_idx = np.mgrid[0:rows, 0:cols]

    # Target pixel centers (offset by 0.5 for pixel-center convention)
    target_x = t_a * (col_idx + 0.5) + t_b * (row_idx + 0.5) + t_c
    target_y = t_d * (col_idx + 0.5) + t_e * (row_idx + 0.5) + t_f

    # Reproject target coordinates to source CRS
    if target_crs != source_crs:
        transformer = Transformer.from_crs(target_crs, source_crs, always_xy=True)
        source_x, source_y = transformer.transform(target_x, target_y)
    else:
        source_x, source_y = target_x, target_y

    # Inverse of source affine to get pixel coords from spatial coords
    det = s_a * s_e - s_b * s_d
    src_col = (s_e * (source_x - s_c) - s_b * (source_y - s_f)) / det
    src_row = (-s_d * (source_x - s_c) + s_a * (source_y - s_f)) / det

    # Subtract 0.5 to go from pixel-center coords to array indices
    return src_row - 0.5, src_col - 0.5


def warp_source_region(
    src_array: np.ndarray,
    src_row: np.ndarray,
    src_col: np.ndarray,
    src_shape: tuple[int, int],
    target_shape: tuple[int, int],
    resampling: str = "nearest",
    nodata: float | int | None = None,
) -> np.ndarray:
    """Sample *src_array* at pre-computed source pixel coordinates.

    *src_row* / *src_col* are in full-source pixel space.  *src_array*
    may be a sub-region; *row_offset* / *col_offset* (derived from
    *src_shape* vs actual array shape) are handled by the caller who
    slices the coordinate arrays.

    Parameters
    ----------
    src_array : 2-D array already read from the source.
    src_row, src_col : fractional pixel coordinates (full-source space),
        already clipped to the region that was read.
    src_shape : (height, width) of the *read* region (== src_array.shape).
    target_shape : output shape.
    resampling : resampling method (only "nearest" supported).
    nodata : source fill value to treat as NaN.
    """
    if resampling != "nearest":
        raise ValueError(f"Unsupported resampling method: {resampling}")

    src_h, src_w = src_shape
    src_col_i = np.round(src_col).astype(int)
    src_row_i = np.round(src_row).astype(int)

    valid = (
        (src_col_i >= 0)
        & (src_col_i < src_w)
        & (src_row_i >= 0)
        & (src_row_i < src_h)
    )

    out_dtype = src_array.dtype if np.issubdtype(src_array.dtype, np.floating) else np.float32
    output = np.full(target_shape, np.nan, dtype=out_dtype)
    output[valid] = src_array[src_row_i[valid], src_col_i[valid]]

    if nodata is not None:
        output[output == nodata] = np.nan

    return output
