from __future__ import annotations

import numpy as np
from pyproj import Transformer


def warp_chunk(
    source_data: np.ndarray,
    source_transform: tuple[float, ...],
    source_crs: str,
    target_transform: tuple[float, ...],
    target_crs: str,
    target_shape: tuple[int, int],
    resampling: str = "nearest",
) -> np.ndarray:
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
    # General inverse for 2x2 matrix [a,b; d,e]:
    det = s_a * s_e - s_b * s_d
    src_col = (s_e * (source_x - s_c) - s_b * (source_y - s_f)) / det
    src_row = (-s_d * (source_x - s_c) + s_a * (source_y - s_f)) / det

    # Subtract 0.5 to go from pixel-center coords to array indices
    src_col = src_col - 0.5
    src_row = src_row - 0.5

    src_h, src_w = source_data.shape

    if resampling == "nearest":
        src_col_i = np.round(src_col).astype(int)
        src_row_i = np.round(src_row).astype(int)

        valid = (
            (src_col_i >= 0)
            & (src_col_i < src_w)
            & (src_row_i >= 0)
            & (src_row_i < src_h)
        )

        output = np.full(target_shape, np.nan, dtype=source_data.dtype)
        output[valid] = source_data[src_row_i[valid], src_col_i[valid]]
        return output
    else:
        raise ValueError(f"Unsupported resampling method: {resampling}")
