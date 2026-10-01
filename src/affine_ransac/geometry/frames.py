"""Coordinate frame conversions (docs/SPEC.md §3). All conversions live here."""

import numpy as np


def pixel_to_tile_nm(uv: np.ndarray, image_shape: tuple[int, int], pixel_size_nm: float) -> np.ndarray:
    """Pixel (u, v) -> tile-local (x, y) in nm.

    Tile-local frame: origin at the image centre, x to the right, y UP.
    The image is v-down, so y flips sign. Pixel centres are at integer (u, v),
    so the image centre is at ((cols - 1) / 2, (rows - 1) / 2).
    """
    rows, cols = image_shape
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    x = (uv[:, 0] - (cols - 1) / 2) * pixel_size_nm
    y = -(uv[:, 1] - (rows - 1) / 2) * pixel_size_nm
    return np.column_stack([x, y])
