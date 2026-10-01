"""Coordinate frame conversions (docs/SPEC.md §3). All conversions live here."""

import numpy as np


def pixel_to_tile_nm(xy_px: np.ndarray, image_shape: tuple[int, int], pixel_size_nm: float) -> np.ndarray:
    """Pixel (x, y) -> tile-local (x, y) in nm.

    Tile-local frame: origin at the image centre, x to the right, y UP.
    The pixel frame is y-down, so y flips sign. Pixel centres are at integer (x, y),
    so the image centre is at ((cols - 1) / 2, (rows - 1) / 2).
    """
    rows, cols = image_shape
    xy_px = np.asarray(xy_px, dtype=np.float64).reshape(-1, 2)
    x = (xy_px[:, 0] - (cols - 1) / 2) * pixel_size_nm
    y = -(xy_px[:, 1] - (rows - 1) / 2) * pixel_size_nm
    return np.column_stack([x, y])


def tile_nm_to_pixel(xy_nm: np.ndarray, image_shape: tuple[int, int], pixel_size_nm: float) -> np.ndarray:
    """Tile-local (x, y) in nm -> pixel (x, y). The exact inverse of pixel_to_tile_nm."""
    rows, cols = image_shape
    xy_nm = np.asarray(xy_nm, dtype=np.float64).reshape(-1, 2)
    x_px = xy_nm[:, 0] / pixel_size_nm + (cols - 1) / 2
    y_px = -xy_nm[:, 1] / pixel_size_nm + (rows - 1) / 2
    return np.column_stack([x_px, y_px])
