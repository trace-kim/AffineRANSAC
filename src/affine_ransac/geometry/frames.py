"""Coordinate frame conversions (docs/SPEC.md §3). All conversions live here."""

import numpy as np


def pixel_to_tile_nm(xy_px: np.ndarray, image_shape: tuple[int, int], pixel_size_nm: float | np.ndarray) -> np.ndarray:
    """Pixel (x, y) -> tile-local (x, y) in nm.

    Tile-local frame: origin at the image centre, x to the right, y UP.
    The pixel frame is y-down, so y flips sign. Pixel centres are at integer (x, y),
    so the image centre is at ((cols - 1) / 2, (rows - 1) / 2).
    """
    rows, cols = image_shape
    xy_px = np.asarray(xy_px, dtype=np.float64).reshape(-1, 2)
    scale = np.array([1.0, -1.0]) * pixel_size_nm  # square pixels or separate x and y sizes
    return (xy_px - [(cols - 1) / 2, (rows - 1) / 2]) * scale


def tile_nm_to_pixel(xy_nm: np.ndarray, image_shape: tuple[int, int], pixel_size_nm: float | np.ndarray) -> np.ndarray:
    """Tile-local (x, y) in nm -> pixel (x, y). The exact inverse of pixel_to_tile_nm."""
    rows, cols = image_shape
    xy_nm = np.asarray(xy_nm, dtype=np.float64).reshape(-1, 2)
    scale = np.array([1.0, -1.0]) * pixel_size_nm
    return xy_nm / scale + [(cols - 1) / 2, (rows - 1) / 2]
