"""Basic polygon measures (shoelace formula). Vertices: (N, 2) array, not repeated at the end."""

import numpy as np


def polygon_area(vertices: np.ndarray) -> float:
    """Area of a simple polygon (positive for either vertex order)."""
    x, y = vertices[:, 0], vertices[:, 1]
    return abs((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()) / 2.0


def polygon_centroid(vertices: np.ndarray) -> np.ndarray:
    """Area centroid (x, y) of a simple polygon."""
    x, y = vertices[:, 0], vertices[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0
    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return np.array([cx, cy])
