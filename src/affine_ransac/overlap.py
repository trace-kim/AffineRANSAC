"""Match the contacts that two neighbouring SEM tiles both see in their overlap strip.

Points are in mask coordinates using the NOMINAL tile placement (tile-local + tile centre).
The same physical contact seen by both tiles should land at almost the same position; the
small difference between the two measurements is what stitching corrects later.
"""

import numpy as np

from affine_ransac.matching import match_points


def overlap_box(center_a, fov_a, center_b, fov_b):
    """Overlap rectangle of two tiles, as (x_min, x_max, y_min, y_max) in nm, or None.

    center_*: (x, y) tile centre in nm; fov_*: (width, height) of the tile in nm.
    """
    x_min = max(center_a[0] - fov_a[0] / 2, center_b[0] - fov_b[0] / 2)
    x_max = min(center_a[0] + fov_a[0] / 2, center_b[0] + fov_b[0] / 2)
    y_min = max(center_a[1] - fov_a[1] / 2, center_b[1] - fov_b[1] / 2)
    y_max = min(center_a[1] + fov_a[1] / 2, center_b[1] + fov_b[1] / 2)
    if x_min >= x_max or y_min >= y_max:
        return None
    return x_min, x_max, y_min, y_max


def points_in_box(points: np.ndarray, box, margin: float = 0.0) -> np.ndarray:
    """Indices of the points inside box, grown by margin on every side."""
    x_min, x_max, y_min, y_max = box
    x, y = points[:, 0], points[:, 1]
    inside = (x >= x_min - margin) & (x <= x_max + margin) & (y >= y_min - margin) & (y <= y_max + margin)
    return np.flatnonzero(inside)


def match_overlap(points_a: np.ndarray, points_b: np.ndarray, box, max_distance: float):
    """Pair the contacts of tile A and tile B that lie in their overlap box.

    The box is grown by max_distance, so a contact right at the box edge is still
    found even if one tile's nominal placement is slightly off.

    Returns:
        (ia, ib): indices into points_a and points_b; points_a[ia[k]] and points_b[ib[k]]
        are the same physical contact.
    """
    in_a = points_in_box(points_a, box, margin=max_distance)
    in_b = points_in_box(points_b, box, margin=max_distance)
    ja, jb = match_points(points_a[in_a], points_b[in_b], max_distance)
    return in_a[ja], in_b[jb]
