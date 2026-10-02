"""2-D affine transforms (docs/SPEC.md §8).

An affine is a 3×3 matrix [[a, b, tx], [c, d, ty], [0, 0, 1]] applied to column vectors,
p' = A @ [x, y, 1]. Fits are done about a reference point (e.g. the design centroid): points are
given relative to it, so the translation is the shift AT the reference point and is not coupled to
rotation or magnification about a distant origin.
"""

import numpy as np


def fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares affine mapping src (N, 2) onto dst (N, 2); exact for N = 3 (non-collinear)."""
    design = np.column_stack([src, np.ones(len(src))])     # (N, 3): x, y, 1
    params, *_ = np.linalg.lstsq(design, dst, rcond=None)  # (3, 2): columns for x', y'
    return np.vstack([params.T, [0.0, 0.0, 1.0]])


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """matrix applied to points (N, 2)."""
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def triangle_area(p0, p1, p2) -> float:
    """Area of the triangle p0, p1, p2 (0 for collinear points)."""
    (x1, y1), (x2, y2) = np.asarray(p1) - p0, np.asarray(p2) - p0
    return abs(x1 * y2 - x2 * y1) / 2


def decompose(matrix: np.ndarray) -> dict:
    """Small-angle mask-metrology terms (SPEC §8.2): translation (nm), magnification (ppm),
    rotation and orthogonality (µrad).

    Mx = a − 1, My = d − 1, Rx = c, Ry = −b, rotation θ = (Rx + Ry) / 2, orthogonality ω = Rx − Ry.
    A positive θ turns counter-clockwise (x toward y), in the y-up mask frame.
    """
    (a, b, tx), (c, d, ty) = matrix[0], matrix[1]
    rx, ry = c, -b
    return {
        "Tx_nm": tx, "Ty_nm": ty,
        "Mx_ppm": (a - 1) * 1e6, "My_ppm": (d - 1) * 1e6,
        "rotation_urad": (rx + ry) / 2 * 1e6, "orthogonality_urad": (rx - ry) * 1e6,
    }
