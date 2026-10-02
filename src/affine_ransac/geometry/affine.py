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


def recentre(matrix: np.ndarray, offset) -> np.ndarray:
    """The same transform written about a reference point moved by offset (2,): the linear part is
    unchanged, the translation becomes the shift at the new point."""
    moved = matrix.copy()
    moved[:2, 2] = matrix[:2, :2] @ offset + matrix[:2, 2] - offset
    return moved


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


def coefficients(matrix: np.ndarray) -> list[tuple[str, float]]:
    """The affine's own coefficients (SPEC §8.1): x' = a·x + b·y + tx, y' = c·x + d·y + ty."""
    (a, b, tx), (c, d, ty) = matrix[0], matrix[1]
    return [("a", a), ("b", b), ("c", c), ("d", d), ("tx (nm)", tx), ("ty (nm)", ty)]


def report_terms(matrix: np.ndarray, points: np.ndarray) -> list[tuple[str, float, float]]:
    """Rows (label, value, edge_nm) for display: translation in nm, magnification in ppm (= nm per
    mm), rotation and orthogonality in degrees. edge_nm = how far that term alone moves the
    furthest of `points` (relative to the reference point, nm).

    For a small rotation θ a point at distance r moves θ·r; a pure orthogonality ω moves it at
    most ω/2·r (each axis turns by ω/2, in opposite directions).
    """
    terms = decompose(matrix)
    x_max, y_max = np.abs(points).max(axis=0)
    r_max = np.linalg.norm(points, axis=1).max()
    rotation, orthogonality = terms["rotation_urad"] * 1e-6, terms["orthogonality_urad"] * 1e-6
    return [
        ("Tx (nm)", terms["Tx_nm"], abs(terms["Tx_nm"])),
        ("Ty (nm)", terms["Ty_nm"], abs(terms["Ty_nm"])),
        ("Mx (ppm)", terms["Mx_ppm"], abs(terms["Mx_ppm"]) * 1e-6 * x_max),
        ("My (ppm)", terms["My_ppm"], abs(terms["My_ppm"]) * 1e-6 * y_max),
        ("rotation (°)", np.degrees(rotation), abs(rotation) * r_max),
        ("orthogonality (°)", np.degrees(orthogonality), abs(orthogonality) / 2 * r_max),
    ]
