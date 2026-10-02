"""Global stitching solve, translation only (first version of docs/SPEC.md S5).

Each tile k gets a correction t_k (nm) that is added to its nominal placement. For a pair (i, j)
the measured shift d_ij is B − A for the same contacts at nominal placement (overlap_fit). After
correction the two measurements coincide when

    (a + t_i) = (b + t_j)   ->   t_i − t_j = d_ij

All pairs together are solved by least squares. A shift common to all tiles cannot be seen in
the overlaps (gauge freedom), so the corrections are fixed to average zero; the later fit to
the design (S7) absorbs any common shift.
"""

import numpy as np


def solve_tile_shifts(n_tiles: int, pairs: list[tuple[int, int]], shifts: np.ndarray, weights=None) -> np.ndarray:
    """Per-tile corrections (n_tiles, 2) in nm, averaging zero.

    pairs: (i, j) tile indices; shifts: (P, 2) measured B − A shift d_ij for each pair;
    weights: optional (P,) weight per pair (e.g. number of inlier contacts).
    Tiles not connected to any pair get a correction of 0.
    """
    shifts = np.asarray(shifts, dtype=float).reshape(-1, 2)
    weights = np.ones(len(pairs)) if weights is None else np.asarray(weights, dtype=float)

    # One equation per pair: t_i − t_j = d_ij, plus one gauge equation: sum of t = 0.
    rows = np.zeros((len(pairs) + 1, n_tiles))
    for p, (i, j) in enumerate(pairs):
        rows[p, i], rows[p, j] = 1.0, -1.0
    rows[-1, :] = 1.0
    right = np.vstack([shifts, [0.0, 0.0]])
    sqrt_w = np.sqrt(np.append(weights, 1.0))[:, None]

    corrections, *_ = np.linalg.lstsq(rows * sqrt_w, right * sqrt_w, rcond=None)
    return corrections
