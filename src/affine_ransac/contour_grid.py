"""Exact grid costs and correlations without whole-image distance transforms (S5, D165).

The distance is capped at a small gate. Paint only that neighbourhood of each edge point, taking the minimum
where they meet. Correlate constant runs of occupied pixels using row sums where this requires less work than
a dense FFT. These are representations of the same raster calculation, not approximations to the contours.
The compiled loops are serial and release the GIL; they do not create another parallel pool beside denoising.
Imported only by contour stitching; numba caches the loops for subsequent processes.
"""

import numba
import numpy as np


def point_coverage(rows, columns, covered_x, covered_y, size: int) -> np.ndarray:
    """Each point contributes one rectangle of allowed shifts; count them on the small search grid."""
    x, y = np.flatnonzero(covered_x), np.flatnonzero(covered_y)
    if not len(x) or not len(y):
        return np.zeros((size, size))
    x0, x1 = np.clip(x[0] - columns, 0, size), np.clip(x[-1] + 1 - columns, 0, size)
    y0, y1 = np.clip(y[0] - rows, 0, size), np.clip(y[-1] + 1 - rows, 0, size)
    changes = np.zeros((size + 1, size + 1), dtype=np.int64)
    np.add.at(changes, (y0, x0), 1)
    np.add.at(changes, (y1, x0), -1)
    np.add.at(changes, (y0, x1), -1)
    np.add.at(changes, (y1, x1), 1)
    return changes.cumsum(axis=0).cumsum(axis=1)[:size, :size].astype(float)


@numba.njit(cache=True, nogil=True)
def _paint_cost(rows, columns, offsets, costs, height, width, cap):
    image = np.full((height, width), cap, dtype=np.float32)
    for k in range(len(rows)):
        for p in range(len(costs)):
            row, col = rows[k] + offsets[p, 0], columns[k] + offsets[p, 1]
            if 0 <= row < height and 0 <= col < width and costs[p] < image[row, col]:
                image[row, col] = costs[p]
    return image


def capped_cost(rows, columns, shape, bin_nm: float, gate_px: float) -> np.ndarray:
    """Exactly the float32 squared pixel-grid distance capped at gate_px, including an empty reference."""
    radius = int(np.ceil(gate_px))
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    squared = xx * xx + yy * yy
    within = squared < gate_px ** 2
    offsets = np.column_stack([yy[within], xx[within]])
    # Keep the distance -> nm -> square -> float32 arithmetic of the former EDT path.
    costs = ((np.sqrt(squared[within]) * bin_nm) ** 2).astype(np.float32)
    points = np.unique(rows * shape[1] + columns)  # upscaled contours can visit one raster cell many times
    return _paint_cost(points // shape[1], points % shape[1], offsets, costs,
                       shape[0], shape[1], np.float32((gate_px * bin_nm) ** 2))


def pixel_runs(rows, columns, width: int) -> tuple[np.ndarray, np.ndarray]:
    """(row, first column, stop column) and point count for each constant nonzero run; duplicates count."""
    flat, weights = np.unique(rows * width + columns, return_counts=True)
    if not len(flat):
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float32)
    starts = np.r_[0, np.flatnonzero((np.diff(flat) != 1) | (np.diff(weights) != 0) |
                                    (np.diff(flat // width) != 0)) + 1]
    stops = np.r_[starts[1:], len(flat)] - 1
    runs = np.column_stack([flat[starts] // width, flat[starts] % width, flat[stops] % width + 1])
    return runs, weights[starts].astype(np.float32)


def compact_runs(rows, columns, shape) -> tuple[np.ndarray, np.ndarray, bool]:
    """Choose the raster orientation with fewer runs, with no restriction on the pattern's direction."""
    horizontal, weights = pixel_runs(rows, columns, shape[1])
    vertical, other_weights = pixel_runs(columns, rows, shape[0])
    if len(vertical) < len(horizontal):
        return vertical, other_weights, True
    return horizontal, weights, False


@numba.njit(cache=True, nogil=True)
def _correlate_runs(sums, runs, weights, size):
    result = np.zeros((size, size))
    for dy in range(size):
        for k in range(len(runs)):
            row, first, stop = runs[k]
            for dx in range(size):
                result[dy, dx] += (sums[row + dy, stop + dx] - sums[row + dy, first + dx]) * weights[k]
    return result


def correlate_runs(cost: np.ndarray, runs: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
    """Every valid shift of the point runs against cost, from exact row sums (float64 accumulation)."""
    sums = np.zeros((cost.shape[0], cost.shape[1] + 1))
    np.cumsum(cost, axis=1, dtype=float, out=sums[:, 1:])
    return _correlate_runs(sums, runs, weights, size)
