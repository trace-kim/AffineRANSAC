"""Drift along y (docs/SPEC.md S8, D54): a smooth curve through the mean error per row, minus its
straight line.

The mean SEM − design error (dx, dy) of each row of contacts (registration.row_means) against the
row's y follows a straight line where the error is affine (shift, scale, rotation, skew): the global
affine removes that line. Drift during the acquisition bends it, and no affine removes a bend.
drift_curve() smooths the row means with a local straight-line fit: at each row, a least-squares line
through the rows within ±window/2 of it (weighted by their contacts), taken at that row. The
correction is that curve minus the straight line through all row means, so the straight part stays
in the data for the global affine (its terms are reported as before). Real mask errors that vary
smoothly along y on the scale of the window are removed too: they cannot be told apart from drift.
"""

from dataclasses import dataclass

import numpy as np

from affine_ransac.registration import row_means


@dataclass
class DriftCurve:
    row_y_nm: np.ndarray     # (R,) mean design y of each row of contacts, mask nm
    row_mean_nm: np.ndarray  # (R, 2) mean error (dx, dy) of each row, nm
    curve_nm: np.ndarray     # (R, 2) the smooth curve through the row means, nm
    line_nm: np.ndarray      # (R, 2) the straight line through the row means (left for the global affine), nm
    window_nm: float         # height of the local straight-line fits, nm
    # The straight line: line = intercept + slope · (y − centre), for dx and dy.
    line_centre_y_nm: float
    line_intercept_nm: np.ndarray  # (2,) nm
    line_slope: np.ndarray         # (2,) nm per nm of y (× 1e6 = ppm)

    def correction(self, design_y: np.ndarray) -> np.ndarray:
        """(N, 2) curve − line at the contacts' design y (the value of their row), nm."""
        bend = self.curve_nm - self.line_nm
        return np.column_stack([np.interp(design_y, self.row_y_nm, bend[:, c]) for c in range(2)])


def drift_curve(design_y: np.ndarray, error_nm: np.ndarray, window_nm: float, gap_nm: float = 10.0) -> DriftCurve:
    """The drift curve of contacts at design y (N,) with errors error_nm (N, 2) = SEM − design, nm
    (e.g. the merged contacts of the translation stitching). window_nm: height of the local fit along
    y; gap_nm: rows as in registration.group_rows."""
    row_y, row_mean, count = row_means(design_y, error_nm, gap_nm)
    curve = np.empty_like(row_mean)
    for i, y in enumerate(row_y):
        near = np.abs(row_y - y) <= window_nm / 2
        curve[i] = _line_value(row_y[near] - y, row_mean[near], count[near])
    centre = float(np.average(row_y, weights=count))
    slope, intercept = np.polyfit(row_y - centre, row_mean, 1, w=np.sqrt(count))  # each (2,): dx, dy
    line = np.outer(row_y - centre, slope) + intercept
    return DriftCurve(row_y, row_mean, curve, line, float(window_nm), centre, intercept, slope)


def _line_value(offsets: np.ndarray, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(2,) value at offset 0 of the least-squares straight line through values (K, 2) at offsets (K,),
    each weighted by its number of contacts; a single row is its own value."""
    if len(offsets) < 2:
        return values[0]
    return np.polyfit(offsets, values, 1, w=np.sqrt(weights))[1]
