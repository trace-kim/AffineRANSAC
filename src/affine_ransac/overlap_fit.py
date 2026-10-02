"""Robust fit of how tile B is placed relative to tile A, from their matched overlap contacts.

Model:  b ≈ R(rotation) · (a − center) + center + shift
- a, b: the same physical contacts measured in tile A and tile B, in mask nm (nominal placement).
- shift: B − A at `center` (the centroid of the inlier A points), in nm.
- rotation: of B relative to A in radians, about `center`; 0 for a translation-only fit.
No scale: both tiles come from the same SEM at the same magnification.

Robustness without RANSAC (too few points per strip): start from the median difference, then
repeatedly fit by least squares to the inliers and mark as outliers the points whose residual is
more than `outlier_factor` times the median residual.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class OverlapFit:
    shift: np.ndarray      # (2,) B − A at center, nm
    rotation: float        # radians, B relative to A, about center (0 if not fitted)
    center: np.ndarray     # (2,) rotation centre: centroid of the inlier A points, nm
    inliers: np.ndarray    # (N,) bool: points used in the final fit
    residuals: np.ndarray  # (N, 2) b − model(a) for every point, nm

    def rms(self, include_outliers: bool = False) -> float:
        """Root-mean-square residual length, nm. Inliers only by default (the measurement noise
        floor); include_outliers=True uses every matched point."""
        residuals = self.residuals if include_outliers else self.residuals[self.inliers]
        return float(np.sqrt((residuals ** 2).sum(axis=1).mean()))


def apply_fit(fit: OverlapFit, a: np.ndarray) -> np.ndarray:
    """Where the model puts tile-A points a in tile B: R · (a − center) + center + shift."""
    cos, sin = np.cos(fit.rotation), np.sin(fit.rotation)
    rotated = (a - fit.center) @ np.array([[cos, sin], [-sin, cos]])  # row vectors: (R · v)ᵀ = vᵀ · Rᵀ
    return rotated + fit.center + fit.shift


def _least_squares(a: np.ndarray, b: np.ndarray, rotation: bool):
    """Least-squares shift (and rotation) about the centroid of a. Returns (shift, rotation, center)."""
    center = a.mean(axis=0)
    shift = b.mean(axis=0) - center
    angle = 0.0
    if rotation and len(a) >= 2:
        a0, b0 = a - center, b - b.mean(axis=0)
        cross = (a0[:, 0] * b0[:, 1] - a0[:, 1] * b0[:, 0]).sum()
        dot = (a0 * b0).sum()
        angle = float(np.arctan2(cross, dot))
    return shift, angle, center


def _outlier_threshold(distances: np.ndarray, outlier_factor: float) -> float:
    # The small floor keeps every point on noise-free data, where the median residual is ~0.
    return max(outlier_factor * float(np.median(distances)), 1e-6)


def fit_overlap(
    a: np.ndarray,
    b: np.ndarray,
    rotation: bool = False,
    outlier_factor: float = 3.0,
    iterations: int = 3,
) -> OverlapFit:
    """Robust B-relative-to-A fit (translation, or translation + rotation) from matched points."""
    if len(a) == 0:
        raise ValueError("No matched points to fit")

    # Start: residuals from the median difference, so a gross outlier cannot pull the start.
    start = np.linalg.norm((b - a) - np.median(b - a, axis=0), axis=1)
    inliers = start <= _outlier_threshold(start, outlier_factor)

    for _ in range(iterations):
        shift, angle, center = _least_squares(a[inliers], b[inliers], rotation)
        fit = OverlapFit(shift, angle, center, inliers, residuals=np.zeros_like(a))
        fit.residuals = b - apply_fit(fit, a)
        distances = np.linalg.norm(fit.residuals, axis=1)
        inliers = distances <= _outlier_threshold(distances, outlier_factor)

    # Final fit on the final inliers, so shift/rotation and inliers agree.
    shift, angle, center = _least_squares(a[inliers], b[inliers], rotation)
    fit = OverlapFit(shift, angle, center, inliers, residuals=np.zeros_like(a))
    fit.residuals = b - apply_fit(fit, a)
    return fit
