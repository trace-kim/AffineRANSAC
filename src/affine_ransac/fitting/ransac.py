"""RANSAC affine from SEM points to design points (docs/SPEC.md S7, §7). No UI code.

1. Draw 3 correspondences at random (seeded). Reject degenerate triplets (triangle area below
   min_area_fraction × the squared RMS spread of the points: collinear or too close together).
2. Exact affine mapping the 3 SEM points onto their 3 design points.
3. Apply it to all SEM points; inliers = residual |G(sem) − design| < threshold_nm.
4. Keep the model with the most inliers (ties: lower sum of inlier residuals).
5. Refit: least squares on all inliers of the best model, recompute the inliers, repeat up to
   refine_iters times or until the inlier set is stable.

Stop condition of the search (Fischler & Bolles 1981, the original RANSAC paper; Hartley &
Zisserman, "Multiple View Geometry", 2nd ed., §4.7.1). One good sample (3 inliers) is enough to
find the right model, so stop once a good sample has almost certainly been drawn:
- If a fraction w of the points are inliers, a random 3-point sample is all inliers with
  probability w³ (sampling with replacement; for hundreds of points or more the difference is
  negligible).
- N samples ALL fail (each contains an outlier) with probability (1 − w³)^N.
- Require that to be at most 1 − p (p = confidence, 0.999):
      (1 − w³)^N <= 1 − p   ->   N >= log(1 − p) / log(1 − w³)
- w is unknown, so the best model's inlier ratio so far is used. It can only underestimate the
  true ratio, so N is conservative; it is recomputed whenever a better model is found, and the
  search stops when the iteration count reaches N, or at max_iters (degenerate samples count).
  Examples: w = 0.9 -> N = 6, 0.5 -> 52, 0.3 -> 253, 0.1 -> 6905.
The precision of the final model comes from the least-squares refit on all inliers (step 5),
not from the number of samples.

Everything is computed relative to a reference point, the centroid of the design points
(SPEC §8), so the model's translation is the shift at that point.

ransac_affine_steps() yields the state after every iteration (for a live monitor);
ransac_affine() runs it to the end and returns the result.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from affine_ransac.geometry.affine import apply_affine, fit_affine, triangle_area


@dataclass
class RansacStep:
    stage: str                 # "search" (random 3-point samples) or "refit" (least squares on inliers)
    iteration: int             # search: 1, 2, ...; refit: 1 .. refine_iters
    sample: np.ndarray | None  # (3,) indices of this iteration's sample (search only)
    degenerate: bool           # True: the sample was rejected (no model this iteration)
    model: np.ndarray | None   # 3×3 affine of this iteration (relative to `reference`)
    inliers: np.ndarray        # (N,) bool, inliers of this iteration's model
    best_model: np.ndarray | None
    best_inliers: np.ndarray
    best_sample: np.ndarray | None
    improved: bool             # True: this iteration became the best so far
    needed_iterations: int     # current N from the adaptive formula (capped at max_iters)
    reference: np.ndarray      # (2,) reference point (design centroid), mask nm


@dataclass
class RansacResult:
    model: np.ndarray        # 3×3 final affine G (least-squares refit), relative to `reference`
    reference: np.ndarray    # (2,) design centroid, mask nm
    inliers: np.ndarray      # (N,) bool
    residuals: np.ndarray    # (N, 2) G(sem) − design for EVERY point, nm (outliers included)
    iterations: int          # search iterations run
    best_sample: np.ndarray  # (3,) indices of the best 3-point sample


def residuals(model: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """model(src) − dst, (N, 2)."""
    return apply_affine(model, src) - dst


def needed_iterations(inlier_ratio: float, confidence: float, max_iters: int) -> int:
    """N = log(1 − p) / log(1 − w³), capped at max_iters."""
    w3 = inlier_ratio ** 3
    if w3 >= 1:
        return 1
    if w3 <= 0:
        return max_iters
    return int(min(max_iters, np.ceil(np.log(1 - confidence) / np.log(1 - w3))))


def ransac_affine_steps(
    sem: np.ndarray,
    design: np.ndarray,
    threshold_nm: float,
    confidence: float = 0.999,
    max_iters: int = 10_000,
    min_area_fraction: float = 1e-3,
    refine_iters: int = 3,
    seed: int = 0,
):
    """Yield a RansacStep after every search and refit iteration. sem, design: (N, 2) matched
    points in mask nm (N >= 3)."""
    reference = design.mean(axis=0)
    src, dst = sem - reference, design - reference
    n = len(src)
    rng = np.random.default_rng(seed)
    min_area = min_area_fraction * (src ** 2).sum(axis=1).mean()
    no_points = np.zeros(n, bool)

    best = {"model": None, "inliers": no_points, "sample": None, "score": (-1, 0.0)}
    needed, iteration = max_iters, 0
    while iteration < min(needed, max_iters):
        iteration += 1
        sample = rng.choice(n, 3, replace=False)
        if triangle_area(*src[sample]) < min_area:
            yield RansacStep("search", iteration, sample, True, None, no_points, best["model"], best["inliers"],
                             best["sample"], False, needed, reference)
            continue
        model = fit_affine(src[sample], dst[sample])
        distance = np.linalg.norm(residuals(model, src, dst), axis=1)
        inliers = distance < threshold_nm
        score = (int(inliers.sum()), -float(distance[inliers].sum()))  # more inliers, then smaller residuals
        improved = score > best["score"]
        if improved:
            best = {"model": model, "inliers": inliers, "sample": sample, "score": score}
            needed = needed_iterations(inliers.mean(), confidence, max_iters)
        yield RansacStep("search", iteration, sample, False, model, inliers, best["model"], best["inliers"],
                         best["sample"], improved, needed, reference)

    inliers = best["inliers"]
    for refit in range(1, refine_iters + 1):
        if inliers.sum() < 3:
            break
        model = fit_affine(src[inliers], dst[inliers])
        new_inliers = np.linalg.norm(residuals(model, src, dst), axis=1) < threshold_nm
        stable = np.array_equal(new_inliers, inliers)
        inliers = new_inliers
        best = {**best, "model": model, "inliers": inliers}
        yield RansacStep("refit", refit, None, False, model, inliers, model, inliers, best["sample"], True,
                         needed, reference)
        if stable:
            break


def ransac_affine(
    sem: np.ndarray,
    design: np.ndarray,
    threshold_nm: float,
    confidence: float = 0.999,
    max_iters: int = 10_000,
    min_area_fraction: float = 1e-3,
    refine_iters: int = 3,
    min_inlier_ratio: float = 0.5,
    seed: int = 0,
) -> RansacResult:
    """Run the RANSAC to the end. Warns if the final inlier ratio is below min_inlier_ratio.
    Residuals are returned for every point; outliers are only excluded from the fit."""
    last, search_iterations = None, 0
    for last in ransac_affine_steps(sem, design, threshold_nm, confidence, max_iters, min_area_fraction,
                                    refine_iters, seed):
        if last.stage == "search":
            search_iterations = last.iteration
    if last is None or last.best_model is None:
        raise ValueError("RANSAC found no model (too few points, or every sample degenerate)")
    ratio = last.best_inliers.mean()
    if ratio < min_inlier_ratio:
        warnings.warn(f"RANSAC inlier ratio {ratio:.2f} is below {min_inlier_ratio} "
                      f"(threshold {threshold_nm} nm may be too tight, or the data too noisy)", stacklevel=2)
    reference = last.reference
    return RansacResult(
        model=last.best_model,
        reference=reference,
        inliers=last.best_inliers,
        residuals=residuals(last.best_model, sem - reference, design - reference),
        iterations=search_iterations,
        best_sample=last.best_sample,
    )
