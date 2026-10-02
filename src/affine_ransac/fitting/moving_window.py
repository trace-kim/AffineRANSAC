"""Moving-window affine along y, SEM -> design (docs/SPEC.md S8, D36). No RANSAC, no UI code.

An alternative to the single global RANSAC affine, mimicking a reference algorithm:
1. Windows of height window_nm slide along the design y in steps of step_nm, from the bottom of the
   field (first window's lower edge at the lowest contact) to the top (last window's upper edge at
   the highest contact).
2. Each window: plain least-squares affine on ALL contacts inside it (no outlier rejection),
   relative to the window's design centroid.
3. Each contact is corrected by the window whose centre is nearest to its design y, so the
   windows overlap for the fits but every contact gets exactly one residual.
"""

from dataclasses import dataclass

import numpy as np

from affine_ransac.fitting.ransac import residuals
from affine_ransac.geometry.affine import fit_affine


@dataclass
class MovingWindowResult:
    residuals: np.ndarray   # (N, 2) G_w(sem) − design for every contact (G_w = its window's affine), nm
    window: np.ndarray      # (N,) index of the window that corrected each contact
    centres_nm: np.ndarray  # (K,) design y at the centre of each window, nm
    models: np.ndarray      # (K, 3, 3) affine of each window, relative to its reference
    references: np.ndarray  # (K, 2) design centroid of the contacts inside each window, nm
    counts: np.ndarray      # (K,) contacts used in each window's fit


def window_centres(y: np.ndarray, window_nm: float, step_nm: float) -> np.ndarray:
    """Centres from the lowest full window to the highest, step_nm apart (the last step may be
    shorter so the top window ends at the highest contact). One centre if the field is shorter
    than a window."""
    low, high = y.min() + window_nm / 2, y.max() - window_nm / 2
    if high <= low:
        return np.array([(y.min() + y.max()) / 2])
    return np.append(np.arange(low, high, step_nm), high)


def moving_window_affine(sem: np.ndarray, design: np.ndarray, window_nm: float, step_nm: float) -> MovingWindowResult:
    """sem, design: (N, 2) matched points in mask nm. Raises if a window holds fewer than 3 contacts."""
    if not 0 < step_nm <= window_nm:
        raise ValueError("need 0 < step_nm <= window_nm (else some contacts lie in no window)")
    y = design[:, 1]
    centres = window_centres(y, window_nm, step_nm)
    nearest = np.searchsorted((centres[1:] + centres[:-1]) / 2, y)  # index of the nearest centre

    models, references, counts = [], [], []
    result = np.full_like(design, np.nan, dtype=float)
    for k, centre in enumerate(centres):
        inside = np.abs(y - centre) <= window_nm / 2
        if inside.sum() < 3:
            raise ValueError(f"window {k} at y = {centre / 1000:.3f} um holds {inside.sum()} contacts (need 3)")
        reference = design[inside].mean(axis=0)
        model = fit_affine(sem[inside] - reference, design[inside] - reference)
        mine = nearest == k
        result[mine] = residuals(model, sem[mine] - reference, design[mine] - reference)
        models.append(model)
        references.append(reference)
        counts.append(int(inside.sum()))

    return MovingWindowResult(result, nearest, centres, np.array(models), np.array(references), np.array(counts))
