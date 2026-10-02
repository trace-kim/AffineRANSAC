"""Registration error: stitched SEM contact centres vs design centres (docs/SPEC.md S6, S8).

v1 reports the RAW error: no affine is removed yet (the RANSAC global affine, S7, comes later),
so a common offset, rotation or scale shows up as a pattern in the error map.

Each tile is matched to its own design contacts (mutual nearest, within a gate). A contact seen
by two tiles in an overlap therefore has two error entries, one per tile (all observations are
kept, SPEC S5). Tiles that cannot be measured are listed with the reason and a warning, never
dropped silently.

Sign: SEM − design (tentative, SPEC §13-8). It is set only in registration_error().
"""

import warnings
from dataclasses import dataclass

import numpy as np

from affine_ransac.matching import match_points


@dataclass
class DesignErrors:
    tile: np.ndarray        # (N,) tile index of each matched contact
    design_nm: np.ndarray   # (N, 2) design centre, mask nm
    sem_nm: np.ndarray      # (N, 2) stitched SEM centre, mask nm
    error_nm: np.ndarray    # (N, 2) registration_error(sem_nm, design_nm), nm
    skipped: dict           # tile index -> reason why it has no errors


def registration_error(sem: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Error vectors for matched SEM and design points (SPEC §13-8: sign still to be confirmed)."""
    return sem - design


def error_summary(error_nm: np.ndarray) -> dict:
    """Count, mean and 3σ per axis, and the largest error length, in nm."""
    if len(error_nm) == 0:
        return {"count": 0}
    return {
        "count": len(error_nm),
        "mean_x_nm": float(error_nm[:, 0].mean()), "mean_y_nm": float(error_nm[:, 1].mean()),
        "3sigma_x_nm": float(3 * error_nm[:, 0].std()), "3sigma_y_nm": float(3 * error_nm[:, 1].std()),
        "max_nm": float(np.linalg.norm(error_nm, axis=1).max()),
    }


def design_errors(
    sem_points: list[np.ndarray],
    design_points: list[np.ndarray],
    corrections: np.ndarray,
    design_ok,
    max_match_nm: float = 25.0,
) -> DesignErrors:
    """Raw registration error of every contact found in both the stitched SEM and the design.

    sem_points: per tile, SEM centres in mask nm at the nominal placement.
    design_points: per tile, design centres in mask nm.
    corrections: (n_tiles, 2) stitching corrections (NaN = tile not stitched).
    design_ok: per tile, whether its design tone matched the SEM (DesignContacts.ok).
    """
    tiles, designs, sems, skipped = [], [], [], {}
    for k, (sem, design) in enumerate(zip(sem_points, design_points)):
        if np.isnan(corrections[k]).any():
            skipped[k] = "not stitched"
            continue
        if not design_ok[k]:
            skipped[k] = "design tone not matched (flagged)"
            continue
        stitched = sem + corrections[k]
        i_design, i_sem = match_points(design, stitched, max_match_nm)
        if len(i_design) == 0:
            skipped[k] = "no SEM contact matched the design"
            continue
        tiles.append(np.full(len(i_design), k))
        designs.append(design[i_design])
        sems.append(stitched[i_sem])

    if skipped:
        warnings.warn(f"No design errors for {len(skipped)} tile(s): {skipped}", stacklevel=2)
    design_nm = np.concatenate(designs) if designs else np.empty((0, 2))
    sem_nm = np.concatenate(sems) if sems else np.empty((0, 2))
    return DesignErrors(
        tile=np.concatenate(tiles) if tiles else np.empty(0, int),
        design_nm=design_nm,
        sem_nm=sem_nm,
        error_nm=registration_error(sem_nm, design_nm),
        skipped=skipped,
    )
