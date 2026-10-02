"""Registration error: stitched SEM contact centres vs stitched design centres (docs/SPEC.md S6, S8).

v1 reports the RAW error: no affine is removed yet (the RANSAC global affine, S7, comes later),
so a common offset, rotation or scale shows up as a pattern in the error map.

1. design_errors(): one OBSERVATION per contact per tile. Both sides are placed with their own
   stitching corrections: the SEM tiles (stitch_tiles) and the per-tile design files
   (stitch_design), which can be offset against each other. Each tile's SEM contacts are matched
   to its own design contacts (mutual nearest, within a gate), so a contact in an overlap gets one
   observation per tile that sees it (2 in a strip, up to 4 in a corner).
2. merge_observations(): one entry per physical CONTACT. Observations whose stitched design
   positions lie within a small radius (and come from different tiles) are the same contact;
   their SEM and design positions are averaged and their spread is kept (SPEC S5).

Tiles that cannot be measured are listed with the reason and a warning, never dropped silently.
Sign: SEM − design (tentative, SPEC §13-8). It is set only in registration_error().
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from affine_ransac.matching import match_points
from affine_ransac.stitching import connected_groups


@dataclass
class DesignErrors:
    tile: np.ndarray               # (N,) tile index of each observation
    design_nm: np.ndarray          # (N, 2) design centre in this placement, mask nm
    sem_nm: np.ndarray             # (N, 2) SEM centre in this placement, mask nm
    error_nm: np.ndarray           # (N, 2) registration_error(sem_nm, design_nm), nm
    design_nominal_nm: np.ndarray  # (N, 2) design centre at the nominal placement (for grouping)
    skipped: dict                  # tile index -> reason why it was not measured (no error computed)


@dataclass
class MergedErrors:
    design_nm: np.ndarray   # (M, 2) mean design centre of each contact, mask nm
    sem_nm: np.ndarray      # (M, 2) mean SEM centre, mask nm
    error_nm: np.ndarray    # (M, 2) registration_error(sem_nm, design_nm), nm
    count: np.ndarray       # (M,) number of observations (1 outside overlaps, 2-4 inside)
    spread_nm: np.ndarray   # (M,) largest distance between the contact's SEM observations (0 if one)
    members: list           # M arrays: indices of the contact's observations in the DesignErrors
    skipped: dict           # tiles not measured, as in the DesignErrors


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
    sem_corrections: np.ndarray,
    design_corrections: np.ndarray,
    design_ok,
    max_match_nm: float = 40.0,
) -> DesignErrors:
    """Raw registration error of every contact found in both the stitched SEM and the design.

    sem_points, design_points: per tile, SEM / design centres in mask nm at the nominal placement.
    sem_corrections, design_corrections: (n_tiles, 2) corrections added to each tile's SEM / design
    points, e.g. one placement of stitching.placement_corrections (NaN = tile not stitched).
    design_ok: per tile, whether its design tone matched the SEM (DesignContacts.ok).
    """
    tiles, designs, nominals, sems, skipped = [], [], [], [], {}
    for k, (sem, design) in enumerate(zip(sem_points, design_points)):
        if not design_ok[k]:
            skipped[k] = "design tone not matched (flagged)"
            continue
        if np.isnan(sem_corrections[k]).any():
            skipped[k] = "SEM tile not stitched"
            continue
        if np.isnan(design_corrections[k]).any():
            skipped[k] = "design tile not stitched"
            continue
        stitched_sem = sem + sem_corrections[k]
        stitched_design = design + design_corrections[k]
        i_design, i_sem = match_points(stitched_design, stitched_sem, max_match_nm)
        if len(i_design) == 0:
            skipped[k] = "no SEM contact matched the design"
            continue
        tiles.append(np.full(len(i_design), k))
        designs.append(stitched_design[i_design])
        nominals.append(design[i_design])
        sems.append(stitched_sem[i_sem])

    if skipped:
        warnings.warn(f"{len(skipped)} tile(s) not measured (no error computed): {skipped}", stacklevel=2)

    def join(parts):
        return np.concatenate(parts) if parts else np.empty((0, 2))

    design_nm, sem_nm = join(designs), join(sems)
    return DesignErrors(
        tile=np.concatenate(tiles) if tiles else np.empty(0, int),
        design_nm=design_nm,
        sem_nm=sem_nm,
        error_nm=registration_error(sem_nm, design_nm),
        design_nominal_nm=join(nominals),
        skipped=skipped,
    )


def merge_observations(errors: DesignErrors, design_corrections: np.ndarray, radius_nm: float = 5.0) -> MergedErrors:
    """Average the observations of the same physical contact (seen by several tiles in an overlap).

    Grouping uses the STITCHED design positions, design_nominal_nm + design_corrections[tile], with
    design_corrections = stitch_design(...).corrections, for every placement, so all placements
    share the same groups. Two observations are the same contact if they come from different
    tiles and lie closer than radius_nm (≪ half the pitch; the design stitching residual is far
    smaller). Grouping never uses SEM positions: they hold the errors being measured.

    Warns if a group holds two observations of the same tile (a tile's own contacts are a pitch
    apart, so that should not happen).
    """
    n = len(errors.tile)
    grouping = errors.design_nominal_nm + design_corrections[errors.tile] if n else np.empty((0, 2))
    pairs = [(i, j) for i, j in cKDTree(grouping).query_pairs(radius_nm) if errors.tile[i] != errors.tile[j]]
    groups = connected_groups(n, pairs)

    label = np.empty(n, int)
    for g, members in enumerate(groups):
        label[members] = g
    count = np.bincount(label, minlength=len(groups))

    def mean(points):
        total = np.zeros((len(groups), 2))
        np.add.at(total, label, points)
        return total / count[:, None]

    spread = np.zeros(len(groups))
    repeated_tile = 0
    for g, members in enumerate(groups):
        if len(members) > 1:
            sem = errors.sem_nm[members]
            spread[g] = np.linalg.norm(sem[:, None] - sem[None], axis=2).max()
            repeated_tile += len(set(errors.tile[members])) < len(members)
    if repeated_tile:
        warnings.warn(f"{repeated_tile} merged contact(s) hold two observations of the same tile "
                      f"(radius {radius_nm} nm too large, or duplicated design contacts)", stacklevel=2)

    design_nm, sem_nm = mean(errors.design_nm), mean(errors.sem_nm)
    return MergedErrors(
        design_nm=design_nm,
        sem_nm=sem_nm,
        error_nm=registration_error(sem_nm, design_nm),
        count=count,
        spread_nm=spread,
        members=[np.asarray(m) for m in groups],
        skipped=errors.skipped,
    )


def binned_mean_2d(points: np.ndarray, values: np.ndarray, bin_nm: float):
    """Mean of values (N,) in square bins of bin_nm over points (N, 2), e.g. a heatmap of one error
    component. Returns (grid, x_edges, y_edges): grid[iy, ix] is the mean in bin (ix, iy), NaN
    where no point falls; row 0 is the lowest y."""
    x_edges = np.arange(points[:, 0].min(), points[:, 0].max() + bin_nm, bin_nm)
    y_edges = np.arange(points[:, 1].min(), points[:, 1].max() + bin_nm, bin_nm)
    total, *_ = np.histogram2d(points[:, 1], points[:, 0], bins=(y_edges, x_edges), weights=values)
    count, *_ = np.histogram2d(points[:, 1], points[:, 0], bins=(y_edges, x_edges))
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(count > 0, total / count, np.nan)
    return grid, x_edges, y_edges


def profile(positions: np.ndarray, values: np.ndarray, bin_nm: float):
    """Values (N,) averaged in bins of bin_nm along positions (N,), e.g. dx against y, averaged
    over x. Returns (centers, mean, std, count) for the bins that hold at least one point."""
    edges = np.arange(positions.min(), positions.max() + bin_nm, bin_nm)
    index = np.clip(np.digitize(positions, edges) - 1, 0, len(edges) - 2)
    count = np.bincount(index, minlength=len(edges) - 1)
    total = np.bincount(index, weights=values, minlength=len(edges) - 1)
    squares = np.bincount(index, weights=values ** 2, minlength=len(edges) - 1)
    used = count > 0
    mean = total[used] / count[used]
    std = np.sqrt(np.maximum(squares[used] / count[used] - mean ** 2, 0))
    centers = (edges[:-1] + edges[1:])[used] / 2
    return centers, mean, std, count[used]
