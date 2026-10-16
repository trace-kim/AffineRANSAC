"""Realignment of an image folder whose images all sit at one wrong offset from their design (docs/SPEC.md D91).

When the SEM alignment went wrong, every image shows the design some µm away from where the metadata says it is,
so no image matches its own .oas. The user's method: stitch the SEM tiles on their own, put the design contacts
of all .oas files together, find the one offset that lines the two up, then give every tile that offset and the
design contacts inside its shifted field of view. The analysis then runs as usual on the realigned tiles.
Used only when settings.realign_search_um > 0. Normal tone only: tone-reversed files are not handled (user).

- find_design_offset(): the offset, scored over every shift at once (FFT correlations of point rasters):
  the fewest MISMATCHES where both sets are complete, as for one tile (matching.scored_shifts, D50).
- merge_close(): one point per group of points closer than a radius (the same contact in two .oas files).
- realign_tiles(): the steps above on loaded tiles (pipeline.process_tile with match_design=False).
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.ndimage import maximum_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from affine_ransac.pipeline import DesignContacts, TileResult, stitch_tiles

MAX_CELLS = 1e8  # raster size limit (~400 MB per float32 raster)


@dataclass
class DesignOffset:
    shift_nm: np.ndarray       # (2,) design − SEM: the SEM contacts + shift_nm lie on the design
    matched: int               # SEM contacts with a design contact within 1.5 cells at that shift
    mismatch_fraction: float   # points without a partner / points compared, where both sets are complete
    runner_up_nm: np.ndarray   # (2,) the best shift at least 3 cells away (NaN if there is none)
    runner_up_fraction: float  # its mismatch fraction (NaN if none)
    tied: bool                 # the runner-up is not clearly worse: periodic content, the offset is a guess


def _points_raster(points, origin, cell_nm, shape) -> np.ndarray:
    """1 in every cell holding a point (rows = y, columns = x); duplicates count once."""
    raster = np.zeros(shape, np.float32)
    index = np.floor((points - origin) / cell_nm).astype(int)
    raster[index[:, 1], index[:, 0]] = 1
    return raster


def _boxes_raster(centers, sizes, margin_nm, origin, cell_nm, shape) -> np.ndarray:
    """1 in every cell inside a box (centre, size) shrunk by margin_nm on every side."""
    raster = np.zeros(shape, np.float32)
    for center, size in zip(centers, sizes):
        low = np.ceil((center - size / 2 + margin_nm - origin) / cell_nm).astype(int)
        high = np.floor((center + size / 2 - margin_nm - origin) / cell_nm).astype(int)
        if np.all(high >= low):
            raster[low[1]:high[1] + 1, low[0]:high[0] + 1] = 1
    return raster


def _correlation(a: np.ndarray, b: np.ndarray, reach: int) -> np.ndarray:
    """c[ky + reach, kx + reach] = sum over cells x of a(x) · b(x + k), for |kx|, |ky| <= reach (no wrap-around)."""
    shape = [next_fast_len(n + reach + 1) for n in a.shape]
    full = irfft2(np.conj(rfft2(a, shape)) * rfft2(b, shape), shape)
    rows, cols = (np.arange(-reach, reach + 1) % n for n in shape)
    return np.rint(full[np.ix_(rows, cols)])


def find_design_offset(sem_nm, sem_centers, sem_sizes, design_nm, design_centers, design_sizes,
                       search_nm: float, cell_nm: float, min_fraction: float) -> DesignOffset:
    """The shift (design − SEM, up to search_nm in x and y) that lines up the SEM contacts with the design.

    sem_nm, design_nm: (N, 2) contacts, mask nm (duplicates in overlaps are fine); *_centers, *_sizes: the tile
    boxes they come from. Each shift is scored by its mismatches (D50): SEM contacts inside the design's tiles
    without a design contact within 1.5 cells, plus design contacts inside the SEM's tiles without an SEM one,
    over the contacts compared; boxes shrunk by the design's typical contact spacing (contacts at a tile edge are
    dropped). Only shifts matching at least min_fraction of the SEM contacts count. cell_nm must stay below a
    third of the pitch. Raises ValueError if the field is too large for the cell or no shift matches enough.
    """
    margin = float(np.median(cKDTree(design_nm).query(design_nm, k=2)[0][:, 1]))
    boxes = np.vstack([sem_centers - sem_sizes / 2, sem_centers + sem_sizes / 2,
                       design_centers - design_sizes / 2, design_centers + design_sizes / 2, sem_nm, design_nm])
    origin = boxes.min(axis=0) - cell_nm
    columns, rows = np.ceil((boxes.max(axis=0) - origin) / cell_nm).astype(int) + 2
    if rows * columns > MAX_CELLS:
        raise ValueError(f"Realignment: the field needs {rows} x {columns} cells of {cell_nm} nm; too large")
    shape = (rows, columns)
    sem = _points_raster(sem_nm, origin, cell_nm, shape)
    design = _points_raster(design_nm, origin, cell_nm, shape)
    sem_boxes = _boxes_raster(sem_centers, sem_sizes, margin, origin, cell_nm, shape)
    design_boxes = _boxes_raster(design_centers, design_sizes, margin, origin, cell_nm, shape)
    reach = int(np.ceil(search_nm / cell_nm))
    # Per shift: contacts of each set inside the other's boxes, and how many of those have a partner nearby.
    sem_inside = _correlation(sem, design_boxes, reach)
    sem_matched = _correlation(sem, design_boxes * maximum_filter(design, size=3), reach)
    design_inside = _correlation(sem_boxes, design, reach)
    design_matched = _correlation(sem_boxes * maximum_filter(sem, size=3), design, reach)
    compared = sem_inside + design_inside
    fraction = (compared - sem_matched - design_matched) / np.maximum(compared, 1)
    matched = _correlation(sem, maximum_filter(design, size=3), reach)
    fraction[matched < min_fraction * sem.sum()] = np.inf
    if not np.isfinite(fraction).any():
        raise ValueError(f"Realignment: no shift up to {search_nm / 1000:g} um matches {min_fraction:.0%} of the "
                         f"SEM contacts to the design")

    steps = np.arange(-reach, reach + 1)
    ky, kx = np.meshgrid(steps, steps, indexing="ij")
    order = np.lexsort(((kx ** 2 + ky ** 2).ravel(), -matched.ravel(), fraction.ravel()))  # fewest mismatches first
    best = order[0]
    # The dilation makes a plateau of equal scores around the true shift: take the shift on it where the most
    # contacts fall in exactly the same cell.
    plateau = (np.maximum(np.abs(kx - kx.flat[best]), np.abs(ky - ky.flat[best])) <= 2) &         (fraction <= fraction.flat[best])
    best = int(np.argmax(np.where(plateau, _correlation(sem, design, reach), -1)))
    far = np.maximum(np.abs(kx - kx.flat[best]), np.abs(ky - ky.flat[best])).ravel() > 3
    others = order[far[order] & np.isfinite(fraction.ravel()[order])]
    shift = np.array([kx.flat[best], ky.flat[best]]) * cell_nm
    if len(others):
        second = others[0]
        runner_up = np.array([kx.flat[second], ky.flat[second]]) * cell_nm
        runner_fraction = float(fraction.flat[second])
        tied = (runner_fraction - fraction.flat[best]) * compared.flat[best] < 2  # as matching.near_ties
    else:
        runner_up, runner_fraction, tied = np.full(2, np.nan), np.nan, False

    # Refine to below one cell: the mean design − SEM of the contacts paired at that shift.
    distance, nearest = cKDTree(design_nm).query(sem_nm + shift, distance_upper_bound=1.5 * cell_nm)
    paired = np.isfinite(distance)
    if paired.any():
        shift = shift + (design_nm[nearest[paired]] - (sem_nm[paired] + shift)).mean(axis=0)
    return DesignOffset(shift, int(matched.flat[best]), float(fraction.flat[best]), runner_up, runner_fraction,
                        bool(tied))


def merge_close(points: np.ndarray, radius_nm: float) -> np.ndarray:
    """One point (the mean) per group of points linked by distances below radius_nm."""
    if len(points) == 0:
        return points.reshape(0, 2)
    pairs = cKDTree(points).query_pairs(radius_nm, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
    _, group = connected_components(graph, directed=False)
    counts = np.bincount(group)
    return np.column_stack([np.bincount(group, points[:, c]) / counts for c in range(2)])


def realign_tiles(tiles: list[TileResult], settings, log) -> list[TileResult]:
    """The tiles realigned to the design (module docstring): each tile's centre moved by the offset, its design
    the merged design contacts inside its shifted field of view (no .oas of its own any more: oas_path None).
    A tile with no design contact there is flagged (design.ok False). settings: analysis.Settings."""
    s = settings
    centers = np.array([t.center_nm for t in tiles])
    fovs = np.array([t.fov_nm for t in tiles])
    sem_points = [t.sem_points_nm(s.stitch_refined) for t in tiles]
    stitch = stitch_tiles(sem_points, centers, fovs, s.max_match_nm, s.outlier_factor, s.min_matched,
                          label="SEM, realignment")
    placed = ~np.isnan(stitch.corrections[:, 0])
    sem = np.vstack([p + c for p, c, ok in zip(sem_points, stitch.corrections, placed) if ok])
    with_design = np.array([len(t.design.centers) > 0 for t in tiles])
    design = np.vstack([t.design_points_nm() for t in tiles if len(t.design.centers)])
    cell = 2 * s.design_tolerance_nm
    log(f"Realignment: SEM of {placed.sum()} of {len(tiles)} stitched tiles, design of {with_design.sum()} .oas "
        f"files; offsets up to {s.realign_search_um:g} um in {cell:g} nm cells")
    offset = find_design_offset(sem, (centers + stitch.corrections)[placed], fovs[placed], design,
                                centers[with_design], fovs[with_design], s.realign_search_um * 1000, cell,
                                s.min_match_fraction)
    dx, dy = offset.shift_nm
    log(f"Realignment: design - SEM offset ({dx:+.1f}, {dy:+.1f}) nm, {offset.matched} SEM contacts matched, "
        f"mismatch fraction {offset.mismatch_fraction:.3f}; next best ({offset.runner_up_nm[0]:+.1f}, "
        f"{offset.runner_up_nm[1]:+.1f}) nm, {offset.runner_up_fraction:.3f}")
    if offset.tied:
        log("!! Realignment: the next best offset fits as well (periodic pattern?): the offset is not certain")

    merged = merge_close(design, s.group_radius_nm)
    realigned = []
    for tile in tiles:
        center = tile.center_nm + offset.shift_nm
        local = merged[np.all(np.abs(merged - center) < tile.fov_nm / 2, axis=1)] - center
        realigned.append(replace(tile, center_nm=center, oas_path=None,
                                 design=DesignContacts(local, False, np.nan, ok=len(local) > 0)))
    return realigned
