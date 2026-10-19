"""Realignment of an image folder whose images all sit at one wrong offset from their design (docs/SPEC.md D91, D93).

When the SEM alignment went wrong, every image shows the design some µm away from where the metadata says it is,
so no image matches its own .oas. The user's method: stitch the SEM tiles on their own, stitch the .oas files on
their own and put their contacts together, find the one offset that lines the two up, then give every tile that
offset and the design contacts inside its shifted field of view. The analysis then runs as usual on the realigned
tiles. Used only when settings.realign is on. Normal tone only: tone-reversed files are not handled (user).

- find_design_offset(): the offset, of any size: every shift at which the images still overlap the .oas files is
  scored at once (FFT correlations of point rasters) by its MISMATCHES where both sets are complete, as for one tile
  (matching.scored_shifts, D50), allowing for the counting noise of the contacts compared (D104).
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

from affine_ransac.pipeline import DesignContacts, TileResult, stitch_design, stitch_tiles

MAX_CELLS = 2e8  # size limit of the correlation results (the shifts searched; ~800 MB per float32 array)
# Counting noise allowed for in the mismatches, in standard deviations: thousands of shifts are compared, so a
# 3-sigma difference happens among them by chance.
NOISE_SIGMA = 4.0


@dataclass
class DesignOffset:
    shift_nm: np.ndarray       # (2,) design − SEM: the SEM contacts + shift_nm lie on the design
    matched: int               # SEM contacts with a design contact within 1.5 cells at that shift
    mismatch_fraction: float   # points without a partner / points compared, where both sets are complete
    # Other shifts more than 2 cells away that fit as well and compare at least half as many contacts: the pattern
    # repeats (they give the same errors, the design being ideal); and the one of them pairing the most contacts.
    others: int
    other_nm: np.ndarray       # (2,) NaN if there is none


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


def _correlation(a: np.ndarray, b: np.ndarray, reach: tuple[int, int]) -> np.ndarray:
    """c[ky + reach_y, kx + reach_x] = sum over cells x of a(x) · b(x + k), for |ky| <= reach_y, |kx| <= reach_x
    (zero padding, so no wrap-around)."""
    shape = [next_fast_len(n + r + 1) for n, r in zip(a.shape, reach)]
    full = irfft2(np.conj(rfft2(a, shape)) * rfft2(b, shape), shape)
    rows, cols = (np.arange(-r, r + 1) % n for n, r in zip(shape, reach))
    return np.rint(full[np.ix_(rows, cols)])


def _same_cell(sem_nm, design_nm, origin, cell_nm, shift_cells) -> int:
    """SEM contacts that land in a cell holding a design contact when shifted by whole cells (x, y)."""
    sem = np.floor((sem_nm - origin) / cell_nm).astype(np.int64) + np.asarray(shift_cells, np.int64)
    design = np.floor((design_nm - origin) / cell_nm).astype(np.int64)
    key = lambda cells: cells[:, 0] * 4_000_000_000 + cells[:, 1]  # one integer per cell
    return int(np.isin(key(sem), key(design)).sum())


def find_design_offset(sem_nm, sem_centers, sem_sizes, design_nm, design_centers, design_sizes,
                       cell_nm: float, min_compared: int) -> DesignOffset | None:
    """The shift (design − SEM) that lines up the SEM contacts with the design, of any size: every shift at which
    the SEM tiles and the design tiles still overlap is searched.

    sem_nm, design_nm: (N, 2) contacts, mask nm (duplicates in overlaps are fine); *_centers, *_sizes: the tile
    boxes they come from. Each shift is scored by its mismatches (D50): SEM contacts inside the design's tiles
    without a design contact within 1.5 cells, plus design contacts inside the SEM's tiles without an SEM one, over
    the contacts compared; boxes shrunk by the design's typical contact spacing (contacts at a tile edge are
    dropped). Only shifts comparing at least min_compared contacts count. The best fit has the lowest fraction with
    NOISE_SIGMA of counting noise added (so a shift comparing few contacts cannot be it by chance); a shift fits as
    well unless its fraction is higher by more than NOISE_SIGMA standard errors of the difference. On a repeating
    pattern every whole-pitch shift of the true offset fits as well, and as the design is ideal they all give the
    same errors (only where on the mask they lie differs, D105): of the shifts that fit as well, the one pairing the
    most contacts is taken, so that the most images are measured. None if no shift compares min_compared contacts.
    cell_nm must stay below a third of the pitch. ValueError if the field is too large for the cell (MAX_CELLS).
    """
    margin = float(np.median(cKDTree(design_nm).query(design_nm, k=2)[0][:, 1]))
    boxes = np.vstack([sem_centers - sem_sizes / 2, sem_centers + sem_sizes / 2,
                       design_centers - design_sizes / 2, design_centers + design_sizes / 2, sem_nm, design_nm])
    origin = boxes.min(axis=0) - cell_nm
    columns, rows = np.ceil((boxes.max(axis=0) - origin) / cell_nm).astype(int) + 2
    reach = (rows, columns)  # every shift at which the two can still overlap: no limit on the offset
    if (2 * rows + 1) * (2 * columns + 1) > MAX_CELLS:
        raise ValueError(f"Realignment: the field needs {rows} x {columns} cells of {cell_nm:g} nm; too large")
    shape = (rows, columns)
    sem = _points_raster(sem_nm, origin, cell_nm, shape)
    design = _points_raster(design_nm, origin, cell_nm, shape)
    sem_boxes = _boxes_raster(sem_centers, sem_sizes, margin, origin, cell_nm, shape)
    design_boxes = _boxes_raster(design_centers, design_sizes, margin, origin, cell_nm, shape)
    # Per shift: contacts of each set inside the other's boxes, minus those with a partner nearby = mismatches.
    unmatched = _correlation(sem, design_boxes * (1 - maximum_filter(design, size=3)), reach)
    unmatched += _correlation(sem_boxes * (1 - maximum_filter(sem, size=3)), design, reach)
    compared = _correlation(sem, design_boxes, reach) + _correlation(sem_boxes, design, reach)
    compared = np.maximum(compared, 1, out=compared)
    # The best fit: the lowest fraction with NOISE_SIGMA added to its mismatch count (Poisson upper bound: the square
    # root of a count has a standard deviation of 1/2, so 0 observed may well be 9), so a shift comparing few
    # contacts cannot be it by chance.
    pessimistic = (np.sqrt(unmatched + 1) + NOISE_SIGMA / 2) ** 2 / compared
    pessimistic[compared < min_compared] = np.inf
    if not np.isfinite(pessimistic.min()):
        return None
    best = np.unravel_index(int(np.argmin(pessimistic)), pessimistic.shape)
    u0, n0 = unmatched[best], compared[best]
    # A shift fits as well unless its fraction is higher than the best's by more than NOISE_SIGMA standard errors of
    # the difference (two fractions, pooled).
    pooled = (unmatched + u0) / (compared + n0)
    error = np.sqrt(pooled * (1 - pooled) * (1 / compared + 1 / n0))
    fits = np.isfinite(pessimistic) & (unmatched / compared - u0 / n0 <= NOISE_SIGMA * error)
    del pooled, error, pessimistic

    # Of the shifts that fit (a repeating pattern: the true offset plus whole pitches), the one where the most SEM
    # contacts have a design contact within one cell (the most measured); equal ones: the smallest shift.
    ky, kx = np.arange(-rows, rows + 1), np.arange(-columns, columns + 1)  # the shifts, in cells
    pair_count = np.where(fits, _correlation(sem, maximum_filter(design, size=3), reach), -1)
    tops = np.argwhere(pair_count == pair_count.max())
    iy, ix = min(tops, key=lambda t: ky[t[0]] ** 2 + kx[t[1]] ** 2)
    around = (slice(max(iy - 2, 0), iy + 3), slice(max(ix - 2, 0), ix + 3))  # it and its dilation plateau
    # The dilation (partners within one cell) makes a plateau of equally good shifts around the true one: take the
    # shift on it where the most SEM contacts fall in the very cell of a design contact.
    plateau = [(iy, ix)] + [tuple(p) for p in np.argwhere(fits[around]) + [around[0].start, around[1].start]]
    iy, ix = max(plateau, key=lambda p: _same_cell(sem_nm, design_nm, origin, cell_nm, (kx[p[1]], ky[p[0]])))
    fraction = float(unmatched[iy, ix] / compared[iy, ix])
    pair_count[around] = -1  # the plateau lies within 2 cells of the shift first taken
    pair_count[compared < compared[iy, ix] / 2] = -1  # others that compare much less are not worth naming
    others = int((pair_count >= 0).sum())
    second = np.unravel_index(int(np.argmax(pair_count)), pair_count.shape)
    other = np.array([kx[second[1]], ky[second[0]]]) * cell_nm if pair_count[second] >= 0 else np.full(2, np.nan)
    shift = np.array([kx[ix], ky[iy]]) * cell_nm

    # Refine to below one cell: the mean design − SEM of the contacts paired at that shift.
    distance, nearest = cKDTree(design_nm).query(sem_nm + shift, distance_upper_bound=1.5 * cell_nm)
    paired = np.isfinite(distance)
    if paired.any():
        shift = shift + (design_nm[nearest[paired]] - (sem_nm[paired] + shift)).mean(axis=0)
    return DesignOffset(shift, int(paired.sum()), fraction, others, other)


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
    the merged design contacts inside its shifted field of view, and oas_center_nm where its own .oas file sits
    (stitched). A tile with no design contact there is flagged (design.ok False). settings: analysis.Settings."""
    s = settings
    centers = np.array([t.center_nm for t in tiles])
    fovs = np.array([t.fov_nm for t in tiles])
    sem_points = [t.sem_points_nm() for t in tiles]
    stitch = stitch_tiles(sem_points, centers, fovs, s.max_match_nm, s.outlier_factor, s.min_matched,
                          label="SEM, realignment")
    placed = ~np.isnan(stitch.corrections[:, 0])
    sem = np.vstack([p + c for p, c, ok in zip(sem_points, stitch.corrections, placed) if ok])

    # The .oas files stitched like the SEM (they can be offset against each other), then put together.
    design_stitch = stitch_design(tiles, s.design_max_match_nm, s.outlier_factor, s.min_matched)
    file_shift = np.nan_to_num(design_stitch.corrections)  # a file that could not be stitched stays nominal
    with_design = np.array([len(t.design.centers) > 0 for t in tiles])
    design = np.vstack([t.design_points_nm() + file_shift[k] for k, t in enumerate(tiles) if with_design[k]])
    cell = 2 * s.design_tolerance_nm
    image_contacts = int(np.median([len(t.design.centers) for t in tiles if len(t.design.centers)]))
    log(f"Realignment: SEM of {placed.sum()} of {len(tiles)} stitched tiles, design of {with_design.sum()} .oas "
        f"files ({len(design_stitch.unplaced)} not stitched, left nominal); {cell:g} nm cells; every offset at which "
        f"at least {image_contacts} contacts (one image's worth) are compared")
    offset = find_design_offset(sem, (centers + stitch.corrections)[placed], fovs[placed], design,
                                centers[with_design], fovs[with_design], cell, image_contacts)
    if offset is None:
        raise ValueError(f"Realignment: at no offset do the images have {image_contacts} contacts (one image's "
                         f"worth) in common with the .oas files: they lie outside the area the .oas files cover")
    dx, dy = offset.shift_nm / 1000
    log(f"Realignment: design - SEM offset ({dx:+.3f}, {dy:+.3f}) µm, {offset.matched} SEM contacts matched, "
        f"mismatch fraction {offset.mismatch_fraction:.4f}")
    if offset.mismatch_fraction > 1 - s.min_match_fraction:
        raise ValueError(f"Realignment: no offset fits: at the best, ({dx:+.3f}, {dy:+.3f}) µm, "
                         f"{offset.mismatch_fraction:.0%} of the contacts compared have no partner (more than "
                         f"1 - min_match_fraction = {1 - s.min_match_fraction:g}): the images do not show the pattern "
                         f"of the .oas files")
    if offset.others:
        ox, oy = offset.other_nm / 1000
        log(f"Realignment: {offset.others} other offset(s) fit as well, e.g. ({ox:+.3f}, {oy:+.3f}) µm: the pattern "
            f"repeats. They give the same errors (the design is ideal); only where on the mask the errors lie is not "
            f"known. The offset above pairs the most contacts")

    merged = merge_close(design, s.group_radius_nm)
    realigned = []
    for k, tile in enumerate(tiles):
        center = tile.center_nm + offset.shift_nm
        local = merged[np.all(np.abs(merged - center) < tile.fov_nm / 2, axis=1)] - center
        realigned.append(replace(tile, center_nm=center, oas_center_nm=tile.center_nm + file_shift[k],
                                 design=DesignContacts(local, False, np.nan, ok=len(local) > 0)))
    return realigned
