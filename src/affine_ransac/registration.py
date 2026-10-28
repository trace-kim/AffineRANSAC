"""Registration error: stitched SEM contact centres vs stitched design centres (docs/SPEC.md S6, S8).

v1 reports the RAW error: no affine is removed yet (the RANSAC global affine, S7, comes later),
so a common offset, rotation or scale shows up as a pattern in the error map.

1. design_errors(): one OBSERVATION per contact per tile. Both sides are placed with their own
   stitching corrections: the SEM tiles (stitch_tiles) and the per-tile design files
   (stitch_design), which can be offset against each other. Each tile's SEM contacts are matched
   to its own design contacts (mutual nearest, within a gate), so a contact in an overlap gets one
   observation per tile that sees it (2 in a strip, up to 4 in a corner).
   design_errors_after_stitching(): the same, each tile's pairs chosen where the stitching put it, on the row its
   neighbours pair on (D143), so that tiles the stitching moved whole pitches cannot pair a row off.
2. merge_observations(): one entry per physical CONTACT. Observations whose stitched design
   positions lie within a small radius (and come from different tiles) are the same contact;
   their SEM and design positions are averaged and their spread is kept (SPEC S5).

Tiles that cannot be measured are listed with the reason and a warning, never dropped silently.
Sign: SEM − design (tentative, SPEC §13-8). It is set only in registration_error().
"""

import warnings
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree

from affine_ransac.matching import (choose_shift, consistent_choice, match_points_tree, match_with_shift, near_ties,
                                    scored_shifts)
from affine_ransac.stitching import apply_correction, connected_groups

# design_errors_after_stitching: a tile following its neighbours must pair at least this share of its SEM contacts
# (fewer pair by chance where its image does not line up with theirs, on a pattern that nearly repeats)
MIN_FOLLOWED = 0.5


@dataclass
class DesignErrors:
    tile: np.ndarray               # (N,) tile index of each observation
    design_nm: np.ndarray          # (N, 2) design centre in this placement, mask nm
    sem_nm: np.ndarray             # (N, 2) SEM centre in this placement, mask nm
    error_nm: np.ndarray           # (N, 2) registration_error(sem_nm, design_nm), nm
    design_nominal_nm: np.ndarray  # (N, 2) design centre at the nominal placement (for grouping)
    skipped: dict                  # tile index -> reason why it was not measured (no error computed)
    # Per tile, from the matching (None for paired_errors): the SEM − design shift used to choose the pairs, at the
    # nominal placement (D86; design_errors_after_stitching: after the corrections, D143; NaN if not matched); whether
    # other shifts matched equally well
    # (tied: no missing contact or other feature decides, so the neighbours did); and the largest
    # difference from a neighbour's
    # shift (NaN without matched neighbours; about one pitch = paired one row off).
    match_shift_nm: np.ndarray | None = None      # (n_tiles, 2) nm
    ambiguous: np.ndarray | None = None           # (n_tiles,) bool
    neighbour_jump_nm: np.ndarray | None = None   # (n_tiles,) nm
    # Points without a partner at the chosen shift, where both sets are complete (NaN if not
    # matched): ~0 = the tile matched its design exactly; more = failed detections or a wrong row.
    match_mismatches: np.ndarray | None = None    # (n_tiles,)
    # Per observation, its row in its tile's SEM / design points (as passed in), e.g. to look up values
    # measured per tile (merge_values).
    sem_index: np.ndarray | None = None     # (N,)
    design_index: np.ndarray | None = None  # (N,)


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
    search_nm: float = 100.0,
    tolerance_nm: float = 10.0,
    neighbours: list[tuple[int, int]] | None = None,
    min_extra: float = 2.0,
    pair_points: list[np.ndarray] | None = None,
    follow_stitching: bool = False,
) -> DesignErrors:
    """Raw registration error of every contact found in both the stitched SEM and the design.

    sem_points, design_points: per tile, SEM / design centres in mask nm at the nominal placement.
    sem_corrections, design_corrections: per tile, the correction applied to its SEM / design points:
    shifts (n_tiles, 2), e.g. one placement of stitching.placement_corrections, or affines
    (n_tiles, 3, 3), e.g. pipeline.stitch_tiles_affine (stitching.apply_correction; NaN = tile not
    stitched).
    design_ok: per tile, whether its design tone matched the SEM (DesignContacts.ok).
    Matching at the NOMINAL placement (D86): each tile's points before its corrections, so search_nm only
    has to cover one tile's offset against its own design (its stage error), not what the stitching adds
    far from the centre; the corrections then move the pairs. pair_points: per tile, the points the pairs
    are chosen on, if not sem_points: the same contacts in the same order, e.g. the nominal SEM points of
    a set whose points already hold a placement (stripe drift corrected). Each tile's candidate shifts
    (matching.scored_shifts, up to search_nm, pairs within tolerance_nm after the shift), scored by their
    mismatches (D50). A candidate with at least
    min_extra fewer mismatches than the others wins outright. Tied tiles: with neighbours ((i, j)
    pairs of overlapping tiles, e.g. pipeline.tile_neighbours) the one agreeing with the
    neighbours (matching.consistent_choice, D49); without, the smallest shift (choose_shift). The shift only chooses the pairs: each error is the full
    stitched SEM − design (D48, option a).
    follow_stitching: tied tiles choose by their shift AFTER the corrections (the candidate shift plus what the
    corrections move the SEM against the design, at the tile's design centre), which agrees between neighbours on a
    consistent stitching, instead of by the shift at the nominal placement, which on a repeating pattern differs
    between neighbours by whole pitches once their images lie more than half a pitch apart (stitching image by image,
    D133). Clear winners keep their candidate either way; the neighbour jumps then compare the shifts after the
    corrections too.
    """
    n = len(sem_points)
    skipped, stitched = _stitched_tiles(sem_points, design_points, sem_corrections, design_corrections, design_ok)
    candidates = [[] for _ in range(n)]
    for k in stitched:
        # Pairs chosen at the nominal placement, where a tile is only its own stage error away
        # from its design; the corrections move the pairs, they do not change them (D86).
        candidates[k] = scored_shifts(design_points[k], sem_points[k] if pair_points is None else pair_points[k],
                                      search_nm, tolerance_nm)
        if not candidates[k]:
            skipped[k] = "no SEM contact matched the design"
    skipped = dict(sorted(skipped.items()))

    moved = np.zeros((n, 2))  # what the corrections move each tile's SEM against its design (follow_stitching)
    if follow_stitching:
        for k in stitched:
            if candidates[k]:
                centre = design_points[k].mean(axis=0, keepdims=True)
                moved[k] = (apply_correction(centre, sem_corrections[k])
                            - apply_correction(centre, design_corrections[k]))[0]
    choosing = [[replace(m, shift=m.shift + moved[k]) for m in c] for k, c in enumerate(candidates)]

    ambiguous = np.array([bool(c) and len(near_ties(c, min_extra)) > 1 for c in candidates])
    if neighbours is not None:
        choice = consistent_choice(choosing, neighbours, min_extra)
    else:
        choice = [choose_shift(c, min_extra)[0] if c else -1 for c in choosing]

    tiles, designs, nominals, sems, indices = [], [], [], [], []
    match_shift = np.full((n, 2), np.nan)
    mismatches = np.full(n, np.nan)
    for k in range(n):
        if not candidates[k]:
            continue
        match = candidates[k][choice[k]]
        stitched_sem, stitched_design = stitched[k]
        match_shift[k], mismatches[k] = match.shift, match.mismatches
        tiles.append(np.full(len(match.ia), k))
        designs.append(stitched_design[match.ia])
        nominals.append(design_points[k][match.ia])
        sems.append(stitched_sem[match.ib])
        indices.append((match.ib, match.ia))

    errors = _collect(tiles, designs, nominals, sems, indices, skipped)
    errors.match_shift_nm, errors.ambiguous = match_shift, ambiguous
    errors.neighbour_jump_nm = neighbour_jumps(match_shift + moved, neighbours or [])
    errors.match_mismatches = mismatches
    return errors


def _stitched_tiles(sem_points, design_points, sem_corrections, design_corrections, design_ok) -> tuple[dict, dict]:
    """(skipped, stitched): per tile that cannot be measured, why; per other tile, its (SEM, design) points with their
    corrections applied."""
    skipped, stitched = {}, {}
    for k, (sem, design) in enumerate(zip(sem_points, design_points)):
        if not design_ok[k]:
            skipped[k] = "design tone not matched (flagged)"
        elif np.isnan(sem_corrections[k]).any():
            skipped[k] = "SEM tile not stitched"
        elif np.isnan(design_corrections[k]).any():
            skipped[k] = "design tile not stitched"
        else:
            stitched[k] = (apply_correction(sem, sem_corrections[k]), apply_correction(design, design_corrections[k]))
    return skipped, stitched


def design_errors_after_stitching(
    sem_points: list[np.ndarray],
    design_points: list[np.ndarray],
    sem_corrections: np.ndarray,
    design_corrections: np.ndarray,
    design_ok,
    neighbours: list[tuple[int, int]],
    search_nm: float = 100.0,
    tolerance_nm: float = 10.0,
    radius_nm: float = 5.0,
) -> DesignErrors:
    """design_errors with each tile's pairs chosen AFTER the corrections, on the row its neighbours pair on (D143).

    Arguments as design_errors (design_corrections: shifts, as merge_observations needs); neighbours: (i, j) pairs of
    overlapping tiles (pipeline.tile_neighbours); radius_nm: design contacts of two files closer than this are the
    same contact (as merge_observations).
    On a repeating pattern the stitching can move an image whole pitches from its stage position, further than
    search_nm, while its overlaps still line up. A tile matched to its design on its own, at the nominal placement,
    then pairs a row away from its neighbours: an abrupt step of a pitch in the errors. So the tiles are paired one at
    a time where the stitching put them, against their own design and their neighbours' (an image moved off its own
    file's field still has design contacts there):
    1. A tile with paired neighbours takes the SEM − design offset of their pairs inside its own stitched image (the
       same contacts, seen by both: their median), pairs its contacts within tolerance_nm of it (mutual nearest), and
       pairs again at the median offset of those pairs. The waiting tile with the most such neighbour pairs goes first.
       The offset is followed from tile to tile, so a scale added up over many tiles, or a drift, is followed too:
       no affine is needed.
    2. Where no waiting tile has such pairs (the first tile of every group of neighbours), the one with the most
       contacts is matched on its own at the nominal placement (matching.match_with_shift: the shift search up to
       search_nm, the smallest of tied shifts), then paired again where its corrections put it, as in 1.
    A tile that pairs fewer than MIN_FOLLOWED of its SEM contacts over the design in 1 (stitched more than
    tolerance_nm off the neighbours it followed, or they off it; on a pattern that nearly repeats a few still pair by
    chance) waits for the pairs of other neighbours, and is matched on its own as in 2 if none come: a step at that
    seam, which neighbour_jump_nm shows. Contacts of an image moved out beyond every design file are not paired.
    Each error is the full stitched SEM − design of a pair; a pair with a neighbour's design contact has
    design_index -1 (no row in the tile's own design). match_shift_nm holds each tile's offset after the corrections
    and neighbour_jump_nm their largest difference between neighbours; ambiguous and match_mismatches are None.
    """
    n = len(sem_points)
    skipped, stitched = _stitched_tiles(sem_points, design_points, sem_corrections, design_corrections, design_ok)
    adjacent = [[] for _ in range(n)]
    for i, j in neighbours:
        adjacent[i].append(j)
        adjacent[j].append(i)
    image = {k: (s.min(axis=0), s.max(axis=0)) for k, (s, _) in stitched.items() if len(s)}  # stitched SEM box
    waiting = np.zeros(n, bool)
    waiting[list(stitched)] = True
    contacts = np.array([min(len(s), len(d)) for s, d in zip(sem_points, design_points)])
    support = np.zeros(n)              # neighbour pairs inside each tile's stitched image ...
    evidence = [[] for _ in range(n)]  # ... and their SEM − design differences
    found, shifts = {}, np.full((n, 2), np.nan)

    while waiting.any():
        first = not (support[waiting] > 0).any()  # no waiting tile has a paired neighbour: a new group starts
        k = int(np.argmax(np.where(waiting, contacts if first else support, -1)))
        waiting[k] = False
        sem = stitched[k][0]
        if first:  # matched on its own, at the nominal placement
            match, _ = match_with_shift(design_points[k], sem_points[k], search_nm, tolerance_nm)
            own = stitched[k][1]
            offset = np.median(sem[match.ib] - own[match.ia], axis=0) if match.score else None
        else:
            offset = np.median(np.concatenate(evidence[k]), axis=0)
        design, rows = _design_around(k, stitched, adjacent, radius_nm)
        ia, ib = _paired_at(design, sem, offset, tolerance_nm) if offset is not None else ([], [])
        if not first and not _followed(sem - offset, design, len(ia)):  # stitched off the neighbours it followed (or
            support[k], evidence[k] = 0, []                               # they off it): again with others' pairs,
            waiting[k] = True                                             # or on its own
            continue
        if len(ia) == 0:
            skipped[k] = "no SEM contact matched the design"
            continue
        found[k] = design[ia], rows[ia], ib
        difference = sem[ib] - design[ia]
        shifts[k] = np.median(difference, axis=0)
        for u in adjacent[k]:  # the same contacts, seen by a waiting neighbour: its evidence
            if waiting[u] and u in image:
                low, high = image[u]
                inside = np.all((sem[ib] >= low) & (sem[ib] <= high), axis=1)
                if inside.any():
                    evidence[u].append(difference[inside])
                    support[u] += inside.sum()

    tiles, designs, nominals, sems, indices = [], [], [], [], []
    for k in sorted(found):
        design, rows, ib = found[k]
        own = rows >= 0
        nominal = design - design_corrections[k]  # a neighbour's contact: where this tile's design would have it
        nominal[own] = design_points[k][rows[own]]
        tiles.append(np.full(len(ib), k))
        designs.append(design)
        nominals.append(nominal)
        sems.append(stitched[k][0][ib])
        indices.append((ib, rows))
    errors = _collect(tiles, designs, nominals, sems, indices, dict(sorted(skipped.items())))
    errors.match_shift_nm = shifts
    errors.neighbour_jump_nm = neighbour_jumps(shifts, neighbours)
    return errors


def _followed(sem: np.ndarray, design: np.ndarray, paired: int) -> bool:
    """Whether `paired` pairs are at least MIN_FOLLOWED of the SEM points (moved by the offset they were paired at)
    over the design's box: an image moved out beyond every design file only counts the contacts with design under
    them."""
    if paired == 0:
        return False
    over = np.all((sem >= design.min(axis=0)) & (sem <= design.max(axis=0)), axis=1)
    return paired >= MIN_FOLLOWED * over.sum()


def _design_around(k: int, stitched: dict, adjacent: list, radius_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """Tile k's stitched design and its neighbours' design contacts beyond it: (points, row of each in tile k's design
    points, -1 for a neighbour's contact). A neighbour's contact closer than radius_nm to one already in is the same
    contact."""
    points = stitched[k][1]
    rows = np.arange(len(points))
    for u in adjacent[k]:
        if u not in stitched:
            continue
        other = stitched[u][1]
        if len(points) and len(other):
            other = other[cKDTree(points).query(other)[0] > radius_nm]
        points = np.vstack([points, other])
        rows = np.concatenate([rows, np.full(len(other), -1)])
    return points, rows


def _paired_at(design: np.ndarray, sem: np.ndarray, offset: np.ndarray, tolerance_nm: float):
    """(design rows, SEM rows) of the pairs of design + offset and sem within tolerance_nm (mutual nearest), paired
    again at the median SEM − design of those pairs."""
    ia, ib = match_points_tree(design + offset, sem, tolerance_nm)
    if len(ia):
        ia, ib = match_points_tree(design + np.median(sem[ib] - design[ia], axis=0), sem, tolerance_nm)
    return ia, ib


def neighbour_jumps(shifts: np.ndarray, neighbours: list[tuple[int, int]]) -> np.ndarray:
    """(n_tiles,) largest distance of each tile's shift from a neighbour's shift, nm (a seam where
    tiles pair one row off shows about one pitch on both sides; a median would hide it where most
    neighbours lie on the same side). NaN for tiles without a shift or without neighbours that
    have one."""
    adjacent = [[] for _ in range(len(shifts))]
    for i, j in neighbours:
        adjacent[i].append(j)
        adjacent[j].append(i)
    jumps = np.full(len(shifts), np.nan)
    for k, others in enumerate(adjacent):
        others = [shifts[j] for j in others if not np.isnan(shifts[j, 0])]
        if others and not np.isnan(shifts[k, 0]):
            jumps[k] = np.linalg.norm(np.array(others) - shifts[k], axis=1).max()
    return jumps


def offset_steps(errors: DesignErrors, neighbours: list[tuple[int, int]], step_nm: float) -> list[tuple[int, int]]:
    """Neighbouring tiles (i, j) whose SEM − design offsets (match_shift_nm) differ by more than step_nm: a step in
    the errors between them, e.g. one of them matched on its own (design_errors_after_stitching). [] without
    offsets (paired_errors)."""
    if errors.match_shift_nm is None:
        return []
    shifts = errors.match_shift_nm
    return [(i, j) for i, j in neighbours if np.linalg.norm(shifts[i] - shifts[j]) > step_nm]  # NaN: not paired


def paired_errors(
    sem_points: list[np.ndarray],
    design_points: list[np.ndarray],
    sem_corrections: np.ndarray,
    design_corrections: np.ndarray,
) -> DesignErrors:
    """Like design_errors, for contacts that are already paired in each tile: row k of
    sem_points[t] is the same contact as row k of design_points[t] (e.g. read_contour_csv).
    No matching and no tone check; tiles not stitched or without contacts are skipped (warned).
    Corrections: shifts or affines, as for design_errors."""
    tiles, designs, nominals, sems, indices, skipped = [], [], [], [], [], {}
    for k, (sem, design) in enumerate(zip(sem_points, design_points)):
        if len(design) == 0:
            skipped[k] = "no contacts"
        elif np.isnan(sem_corrections[k]).any():
            skipped[k] = "SEM tile not stitched"
        elif np.isnan(design_corrections[k]).any():
            skipped[k] = "design tile not stitched"
        else:
            tiles.append(np.full(len(design), k))
            designs.append(apply_correction(design, design_corrections[k]))
            nominals.append(design)
            sems.append(apply_correction(sem, sem_corrections[k]))
            indices.append((np.arange(len(sem)), np.arange(len(design))))
    return _collect(tiles, designs, nominals, sems, indices, skipped)


def _collect(tiles, designs, nominals, sems, indices, skipped) -> DesignErrors:
    """DesignErrors from per-tile lists of observations and their (SEM, design) rows; warns about the
    skipped tiles."""
    if skipped:
        warnings.warn(f"{len(skipped)} tile(s) not measured (no error computed): {skipped}", stacklevel=3)

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
        sem_index=np.concatenate([i for i, _ in indices]).astype(int) if indices else np.empty(0, int),
        design_index=np.concatenate([j for _, j in indices]).astype(int) if indices else np.empty(0, int),
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


def merge_values(merged: MergedErrors, errors: DesignErrors, sem_values: list[np.ndarray],
                 design_values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per merged contact, the mean of values measured per tile (e.g. contact sizes) over its
    observations: sem_values[t] (n_t, k) has a row per SEM point of tile t, design_values[t] per design
    point, in the order passed to design_errors / paired_errors. NaN values are left out of a mean
    (NaN if all are), as are observations without a row (index -1: paired with a neighbour's design contact,
    design_errors_after_stitching). merged must be merge_observations(errors, ...). Returns (M, k) SEM and design."""
    labels = np.repeat(np.arange(len(merged.members)), [len(m) for m in merged.members])
    rows = np.concatenate(merged.members) if merged.members else np.empty(0, int)

    def mean(values, index):
        starts = np.concatenate([[0], np.cumsum([len(v) for v in values])[:-1]])
        flat = np.concatenate(values)
        with_row = index[rows] >= 0
        observed = np.full((len(rows), flat.shape[1]), np.nan)
        observed[with_row] = flat[starts[errors.tile[rows[with_row]]] + index[rows[with_row]]]
        found = np.isfinite(observed)
        total = np.zeros((len(merged.members), flat.shape[1]))
        count = np.zeros_like(total)
        np.add.at(total, labels, np.where(found, observed, 0))
        np.add.at(count, labels, found)
        with np.errstate(invalid="ignore"):
            return np.where(count > 0, total / count, np.nan)

    return mean(sem_values, errors.sem_index), mean(design_values, errors.design_index)


def group_rows(design_y: np.ndarray, gap_nm: float = 10.0) -> np.ndarray:
    """Row index (N,) of each contact, from its design y position (nm).

    Contacts of one row share the same design y (after design stitching, rows from neighbouring
    tiles agree to well below a nm). Sorted y values start a new row wherever they jump by more
    than gap_nm, which must stay well below the row pitch. Rows are numbered from the lowest y.
    """
    order = np.argsort(design_y)
    new_row = np.diff(design_y[order]) > gap_nm
    rows = np.empty(len(design_y), int)
    rows[order] = np.concatenate([[0], np.cumsum(new_row)])
    return rows


def row_means(design_y: np.ndarray, values: np.ndarray, gap_nm: float = 10.0):
    """Mean of values (N,) or (N, k) over the contacts of each row (group_rows).

    Returns (row_y, mean, count): the mean design y of each row (nm), the mean value(s) per row
    and the number of contacts per row, from the lowest row up.
    """
    rows = group_rows(design_y, gap_nm)
    count = np.bincount(rows)
    row_y = np.bincount(rows, weights=design_y) / count
    values = np.asarray(values, dtype=float)
    columns = values.reshape(len(values), -1)
    mean = np.column_stack([np.bincount(rows, weights=c) / count for c in columns.T]).reshape(
        (len(count),) + values.shape[1:])
    return row_y, mean, count


def grid_means(points: np.ndarray, values: np.ndarray, cell_nm: float):
    """Mean of values (N, k) over the points (N, 2) in each square cell of a grid of cell_nm, whose first cell
    starts at the lowest x and y of the points.

    Returns (corner, mean, count): the grid's lower-left corner (2,) nm, the mean values per cell
    (rows, cols, k), NaN where a cell holds no point, and the points per cell (rows, cols); row 0 is the lowest.
    """
    corner = points.min(axis=0)
    index = np.floor((points - corner) / cell_nm).astype(int)
    cols, rows = index.max(axis=0) + 1
    cell = index[:, 1] * cols + index[:, 0]
    count = np.bincount(cell, minlength=rows * cols)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.column_stack([np.bincount(cell, weights=v, minlength=rows * cols) / count for v in values.T])
    return corner, mean.reshape(rows, cols, -1), count.reshape(rows, cols)


def band_means(position: np.ndarray, values: np.ndarray, width_nm: float):
    """Mean of values (N, k) over the points in each band of width_nm along one axis, the first band starting at the
    lowest position: the rows (position y) or the columns (x) of grid_means' cells.

    Returns (centre, mean, count) of the bands that hold a point: the band's centre (nm), the mean values (B, k) and
    the points per band, from the lowest band up.
    """
    start = position.min()
    band = np.floor((position - start) / width_nm).astype(int)
    count = np.bincount(band)
    held = np.flatnonzero(count)
    mean = np.column_stack([np.bincount(band, weights=v)[held] / count[held] for v in values.T])
    return start + (held + 0.5) * width_nm, mean, count[held]


def turning_points(values: np.ndarray, height: float) -> np.ndarray:
    """Indices where a profile (values in order along it) turns, with its first and last index, so that neighbouring
    indices bound its ramps. A top counts once the values fall more than height below it, a bottom once they rise
    more than height above it (a zigzag), so wiggles up to height do not end a ramp."""
    points, direction, low, high, extreme = [0], 0, 0, 0, 0  # direction: +1 rising, -1 falling, 0 not known yet
    for k in range(1, len(values)):
        if direction == 0:  # the first ramp starts at the lowest or highest value before it
            low = k if values[k] < values[low] else low
            high = k if values[k] > values[high] else high
            if values[high] - values[low] > height:
                direction = 1 if low < high else -1
                if min(low, high):
                    points.append(min(low, high))
                extreme = k
        elif (values[k] - values[extreme]) * direction > 0:  # further up (down) the ramp
            extreme = k
        elif (values[extreme] - values[k]) * direction > height:  # turned
            points.append(extreme)
            direction, extreme = -direction, k
    if direction and extreme != points[-1]:
        points.append(extreme)
    if points[-1] != len(values) - 1:
        points.append(len(values) - 1)
    return np.array(points)


def largest_ramp(values: np.ndarray, height: float) -> tuple[int, int]:
    """(first, last) index of a profile's tallest ramp (turning_points): the one whose values span the most."""
    points = turning_points(values, height)
    if len(points) < 2:
        return 0, 0
    return max(zip(points[:-1], points[1:]), key=lambda ends: np.ptp(values[ends[0]:ends[1] + 1]))


def noise_sigma(values: np.ndarray) -> float:
    """Noise of a profile's values: robust σ (1.4826 × MAD) of the steps between neighbours, / √2 (a step is the
    difference of two noisy values). A slope, or a few jumps, hardly change it."""
    steps = np.diff(values)
    if len(steps) == 0:
        return 0.0
    return 1.4826 * float(np.median(np.abs(steps - np.median(steps)))) / np.sqrt(2)


def row_pitch(design_y: np.ndarray, y: np.ndarray, gap_nm: float = 10.0):
    """Row-to-row pitch along y (nm) of one set of positions y (N,), rows grouped by design y.

    Each row's position is the mean y of its contacts (row_means); the pitch is the difference
    between neighbouring rows, from the lowest up. Returns (mid_y, pitch): the mean of the two rows'
    DESIGN y (nm, so every set is plotted at the same places) and the pitch, both (R − 1,).
    """
    row_design, _, _ = row_means(design_y, design_y, gap_nm)
    _, row_y, _ = row_means(design_y, y, gap_nm)
    return (row_design[1:] + row_design[:-1]) / 2, np.diff(row_y)
