"""Global stitching solve, translation only (first version of docs/SPEC.md S5).

Each tile k gets a correction t_k (nm) that is added to its nominal placement. For a pair (i, j)
the measured shift d_ij is B − A for the same contacts at nominal placement (overlap_fit). After
correction the two measurements coincide when

    (a + t_i) = (b + t_j)   ->   t_i − t_j = d_ij

All pairs together are solved by least squares. A shift common to all tiles cannot be seen in
the overlaps (gauge freedom), so the corrections are fixed to average zero; the later fit to
the design (S7) absorbs any common shift.

Only tiles connected to each other through pairs can be placed relative to each other. The
largest connected group is solved; every other tile gets a NaN correction (never a silent 0),
so a stitching failure cannot pass unnoticed into later steps.
"""

import numpy as np


def connected_groups(n_tiles: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Groups of tiles linked through pairs, largest first (ties: lowest tile index first)."""
    neighbours = [[] for _ in range(n_tiles)]
    for i, j in pairs:
        neighbours[i].append(j)
        neighbours[j].append(i)

    group_of = [-1] * n_tiles
    groups = []
    for start in range(n_tiles):
        if group_of[start] >= 0:
            continue
        group, todo = [], [start]
        group_of[start] = len(groups)
        while todo:
            k = todo.pop()
            group.append(k)
            for m in neighbours[k]:
                if group_of[m] < 0:
                    group_of[m] = len(groups)
                    todo.append(m)
        groups.append(sorted(group))
    return sorted(groups, key=lambda g: (-len(g), g[0]))


def solve_tile_shifts(n_tiles: int, pairs: list[tuple[int, int]], shifts: np.ndarray, weights=None) -> np.ndarray:
    """Per-tile corrections (n_tiles, 2) in nm.

    pairs: (i, j) tile indices; shifts: (P, 2) measured B − A shift d_ij for each pair;
    weights: optional (P,) weight per pair (e.g. number of inlier contacts).
    The largest connected group of tiles is solved, with corrections averaging zero over that
    group. Tiles outside it cannot be placed: their correction is NaN.
    """
    shifts = np.asarray(shifts, dtype=float).reshape(-1, 2)
    weights = np.ones(len(pairs)) if weights is None else np.asarray(weights, dtype=float)
    placed = connected_groups(n_tiles, pairs)[0] if n_tiles else []
    in_group = np.zeros(n_tiles, bool)
    in_group[placed] = True
    use = [p for p, (i, j) in enumerate(pairs) if in_group[i]]  # a pair is inside or outside the group

    # One equation per pair: t_i − t_j = d_ij, plus one gauge equation: sum of t over the group = 0.
    rows = np.zeros((len(use) + 1, n_tiles))
    for row, p in enumerate(use):
        i, j = pairs[p]
        rows[row, i], rows[row, j] = 1.0, -1.0
    rows[-1, in_group] = 1.0
    right = np.vstack([shifts[use], [0.0, 0.0]])
    sqrt_w = np.sqrt(np.append(weights[use], 1.0))[:, None]

    corrections = np.full((n_tiles, 2), np.nan)
    solved, *_ = np.linalg.lstsq((rows * sqrt_w)[:, in_group], right * sqrt_w, rcond=None)
    corrections[in_group] = solved
    return corrections


def fix_tile(corrections: np.ndarray, k: int) -> np.ndarray:
    """The same stitching with tile k kept at its nominal position (its correction becomes 0).

    Moving every tile by the same amount does not change how the tiles fit together, so this is
    an equally valid solution, convenient to compare against one tile's own placement.
    """
    return corrections - corrections[k]


def first_stitched_tile(*corrections: np.ndarray) -> int | None:
    """Index of the first tile with a correction (not NaN) in every given set, or None."""
    placed = np.all([~np.isnan(c[:, 0]) for c in corrections], axis=0)
    return int(np.argmax(placed)) if placed.any() else None


def placement_corrections(sem_corrections: np.ndarray, design_corrections: np.ndarray) -> dict:
    """Per placement, the (SEM, design) corrections added to the nominal tile centres:

    - "nominal": none (metadata centres),
    - "mean": the stitching solutions (each averages zero),
    - "first": both solutions shifted so that the first tile stitched in both stays nominal.
    NaN marks a tile that could not be stitched. Both sets are (n_tiles, 2), nm.
    """
    zero = np.zeros_like(sem_corrections)
    placements = {"nominal": (zero, zero), "mean": (sem_corrections, design_corrections)}
    k = first_stitched_tile(sem_corrections, design_corrections)
    if k is not None:
        placements["first"] = (fix_tile(sem_corrections, k), fix_tile(design_corrections, k))
    return placements
