"""Global stitching solve (docs/SPEC.md S5): translation per tile, translation + rotation per tile, or
a full affine per tile.

Translation (solve_tile_shifts): each tile k gets a correction t_k (nm) that is added to its nominal placement. For a pair (i, j)
the measured shift d_ij is B − A for the same contacts at nominal placement (overlap_fit). After
correction the two measurements coincide when

    (a + t_i) = (b + t_j)   ->   t_i − t_j = d_ij

All pairs together are solved by least squares. A shift common to all tiles cannot be seen in
the overlaps (gauge freedom), so the corrections are fixed to average zero; the later fit to
the design (S7) absorbs any common shift.

Only tiles connected to each other through pairs can be placed relative to each other. The
largest connected group is solved; every other tile gets a NaN correction (never a silent 0),
so a stitching failure cannot pass unnoticed into later steps.

Affine (solve_tile_affines, D51): each tile k gets T_k(p) = p + D_k (p − c_k) + t_k, a 2×2 linear
part D_k about its centre c_k plus a shift t_k. Every tie contact (a in tile i, b in tile j)
gives T_i(a) = T_j(b). Gauge: an affine common to all tiles is not seen in the overlaps, so the
D_k and the t_k average zero over the stitched group (mean tile affine = identity, SPEC D6).

Translation + rotation (solve_tile_rigid, D58): T_k(p) = R(θ_k)(p − c_k) + c_k + t_k, solved with
R(θ) ≈ I + θ·[[0, −1], [1, 0]] (θ ~ 1e-3 rad); gauge: the θ_k and the t_k average zero. The result
is written as affines (3, 3) with the exact rotation.

A tile's correction is either a shift (2,) or an affine (3, 3) in mask nm (apply_correction).
"""

import warnings

import numpy as np
from scipy.sparse import bmat, coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from affine_ransac.geometry.affine import apply_affine


def apply_correction(points: np.ndarray, correction: np.ndarray) -> np.ndarray:
    """points (N, 2) mask nm moved by one tile's correction: a shift (2,) or an affine (3, 3)."""
    correction = np.asarray(correction, dtype=float)
    return apply_affine(correction, points) if correction.shape == (3, 3) else points + correction


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


def solve_tile_affines(
    n_tiles: int,
    pairs: list[tuple[int, int]],
    ties: list[tuple[np.ndarray, np.ndarray]],
    centers: np.ndarray,
    min_spread_nm: float = 10.0,
) -> np.ndarray:
    """Per-tile affines (n_tiles, 3, 3), mapping nominal mask nm to stitched mask nm.

    pairs: (i, j) tile indices; ties: per pair, (a, b) of shape (K, 2): the same K contacts at
    nominal placement in tile i and in tile j; centers: (n_tiles, 2) tile centres (the point each
    tile's scale and rotation act about). Least squares over all ties of T_i(a) − T_j(b), each tie
    weighted 1; gauge: D and t average zero over the largest connected group. Tiles outside it get
    NaN. A tile whose tie contacts lie (nearly) on one line, spread less than min_spread_nm across
    it, cannot have its scale across that line measured: it is kept near nominal by a tiny pull of
    D towards 0 (negligible for measured tiles), and a warning lists it.
    """
    centers = np.asarray(centers, dtype=float)
    placed = connected_groups(n_tiles, pairs)[0] if n_tiles else []
    index = np.full(n_tiles, -1)
    index[placed] = np.arange(len(placed))
    m = len(placed)

    # Unknowns per tile and axis: [D row (nm per µm), t (nm)]; x and y share the same matrix.
    def design(points, k):
        return np.column_stack([(points - centers[k]) / 1000, np.ones(len(points))])  # µm, 1

    rows, cols, values = [], [], []
    right = np.zeros((3 * m, 2))
    tie_points = [[] for _ in range(n_tiles)]
    for (i, j), (a, b) in zip(pairs, ties):
        if index[i] < 0:
            continue  # a pair is inside or outside the group
        u, v = design(a, i), design(b, j)
        for (p, q), block in (((i, i), u.T @ u), ((j, j), v.T @ v), ((i, j), -u.T @ v), ((j, i), -v.T @ u)):
            r, c = np.meshgrid(3 * index[p] + np.arange(3), 3 * index[q] + np.arange(3), indexing="ij")
            rows.append(r.ravel())
            cols.append(c.ravel())
            values.append(block.ravel())
        right[3 * index[i]:3 * index[i] + 3] += u.T @ (b - a)
        right[3 * index[j]:3 * index[j] + 3] -= v.T @ (b - a)
        tie_points[i].append(a - centers[i])
        tie_points[j].append(b - centers[j])

    weak = [k for k in placed if _line_spread(tie_points[k]) < min_spread_nm]
    if weak:
        warnings.warn(f"{len(weak)} tile(s) with overlap contacts (nearly) on one line: scale and skew across "
                      f"it are not measured and stay nominal: {weak}", stacklevel=2)

    ridge = np.zeros(3 * m)
    ridge[np.arange(3 * m) % 3 != 2] = 1e-6  # tiny pull of D (not t) towards 0, µm²
    rows.append(np.arange(3 * m))
    cols.append(np.arange(3 * m))
    values.append(ridge)
    normal = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * m, 3 * m))
    gauge = csc_matrix(np.tile(np.eye(3), m))  # sum of each unknown over the group = 0
    kkt = bmat([[normal, gauge.T], [gauge, None]], format="csc")
    solution = spsolve(kkt, np.vstack([right, np.zeros((3, 2))]))[:3 * m].reshape(m, 3, 2)

    affines = np.full((n_tiles, 3, 3), np.nan)
    for k in placed:
        theta = solution[index[k]]  # column = output axis (x', y'); rows = factor of x, factor of y, shift t
        linear = np.eye(2) + theta[:2].T / 1000       # nm per µm -> nm per nm
        affines[k] = np.eye(3)
        affines[k, :2, :2] = linear
        affines[k, :2, 2] = centers[k] + theta[2] - linear @ centers[k]  # T(c) = c + t
    return affines


def solve_tile_rigid(
    n_tiles: int,
    pairs: list[tuple[int, int]],
    ties: list[tuple[np.ndarray, np.ndarray]],
    centers: np.ndarray,
) -> np.ndarray:
    """Per-tile translation + rotation (n_tiles, 3, 3), nominal mask nm -> stitched mask nm.

    pairs, ties, centers as for solve_tile_affines. Least squares over all ties of T_i(a) − T_j(b)
    (each tie weighted 1) with T_k(p) = R(θ_k)(p − c_k) + c_k + t_k; gauge: θ and t average zero over
    the largest connected group, other tiles NaN. A tiny pull of θ towards 0 keeps a tile solvable if
    its ties sit at one point (negligible otherwise). The rotation is solved to first order: the error
    is about θ²/2 · r (r = distance from the tile centre), 0.0005 nm for θ = 1 mrad at 1 µm.
    """
    centers = np.asarray(centers, dtype=float)
    placed = connected_groups(n_tiles, pairs)[0] if n_tiles else []
    index = np.full(n_tiles, -1)
    index[placed] = np.arange(len(placed))
    m = len(placed)

    # Unknowns per tile: [θ (nm per µm = mrad), t_x, t_y (nm)]. Per tie contact one row for x and one for y.
    def design(points, k):
        u = (points - centers[k]) / 1000  # µm
        rows = np.zeros((len(points), 2, 3))
        rows[:, 0, 0], rows[:, 0, 1] = -u[:, 1], 1.0  # x' = x − θ·u_y + t_x
        rows[:, 1, 0], rows[:, 1, 2] = u[:, 0], 1.0   # y' = y + θ·u_x + t_y
        return rows

    rows, cols, values = [], [], []
    right = np.zeros(3 * m)
    for (i, j), (a, b) in zip(pairs, ties):
        if index[i] < 0:
            continue  # a pair is inside or outside the group
        u, v, d = design(a, i), -design(b, j), b - a
        for (p, q), block in (((i, i), np.einsum("kca,kcb->ab", u, u)), ((j, j), np.einsum("kca,kcb->ab", v, v)),
                              ((i, j), np.einsum("kca,kcb->ab", u, v)), ((j, i), np.einsum("kca,kcb->ab", v, u))):
            r, c = np.meshgrid(3 * index[p] + np.arange(3), 3 * index[q] + np.arange(3), indexing="ij")
            rows.append(r.ravel())
            cols.append(c.ravel())
            values.append(block.ravel())
        right[3 * index[i]:3 * index[i] + 3] += np.einsum("kca,kc->a", u, d)
        right[3 * index[j]:3 * index[j] + 3] += np.einsum("kca,kc->a", v, d)

    ridge = np.where(np.arange(3 * m) % 3 == 0, 1e-6, 0.0)  # tiny pull of θ (not t) towards 0, µm²
    rows.append(np.arange(3 * m))
    cols.append(np.arange(3 * m))
    values.append(ridge)
    normal = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * m, 3 * m))
    gauge = csc_matrix(np.tile(np.eye(3), m))  # sum of each unknown over the group = 0
    kkt = bmat([[normal, gauge.T], [gauge, None]], format="csc")
    solution = spsolve(kkt, np.concatenate([right, np.zeros(3)]))[:3 * m].reshape(m, 3)

    affines = np.full((n_tiles, 3, 3), np.nan)
    for k in placed:
        theta, shift = solution[index[k], 0] / 1000, solution[index[k], 1:]  # rad, nm
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        affines[k] = np.eye(3)
        affines[k, :2, :2] = rotation
        affines[k, :2, 2] = centers[k] + shift - rotation @ centers[k]  # T(c) = c + t
    return affines


def _line_spread(points: list[np.ndarray]) -> float:
    """Spread (std, nm) of the points across their best-fitting line; 0 for fewer than 3 points."""
    if not points or sum(len(p) for p in points) < 3:
        return 0.0
    all_points = np.concatenate(points)
    return float(np.sqrt(np.linalg.eigvalsh(np.cov(all_points.T))[0]))


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
