"""Stitch images by their contours (docs/SPEC.md S5, D156): for patterns without contact centres (pattern contours).

An edge point knows where it is across its edge only, so two images are compared by how far their edges lie from each
other across the edges, edges facing the same way only. Nothing is assumed about the pattern: straight lines fix the
position across them, any other shape both ways.

- edge_points(): the contours of one image (features.contours) as edge points in nm, each with the way it faces.
- mismatch(): for every shift of an image in the search window, how far its edges then lie from a placed
  neighbour's (one FFT per facing direction).
- coarse_fit(), exact_fit(), as_good(): of those shifts, the ones that fit as well as the best one, up to the
  noise: picked on the pixel grid, then compared exactly.
- lined_up(): a shift made exact, moved only the ways the edges measure.
- placed_by_contours(): the images placed one at a time (as pipeline.placed_image_by_image places contacts), each at
  the fitting shift nearest its stage position.
- overlap_distances(): how far the edges of each overlap lie apart at a placement.
"""

from dataclasses import dataclass

import numpy as np
from scipy import fft, ndimage
from scipy.spatial import cKDTree

from affine_ransac.features.contours import Contours
from affine_ransac.geometry.frames import pixel_to_tile_nm
from affine_ransac.overlap import overlapping_pairs, points_in_box
from affine_ransac.step_log import StepLog, timed

SECTORS = 8            # the facing directions the edges are sorted into for the search (45 degrees each)
GATE_PX = 3.0          # edges farther apart than this many pixels are not the same edge; a distance counts up to it
SHARED_NOISE_PX = 4.0  # edge points closer than this along an edge share noise (the edge blur): they count as one
MIN_POINTS = 40        # an image is placed against at least this many edge points of its placed neighbours
MIN_SHARE = 0.25       # a shift is tried only if it keeps this share of the most edge points any shift compares
WELL_MEASURED = 0.1    # lined_up moves a direction only if the edges measure it this well, relative to the best one
FIT_BATCH = 64         # bound the memory of exact candidate queries (at most 600 points per shift)


@dataclass
class EdgePoints:
    """The edge points of one image's contours."""
    xy: np.ndarray       # (M, 2) nm (x right, y up)
    normals: np.ndarray  # (M, 2) unit: the way each point's edge faces, out of its region
    step_nm: float       # the typical distance between neighbouring points along a contour, nm

    def moved(self, shift) -> "EdgePoints":
        return EdgePoints(self.xy + shift, self.normals, self.step_nm)

    def inside(self, box, margin: float = 0.0) -> "EdgePoints":
        keep = points_in_box(self.xy, box, margin)
        return EdgePoints(self.xy[keep], self.normals[keep], self.step_nm)


def edge_points(contours: Contours, image_shape: tuple[int, int], pixel_size_nm: float) -> EdgePoints:
    """An image's contours (pixel frame) as edge points in tile-local nm; points the refinement found no edge for
    (NaN) are left out."""
    if not contours.points:
        return EdgePoints(np.empty((0, 2)), np.empty((0, 2)), pixel_size_nm)
    xy, normals = np.concatenate(contours.points), np.concatenate(contours.normals)
    steps = np.concatenate([np.linalg.norm(np.diff(p, axis=0), axis=1) for p in contours.points])
    steps = steps[np.isfinite(steps)]
    found = np.isfinite(xy).all(axis=1)
    return EdgePoints(pixel_to_tile_nm(xy[found], image_shape, pixel_size_nm), normals[found] * [1.0, -1.0],
                      float(np.median(steps)) * pixel_size_nm if len(steps) else pixel_size_nm)


def tile_box(center, fov) -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) of a tile, nm."""
    return (center[0] - fov[0] / 2, center[0] + fov[0] / 2, center[1] - fov[1] / 2, center[1] + fov[1] / 2)


def facing_sectors(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per edge point, the sector its facing direction falls in and the next nearest one (0 .. SECTORS - 1)."""
    position = np.arctan2(normals[:, 1], normals[:, 0]) / (2 * np.pi / SECTORS)
    nearest = np.round(position)
    other = nearest + np.where(position >= nearest, 1, -1)
    return nearest.astype(int) % SECTORS, other.astype(int) % SECTORS


def covered_counts(image: np.ndarray, covered_x: np.ndarray, covered_y: np.ndarray, size: int) -> np.ndarray:
    """Count image points inside a rectangular field for each valid correlation shift, using rectangle sums."""
    x, y = np.flatnonzero(covered_x), np.flatnonzero(covered_y)
    if not len(x) or not len(y):
        return np.zeros((size, size))
    sums = np.pad(image.cumsum(axis=0, dtype=float).cumsum(axis=1), ((1, 0), (1, 0)))
    shifts = np.arange(size)
    x0, x1 = np.clip(x[0] - shifts, 0, image.shape[1]), np.clip(x[-1] + 1 - shifts, 0, image.shape[1])
    y0, y1 = np.clip(y[0] - shifts, 0, image.shape[0]), np.clip(y[-1] + 1 - shifts, 0, image.shape[0])
    return sums[np.ix_(y1, x1)] - sums[np.ix_(y0, x1)] - sums[np.ix_(y1, x0)] + sums[np.ix_(y0, x0)]


def mismatch(theirs: EdgePoints, their_box, mine: EdgePoints, my_box, reach: int, bin_nm: float
             ) -> tuple[np.ndarray, np.ndarray]:
    """How well my image's edges lie on those of a placed image, for every shift of mine up to `reach` bins each way.

    theirs, their_box: the placed image's edge points and box where it is placed; mine, my_box: mine at the stage
    position; bin_nm: the grid the shifts are tried on (the pixel size).
    Returns (total, count), each (2 reach + 1, 2 reach + 1): at [a, b], mine shifted by ((b - reach) bin_nm,
    (a - reach) bin_nm): count, my edge points that then lie inside their image; total, the sum over them of the
    squared distance (nm², at most the gate, GATE_PX bins) to their nearest edge facing the same way (the same
    SECTORS sector or the one beside it). total / count is small where the edges lie on each other.
    The total comes from correlations by facing direction: my points as an image, against the image of the distance
    to their edges (chamfer matching). Rectangle sums count covered points without an FFT."""
    size = 2 * reach + 1
    total, count = np.zeros((size, size)), np.zeros((size, size))
    gate, grown = GATE_PX * bin_nm, reach * bin_nm
    # the part of my image that some shift brings into theirs
    x0, x1 = max(my_box[0], their_box[0] - grown), min(my_box[1], their_box[1] + grown)
    y0, y1 = max(my_box[2], their_box[2] - grown), min(my_box[3], their_box[3] + grown)
    if x1 <= x0 or y1 <= y0 or len(theirs.xy) == 0:
        return total, count
    w, h = int(np.ceil((x1 - x0) / bin_nm)) + 1, int(np.ceil((y1 - y0) / bin_nm)) + 1
    wide, high = w + 2 * reach, h + 2 * reach  # their grid: that part grown by the reach

    ix, iy = np.round((mine.xy - [x0, y0]) / bin_nm).astype(int).T
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    if not inside.any():
        return total, count
    ix, iy, my_sector = ix[inside], iy[inside], facing_sectors(mine.normals[inside])[0]
    jx, jy = np.round((theirs.xy - [x0, y0]) / bin_nm).astype(int).T + reach
    there = (jx >= 0) & (jx < wide) & (jy >= 0) & (jy < high)
    jx, jy = jx[there], jy[there]
    their_sector, their_other = facing_sectors(theirs.normals[there])

    cell_x, cell_y = x0 + (np.arange(wide) - reach) * bin_nm, y0 + (np.arange(high) - reach) * bin_nm
    covered_x = (cell_x >= their_box[0]) & (cell_x <= their_box[1])
    covered_y = (cell_y >= their_box[2]) & (cell_y <= their_box[3])
    covered = np.outer(covered_y, covered_x).astype(np.float32)

    def as_image(rows, columns):
        image = np.zeros((h, w), np.float32)
        np.add.at(image, (rows, columns), 1)
        return image
    count = covered_counts(as_image(iy, ix), covered_x, covered_y, size)
    # Only valid shifts are needed: no point wraps at these shifts, so padding to the larger image
    # is sufficient. Full linear correlation pads to the SUM of the two image sizes instead.
    shape = tuple(fft.next_fast_len(n, real=True) for n in (high, wide))
    spectrum = np.zeros((shape[0], shape[1] // 2 + 1), dtype=np.complex64)
    for sector in np.unique(my_sector):
        edges = np.zeros((high, wide), bool)
        facing = (their_sector == sector) | (their_other == sector)
        edges[jy[facing], jx[facing]] = True
        distance = ndimage.distance_transform_edt(~edges) * bin_nm if edges.any() else np.full(edges.shape, gate)
        cost = (np.minimum(distance, gate) ** 2).astype(np.float32) * covered
        mine_here = as_image(iy[my_sector == sector], ix[my_sector == sector])
        spectrum += fft.rfftn(cost, shape) * fft.rfftn(mine_here, shape).conj()
    # Correlation is linear: sum the sectors before doing the single inverse transform.
    total = fft.irfftn(spectrum, shape)[:size, :size].astype(float)
    return np.maximum(total, 0), count


def coarse_fit(total: np.ndarray, count: np.ndarray, bin_nm: float, step_nm: float) -> np.ndarray:
    """Of the shifts of mismatch() (total, count summed over the placed neighbours), those whose edges fit as well as
    the best one's on the search grid: a bool array of their shape.

    A shift is tried if it keeps MIN_POINTS edge points and MIN_SHARE of the most any shift keeps. It fits as well
    as the best if their mean squared distances differ by less than three times the spread of that difference
    (as_good; n: its edge points counted as independent, one per SHARED_NOISE_PX pixels of edge; the rounding to
    bins, bin²/12, added to both). The grid is a pixel wide, so this test is blind to what is smaller: it only
    picks the shifts that exact_fit then compares."""
    valid = count >= max(MIN_POINTS, MIN_SHARE * count.max())
    if not valid.any():
        return valid
    mean = np.where(valid, total / np.maximum(count, 1), np.inf) + bin_nm ** 2 / 12
    independent = np.maximum(count * step_nm / (SHARED_NOISE_PX * bin_nm), 1)
    return valid & as_good(mean, independent)


def as_good(mean_squared: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Which fits are as good as the best one, up to their noise: the log of the ratio of their mean squared
    distances within three times its spread (2 / n for each, n its independent points)."""
    best = np.unravel_index(np.argmin(mean_squared), mean_squared.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(mean_squared / mean_squared[best])
    return np.nan_to_num(ratio, nan=0.0) < 3 * np.sqrt(2 / n[best] + 2 / n)


def grid_shifts(grid: np.ndarray, bin_nm: float) -> np.ndarray:
    """The shifts (N, 2) nm of the True cells of a mismatch() grid, row by row."""
    rows, columns = np.nonzero(grid)
    reach = (grid.shape[0] - 1) // 2
    return np.column_stack([columns - reach, rows - reach]) * bin_nm


def separate_groups(fitting: np.ndarray) -> int:
    """In how many groups the fitting shifts of a mismatch() grid lie, groups less than two gates apart counted as
    one: one position or one run of shifts along straight edges is one group, even where noise breaks the run."""
    reach = int(np.ceil(GATE_PX))
    joined = ndimage.binary_dilation(fitting, structure=np.ones((2 * reach + 1, 2 * reach + 1), bool))
    return int(ndimage.label(joined, structure=np.ones((3, 3)))[1])


def thinned(points: "EdgePoints", bin_nm: float, most: int = 600) -> "EdgePoints":
    """Edge points far enough apart along their contours not to share noise (SHARED_NOISE_PX pixels), at most
    `most` of them, evenly."""
    every = max(1, int(round(SHARED_NOISE_PX * bin_nm / points.step_nm)))
    keep = np.arange(0, len(points.xy), every)
    if len(keep) > most:
        keep = keep[np.linspace(0, len(keep) - 1, most).astype(int)]
    return EdgePoints(points.xy[keep], points.normals[keep], points.step_nm * every)


def exact_fit(tree: cKDTree, theirs: "EdgePoints", mine: "EdgePoints", their_boxes: list, shift, gate: float
              ) -> tuple[float, int]:
    """How well my edges fit theirs at this shift, without the search grid: (mean squared distance across the
    edges in nm², my points compared).

    Compared are my points that the shift brings into one of their images (their_boxes), two gates or more from
    its border (an image's contours end short of its border). Each counts its distance
    to their edge facing the same way (across_distances) after the small move across the edges that fits them best
    (across_step: the shift is a whole number of bins, the edges are not), at most the gate; one without such an
    edge within the gate counts the gate."""
    moved = mine.xy + shift
    expected = np.zeros(len(moved), bool)
    for box in their_boxes:
        expected[points_in_box(moved, box, margin=-2 * gate)] = True
    if not expected.any():
        return np.inf, 0
    apart, normals = across_distances(tree, theirs, EdgePoints(mine.xy[expected], mine.normals[expected], 0), shift,
                                      gate)
    left = apart - normals @ across_step(apart, normals)[0] if len(apart) >= 3 else apart
    unpaired = expected.sum() - len(apart)
    return float((np.minimum(left ** 2, gate ** 2).sum() + unpaired * gate ** 2) / expected.sum()), int(expected.sum())


def exact_fits(tree: cKDTree, theirs: EdgePoints, mine: EdgePoints, their_boxes: list,
               shifts: np.ndarray, gate: float) -> np.ndarray:
    """exact_fit for all shifts, with bounded batches of tree queries and 2 x 2 least-squares solves.

    The sample, gate, border test and measured directions are unchanged. The scalar exact_fit remains the
    reference for callers checking one shift. Empty and unpaired samples still count as they do there.
    """
    fits = np.empty((len(shifts), 2))
    for start in range(0, len(shifts), FIT_BATCH):
        batch = shifts[start:start + FIT_BATCH]
        moved = mine.xy[None, :, :] + batch[:, None, :]
        expected = np.zeros(moved.shape[:2], bool)
        for x0, x1, y0, y1 in their_boxes:
            expected |= ((moved[..., 0] >= x0 + 2 * gate) & (moved[..., 0] <= x1 - 2 * gate) &
                         (moved[..., 1] >= y0 + 2 * gate) & (moved[..., 1] <= y1 - 2 * gate))
        rows, columns = np.nonzero(expected)
        query = np.column_stack([moved[rows, columns], 2 * gate * mine.normals[columns]])
        distance, index = tree.query(query, distance_upper_bound=gate)
        found = np.isfinite(distance)
        rows, columns, index = rows[found], columns[found], index[found]
        normals = np.zeros(moved.shape)
        normals[rows, columns] = theirs.normals[index]
        apart = np.zeros(expected.shape)
        apart[rows, columns] = ((theirs.xy[index] - mine.xy[columns] - batch[rows]) * theirs.normals[index]).sum(axis=1)
        paired = np.bincount(rows, minlength=len(batch))
        values, vectors = np.linalg.eigh(normals.transpose(0, 2, 1) @ normals)
        well = (values >= WELL_MEASURED * values[:, -1:]) & (values > 0)
        rhs = normals.transpose(0, 2, 1) @ apart[..., None]
        component = (vectors.transpose(0, 2, 1) @ rhs)[..., 0] / np.where(well, values, 1)
        step = (vectors @ np.where(well, component, 0)[..., None])[..., 0]
        step[paired < 3] = 0
        left = apart - (normals @ step[..., None])[..., 0]
        count = expected.sum(axis=1)
        total = np.minimum(left ** 2, gate ** 2).sum(axis=1) + (count - paired) * gate ** 2
        fits[start:start + len(batch), 0] = np.where(count > 0, total / np.maximum(count, 1), np.inf)
        fits[start:start + len(batch), 1] = count
    return fits


def across_step(apart: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, int]:
    """The move (2,) that brings paired edges onto each other in the least squares sense (apart: their distances
    across the edges; normals: the way each faces), and how many directions it moves in. It moves only the ways the
    edges measure: a direction they measure less than WELL_MEASURED as well as the best measured one is left alone
    (straight lines: along them)."""
    values, vectors = np.linalg.eigh(normals.T @ normals)
    well = values >= WELL_MEASURED * values.max()
    return vectors[:, well] @ ((vectors[:, well].T @ (normals.T @ apart)) / values[well]), int(well.sum())


def across_distances(tree: cKDTree, theirs: EdgePoints, mine: EdgePoints, shift, gate: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """My edge points (shifted) paired with their nearest edge point within gate that faces the same way (within
    about 30 degrees): (distances from mine to theirs across their edge, nm; their normals)."""
    facing = 2 * gate  # points facing differently lie this far apart in the tree, so beyond the gate
    distance, index = tree.query(np.hstack([mine.xy + shift, facing * mine.normals]), distance_upper_bound=gate)
    found = np.isfinite(distance)
    normals = theirs.normals[index[found]]
    return ((theirs.xy[index[found]] - mine.xy[found] - shift) * normals).sum(axis=1), normals


def facing_tree(points: EdgePoints, gate: float) -> cKDTree:
    """The tree across_distances pairs in: position and, scaled, the facing direction."""
    return cKDTree(np.hstack([points.xy, 2 * gate * points.normals]))


def lined_up(theirs: EdgePoints, mine: EdgePoints, shift, gate: float, steps: int = 10, *, tree: cKDTree | None = None
             ) -> tuple[np.ndarray, int, float, int]:
    """`shift` of my edge points made exact: moved until my edges lie on theirs (least squares of across_distances,
    distances beyond three robust spreads left out; again until it moves less than 0.01 nm).

    It moves only the ways the edges measure (across_step; straight lines: not along them). Returns (shift, pairs, the rms distance
    left across the edges in nm with each pair counted at most at three spreads, directions measured: 2 or 1)."""
    shift = np.asarray(shift, dtype=float)
    tree = facing_tree(theirs, gate) if tree is None else tree
    measured = 0
    for _ in range(steps):
        apart, normals = across_distances(tree, theirs, mine, shift, gate)
        if len(apart) < 3:
            return shift, len(apart), np.nan, 0
        centred = np.abs(apart - np.median(apart))
        keep = centred <= 3 * 1.4826 * np.median(centred) + 1e-9
        step, measured = across_step(apart[keep], normals[keep])
        shift = shift + step
        if np.linalg.norm(step) < 0.01:
            break
    apart, _ = across_distances(tree, theirs, mine, shift, gate)
    return shift, len(apart), capped_rms(apart), measured


def capped_rms(apart: np.ndarray) -> float:
    """The rms of the distances, each counted at most at three robust spreads (an outlier counts, but cannot rule)."""
    if len(apart) == 0:
        return np.nan
    limit = 3 * 1.4826 * np.median(np.abs(apart - np.median(apart))) + 1e-9
    return float(np.sqrt(np.mean(np.minimum(np.abs(apart), limit) ** 2)))


@dataclass
class ContourChoice:
    """How one image was placed against its placed neighbours (placed_by_contours)."""
    tile: int
    correction: np.ndarray  # (2,) added to its stage position, nm
    pairs: int              # its edge points paired with theirs
    rms_nm: float           # the distance left between its edges and theirs, across the edges (capped_rms)
    positions: int          # separate groups of shifts that fit as well as the best (separate_groups); > 1: the edges cannot tell
                            # them apart (a repeating pattern), the one nearest the stage position was taken
    directions: int         # ways the edges fixed its position: 2, or 1 (across straight edges; along from the stage)


def placed_by_contours(edges: list[EdgePoints], centers: np.ndarray, fovs: np.ndarray, search_nm: float,
                       bin_nm: float, progress=lambda placed, total: None, *, log: StepLog | None = None
                       ) -> tuple[np.ndarray, list[ContourChoice]]:
    """(n_tiles, 2) corrections that place the images one at a time by their contours, and how each was placed.

    edges: per tile, its edge points in mask nm at the stage position; centers, fovs: (n_tiles, 2), nm; search_nm:
    how far an image may lie from its stage position, each way; bin_nm: the pixel size.
    The image with the most edge points in its overlaps stays (correction 0). Then, again and again, one image is
    placed against all its placed neighbours at once: of the shifts whose edges fit as well as the best, the one
    nearest its stage position, made exact by lined_up. The shifts are found on the pixel grid (mismatch,
    coarse_fit) and those then compared exactly, on edge points that do not share noise (thinned, exact_fit,
    as_good), so that edge roughness smaller than a pixel counts. So where the edges fix
    one position, it is taken; along straight lines the image stays at its stage position; and where positions apart
    fit equally (a repeating pattern), the nearest is taken and the choice says so (positions > 1). Images with one
    position are placed first, so that an undecided one does not decide for its neighbours (as
    pipeline.placed_image_by_image); either way the one sharing the most edge points with the placed ones goes
    first. An image never placed (no placed neighbour with MIN_POINTS edge points to compare) gets NaN.
    progress(placed, total) is called after every image placed. log optionally records the work and time of each
    stitching phase; it does not change placement."""
    n = len(edges)
    corrections, choices = np.full((n, 2), np.nan), []
    if n == 0:
        return corrections, choices
    reach, gate = int(np.ceil(search_nm / bin_nm)), GATE_PX * bin_nm
    boxes = [tile_box(c, f) for c, f in zip(centers, fovs)]
    neighbours, shared = [[] for _ in range(n)], {}
    with timed(log, "find overlaps and count shared edges"):
        for i, j, box in overlapping_pairs(centers, fovs):
            shared[i, j] = shared[j, i] = min(len(points_in_box(edges[i].xy, box, search_nm)),
                                              len(points_in_box(edges[j].xy, box, search_nm)))
            neighbours[i].append(j)
            neighbours[j].append(i)
    if log is not None:
        log.counts.update(images=n, overlaps=len(shared) // 2, edge_points=sum(len(e.xy) for e in edges),
                          search_shifts=(2 * reach + 1) ** 2, surfaces=0, choices=0, exact_shifts=0)
    surfaces = {}  # (k, o): mismatch of tile k against the placed tile o
    options = {}   # tile -> (its placed neighbours when its choice was found, the choice or None)
    gain = np.zeros(n)  # edge points each tile shares with the placed tiles
    placed_edges, placed_boxes = {}, {}

    def place(k, correction):
        corrections[k] = correction
        placed_edges[k] = edges[k].moved(corrections[k])
        placed_boxes[k] = tuple(np.add(boxes[k], np.repeat(corrections[k], 2)))
        options.pop(k, None)
        for o in neighbours[k]:
            gain[o] += shared[k, o]
            surfaces.pop((k, o), None)  # a placed tile is never a candidate again
        for o in [k] + neighbours[k]:
            if all(np.isfinite(corrections[j, 0]) for j in neighbours[o]):
                placed_edges.pop(o, None)
                placed_boxes.pop(o, None)

    def choice(k, placed) -> ContourChoice | None:
        total, count = 0.0, 0.0
        with timed(log, "grid search against placed neighbours"):
            for o in placed:
                if (k, o) not in surfaces:
                    surfaces[k, o] = mismatch(placed_edges[o], placed_boxes[o], edges[k], boxes[k], reach, bin_nm)
                    if log is not None:
                        log.counts["surfaces"] += 1
                total, count = total + surfaces[k, o][0], count + surfaces[k, o][1]
            coarse = coarse_fit(total, count, bin_nm, edges[k].step_nm)
        if log is not None:
            log.counts["choices"] += 1
        if not coarse.any():
            return None
        with timed(log, "prepare overlap samples and facing tree"):
            near = [placed_edges[o].inside(boxes[k], search_nm + gate) for o in placed]
            theirs = EdgePoints(np.concatenate([e.xy for e in near]), np.concatenate([e.normals for e in near]), bin_nm)
            their_boxes = [placed_boxes[o] for o in placed]
            tree = facing_tree(theirs, gate)
            mine = [edges[k].inside(box, search_nm) for box in their_boxes]  # some shift brings my points into theirs
            mine = EdgePoints(np.concatenate([e.xy for e in mine]), np.concatenate([e.normals for e in mine]),
                              edges[k].step_nm)
            sample = thinned(mine, bin_nm)
        with timed(log, "compare candidate shifts exactly"):
            fits = exact_fits(tree, theirs, sample, their_boxes, grid_shifts(coarse, bin_nm), gate)
            if log is not None:
                log.counts["exact_shifts"] += len(fits)
            exact = np.zeros_like(coarse)
            exact[coarse] = as_good(fits[:, 0], np.maximum(fits[:, 1], 1))
            shifts = grid_shifts(exact, bin_nm)
            nearest = shifts[np.argmin(np.linalg.norm(shifts, axis=1))]
            groups = separate_groups(exact)
        with timed(log, "refine selected shifts"):
            shift, pairs, rms, directions = lined_up(theirs, mine, nearest, gate, tree=tree)
        return ContourChoice(k, shift, pairs, rms, groups, directions) if pairs >= MIN_POINTS else None

    place(int(np.argmax([sum(shared[k, o] for o in neighbours[k]) for k in range(n)])), 0.0)
    while True:
        waiting = np.flatnonzero(np.isnan(corrections[:, 0]) & (gain >= MIN_POINTS))
        for k in waiting:
            placed = [o for o in neighbours[k] if not np.isnan(corrections[o, 0])]
            if k not in options or options[k][0] != placed:  # found again only once more neighbours are placed
                options[k] = placed, choice(k, placed)
        found = [k for k in waiting if options[k][1] is not None]
        if not found:
            return corrections, choices
        k = max([k for k in found if options[k][1].positions == 1] or found, key=lambda t: gain[t])
        choices.append(options[k][1])
        place(k, options[k][1].correction)
        progress(len(choices) + 1, n)


@dataclass
class OverlapDistance:
    i: int
    j: int
    pairs: int     # edge points of tile j paired with tile i's
    rms_nm: float  # how far the paired edges lie apart, across the edges (capped_rms); NaN without pairs


def overlap_distances(edges: list[EdgePoints], centers: np.ndarray, fovs: np.ndarray, corrections: np.ndarray,
                      bin_nm: float) -> list[OverlapDistance]:
    """Per overlap of two placed images, how far their edges lie apart where the placement puts them: the check of
    placed_by_contours that needs no trust in it (small: the same edges on each other; near the gate, GATE_PX
    pixels: they are not)."""
    gate, found = GATE_PX * bin_nm, []
    for i, j, box in overlapping_pairs(centers, fovs):
        if np.isnan(corrections[[i, j]]).any():
            continue
        theirs = edges[i].moved(corrections[i]).inside(box, 2 * gate)
        mine = edges[j].moved(corrections[j]).inside(box, 2 * gate)
        if len(theirs.xy) == 0 or len(mine.xy) == 0:
            found.append(OverlapDistance(i, j, 0, np.nan))
            continue
        apart, _ = across_distances(facing_tree(theirs, gate), theirs, mine, 0.0, gate)
        found.append(OverlapDistance(i, j, len(apart), capped_rms(apart)))
    return found
