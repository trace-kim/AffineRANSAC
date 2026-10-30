"""In-image (intrafield) distortion shared by all SEM images (docs/SPEC.md S5, D53, D145, D146).

If every image is distorted the same way (e.g. a nonlinearity of the SEM scan), a contact at the
position u inside an image (relative to the image centre) is measured at its true position + f(u).

- estimate_formula(): f from the design, as one smooth formula over the whole image (powers 2 to 3
  of x and y by default: the distortion of the beam's deflection and lenses is of 3rd order),
  fitted to all images at once, each with its own affine (shift, scale, rotation, skew: the affine
  part of f is the same in all images and is left to the global affine). Kept as a map of nodes
  over the image, bilinear in between. Mask errors sit at fixed places on the mask, so at a
  different u in every image, and average out. Mask errors that repeat at the image spacing do NOT
  average out: they would be taken for SEM distortion.
- correct_points(): an image's SEM points with the map removed (before stitching).
- overlap_slopes(): the check without the design. In an overlap the same contact is measured by two
  images, so mask errors cancel. A distortion common to all images makes B − A change across the
  overlap (it sits at the top of one image and at the bottom of the next); after a correct map it
  does not.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from affine_ransac.pipeline import StitchResult, pair_kind, stitch_ties

FORMULA_NODES = 65  # the formula is kept as a map of this many nodes per axis: bilinear between them, a 3rd-order
                    # formula is off by < 0.1 % of its size
DETERMINED = 1e-2   # the formula's terms are determined if the weakest combination of them is at least this fraction
                    # of the strongest (full images: >= 0.05 up to order 5; contacts in a few rows: ~1e-4)


@dataclass
class IntrafieldMap:
    x_nm: np.ndarray      # (nodes,) node positions along x, relative to the image centre, nm
    y_nm: np.ndarray      # (nodes,) node positions along y, nm
    shift_nm: np.ndarray  # (nodes, nodes, 2) [row = y, column = x]: (SEM after its image's affine) − design
                          # at the node, nm (the formula there)
    count: np.ndarray     # (nodes, nodes) contacts nearest to each node
    images: int           # images used (images with 3 contacts or fewer are left out)


def _nodes(positions: np.ndarray, nodes: int):
    """nodes x and nodes y node positions from the lowest to the highest of positions (N, 2) (on the image edges: the
    overlaps lie there, and nothing has to be extrapolated), and the row and column of each position's nearest node."""
    x_nodes = np.linspace(positions[:, 0].min(), positions[:, 0].max(), nodes)
    y_nodes = np.linspace(positions[:, 1].min(), positions[:, 1].max(), nodes)
    column = np.rint((positions[:, 0] - x_nodes[0]) / (x_nodes[1] - x_nodes[0])).astype(int)
    row = np.rint((positions[:, 1] - y_nodes[0]) / (y_nodes[1] - y_nodes[0])).astype(int)
    return x_nodes, y_nodes, row, column


def formula_terms(order: int) -> list[tuple[int, int]]:
    """(power of x, power of y) of each term of the in-image formula: every power from 2 to order (3: x², xy, y², x³,
    x²y, xy², y³). Powers 0 and 1 are left out: each image's own affine takes them."""
    return [(n - j, j) for n in range(2, order + 1) for j in range(n + 1)]


def estimate_formula(design_local: list[np.ndarray], sem_local: list[np.ndarray], order: int = 3,
                     nodes: int = FORMULA_NODES) -> IntrafieldMap | None:
    """The distortion shared by all images, measured against the design as one smooth formula f (D145), kept as a map
    of nodes × nodes nodes (map_shift, correct_points).

    design_local, sem_local: per image, (N_k, 2) design and SEM positions of the same contacts (row k of both is one
    contact), relative to the image centre, nm. f(s) = the sum of c · u^p v^q over formula_terms(order), (u, v) the SEM
    position s relative to the image centre, divided by the largest |coordinate| of the contacts (so each c is the
    term's nm at the image edge). Fitted to all images at once: in every image the SEM contacts with f removed are an
    affine image of its design, s − f(s) = B(d), B that image's own affine. Images with 3 contacts or fewer add nothing
    (their own affine fits them exactly); partial images take part like the others. Contacts more than 5 median
    absolute deviations off (per axis) are left out and f is fitted again. The map is f without its affine part (the
    affine part is left to the global affine). None if the contacts cannot determine every term (e.g. all of them in
    a few rows).
    """
    used = [k for k, d in enumerate(design_local) if len(d) > 3]
    if not used:
        return None
    design, sem = [design_local[k] for k in used], [sem_local[k] for k in used]
    terms, scale = formula_terms(order), max(np.abs(s).max() for s in sem)
    fit = _fit_formula(design, sem, terms, scale)
    if fit is None:
        return None
    deviation = np.abs(fit[1] - np.median(fit[1], axis=0))
    keep = np.all(deviation <= 5 * np.median(deviation, axis=0) + 1e-9, axis=1)
    keep = np.split(keep, np.cumsum([len(d) for d in design])[:-1])  # per image
    fit = _fit_formula([d[k] for d, k in zip(design, keep)], [s[k] for s, k in zip(sem, keep)], terms, scale)
    if fit is None:
        return None

    positions = np.concatenate(sem)
    x_nodes, y_nodes, row, column = _nodes(positions, nodes)
    grid = np.stack(np.meshgrid(x_nodes, y_nodes), axis=-1).reshape(-1, 2)  # row = y, column = x
    shift = _without_affine(grid, _terms_at(grid / scale, terms) @ fit[0])
    count = np.zeros((nodes, nodes), int)
    np.add.at(count, (row, column), 1)
    return IntrafieldMap(x_nodes, y_nodes, shift.reshape(nodes, nodes, 2), count, len(used))


def _fit_formula(design: list[np.ndarray], sem: list[np.ndarray], terms, scale: float):
    """The coefficients (terms, 2) of the formula fitted to all images at once, and every contact's residual (N, 2);
    None if the contacts cannot determine every term.

    Each image's own affine is not solved for: in each image, what an affine of its design explains is removed from its
    SEM positions and from every term first (the same fit as with all affines free, without 6 unknowns per image)."""
    x = np.concatenate([_without_affine(d, _terms_at(s / scale, terms)) for d, s in zip(design, sem)])
    y = np.concatenate([_without_affine(d, s) for d, s in zip(design, sem)])
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=DETERMINED)
    if rank < len(terms):
        return None
    return coefficients, y - x @ coefficients


def _terms_at(uv: np.ndarray, terms) -> np.ndarray:
    """(N, terms) each term u^p v^q at the positions uv (N, 2)."""
    return np.column_stack([uv[:, 0] ** p * uv[:, 1] ** q for p, q in terms])


def _without_affine(positions: np.ndarray, values: np.ndarray) -> np.ndarray:
    """values (N, m) minus their least-squares fit by an affine function of positions (N, 2): what no affine explains."""
    q = np.column_stack([np.ones(len(positions)), positions])
    return values - q @ np.linalg.lstsq(q, values, rcond=None)[0]


def map_shift(m: IntrafieldMap, local_nm: np.ndarray) -> np.ndarray:
    """(N, 2) the map's shift at positions local_nm (relative to the image centre, nm): bilinear
    between the nodes (extrapolated linearly outside them, e.g. a contact just beyond the area the
    map's contacts covered)."""
    query = np.column_stack([local_nm[:, 1], local_nm[:, 0]])
    return np.column_stack([RegularGridInterpolator((m.y_nm, m.x_nm), m.shift_nm[..., c], bounds_error=False,
                                                    fill_value=None)(query) for c in range(2)])


def correct_points(points_nm: np.ndarray, center_nm, m: IntrafieldMap) -> np.ndarray:
    """One image's SEM points (N, 2), mask nm, with the map's shift removed; center_nm: the image
    centre in mask nm."""
    return points_nm - map_shift(m, points_nm - np.asarray(center_nm, dtype=float))


def overlap_slopes(points: list[np.ndarray], stitch: StitchResult, centers: np.ndarray) -> dict:
    """How B − A of the tie contacts changes across each used overlap of stitch, in ppm:
    (b − a) = const + S · (a − mean a), with S = [[d(dx)/dx, d(dx)/dy], [d(dy)/dx, d(dy)/dy]].

    points: per image, the (N, 2) points stitch was made from; centers: (n_images, 2) image centres,
    nm, to tell the overlaps apart. Returns {"vertical": (P, 2, 2), "horizontal": (Q, 2, 2)}:
    A is the lower image of a vertical pair and the left image of a horizontal pair; corner pairs
    are left out. A distortion common to all images gives about the same S in every pair of a kind.
    """
    slopes = {"vertical": [], "horizontal": []}
    for p, (a, b) in zip(stitch.pairs, stitch_ties(points, stitch)):
        kind = pair_kind(centers, p.i, p.j)
        if kind == "corner":
            continue
        step = centers[p.j] - centers[p.i]
        if step[1 if kind == "vertical" else 0] < 0:  # A = the lower / left image
            a, b = b, a
        if len(a) >= 3:
            design = np.column_stack([a - a.mean(axis=0), np.ones(len(a))])
            coef, *_ = np.linalg.lstsq(design, b - a, rcond=None)  # rows: d/dx, d/dy, const
            slopes[kind].append(coef[:2].T * 1e6)
    return {kind: np.array(v).reshape(-1, 2, 2) for kind, v in slopes.items()}
