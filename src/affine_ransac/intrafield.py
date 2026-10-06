"""In-image (intrafield) distortion shared by all SEM images (docs/SPEC.md S5, D53).

If every image is distorted the same way (e.g. a nonlinearity of the SEM scan), a contact at the
position u inside an image (relative to the image centre) is measured at its true position + f(u).

- estimate_map(): f from the design. In each image, SEM − design after that image's own affine
  (shift, scale, rotation, skew: the affine part of f is the same in all images and is left to the
  global affine), on a grid of nodes over the image: each node's value comes from the residuals of
  all images nearest to it (a plane fitted to them, strongly deviating ones left out); in between,
  bilinear. Mask errors sit at fixed places on the mask, so at a different u in every image, and
  average out. Mask errors that repeat at the image spacing do NOT average out: they would be taken
  for SEM distortion.
- correct_points(): an image's SEM points with the map removed (before stitching).
- overlap_slopes(): the check without the design. In an overlap the same contact is measured by two
  images, so mask errors cancel. A distortion common to all images makes B − A change across the
  overlap (it sits at the top of one image and at the bottom of the next); after a correct map it
  does not.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from affine_ransac.geometry.affine import apply_affine, fit_affine
from affine_ransac.pipeline import StitchResult, pair_kind, stitch_ties


@dataclass
class IntrafieldMap:
    x_nm: np.ndarray      # (nodes,) node positions along x, relative to the image centre, nm
    y_nm: np.ndarray      # (nodes,) node positions along y, nm
    shift_nm: np.ndarray  # (nodes, nodes, 2) [row = y, column = x]: (SEM after its image's affine) − design
                          # at the node, nm (_value_at_node); 0 at a node without contacts
    count: np.ndarray     # (nodes, nodes) contacts nearest to each node
    images: int           # images used (images with too few contacts are left out)


def estimate_map(design_local: list[np.ndarray], sem_local: list[np.ndarray], nodes: int = 9,
                 full_fraction: float = 0.9) -> IntrafieldMap:
    """The distortion shared by all images, measured against the design.

    design_local, sem_local: per image, (N_k, 2) design and SEM positions of the same contacts (row k
    of both is one contact), relative to the image centre, nm. Images with fewer contacts than
    full_fraction × the median are left out: they cover only part of the image, so their own affine
    would take up part of the distortion. nodes × nodes nodes (nodes >= 2) over the area the contacts
    cover, from edge to edge.
    """
    counts = np.array([len(d) for d in design_local])
    use = [k for k in range(len(counts)) if counts[k] >= max(3, full_fraction * np.median(counts))]
    positions, residuals = [], []
    for k in use:
        model = fit_affine(sem_local[k], design_local[k])  # this image's own affine, SEM -> design
        positions.append(design_local[k])
        residuals.append(apply_affine(model, sem_local[k]) - design_local[k])
    positions, residuals = np.concatenate(positions), np.concatenate(residuals)

    # Nodes from the lowest to the highest contact position (on the image edges: the overlaps lie
    # there, and nothing has to be extrapolated); each contact belongs to its nearest node.
    x_nodes = np.linspace(positions[:, 0].min(), positions[:, 0].max(), nodes)
    y_nodes = np.linspace(positions[:, 1].min(), positions[:, 1].max(), nodes)
    column = np.rint((positions[:, 0] - x_nodes[0]) / (x_nodes[1] - x_nodes[0])).astype(int)
    row = np.rint((positions[:, 1] - y_nodes[0]) / (y_nodes[1] - y_nodes[0])).astype(int)
    shift, count = np.zeros((nodes, nodes, 2)), np.zeros((nodes, nodes), int)
    for r in range(nodes):
        for c in range(nodes):
            near = (row == r) & (column == c)
            count[r, c] = near.sum()
            if count[r, c]:
                shift[r, c] = _value_at_node(positions[near] - (x_nodes[c], y_nodes[r]), residuals[near])
    return IntrafieldMap(x_nodes, y_nodes, shift, count, len(use))


def _value_at_node(offsets: np.ndarray, values: np.ndarray) -> np.ndarray:
    """(2,) value at offset 0 of a plane fitted to values (K, 2) at offsets (K, 2) from a node.

    A plane, not the median: the contacts near a node can sit at a few positions away from it (e.g.
    when the image step is a whole number of pitches, every image has them at the same place), and
    at the image edge they all lie on one side. Values more than 5 median absolute deviations from
    the median are left out first, so strongly shifted contacts do not pull the plane.
    """
    deviation = np.abs(values - np.median(values, axis=0))
    keep = np.all(deviation <= 5 * np.median(deviation, axis=0) + 1e-9, axis=1)
    design = np.column_stack([np.ones(keep.sum()), offsets[keep]])
    coef, *_ = np.linalg.lstsq(design, values[keep], rcond=None)  # rows: value, d/dx, d/dy
    return coef[0]


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
