"""Find the contours of any pattern in an SEM image (docs/SPEC.md S2, D155).

Nothing is assumed about the pattern: no shape, size or direction, and a region need not lie inside the image
(contacts: features.contact; horizontal lines: features.lines). A contour is a run of edge points in order, each with
the direction its edge faces: out of its region.

- detect_contours(): the same threshold as for contacts (features.contact.contact_mask). Every region of the mask
  (area limits as for contacts) gives its whole boundary: its outer one and that of each hole in it. The points on
  the image's outermost rows and columns are not edges (the region runs out of the image there) and are taken out:
  a boundary that never gets there is one closed contour, the others fall into open contours, each ending at the
  image border.
- refine_contours(): each point moves along its direction to the edge by the rule used for contacts
  (features.edges: mid-level, steepest gradient, threshold). Mid-level: each region has its own two levels, as each
  contact.
- in_image_contours(): contours found in an upscaled image (features.contact.upscaled_image) in the image's own
  pixels.

Pixel frame: (x, y) = (column, row), y pointing down, pixel centres at whole numbers (docs/SPEC.md §3).
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

from affine_ransac.features.contact import contact_mask
from affine_ransac.features.edges import RULES, crossings, edge_offsets, sample_profiles, without_isolated
from affine_ransac.step_log import StepLog, timed

LEFT_OUT = ("small", "large")  # why a region gives no contour


@dataclass
class Contours:
    """The contours found in one image."""
    points: list[np.ndarray]   # N arrays (M, 2): the points of each contour, in order, (x, y) px
    normals: list[np.ndarray]  # N arrays (M, 2): at each point the unit direction its edge faces, out of its region
    closed: np.ndarray         # (N,) bool: the last point joins the first; False: both ends at the image border
    region: np.ndarray         # (N,) number of the region each contour bounds (with a hole, or cut: several of one)
    threshold: float           # the detection's threshold (grey level)
    left_out: dict[str, int] = field(default_factory=dict)  # regions without a contour, per reason (LEFT_OUT)


def detect_contours(image: np.ndarray, dark: bool = True, blur_sigma: float = 1.5, min_area_px: int = 20,
                    max_area_px: int | None = None, method: str = "otsu",
                    normal_span: int = 3, log: StepLog | None = None, equalize: str = "none",
                    clahe_clip: float = 2.0, clahe_tiles: int = 8) -> tuple[Contours, np.ndarray]:
    """The contours of a grayscale uint8 image at the pixels of the threshold mask (module docstring), and the region
    number of every pixel (0: none), which refine_contours needs. dark: the regions are the darker side.
    log: if given, gets the time of each step and the counts they depend on: the regions of the mask and those kept,
    the pixels in the kept regions' bounding boxes (each region is traced in its own), the boundaries around holes.
    equalize, clahe_clip, clahe_tiles: as for contact_mask (D167); the threshold mask only."""
    with timed(log, "threshold mask"):
        mask, threshold = contact_mask(image, dark, blur_sigma, method, equalize, clahe_clip, clahe_tiles)
    with timed(log, "regions of the mask"):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4 if method == "band" else 8)
    points, normals, closed, region = [], [], [], []
    left_out = dict.fromkeys(LEFT_OUT, 0)
    loop_start = time.perf_counter()
    tracing = cutting = 0.0          # seconds in the two steps of the kept regions
    kept = box_pixels = holes = 0    # kept regions, pixels in their bounding boxes, their boundaries around holes
    for label in range(1, count):  # label 0 is not mask
        left, top, width, height, area = stats[label]
        if area < min_area_px:
            left_out["small"] += 1
            continue
        if max_area_px is not None and area > max_area_px:
            left_out["large"] += 1
            continue
        start = time.perf_counter()
        pixels = (labels[top:top + height, left:left + width] == label).astype(np.uint8)
        boundaries, hierarchy = cv2.findContours(pixels, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        traced = time.perf_counter()
        kept, box_pixels = kept + 1, box_pixels + int(width) * int(height)
        holes += int((hierarchy[0][:, 3] >= 0).sum())
        for boundary, (_, _, _, parent) in zip(boundaries, hierarchy[0]):
            boundary = boundary.reshape(-1, 2).astype(float) + [left, top]  # crop -> image coordinates
            side = out_of_polygon(boundary) * (-1.0 if parent >= 0 else 1.0)  # a hole: out of the region is into it
            for piece, whole in inside_image(boundary, mask.shape):
                points.append(piece)
                normals.append(side * right_of_tangent(piece, normal_span, whole))
                closed.append(whole)
                region.append(label)
        tracing, cutting = tracing + traced - start, cutting + time.perf_counter() - traced
    if log is not None:
        log.add("go through the regions, leave out by size", time.perf_counter() - loop_start - tracing - cutting)
        log.add("trace every kept region in its bounding box", tracing)
        log.add("cut the boundaries at the image border, direction of every point", cutting)
        log.counts.update(regions=count - 1, kept=kept, box_pixels=box_pixels, holes=holes)
    return Contours(points, normals, np.array(closed, bool), np.array(region, int), float(threshold), left_out), labels


def out_of_polygon(boundary: np.ndarray) -> float:
    """+1 if right_of_tangent's vectors point out of a closed boundary (M, 2), -1 if into it: by the sign of its
    area, as features.edges.outward_normals."""
    x, y = boundary[:, 0], boundary[:, 1]
    signed_area = (x * np.roll(y, -1) - np.roll(x, -1) * y).sum() / 2
    return 1.0 if signed_area > 0 else -1.0


def inside_image(boundary: np.ndarray, shape: tuple[int, int]) -> list[tuple[np.ndarray, bool]]:
    """A closed boundary (M, 2) without its points on the outermost rows and columns of an image of this shape:
    [(points, closed)]. None there: the boundary itself, closed. Otherwise the runs of points between them, each
    open (runs of fewer than 2 points are dropped: no direction)."""
    rows, cols = shape
    x, y = boundary[:, 0], boundary[:, 1]
    on_border = (x == 0) | (y == 0) | (x == cols - 1) | (y == rows - 1)
    if not on_border.any():
        return [(boundary, True)]
    start = np.flatnonzero(on_border)[0]  # begin on the border, so that no run goes on from the last point to the first
    boundary, on_border = np.roll(boundary, -start, axis=0), np.roll(on_border, -start)
    runs = np.split(boundary, np.flatnonzero(np.diff(on_border)) + 1)  # in turn on the border, off it, on it, ...
    return [(run, False) for run in runs[1::2] if len(run) >= 2]


def right_of_tangent(points: np.ndarray, span: int, closed: bool) -> np.ndarray:
    """(M, 2) unit vectors at right angles to a contour (M, 2): (ty, -tx) of its unit tangent (tx, ty), all on the
    same side of it. The tangent at point k runs from point k - span to point k + span, which smooths the pixel
    staircase; an open contour's stops at its ends. Where the two coincide (the tip of a 1 px wide spur) there is no
    tangent: (0, 0), so the point is not moved."""
    k = np.arange(len(points))
    if closed:
        ahead, behind = points[(k + span) % len(points)], points[(k - span) % len(points)]
    else:
        ahead, behind = points[np.minimum(k + span, len(points) - 1)], points[np.maximum(k - span, 0)]
    tangent = ahead - behind
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = np.divide(tangent, length, out=np.zeros_like(tangent), where=length > 0)
    return np.column_stack([tangent[:, 1], -tangent[:, 0]])


def refine_contours(image: np.ndarray, contours: Contours, labels: np.ndarray, dark: bool = True,
                    search_px: float = 5.0, step_px: float = 0.25, blur_sigma: float = 1.0,
                    rule: str = "mid-level", log: StepLog | None = None) -> Contours:
    """The contours with each point moved along its direction (searched from -search_px to +search_px in steps of
    step_px) to its edge by `rule`, on the image lightly smoothed (Gaussian, sigma blur_sigma), as
    features.edges.refine_edges moves a contact's. labels: detect_contours' region numbers. log: if given, gets the
    time of each step."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, not {rule!r}")
    if not contours.points:
        return contours
    sign = 1.0 if dark else -1.0  # bright regions: the rules on the negative image, as for contacts
    with timed(log, "smooth the image for the edges"):
        signed = image.astype(np.float32)
        if blur_sigma > 0:
            cv2.GaussianBlur(signed, (0, 0), blur_sigma, dst=signed)
        if sign < 0:
            np.negative(signed, out=signed)
    offsets = np.arange(-search_px, search_px + step_px / 2, step_px)
    all_points, all_normals = np.concatenate(contours.points), np.concatenate(contours.normals)
    with timed(log, "sample the image across the edge at every point"):
        profiles = sample_profiles(signed, all_points, all_normals, offsets)  # each rises going out of its region
    starts = np.cumsum([len(p) for p in contours.points])[:-1]
    parts = np.split(profiles, starts)
    if rule == "steepest gradient":
        with timed(log, "edge position of every point"):
            shifts = np.split(edge_offsets(profiles, offsets), starts)
    else:
        levels = [sign * contours.threshold] * len(parts) if rule == "threshold" else \
            mid_levels(signed, labels, contours.region, parts, offsets, log)
        with timed(log, "edge position of every point, contour by contour"):
            shifts = [without_isolated(crossings(part, offsets, level), closed=whole)
                      for part, level, whole in zip(parts, levels, contours.closed)]
    with timed(log, "move the points to their edges"):
        moved = [p + shift[:, None] * n for p, shift, n in zip(contours.points, shifts, contours.normals)]
    return Contours(moved, contours.normals, contours.closed, contours.region, contours.threshold, contours.left_out)


def mid_levels(signed: np.ndarray, labels: np.ndarray, region: np.ndarray, parts: list[np.ndarray],
               offsets: np.ndarray, log: StepLog | None = None) -> list[float]:
    """Per contour, the edge level of its region: half-way between the region's interior (the median of `signed`
    over its pixels) and its band (the median over the profiles of all its contours of their brightest value from
    the start outward), as features.edges.level_offsets for one contact. parts: the profiles (M, T) of each contour."""
    numbers = np.unique(region)
    with timed(log, "interior level of every region (group pixels and select medians)"):
        from affine_ransac.features.region_medians import region_medians
        interior = dict(zip(numbers, region_medians(signed, labels, numbers)))
    with timed(log, "band level of every contour (from all edge points of its region)"):
        brightest = defaultdict(list)
        for number, part in zip(region, parts):
            brightest[number].append(part[:, offsets >= 0].max(axis=1))
        levels = {number: (interior[number] + np.median(np.concatenate(values))) / 2
                  for number, values in brightest.items()}
        return [levels[number] for number in region]


def in_image_contours(contours: Contours, factor: int) -> Contours:
    """Contours found in an image upscaled by `factor` (upscaled_image), in the pixel frame of the image itself."""
    if factor == 1:
        return contours
    points = [(p + 0.5) / factor - 0.5 for p in contours.points]
    return Contours(points, contours.normals, contours.closed, contours.region, contours.threshold, contours.left_out)
