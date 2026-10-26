"""Refine detected contact contours to sub-pixel edges.

Works for any contact shape (circle, ellipse, rectangle, ...): every point of the detected
contour is moved along its own outward normal, within a fixed search distance in pixels, to
the edge on a lightly smoothed copy of the image. Where the edge is depends on the rule
(docs/SPEC.md S2):
- "mid-level" (D122, default): where the profile rises (going outward, for dark contacts) through
  half-way between the contact's interior and its bright band, each contact with its own two levels.
  A level crossing does not differentiate the noise, and own levels follow contrast that changes
  across the image. Points without a crossing, or isolated far from their neighbours, take the
  median of their neighbours; a real deviation spans several neighbouring points and is kept.
- "steepest gradient" (D22, D23): where the intensity changes fastest, dark -> bright going outward
  (bright -> dark for bright contacts). On noisy, low-contrast images single points jump to noise.
- "threshold" (D130): where the profile crosses the detection's threshold (DetectedContacts.threshold), one level
  for the whole image: the detected outline, placed sub-pixel. For images without a bright band (whose band level
  the mid-level rule would take from noise); one level fits only an even background (pipeline background_px).
  Points without a crossing, or isolated, are replaced as for the mid-level.
Edge positions are found to sub-pixel precision; the refined centre is the area centroid of the
refined contour.

Coordinates are pixels (x, y), y pointing down, pixel centres at integers (docs/SPEC.md §3).
"""

import warnings

import cv2
import numpy as np

from affine_ransac.features.contact import DetectedContacts
from affine_ransac.geometry.polygon import polygon_area, polygon_centroid

RULES = ("mid-level", "steepest gradient", "threshold")


def outward_normals(contour: np.ndarray, span: int = 3) -> np.ndarray:
    """Unit outward normals (M, 2) of a closed contour (M, 2).

    The tangent at point k is taken from point k - span to point k + span, which smooths
    the pixel staircase of an Otsu contour. Works for either point order.
    Where the two points coincide (the tip of a 1 px wide spur, which the contour runs out
    and back along), there is no tangent: the normal is (0, 0), so the point is not moved.
    """
    tangent = np.roll(contour, -span, axis=0) - np.roll(contour, span, axis=0)
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = np.divide(tangent, length, out=np.zeros_like(tangent), where=length > 0)
    x, y = contour[:, 0], contour[:, 1]
    signed_area = (x * np.roll(y, -1) - np.roll(x, -1) * y).sum() / 2
    # For a positive signed area the inside is to the left of the tangent, so the outside
    # is to the right: (ty, -tx). Flip for the opposite point order.
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    return normal if signed_area > 0 else -normal


def sample_profiles(image: np.ndarray, points: np.ndarray, directions: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Bilinearly sampled intensity at points[m] + offsets[t] * directions[m], shape (M, T)."""
    x = (points[:, :1] + directions[:, :1] * offsets).astype(np.float32)
    y = (points[:, 1:] + directions[:, 1:] * offsets).astype(np.float32)
    image = image.astype(np.float32, copy=False)  # no second copy of a smoothed (upscaled) image
    chunk = 30000  # cv2.remap allows at most 32767 output rows per call
    return np.concatenate([
        cv2.remap(image, x[i:i + chunk], y[i:i + chunk], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        for i in range(0, len(x), chunk)
    ]).reshape(len(x), len(offsets))


def edge_offsets(profiles: np.ndarray, offsets: np.ndarray, rising: bool = True) -> np.ndarray:
    """Offset of the steepest rise (or fall) along each profile (M, T), refined by a parabola fit.

    offsets must be evenly spaced. Returns (M,) offsets.
    """
    gradient = np.gradient(profiles, offsets, axis=1)
    if not rising:
        gradient = -gradient
    k = np.argmax(gradient, axis=1)
    rows = np.arange(len(k))
    left = gradient[rows, np.clip(k - 1, 0, None)]
    mid = gradient[rows, k]
    right = gradient[rows, np.clip(k + 1, None, gradient.shape[1] - 1)]
    curvature = left - 2 * mid + right
    # Move to the parabola's top (within +-0.5 step), only for a real peak away from the ends.
    is_peak = (k > 0) & (k < gradient.shape[1] - 1) & (curvature < 0)
    shift = np.where(is_peak, 0.5 * (left - right) / np.where(is_peak, curvature, 1.0), 0.0)
    step = offsets[1] - offsets[0]
    return offsets[0] + (k + shift) * step


def interior_level(image: np.ndarray, contour: np.ndarray) -> float:
    """Median grey level of the image over the pixels inside a contour (M, 2); NaN if none."""
    rows, cols = image.shape
    low = np.maximum(np.floor(contour.min(axis=0)).astype(int), 0)
    high = np.minimum(np.ceil(contour.max(axis=0)).astype(int) + 1, (cols, rows))
    inside = np.zeros((high[1] - low[1], high[0] - low[0]), np.uint8)
    cv2.fillPoly(inside, [np.round(contour - low).astype(np.int32)], 1)
    values = image[low[1]:high[1], low[0]:high[0]][inside > 0]
    return float(np.median(values)) if len(values) else np.nan


def level_offsets(profiles: np.ndarray, offsets: np.ndarray, interior: float) -> np.ndarray:
    """Mid-level edge of one contact, (M,) offsets along its profiles (M, T) (rising outward, a dark contact): where
    each profile rises through half-way between `interior` (its interior grey level) and its band (the median over
    its profiles of their brightest value from the start outward, offsets >= 0); see crossings."""
    band = np.median(profiles[:, offsets >= 0].max(axis=1))
    return crossings(profiles, offsets, (interior + band) / 2)


def crossings(profiles: np.ndarray, offsets: np.ndarray, level: float) -> np.ndarray:
    """(M,) offsets where each profile (M, T) rises through `level` going outward: the crossing nearest the start,
    sub-sample by linear interpolation. NaN where a profile does not cross."""
    before, after = profiles[:, :-1], profiles[:, 1:]
    rise = after - before
    at = offsets[:-1] + (offsets[1] - offsets[0]) * (level - before) / np.where(rise != 0, rise, 1)
    distance = np.where((before < level) & (after >= level), np.abs(at), np.inf)
    nearest = distance.argmin(axis=1)
    return np.where(np.isfinite(distance.min(axis=1)), at[np.arange(len(at)), nearest], np.nan)


def without_isolated(shifts: np.ndarray, neighbours: int = 3, tolerance_px: float = 0.5) -> np.ndarray:
    """Shifts (M,) of the points of one closed contour, where a shift that is NaN, or further from the median of its
    `neighbours` on each side than 4 robust sigma (of all its points) + tolerance_px, is replaced by that median (0 if
    its neighbours have none either)."""
    steps = np.array([k for k in range(-neighbours, neighbours + 1) if k])
    around = np.sort(shifts[(np.arange(len(shifts))[:, None] + steps) % len(shifts)], axis=1)  # NaN sort last
    found = np.isfinite(around).sum(axis=1, keepdims=True)
    # the median of each point's finite neighbours: the middle one or two of them (none: NaN, as every value is)
    median = (np.take_along_axis(around, (found - 1) // 2, axis=1) + np.take_along_axis(around, found // 2, axis=1))
    median = median[:, 0] / 2
    with warnings.catch_warnings():  # every point NaN: sigma NaN, handled below
        warnings.simplefilter("ignore", RuntimeWarning)
        sigma = 1.4826 * np.nanmedian(np.abs(shifts - median))
    isolated = ~np.isfinite(shifts) | (np.abs(shifts - median) > 4 * sigma + tolerance_px)
    return np.nan_to_num(np.where(isolated, median, shifts))


def mid_level_shifts(image: np.ndarray, contour: np.ndarray, profiles: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Shifts (M,) along the normals of one dark contact's contour (M, 2) to its mid-level edge (level_offsets), the
    isolated ones replaced (without_isolated). image: the smoothed image the profiles (M, T) were sampled from."""
    interior = interior_level(image, contour)
    if not np.isfinite(interior):  # a contour enclosing no pixel centre: its innermost samples instead
        interior = float(np.median(profiles[:, 0]))
    return without_isolated(level_offsets(profiles, offsets, interior))


def refine_edges(
    image: np.ndarray,
    contacts: DetectedContacts,
    dark_contacts: bool = True,
    search_px: float = 5.0,
    step_px: float = 0.25,
    blur_sigma: float = 1.0,
    normal_span: int = 3,
    rule: str = "mid-level",
) -> DetectedContacts:
    """Refined contours and centres for contacts found by detect_contacts.

    Each detected contour point is searched along its outward normal from -search_px to
    +search_px (in steps of step_px) for its edge by `rule` (module docstring).
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, not {rule!r}")
    smoothed = cv2.GaussianBlur(image.astype(np.float32), (0, 0), blur_sigma) if blur_sigma > 0 else image
    offsets = np.arange(-search_px, search_px + step_px / 2, step_px)

    if not contacts.contours:
        return contacts

    # All contour points of all contacts are sampled in one go (fast); then split per contact.
    points = [c.astype(np.float64) for c in contacts.contours]
    normals = [outward_normals(p, normal_span) for p in points]
    all_points, all_normals = np.concatenate(points), np.concatenate(normals)
    profiles = sample_profiles(smoothed, all_points, all_normals, offsets)
    starts = np.cumsum([len(p) for p in points])[:-1]
    if rule == "steepest gradient":
        shifts = edge_offsets(profiles, offsets, rising=dark_contacts)
    else:
        sign = 1.0 if dark_contacts else -1.0  # bright contacts: the same rule on the negative image
        signed = sign * np.asarray(smoothed, dtype=np.float32)
        parts = np.split(sign * profiles, starts)
        if rule == "threshold":  # one level for every contact: the detection's threshold
            shifts = np.concatenate([without_isolated(crossings(p, offsets, sign * contacts.threshold)) for p in parts])
        else:
            shifts = np.concatenate([mid_level_shifts(signed, c, p, offsets) for c, p in zip(points, parts)])
    refined_points = all_points + shifts[:, None] * all_normals
    contours = np.split(refined_points, starts)

    centers = [polygon_centroid(c) for c in contours]
    areas = [polygon_area(c) for c in contours]

    return DetectedContacts(
        centers=np.array(centers, dtype=np.float64).reshape(-1, 2),
        areas=np.array(areas, dtype=np.float64),
        contours=contours,
        threshold=contacts.threshold,
        left_out=contacts.left_out,
    )
