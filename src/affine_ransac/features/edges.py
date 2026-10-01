"""Refine Otsu contact contours to the edge of maximum intensity gradient.

Works for any contact shape (circle, ellipse, rectangle, ...): every point of the Otsu
contour is moved along its own outward normal, within a fixed search distance in pixels,
to where the intensity changes fastest in the expected direction: dark -> bright going
outward for dark contacts (bright -> dark for bright ones). Edge positions are found to
sub-pixel precision; the refined centre is the area centroid of the refined contour.

Coordinates are pixels (x, y), y pointing down, pixel centres at integers (docs/SPEC.md §3).
"""

import cv2
import numpy as np

from affine_ransac.features.contact import DetectedContacts
from affine_ransac.geometry.polygon import polygon_area, polygon_centroid


def outward_normals(contour: np.ndarray, span: int = 3) -> np.ndarray:
    """Unit outward normals (M, 2) of a closed contour (M, 2).

    The tangent at point k is taken from point k - span to point k + span, which smooths
    the pixel staircase of an Otsu contour. Works for either point order.
    """
    tangent = np.roll(contour, -span, axis=0) - np.roll(contour, span, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
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
    image = image.astype(np.float32)
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


def refine_edges(
    image: np.ndarray,
    contacts: DetectedContacts,
    dark_contacts: bool = True,
    search_px: float = 5.0,
    step_px: float = 0.25,
    blur_sigma: float = 1.0,
    normal_span: int = 3,
) -> DetectedContacts:
    """Refined contours and centres for contacts found by detect_contacts.

    Each Otsu contour point is searched along its outward normal from -search_px to
    +search_px (in steps of step_px) for the steepest intensity change.
    """
    smoothed = cv2.GaussianBlur(image.astype(np.float32), (0, 0), blur_sigma) if blur_sigma > 0 else image
    offsets = np.arange(-search_px, search_px + step_px / 2, step_px)

    if not contacts.contours:
        return contacts

    # All contour points of all contacts are searched in one go (fast); then split per contact.
    points = [c.astype(np.float64) for c in contacts.contours]
    normals = [outward_normals(p, normal_span) for p in points]
    all_points, all_normals = np.concatenate(points), np.concatenate(normals)
    profiles = sample_profiles(smoothed, all_points, all_normals, offsets)
    shifts = edge_offsets(profiles, offsets, rising=dark_contacts)
    refined_points = all_points + shifts[:, None] * all_normals
    contours = np.split(refined_points, np.cumsum([len(p) for p in points])[:-1])

    centers = [polygon_centroid(c) for c in contours]
    areas = [polygon_area(c) for c in contours]

    return DetectedContacts(
        centers=np.array(centers, dtype=np.float64).reshape(-1, 2),
        areas=np.array(areas, dtype=np.float64),
        contours=contours,
        threshold=contacts.threshold,
    )
