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
    return cv2.remap(image.astype(np.float32), x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def edge_offset(profile: np.ndarray, offsets: np.ndarray, rising: bool = True) -> float:
    """Offset of the steepest rise (or fall) along one profile, refined by a parabola fit."""
    gradient = np.gradient(profile, offsets)
    if not rising:
        gradient = -gradient
    k = int(np.argmax(gradient))
    if 0 < k < len(gradient) - 1:
        left, mid, right = gradient[k - 1], gradient[k], gradient[k + 1]
        curvature = left - 2 * mid + right
        if curvature < 0:  # a real peak: move to the parabola's top (within +-0.5 step)
            k = k + 0.5 * (left - right) / curvature
    return float(np.interp(k, np.arange(len(offsets)), offsets))


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

    centers, areas, contours = [], [], []
    for otsu_contour in contacts.contours:
        points = otsu_contour.astype(np.float64)
        normals = outward_normals(points, normal_span)
        profiles = sample_profiles(smoothed, points, normals, offsets)
        shifts = np.array([edge_offset(p, offsets, rising=dark_contacts) for p in profiles])

        contour = points + shifts[:, None] * normals
        contours.append(contour)
        centers.append(polygon_centroid(contour))
        areas.append(polygon_area(contour))

    return DetectedContacts(
        centers=np.array(centers, dtype=np.float64).reshape(-1, 2),
        areas=np.array(areas, dtype=np.float64),
        contours=contours,
        threshold=contacts.threshold,
    )
