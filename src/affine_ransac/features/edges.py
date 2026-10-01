"""Refine Otsu contact contours to the edge of maximum intensity gradient.

For each contact, rays go out from its Otsu centre. Along each ray the (lightly smoothed)
intensity is sampled, and the edge is placed where the intensity changes fastest in the
expected direction: dark -> bright going outward for dark contacts (bright -> dark for
bright ones). The search is limited to a band around the Otsu radius. Edge points are
found to sub-pixel precision; the refined centre is the area centroid of the polygon
through them.

Coordinates are pixels (x, y), y pointing down, pixel centres at integers (docs/SPEC.md §3).
"""

import cv2
import numpy as np

from affine_ransac.features.contact import DetectedContacts
from affine_ransac.geometry.polygon import polygon_area, polygon_centroid


def sample_rays(image: np.ndarray, center, angles: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Bilinearly sampled intensity at center + r * (cos a, sin a), shape (len(angles), len(radii))."""
    x = (center[0] + np.outer(np.cos(angles), radii)).astype(np.float32)
    y = (center[1] + np.outer(np.sin(angles), radii)).astype(np.float32)
    return cv2.remap(image.astype(np.float32), x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def edge_radius(profile: np.ndarray, radii: np.ndarray, rising: bool = True) -> float:
    """Radius of the steepest rise (or fall) along one ray, refined by a parabola fit."""
    gradient = np.gradient(profile, radii)
    if not rising:
        gradient = -gradient
    k = int(np.argmax(gradient))
    if 0 < k < len(gradient) - 1:
        left, mid, right = gradient[k - 1], gradient[k], gradient[k + 1]
        curvature = left - 2 * mid + right
        if curvature < 0:  # a real peak: move to the parabola's top (within +-0.5 step)
            k = k + 0.5 * (left - right) / curvature
    return float(np.interp(k, np.arange(len(radii)), radii))


def refine_edges(
    image: np.ndarray,
    contacts: DetectedContacts,
    dark_contacts: bool = True,
    n_rays: int = 64,
    search_fraction: float = 0.5,
    blur_sigma: float = 1.0,
    step_px: float = 0.25,
) -> DetectedContacts:
    """Refined contours and centres for contacts found by detect_contacts.

    Each ray searches radii within +-search_fraction of the contact's Otsu radius
    (radius of a circle with the same area).
    """
    smoothed = cv2.GaussianBlur(image.astype(np.float32), (0, 0), blur_sigma) if blur_sigma > 0 else image
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

    centers, areas, contours = [], [], []
    for center, area in zip(contacts.centers, contacts.areas):
        otsu_radius = np.sqrt(area / np.pi)
        radii = np.arange((1 - search_fraction) * otsu_radius, (1 + search_fraction) * otsu_radius, step_px)
        profiles = sample_rays(smoothed, center, angles, radii)
        r_edge = np.array([edge_radius(p, radii, rising=dark_contacts) for p in profiles])

        contour = center + np.column_stack([r_edge * np.cos(angles), r_edge * np.sin(angles)])
        contours.append(contour)
        centers.append(polygon_centroid(contour))
        areas.append(polygon_area(contour))

    return DetectedContacts(
        centers=np.array(centers, dtype=np.float64).reshape(-1, 2),
        areas=np.array(areas, dtype=np.float64),
        contours=contours,
        threshold=contacts.threshold,
    )
