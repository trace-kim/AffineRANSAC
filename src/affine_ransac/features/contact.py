"""Detect contact holes in an SEM image with a simple Otsu threshold.

Steps:
  1. Blur slightly to suppress pixel noise.
  2. Otsu threshold -> binary mask of contact pixels (dark contacts by default).
  3. Split the mask into connected regions; each region is one candidate contact.
  4. Drop regions that touch the image border (cut-off contacts, the data bar)
     or whose area is outside [min_area_px, max_area_px].
  5. Centre = centroid of the region's pixels; contour = its outer boundary.

All coordinates are in the pixel frame: (u, v) = (column, row), v pointing down,
pixel centres at integer values (docs/SPEC.md §3).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectedContacts:
    centers: np.ndarray  # (N, 2) centroid (u, v) in pixels
    areas: np.ndarray  # (N,) area in pixels
    contours: list[np.ndarray]  # N arrays of shape (M, 2), boundary points (u, v)
    threshold: float  # Otsu threshold that was used (grey level)


def otsu_mask(image: np.ndarray, dark_contacts: bool = True, blur_sigma: float = 1.5):
    """Blur, then Otsu threshold. Returns (mask, threshold); mask is 1 on contact pixels."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=blur_sigma) if blur_sigma > 0 else image
    mode = cv2.THRESH_BINARY_INV if dark_contacts else cv2.THRESH_BINARY
    threshold, mask = cv2.threshold(blurred, 0, 1, mode + cv2.THRESH_OTSU)
    return mask, threshold


def detect_contacts(
    image: np.ndarray,
    dark_contacts: bool = True,
    blur_sigma: float = 1.5,
    min_area_px: int = 20,
    max_area_px: int | None = None,
) -> DetectedContacts:
    """Find contacts in a grayscale uint8 image. See the module docstring for the steps."""
    mask, threshold = otsu_mask(image, dark_contacts, blur_sigma)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rows, cols = image.shape

    centers, areas, contours = [], [], []
    for label in range(1, count):  # label 0 is the background
        left, top, width, height, area = stats[label]
        touches_border = left == 0 or top == 0 or left + width == cols or top + height == rows
        too_small = area < min_area_px
        too_large = max_area_px is not None and area > max_area_px
        if touches_border or too_small or too_large:
            continue

        centers.append(centroids[label])
        areas.append(area)
        contours.append(_outer_contour(labels, label, left, top, width, height))

    return DetectedContacts(
        centers=np.array(centers, dtype=np.float64).reshape(-1, 2),
        areas=np.array(areas, dtype=np.float64),
        contours=contours,
        threshold=float(threshold),
    )


def _outer_contour(labels, label, left, top, width, height) -> np.ndarray:
    """Outer boundary of one labelled region as an (M, 2) array of (u, v) pixel coordinates."""
    region = (labels[top:top + height, left:left + width] == label).astype(np.uint8)
    found, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = max(found, key=len).reshape(-1, 2)  # one region -> one outer contour
    return boundary + [left, top]  # crop coordinates -> image coordinates
