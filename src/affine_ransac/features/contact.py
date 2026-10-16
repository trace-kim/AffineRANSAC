"""Detect contact holes in an SEM image by thresholding.

Steps:
  1. Blur slightly to suppress pixel noise.
  2. Threshold -> binary mask of candidate contact pixels (dark contacts by default), by method:
     "otsu":  plain Otsu (two grey-level classes).
     "otsu3": three classes (hole interior | background | bright band at the hole edge,
              multi-Otsu); the threshold between the hole and the background.
     "band":  the bright band: pixels above the threshold between background and band (three
              classes). The mask is everything that is NOT band; the background is the large region
              touching the border (dropped in step 4), so the regions left are those enclosed by a
              band. It does not need the hole interior to be darker than the background.
  3. Split the mask into connected regions; each region is one candidate contact ("band": 4-connected,
     so a region cannot leak diagonally through a thin band).
  4. Drop regions that touch the image border (cut-off contacts, the data bar, the background for
     "band"), whose area is outside [min_area_px, max_area_px], or whose solidity (area / convex hull
     area) is below min_solidity. Contacts are convex (solidity ~1); background pinched off between
     nearly touching bands is concave (~0.7-0.88). "band" uses min_solidity 0.9 unless given.
  5. Centre = centroid of the region's pixels; contour = its outer boundary.
contact_contrast() then tells real contacts from regions cut out of plain noise (images without pattern):
the contrast of each contour against a ring around it, in noise sigma (keep_contacts drops the others).

All coordinates are in the pixel frame: (x, y) = (column, row) in pixels, y pointing DOWN,
pixel centres at integer values (docs/SPEC.md §3).
"""

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu

METHODS = ("otsu", "otsu3", "band")
BAND_MIN_SOLIDITY = 0.9


@dataclass
class DetectedContacts:
    centers: np.ndarray  # (N, 2) centroid (x, y) in pixels
    areas: np.ndarray  # (N,) area in pixels
    contours: list[np.ndarray]  # N arrays of shape (M, 2), boundary points (x, y) in pixels
    threshold: float  # threshold that was used (grey level; for "band" the band threshold)


def contact_mask(image: np.ndarray, dark_contacts: bool = True, blur_sigma: float = 1.5, method: str = "otsu"):
    """Blur, then threshold by method (module docstring). Returns (mask, threshold); mask is 1 on
    candidate contact pixels. Comparisons as in cv2: <= threshold is the dark side."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=blur_sigma) if blur_sigma > 0 else image
    if method == "otsu":
        mode = cv2.THRESH_BINARY_INV if dark_contacts else cv2.THRESH_BINARY
        threshold, mask = cv2.threshold(blurred, 0, 1, mode + cv2.THRESH_OTSU)
        return mask, threshold
    low, high = threshold_multiotsu(blurred, classes=3)
    if method == "otsu3":
        threshold = low if dark_contacts else high
        mask = blurred <= threshold if dark_contacts else blurred > threshold
    else:  # band: bright band around dark contacts, dark band around bright ones
        threshold = high if dark_contacts else low
        mask = blurred <= threshold if dark_contacts else blurred > threshold
    return mask.astype(np.uint8), float(threshold)


def solidity(contour: np.ndarray, area: float) -> float:
    """Pixel area / area of the convex hull of the contour (slightly above 1 for convex pixel regions)."""
    hull_area = cv2.contourArea(cv2.convexHull(contour.astype(np.int32)))
    return area / hull_area if hull_area > 0 else 0.0


def detect_contacts(
    image: np.ndarray,
    dark_contacts: bool = True,
    blur_sigma: float = 1.5,
    min_area_px: int = 20,
    max_area_px: int | None = None,
    method: str = "otsu",
    min_solidity: float | None = None,
) -> DetectedContacts:
    """Find contacts in a grayscale uint8 image. See the module docstring for the steps."""
    mask, threshold = contact_mask(image, dark_contacts, blur_sigma, method)
    if min_solidity is None and method == "band":
        min_solidity = BAND_MIN_SOLIDITY
    connectivity = 4 if method == "band" else 8
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    rows, cols = image.shape

    centers, areas, contours = [], [], []
    for label in range(1, count):  # label 0 is the background
        left, top, width, height, area = stats[label]
        touches_border = left == 0 or top == 0 or left + width == cols or top + height == rows
        too_small = area < min_area_px
        too_large = max_area_px is not None and area > max_area_px
        if touches_border or too_small or too_large:
            continue
        contour = _outer_contour(labels, label, left, top, width, height)
        if min_solidity is not None and solidity(contour, area) < min_solidity:
            continue

        centers.append(centroids[label])
        areas.append(area)
        contours.append(contour)

    return DetectedContacts(
        centers=np.array(centers, dtype=np.float64).reshape(-1, 2),
        areas=np.array(areas, dtype=np.float64),
        contours=contours,
        threshold=float(threshold),
    )


def noise_sigma(image: np.ndarray, blur_sigma: float = 1.5) -> float:
    """Robust pixel noise of the image (grey levels): 1.4826 x the median absolute deviation of the image minus
    its blurred copy. Spatially correlated noise makes it come out lower than the real noise."""
    detail = image.astype(np.float64) - cv2.GaussianBlur(image.astype(np.float64), (0, 0), sigmaX=blur_sigma)
    return 1.4826 * float(np.median(np.abs(detail - np.median(detail))))


def contact_contrast(image: np.ndarray, contacts: DetectedContacts, ring_px: int = 3) -> np.ndarray:
    """(N,) |mean grey level inside each contour - mean in a ring ring_px wide around it| / noise_sigma.
    Real contacts stand out by many noise sigma; regions that thresholding cut out of plain noise (an image
    without pattern) by about one."""
    sigma = max(noise_sigma(image), 1e-6)
    kernel = np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)
    rows, cols = image.shape
    contrast = np.zeros(len(contacts.contours))
    for k, contour in enumerate(contacts.contours):
        low = np.maximum(contour.min(axis=0) - ring_px - 1, 0).astype(int)
        high = np.minimum(contour.max(axis=0) + ring_px + 2, (cols, rows)).astype(int)
        crop = image[low[1]:high[1], low[0]:high[0]].astype(np.float64)
        inside = np.zeros(crop.shape, np.uint8)
        cv2.fillPoly(inside, [np.round(contour - low).astype(np.int32)], 1)
        ring = cv2.dilate(inside, kernel) - inside
        if inside.any() and ring.any():
            contrast[k] = abs(crop[inside > 0].mean() - crop[ring > 0].mean()) / sigma
    return contrast


def keep_contacts(contacts: DetectedContacts, keep: np.ndarray) -> DetectedContacts:
    """The contacts where keep (N,) bool is True, in the same order."""
    return DetectedContacts(contacts.centers[keep], contacts.areas[keep],
                            [c for c, k in zip(contacts.contours, keep) if k], contacts.threshold)


def _outer_contour(labels, label, left, top, width, height) -> np.ndarray:
    """Outer boundary of one labelled region as an (M, 2) array of (x, y) pixel coordinates."""
    region = (labels[top:top + height, left:left + width] == label).astype(np.uint8)
    found, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = max(found, key=len).reshape(-1, 2)  # one region -> one outer contour
    return boundary + [left, top]  # crop coordinates -> image coordinates
