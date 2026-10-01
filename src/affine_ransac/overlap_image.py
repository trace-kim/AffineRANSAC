"""B − A shift of two overlapping tiles from their overlap PIXELS (phase correlation).

Independent of contact detection and contour extraction. The same nominal mask rectangle (the
overlap box) is cut from both images, and an upsampled cross-correlation (scikit-image,
Guizar-Sicairos method; mean removed, Hann window) measures how far tile B's content is moved
relative to tile A's. Translation only.

Precision: exact for pure shifts, but on real-like overlap strips (100–200 px tall) about
0.1–0.2 px, because the two crops never differ by a pure shift (noise, content entering and
leaving the crop). Tried and rejected: cv2.phaseCorrelate (off by up to 0.38 px even for exact
shifts) and scikit-image's "phase" normalisation (up to 0.6 px on narrow strips).

The result is in the same convention as overlap_fit: B − A in mask nm (y up), i.e. for a
contact seen by both tiles, (its nominal mask position in B) − (its nominal position in A).
"""

import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

from affine_ransac.geometry.frames import pixel_to_tile_nm, tile_nm_to_pixel


def overlap_crops(image_a, center_a, image_b, center_b, pixel_size_nm: float, box):
    """Cut the overlap box from both images.

    Returns (crop_a, crop_b, remainder_px). crop_b starts at the whole pixel nearest to where
    the box corner should be in B; remainder_px = (that pixel − the exact position), (x, y).
    """
    x_min, x_max, y_min, y_max = box
    # Box corners in A pixels: top-left is (x_min, y_max) because pixel y points down.
    corners = np.array([[x_min, y_max], [x_max, y_min]]) - center_a
    (left, top), (right, bottom) = tile_nm_to_pixel(corners, image_a.shape, pixel_size_nm)
    x0, y0 = int(np.ceil(left)), int(np.ceil(top))
    x1, y1 = int(np.floor(right)) + 1, int(np.floor(bottom)) + 1
    rows, cols = image_a.shape
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, cols), min(y1, rows)

    # The same mask point as A's crop corner, in B pixels (generally not a whole pixel).
    corner_mask = pixel_to_tile_nm([[x0, y0]], image_a.shape, pixel_size_nm)[0] + center_a
    exact_b = tile_nm_to_pixel([corner_mask - center_b], image_b.shape, pixel_size_nm)[0]
    xb0, yb0 = np.round(exact_b).astype(int)
    width, height = x1 - x0, y1 - y0
    if xb0 < 0 or yb0 < 0 or xb0 + width > image_b.shape[1] or yb0 + height > image_b.shape[0]:
        raise ValueError("The overlap box is not inside tile B")

    crop_a = image_a[y0:y1, x0:x1]
    crop_b = image_b[yb0:yb0 + height, xb0:xb0 + width]
    return crop_a, crop_b, np.array([xb0, yb0]) - exact_b


def image_overlap_shift(image_a, center_a, image_b, center_b, pixel_size_nm: float, box):
    """B − A shift (nm, mask frame) of the overlap content, and the match error.

    center_*: nominal tile centres (x, y) in mask nm; box: overlap box from overlap_box().
    The error (0..1, from scikit-image) is low for a good match and high for an unreliable one.
    """
    crop_a, crop_b, remainder_px = overlap_crops(image_a, center_a, image_b, center_b, pixel_size_nm, box)
    window = cv2.createHanningWindow((crop_a.shape[1], crop_a.shape[0]), cv2.CV_32F)
    a = (crop_a - crop_a.mean()) * window
    b = (crop_b - crop_b.mean()) * window
    register, error, _ = phase_cross_correlation(a, b, upsample_factor=100, normalization=None)

    # scikit-image returns the (row, col) shift that moves crop B back onto crop A, so crop B's
    # content moved by its negative, as (x, y) px, right/down positive. Crop B started
    # remainder_px later than exact, which moved its content back by that much: add it back.
    move_px = -np.array([register[1], register[0]]) + remainder_px
    return np.array([move_px[0], -move_px[1]]) * pixel_size_nm, float(error)  # px (y down) -> nm (y up)
