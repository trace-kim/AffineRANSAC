"""Stitch tile images into one mosaic image at given mask positions.

Tiles are placed with sub-pixel accuracy (bilinear interpolation). Where tiles overlap, their
pixels are averaged, so a misplacement shows up as doubled ("ghost") patterns.
All tiles must have the same pixel size.
"""

import cv2
import numpy as np


def build_mosaic(images: list[np.ndarray], centers_nm: np.ndarray, pixel_size_nm: float):
    """Mosaic of the images, each placed with its centre at centers_nm[k] (mask nm).

    Returns (mosaic, extent_nm): mosaic is float32 with NaN where no tile covers; extent_nm is
    (left, right, bottom, top) of the mosaic's outer pixel edges in mask nm, for
    matplotlib's imshow(extent=...).
    """
    s = pixel_size_nm
    # Mask position of each tile's top-left pixel centre (pixel y points down, mask y up).
    top_left = np.array([
        (cx - (img.shape[1] - 1) / 2 * s, cy + (img.shape[0] - 1) / 2 * s)
        for img, (cx, cy) in zip(images, centers_nm)
    ])
    x0, y0 = top_left[:, 0].min(), top_left[:, 1].max()  # mosaic pixel (0, 0) centre
    width = int(np.ceil(max((tl[0] - x0) / s + img.shape[1] for tl, img in zip(top_left, images)) - 1e-9))
    height = int(np.ceil(max((y0 - tl[1]) / s + img.shape[0] for tl, img in zip(top_left, images)) - 1e-9))

    total = np.zeros((height, width), np.float32)
    count = np.zeros((height, width), np.float32)
    for img, (tlx, tly) in zip(images, top_left):
        offset_x, offset_y = (tlx - x0) / s, (y0 - tly) / s  # tile pixel (0, 0) in mosaic pixels
        ix, iy = int(np.floor(offset_x)), int(np.floor(offset_y))
        move = np.float32([[1, 0, offset_x - ix], [0, 1, offset_y - iy]])  # sub-pixel remainder
        # One extra row/column holds the part pushed out by the sub-pixel move; clip it at the edge.
        w, h = min(img.shape[1] + 1, width - ix), min(img.shape[0] + 1, height - iy)
        placed = cv2.warpAffine(img.astype(np.float32), move, (w, h), flags=cv2.INTER_LINEAR)
        covered = cv2.warpAffine(np.ones(img.shape, np.float32), move, (w, h), flags=cv2.INTER_LINEAR)
        total[iy:iy + h, ix:ix + w] += placed
        count[iy:iy + h, ix:ix + w] += covered

    mosaic = np.full((height, width), np.nan, np.float32)
    inside = count > 0.5
    mosaic[inside] = total[inside] / count[inside]
    extent = (x0 - s / 2, x0 + (width - 0.5) * s, y0 - (height - 0.5) * s, y0 + s / 2)
    return mosaic, extent
