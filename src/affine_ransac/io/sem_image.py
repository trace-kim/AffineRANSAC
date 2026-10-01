"""Read SEM images (JPEG) as grayscale arrays.

Pixel frame convention (see docs/SPEC.md §3): the array is indexed image[y, x],
where x is the column (to the right) and y is the row (pointing DOWN), in pixels.
(0, 0) is the centre of the top-left pixel; pixel centres are at integer (x, y).
"""

from pathlib import Path

import cv2
import numpy as np


def load_sem_image(path: str | Path) -> np.ndarray:
    """Load an image file as a 2-D uint8 grayscale array, shape (rows, cols).

    Colour images are converted to grayscale. Values are kept exactly as stored
    in the file (uint8); convert to float in the processing step when needed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    # cv2.imread fails silently on Windows for non-ASCII paths (e.g. Korean folder
    # names), so read the raw bytes with numpy and decode them in memory instead.
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def crop_databar(image: np.ndarray, databar_rows: int) -> np.ndarray:
    """Remove the SEM info/data bar from the bottom of the image.

    Removing BOTTOM rows does not change the (x, y) pixel coordinates of the pixels
    that remain, so results on the cropped image need no correction.
    """
    if databar_rows <= 0:
        return image
    return image[: image.shape[0] - databar_rows]
