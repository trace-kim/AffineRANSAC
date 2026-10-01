"""Write synthetic SEM-like JPEG images for tests and for trying the viewer.

Run directly to create a sample image:
    python tests/sample_sem.py sample_sem.jpg
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def write_sem_like(
    path,
    rows=768,
    cols=1024,
    pitch_px=64,
    radius_px=14,
    databar_px=64,
    seed=0,
):
    """Write a grayscale JPEG of a contact-hole array and return the true centres.

    Contacts are dark discs with a bright rim, on a grey background, with blur and
    noise. A black data bar with text is drawn along the bottom edge.

    Returns:
        (N, 2) array of contact centres (x, y) in pixels.
    """
    image = np.full((rows, cols), 110, dtype=np.uint8)

    usable_rows = rows - databar_px
    xs = np.arange(pitch_px // 2, cols - radius_px, pitch_px)
    ys = np.arange(pitch_px // 2, usable_rows - radius_px, pitch_px)
    centers = np.array([(x, y) for y in ys for x in xs])
    for x, y in centers:
        cv2.circle(image, (int(x), int(y)), radius_px + 2, 200, thickness=3)  # bright rim
        cv2.circle(image, (int(x), int(y)), radius_px, 40, thickness=-1)  # dark hole

    image = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
    noise = np.random.default_rng(seed).normal(0, 8, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    image[usable_rows:] = 0
    cv2.putText(image, "SEM  HV 1.0 kV  x100k  FOV 3.0 um", (10, rows - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)

    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    encoded.tofile(str(path))  # works for non-ASCII paths, unlike cv2.imwrite
    return centers.astype(np.float64)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_sem.jpg")
    n = len(write_sem_like(out))
    print(f"Wrote {out} ({n} contacts)")
