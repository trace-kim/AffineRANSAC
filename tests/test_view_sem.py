"""Tests for the SEM viewer, run without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import numpy as np
import pyqtgraph as pg

from affine_ransac.view_sem import build_window, pixel_at
from sample_sem import write_sem_like


def test_pixel_at_uses_integer_pixel_centres():
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)  # 3 rows, 4 cols

    assert pixel_at(image, 0.0, 0.0) == (0, 0, 0)
    assert pixel_at(image, 0.49, 0.0) == (0, 0, 0)
    assert pixel_at(image, 0.51, 0.0) == (1, 0, 1)  # crossed into the next column
    assert pixel_at(image, 3.0, 2.0) == (3, 2, 11)  # last pixel: row 2, col 3
    assert pixel_at(image, -0.51, 0.0) is None
    assert pixel_at(image, 0.0, 2.51) is None


def test_build_window_places_pixel_centres_on_integers(tmp_path):
    path = tmp_path / "tile.jpg"
    write_sem_like(path, rows=120, cols=160, databar_px=20)
    app = pg.mkQApp()

    win = build_window(path)
    win.show()
    app.processEvents()

    image_item = win.image_view.getImageItem()
    assert image_item.image.shape == (120, 160)  # not transposed
    # Top-left corner of pixel (0, 0) must sit at (-0.5, -0.5), so its centre is at (0, 0).
    corner = image_item.mapToView(pg.QtCore.QPointF(0, 0))
    assert (corner.x(), corner.y()) == (-0.5, -0.5)
    win.close()
