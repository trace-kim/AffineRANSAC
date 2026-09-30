"""Smoke test: the viewer window builds and draws without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import numpy as np
import pyqtgraph as pg

from affine_ransac.view_design import build_window, polygons_to_path
from sample_design import write_contact_array


def test_polygons_to_path_keeps_polygons_separate():
    squares = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float) + dx for dx in (0, 10)]
    path = polygons_to_path(squares)
    # Two closed squares -> two subpaths, and the outline spans both.
    assert len(path.toSubpathPolygons()) == 2
    rect = path.boundingRect()
    assert (rect.left(), rect.right()) == (0, 11)


def test_build_window(tmp_path):
    path = tmp_path / "a.oas"
    write_contact_array(path)
    app = pg.mkQApp()

    win = build_window(path)
    win.show()
    app.processEvents()

    plot = win.getItem(1, 0)
    view_range = plot.vb.viewRange()  # autoRange should cover the 3x2 array (x 0..400 nm)
    assert view_range[0][0] < 0 and view_range[0][1] > 400
    win.close()
