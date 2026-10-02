"""Tests for the row pitch viewer, run without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import numpy as np
import pyqtgraph as pg

from affine_ransac.view_pitch import PitchView

# 5 rows of 4 contacts, 100 nm pitch (nm).
DESIGN = np.array([(x, y) for y in np.arange(0, 500, 100.0) for x in np.arange(0, 400, 100.0)])


def test_pitch_of_each_set_and_its_difference_from_the_design():
    bare = DESIGN.copy()
    bare[DESIGN[:, 1] >= 300, 1] += 2.0  # a 2 nm seam between rows 2 and 3
    pg.mkQApp()
    view = PitchView(DESIGN, {"stitched": bare, "affine": DESIGN}, DESIGN.mean(axis=0),
                     tile_edges_y_nm=[250.0], use_opengl=False)

    np.testing.assert_allclose(view.design_pitch, 100.0)
    np.testing.assert_allclose(view.mid_um, (np.array([50, 150, 250, 350]) - 200) / 1000)
    np.testing.assert_allclose(view.pitch["stitched"], [100, 100, 102, 100])
    np.testing.assert_allclose(view.pitch["affine"], 100.0)
    view.show()
    pg.mkQApp().processEvents()
    view.close()
