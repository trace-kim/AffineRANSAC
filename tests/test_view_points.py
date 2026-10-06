"""Tests for the points-only stitching viewer, run without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import numpy as np
import pyqtgraph as pg

from affine_ransac.pipeline import stitch_tiles, tile_boxes_from_points
from affine_ransac.view_points import PointsStitchView, box_outlines
from test_contour_csv import write_csv_tiles
from affine_ransac.io.contour_csv import read_contour_folder


def test_box_outlines_are_closed_rectangles_separated_by_nan():
    x, y = box_outlines([(0, 2, 0, 1), (5, 6, 5, 6)])
    assert len(x) == 12 and np.isnan(x[5]) and np.isnan(y[11])
    np.testing.assert_array_equal(x[:5], [0, 2, 2, 0, 0])
    np.testing.assert_array_equal(y[:5], [0, 0, 1, 1, 0])


def make_view(tmp_path):
    write_csv_tiles(tmp_path)
    tiles = read_contour_folder(tmp_path)
    sem, design = [t.sem_nm for t in tiles], [t.design_nm for t in tiles]
    centers, sizes = tile_boxes_from_points(design, margin_nm=25.0)
    stitch, design_stitch = stitch_tiles(sem, centers, sizes), stitch_tiles(design, centers, sizes)
    pg.mkQApp()
    view = PointsStitchView([t.name for t in tiles], sem, design, centers, sizes, stitch, design_stitch,
                            use_opengl=False)
    return view, sem, design, stitch


def test_shows_every_contact_and_the_overlaps_relative_to_a_whole_um_origin(tmp_path):
    view, sem, design, stitch = make_view(tmp_path)
    assert view.origin[0] % 1000 == 0 and view.origin[1] % 1000 == 0
    x, y = view.items["Design centres"].getData()
    np.testing.assert_allclose(np.column_stack([x, y]) + view.origin, np.concatenate(design))
    assert len(view.items["SEM centres"].getData()[0]) == sum(len(p) for p in sem)
    x, _ = view.items["Overlaps used (SEM)"].getData()
    assert np.isnan(x).sum() == len(stitch.pairs) == 4
    assert view.items["Tiles not stitched"].getData()[0] is None or len(view.items["Tiles not stitched"].getData()[0]) == 0
    assert "4 of 4 tiles stitched" in view.status.text()
    view.close()


def test_tile_names_are_off_until_checked(tmp_path):
    view, *_ = make_view(tmp_path)
    assert len(view.names) == 4 and not any(t.isVisible() for t in view.names)
    view.names_box.setChecked(True)
    assert all(t.isVisible() for t in view.names)
    view.close()
