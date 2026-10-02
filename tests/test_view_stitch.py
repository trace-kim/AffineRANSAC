"""Tests for the stitch viewer, run without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from affine_ransac.features.contact import detect_contacts
from affine_ransac.geometry.frames import pixel_to_tile_nm
from affine_ransac.overlap_fit import OverlapFit
from affine_ransac.pipeline import PairResult, StitchResult, TileResult
from affine_ransac.view_stitch import DESIGN, IMAGES, OUTLIERS, REFINED, StitchViewer, outlier_indices

PIXEL = 2.0  # nm


def make_tile(center):
    image = np.full((40, 60), 200, np.uint8)
    image[10:16, 20:26] = 20  # one dark contact
    found = detect_contacts(image, blur_sigma=0)
    return TileResult(image=image, center_nm=np.array(center, float), fov_nm=np.array([120.0, 80.0]),
                      pixel_size_nm=PIXEL, otsu=found, refined=found, design_centers=np.array([[0.0, 0.0]]))


def make_viewer():
    # Mask-scale centres: the viewer must draw relative to its local origin.
    tiles = [make_tile((5_000_000.0, 2_000_000.0)), make_tile((5_000_100.0, 2_000_000.0))]
    fit = OverlapFit(shift=np.zeros(2), rotation=0.0, center=np.zeros(2),
                     inliers=np.array([False]), residuals=np.zeros((1, 2)))
    pair = PairResult(0, 1, (5_000_040.0, 5_000_060.0, 1_999_960.0, 2_000_040.0), np.array([0]), np.array([0]), fit)
    stitch = StitchResult([pair], corrections=np.array([[1.5, -0.5], [-1.5, 0.5]]))
    square = [np.array([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]])]
    pg.mkQApp()
    return tiles, StitchViewer(tiles, ["A", "B"], [square, square], stitch)


def item(viewer, layer, k):
    return next(i for name, tile, i in viewer.items if name == layer and tile == k)


def view_xy(graphics_item, x, y):
    point = graphics_item.mapToParent(QtCore.QPointF(x, y))
    return np.array([point.x(), point.y()])


def test_image_pixel_centres_land_on_their_mask_positions():
    tiles, viewer = make_viewer()
    # Pixel (col 3, row 5): its centre is at item coordinates (3.5, 5.5).
    expected = pixel_to_tile_nm([[3, 5]], (40, 60), PIXEL)[0] + tiles[1].center_nm - viewer.origin
    np.testing.assert_allclose(view_xy(item(viewer, IMAGES, 1), 3.5, 5.5), expected)
    viewer.close()


def test_stitched_placement_moves_sem_items_not_the_design():
    tiles, viewer = make_viewer()
    nominal = tiles[0].center_nm - viewer.origin

    viewer.stitched_button.setChecked(True)
    np.testing.assert_allclose(view_xy(item(viewer, REFINED, 0), 0, 0), nominal + [1.5, -0.5])
    np.testing.assert_allclose(view_xy(item(viewer, DESIGN, 0), 0, 0), nominal)

    viewer.nominal_button.setChecked(True)
    np.testing.assert_allclose(view_xy(item(viewer, REFINED, 0), 0, 0), nominal)
    viewer.close()


def test_layers_and_tiles_can_be_hidden():
    _, viewer = make_viewer()
    assert item(viewer, REFINED, 0).isVisible()

    viewer.layer_boxes[REFINED].setChecked(False)
    assert not item(viewer, REFINED, 0).isVisible() and item(viewer, IMAGES, 0).isVisible()

    viewer.tile_list.item(0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert not item(viewer, IMAGES, 0).isVisible() and item(viewer, IMAGES, 1).isVisible()
    viewer.close()


def test_outliers_are_marked_in_both_tiles():
    _, viewer = make_viewer()
    assert item(viewer, OUTLIERS, 0) is not None and item(viewer, OUTLIERS, 1) is not None
    viewer.close()


def test_outlier_indices():
    fit = OverlapFit(np.zeros(2), 0.0, np.zeros(2), np.array([True, False, True]), np.zeros((3, 2)))
    stitch = StitchResult([PairResult(0, 2, (0, 1, 0, 1), np.array([4, 7, 9]), np.array([1, 2, 3]), fit)], np.zeros((3, 2)))
    found = outlier_indices(stitch, 3)
    assert [list(f) for f in found] == [[7], [], [2]]
