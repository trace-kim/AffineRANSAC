"""Tests for the stitch viewer, run without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from affine_ransac.features.contact import detect_contacts
from affine_ransac.geometry.frames import pixel_to_tile_nm
from affine_ransac.overlap_fit import OverlapFit
from affine_ransac.pipeline import DesignContacts, PairResult, RejectedPair, StitchResult, TileResult
from affine_ransac.registration import MergedErrors
from affine_ransac.view_stitch import (BOXES, DESIGN, DESIGN_CENTRES, DESIGN_FAILED, ERRORS, FAILED, IMAGES,
                                       NOT_MEASURED, OUTLIERS, REFINED, REFINED_CENTRES, SPREAD, TILE_FRAMES,
                                       UNSTITCHED,
                                       StitchViewer, outlier_indices)

PIXEL = 2.0  # nm
CENTERS = [(5_000_000.0, 2_000_000.0), (5_000_100.0, 2_000_000.0), (5_000_000.0, 2_000_070.0)]
SEM_CORRECTIONS = np.array([[1.5, -0.5], [-1.5, 0.5], [np.nan, np.nan]])  # tile C: SEM not stitched
DESIGN_CORRECTIONS = np.array([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.0]])


def make_tile(center):
    image = np.full((40, 60), 200, np.uint8)
    image[10:16, 20:26] = 20  # one dark contact
    found = detect_contacts(image, blur_sigma=0)
    return TileResult(image=image, center_nm=np.array(center, float), fov_nm=np.array([120.0, 80.0]),
                      pixel_size_nm=PIXEL, otsu=found, refined=found,
                      design=DesignContacts(np.array([[0.0, 0.0]]), False, 1.0, ok=True))


def errors_at(error, skipped):
    """Two contacts: one at tile A's centre with this error (seen once), one in the A|B overlap
    seen twice with a 3 nm spread; the listed tiles not measured."""
    design = np.array([CENTERS[0], (5_000_050.0, 2_000_000.0)])
    errors = np.array([error, (0.5, 0.5)], float)
    return MergedErrors(design_nm=design, sem_nm=design + errors, error_nm=errors, count=np.array([1, 2]),
                        spread_nm=np.array([0.0, 3.0]), members=[np.array([0]), np.array([1, 2])], skipped=skipped)


ERRORS_BY_PLACEMENT = {
    "nominal": errors_at((5.0, 0.0), {1: "design tone not matched (flagged)"}),
    "mean": errors_at((2.0, -1.0), {2: "SEM tile not stitched"}),
    "first": errors_at((1.0, 1.0), {2: "SEM tile not stitched"}),
}


def make_viewer(errors=None):
    # Mask-scale centres: the viewer must draw relative to its local origin.
    tiles = [make_tile(c) for c in CENTERS]
    fit = OverlapFit(shift=np.zeros(2), rotation=0.0, center=np.zeros(2),
                     inliers=np.array([False]), residuals=np.zeros((1, 2)))
    pair = PairResult(0, 1, (5_000_040.0, 5_000_060.0, 1_999_960.0, 2_000_040.0), np.array([0]), np.array([0]), fit)
    failed = RejectedPair(0, 2, (4_999_940.0, 5_000_060.0, 2_000_030.0, 2_000_040.0), in_box=6, matched=0, failed=True)
    stitch = StitchResult([pair], [failed], corrections=SEM_CORRECTIONS)
    design_stitch = StitchResult([pair], [failed], corrections=DESIGN_CORRECTIONS)
    square = [np.array([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]])]
    pg.mkQApp()
    return tiles, StitchViewer(tiles, ["A", "B", "C"], [square] * 3, stitch, design_stitch, errors=errors)


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


def test_placements_move_sem_and_design_by_their_own_corrections():
    tiles, viewer = make_viewer()
    nominal = tiles[0].center_nm - viewer.origin
    np.testing.assert_allclose(view_xy(item(viewer, REFINED, 0), 0, 0), nominal)
    np.testing.assert_allclose(view_xy(item(viewer, DESIGN, 0), 0, 0), nominal)

    viewer.placement_buttons["mean"].setChecked(True)
    np.testing.assert_allclose(view_xy(item(viewer, REFINED, 0), 0, 0), nominal + SEM_CORRECTIONS[0])
    np.testing.assert_allclose(view_xy(item(viewer, DESIGN, 0), 0, 0), nominal + DESIGN_CORRECTIONS[0])

    viewer.placement_buttons["first"].setChecked(True)  # tile A stitched in both: it stays nominal
    np.testing.assert_allclose(view_xy(item(viewer, REFINED, 0), 0, 0), nominal)
    np.testing.assert_allclose(view_xy(item(viewer, DESIGN, 0), 0, 0), nominal)
    b_nominal = tiles[1].center_nm - viewer.origin
    np.testing.assert_allclose(view_xy(item(viewer, REFINED, 1), 0, 0), b_nominal + [-3.0, 1.0])
    np.testing.assert_allclose(view_xy(item(viewer, DESIGN, 1), 0, 0), b_nominal + [-1.0, 0.0])
    viewer.close()


def test_unstitched_tile_is_marked_and_stays_nominal():
    tiles, viewer = make_viewer()
    assert "Stitching incomplete" in viewer.status_label.text()
    assert "SEM not stitched" in viewer.tile_list.item(2).text()
    assert item(viewer, UNSTITCHED, 2) is not None
    assert item(viewer, FAILED, None) is not None and item(viewer, DESIGN_FAILED, None) is not None
    for mode in ("mean", "first"):
        viewer.placement_buttons[mode].setChecked(True)
        np.testing.assert_allclose(view_xy(item(viewer, REFINED, 2), 0, 0), tiles[2].center_nm - viewer.origin)
    viewer.close()


def test_error_lines_follow_the_selected_placement():
    tiles, viewer = make_viewer(ERRORS_BY_PLACEMENT)
    lines = viewer.error_lines
    start = tiles[0].center_nm - viewer.origin

    def end():
        x, y = lines.getData()
        return view_xy(lines, x[1], y[1])

    np.testing.assert_allclose(end(), start + 10 * np.array([5.0, 0.0]))  # nominal, default scale x10
    viewer.placement_buttons["mean"].setChecked(True)
    np.testing.assert_allclose(end(), start + 10 * np.array([2.0, -1.0]))
    viewer.placement_buttons["first"].setChecked(True)
    np.testing.assert_allclose(end(), start + 10 * np.array([1.0, 1.0]))
    viewer.scale_box.setValue(100)
    np.testing.assert_allclose(end(), start + 100 * np.array([1.0, 1.0]))
    assert "×100" in viewer.error_label.text() and "first tile fixed" in viewer.error_label.text()
    viewer.close()


def test_not_measured_frames_follow_the_placement():
    _, viewer = make_viewer(ERRORS_BY_PLACEMENT)
    assert item(viewer, NOT_MEASURED, 1).isVisible() and not item(viewer, NOT_MEASURED, 2).isVisible()
    viewer.placement_buttons["mean"].setChecked(True)
    assert not item(viewer, NOT_MEASURED, 1).isVisible() and item(viewer, NOT_MEASURED, 2).isVisible()
    assert "not measured" in viewer.error_label.text()
    viewer.close()


def test_error_map_shares_zoom():
    _, viewer = make_viewer(ERRORS_BY_PLACEMENT)
    viewer.show()
    viewer.plot.setXRange(0, 50, padding=0)
    assert np.allclose(viewer.error_plot.getPlotItem().vb.viewRect().left(), viewer.plot.getPlotItem().vb.viewRect().left())
    viewer.close()


def test_start_with_heavy_layers_off():
    _, viewer = make_viewer(ERRORS_BY_PLACEMENT)
    shown = {layer for layer, box in viewer.layer_boxes.items() if box.isChecked()}
    assert shown == {IMAGES, DESIGN_CENTRES, REFINED_CENTRES, BOXES, ERRORS, TILE_FRAMES,
                     FAILED, DESIGN_FAILED, UNSTITCHED, NOT_MEASURED, SPREAD}
    viewer.close()


def test_layers_and_tiles_can_be_hidden():
    _, viewer = make_viewer()
    assert not item(viewer, REFINED, 0).isVisible()  # off at start
    viewer.layer_boxes[REFINED].setChecked(True)
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
    stitch = StitchResult([PairResult(0, 2, (0, 1, 0, 1), np.array([4, 7, 9]), np.array([1, 2, 3]), fit)], [], np.zeros((3, 2)))
    found = outlier_indices(stitch, 3)
    assert [list(f) for f in found] == [[7], [], [2]]


def test_merged_overlap_contacts_are_ringed_and_large_spread_flagged():
    _, viewer = make_viewer(ERRORS_BY_PLACEMENT)
    assert len(viewer.merged_rings.data) == 1 and len(viewer.spread_rings.data) == 1  # spread 3 nm > 2 nm
    assert "1 merged" in viewer.error_label.text() and "spread > 2 nm" in viewer.error_label.text()
    viewer.close()
