"""Tests for the registration error viewer, run without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from affine_ransac.fitting.ransac import ransac_affine
from affine_ransac.view_registration import RegistrationView, colorize, rasterize
from test_ransac import synthetic


def test_rasterize_one_pixel_per_point_and_mean_where_points_share_a_pixel():
    points = np.array([[0.5, 0.5], [0.6, 0.6], [3.5, 1.5]])
    grid = rasterize(points, np.array([1.0, 3.0, 5.0]), rect=(0, 0, 4, 2), shape=(2, 4), radius_px=0)
    assert grid[0, 0] == 2.0  # two points in the bottom-left pixel: their mean
    assert grid[1, 3] == 5.0  # row 0 is the bottom, so y = 1.5 is row 1
    assert np.isnan(grid).sum() == 6


def test_rasterize_draws_disks_when_zoomed_in():
    grid = rasterize(np.array([[5.5, 5.5]]), np.array([2.0]), rect=(0, 0, 11, 11), shape=(11, 11), radius_px=2)
    assert np.nansum(grid > 0) == 13  # a radius-2 disk: 13 pixels
    assert grid[5, 7] == 2.0 and np.isnan(grid[5, 8])


def make_view():
    sem, design, *_ = synthetic(n=600)
    result = ransac_affine(sem, design, threshold_nm=0.5, seed=1)
    pg.mkQApp()
    view = RegistrationView(design, result.residuals, result.reference, use_opengl=False)
    return view, design, result


def test_maps_show_every_contact_and_share_the_view():
    view, design, result = make_view()
    view.show()
    pg.mkQApp().processEvents()
    (plot_x, image_x, _), (plot_y, _, _) = view.maps
    shown = (image_x.image.max(axis=2) > 0).sum()  # coloured (not background) pixels
    assert shown >= len(design) * 0.5  # zoomed out: about one pixel per contact (some share a pixel)
    plot_x.vb.setRange(xRange=(-1, 1), yRange=(-1, 1), padding=0)
    assert np.allclose(plot_y.vb.viewRect().getRect(), plot_x.vb.viewRect().getRect())
    view.rasterize_maps()  # the timer would do this after a zoom
    low, high = view.color_bar.levels()
    assert low == -high and high > 0
    view.close()


def test_row_profile_is_the_mean_error_of_each_row():
    sem, design, *_ = synthetic(n=600)
    # Put the contacts on 5 rows; add a row-dependent dx to the error.
    rows_y = design[:, 1].min() + 4000.0 * np.arange(5)
    design[:, 1] = rows_y[np.arange(len(design)) % 5]
    error = np.zeros_like(design)
    error[:, 0] = np.arange(len(design)) % 5 * 0.1  # row k: +0.1 k nm in dx
    pg.mkQApp()
    view = RegistrationView(design, error, design.mean(axis=0), use_opengl=False)
    assert len(view.row_y) == 5 and view.row_count.sum() == len(design)
    np.testing.assert_allclose(view.row_mean[:, 0], 0.1 * np.arange(5), atol=1e-9)
    view.close()


def test_colorize_jet_and_background_where_empty():
    rgba = colorize(np.array([[-1.0, 0.0, 1.0, np.nan]]), levels=(-1, 1))
    assert rgba.shape == (1, 4, 3) and rgba.dtype == np.uint8
    np.testing.assert_array_equal(rgba[0, 0, :3], [0, 0, 128])    # low end: dark blue
    np.testing.assert_array_equal(rgba[0, 2, :3], [128, 0, 0])    # high end: dark red
    assert (rgba[0, 3] == 0).all()                                # NaN: black background


def test_set_errors_updates_rows_and_summary_but_keeps_the_colour_scale():
    view, design, result = make_view()
    levels = view.color_bar.levels()
    view.set_errors(np.zeros_like(result.residuals), "nothing")
    np.testing.assert_allclose(view.row_mean, 0)
    assert "after nothing" in view.label.text()
    assert view.color_bar.levels() == levels
    view.close()


def test_raw_reference_row_profile_is_optional_and_shares_the_row_axis():
    view, design, result = make_view()
    assert view.raw_plot is None
    view.close()

    sem, design, *_ = synthetic(n=600)
    rows_y = design[:, 1].min() + 4000.0 * np.arange(5)
    design[:, 1] = rows_y[np.arange(len(design)) % 5]
    raw = np.zeros_like(design)
    raw[:, 1] = 10.0 + np.arange(len(design)) % 5  # row k: dy = 10 + k nm, no affine removed
    pg.mkQApp()
    view = RegistrationView(design, np.zeros_like(design), design.mean(axis=0), use_opengl=False, raw_error_nm=raw)
    np.testing.assert_allclose(view.raw_row_mean[:, 1], 10.0 + np.arange(5), atol=1e-9)
    np.testing.assert_allclose(view.row_mean, 0)  # the corrected profile is unchanged
    assert view.raw_plot.getViewBox().linkedView(0) is view.row_plot.getViewBox()
    view.close()


def test_add_rows_draws_another_set_beside_the_original_lines():
    view, design, result = make_view()
    other = design[::2]
    error = np.zeros_like(other)
    error[:, 0] = 1.5
    view.add_rows(other, error, "corrected")

    assert list(view.line_boxes) == [("contacts", "dx"), ("contacts", "dy"), ("corrected", "dx"), ("corrected", "dy")]
    (line,) = view.line_items["corrected", "dx"]  # no reference plot in this view: the row profile only
    np.testing.assert_allclose(line.getData()[1], 1.5)
    np.testing.assert_allclose(view.error_nm, result.residuals)  # the view's own errors are unchanged
    first, second = view.label.text().splitlines()
    assert first.startswith("Registration error after") and second.startswith(f"corrected: {len(other)} contacts")
    view.close()


def test_show_bar_holds_three_sets_per_row_in_the_given_colours():
    view, design, result = make_view()
    for k in range(4):
        view.add_rows(design, result.residuals, f"set {k}")
    view.add_rows(design, result.residuals, "chosen", colors=("#123456", "#654321"))
    box = view.line_boxes["set 2", "dx"]  # the 4th set of the bar (the view's own contacts first)
    assert view.show_bar.getItemPosition(view.show_bar.indexOf(box))[:2] == (1, 2)  # 2nd row, 1st set's dx
    (line,) = view.line_items["chosen", "dy"]
    assert line.opts["pen"].color().name() == "#654321"
    view.close()


def test_a_check_box_hides_its_lines_in_both_row_plots_and_the_y_axis_follows():
    sem, design, *_ = synthetic(n=600)
    pg.mkQApp()
    view = RegistrationView(design, np.zeros_like(design), design.mean(axis=0), use_opengl=False,
                            raw_error_nm=np.zeros_like(design), name="uncorrected")
    big = np.zeros_like(design)
    big[:, 1] = 50.0
    view.add_rows(design, big, "corrected", raw_error_nm=big)

    row_line, raw_line = view.line_items["corrected", "dy"]  # one line in each row plot
    assert row_line in view.row_plot.listDataItems() and raw_line in view.raw_plot.listDataItems()
    view.line_boxes["corrected", "dy"].setChecked(False)
    assert not row_line.isVisible() and not raw_line.isVisible()
    assert all(line.isVisible() for line in view.line_items["uncorrected", "dy"])
    view.row_plot.getViewBox().updateAutoRange()
    assert view.row_plot.getViewBox().viewRange()[1][1] < 50  # rescaled to the visible lines
    view.line_boxes["corrected", "dy"].setChecked(True)
    assert row_line.isVisible() and raw_line.isVisible()
    view.close()


def test_external_reference_adds_row_lines_and_rings_in_the_colour_scale():
    sem, design, *_ = synthetic(n=600)
    rows_y = design[:, 1].min() + 4000.0 * np.arange(5)
    design[:, 1] = rows_y[np.arange(len(design)) % 5]
    pg.mkQApp()
    view = RegistrationView(design, np.zeros_like(design), design.mean(axis=0), use_opengl=False)
    view.color_bar.setLevels((-1.0, 1.0))
    # Three sites on two of the rows (x spread out): dx -1 / +1 on row 0, dy 0.5 on row 3.
    sites = np.array([[design[:, 0].min(), rows_y[0]], [design[:, 0].max(), rows_y[0]], [design[:, 0].mean(), rows_y[3]]])
    errors = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.5]])
    view.add_external("tool A", sites, errors)

    row_dx, outline_dx, ring_dx = view.line_items["tool A", "dx"]  # its row line, its dx map rings
    row_y_um, mean_dx = row_dx.getData()
    np.testing.assert_allclose(row_y_um, (rows_y[[0, 3]] - view.reference_nm[1]) / 1000)
    np.testing.assert_allclose(mean_dx, [0.0, 0.0])  # row 0: mean of -1 and +1
    assert ring_dx in view.maps[0][0].items and len(ring_dx.data) == 3
    _, _, ring_dy = view.line_items["tool A", "dy"]
    assert ring_dy in view.maps[1][0].items
    # Hollow rings in the maps' jet colours: -1 = low end (dark blue), +1 = high end (dark red).
    assert ring_dx.points()[0].pen().color().getRgb()[:3] == (0, 0, 128)
    assert ring_dx.points()[1].pen().color().getRgb()[:3] == (128, 0, 0)
    assert ring_dx.points()[0].brush().style() == QtCore.Qt.BrushStyle.NoBrush  # no fill: the contact shows
    view.color_bar.setLevels((-2.0, 2.0))  # recoloured when the user drags the colour bar:
    view.color_bar.sigLevelsChanged.emit(view.color_bar)  # (setLevels alone does not emit it)
    assert ring_dx.points()[1].pen().color().getRgb()[:3] != (128, 0, 0)
    assert "tool A (external reference): 3 sites" in view.label.text()
    assert "dx +1.000 nm" in ring_dx.points()[1].data()  # hover tip

    view.dot_radius_px = 20  # zoomed in: the rings grow to enclose the contact disks
    view.size_external()
    assert ring_dx.points()[0].size() > 40 and outline_dx.points()[0].size() == ring_dx.points()[0].size()

    view.line_boxes["tool A", "dx"].setChecked(False)  # hides its dx line and dx rings, not dy
    assert not row_dx.isVisible() and not ring_dx.isVisible() and not outline_dx.isVisible() and ring_dy.isVisible()
    view.close()
