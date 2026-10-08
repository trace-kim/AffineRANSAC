"""Tests for the tabbed analysis window, run without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from affine_ransac.fitting.moving_window import moving_window_affine
from affine_ransac.fitting.ransac import ransac_affine
from affine_ransac.registration import row_means
from affine_ransac.view_analysis import AnalysisWindow
from test_ransac import synthetic


def analysed(seed=0, n=600):
    sem, design, *_ = synthetic(n=n, seed=seed)
    return sem, design, ransac_affine(sem, design, threshold_nm=0.5, seed=1), moving_window_affine(sem, design, 20_000, 5_000)


def make_window(stitch_view=None, extra=None):
    sem, design, ransac, moving = analysed()
    pg.mkQApp()
    window = AnalysisWindow(sem, design, ransac, moving, window_um=20, step_um=5, seed=1,
                            stitch_view=stitch_view, use_opengl=False, extra=extra)
    return window, sem, design, ransac, moving


def test_one_tab_per_viewer_with_the_given_results():
    window, sem, design, ransac, moving = make_window()
    tabs = [window.tabText(i) for i in range(window.count())]
    assert tabs == ["RANSAC monitor", "Registration, RANSAC", "Registration, moving window",
                    "Moving-window tuner", "Row pitch"]
    np.testing.assert_allclose(window.ransac_view.error_nm, ransac.residuals)
    np.testing.assert_allclose(window.moving_view.error_nm, moving.residuals)
    assert not window.monitor.timer.isActive()  # RANSAC monitor waits for Run
    window.close()


def test_row_profiles_show_the_stitched_error_without_affine():
    window, sem, design, *_ = make_window()
    for view in (window.ransac_view, window.moving_view, window.tuner.view):
        assert view.raw_plot is not None
    expected = row_means(design[:, 1], sem - design, 10.0)[1]
    np.testing.assert_allclose(window.ransac_view.raw_row_mean, expected)
    window.close()


def test_stitch_view_is_the_first_tab_when_given():
    stitch_view = QtWidgets.QLabel("stitch")
    window, *_ = make_window(stitch_view)
    assert window.tabText(0) == "Stitching" and window.widget(0) is stitch_view
    window.close()


def line_data(view, plot, name, axis):
    """(x, y) of set name's line of component axis ("dx" / "dy") in one of the view's row plots."""
    (line,) = [line for line in view.line_items[name, axis] if line in plot.listDataItems()]
    return line.getData()


def test_extra_sets_add_lines_and_one_tab_each():
    corrected = analysed(seed=5, n=500)  # other contacts than the main set
    c_sem, c_design, c_ransac, c_moving = corrected
    d_sem, d_design, d_ransac, _ = analysed(seed=6, n=400)
    window, sem, design, ransac, moving = make_window(
        extra={"in-image corrected": corrected, "drift corrected": (d_sem, d_design, d_ransac, None)})

    tabs = [window.tabText(i) for i in range(window.count())]
    assert tabs == ["RANSAC monitor", "Registration, RANSAC", "Registration, moving window",
                    "Moving-window tuner", "Row pitch", "In-image corrected", "Drift corrected"]
    # The original lines stay; each set's rows are added beside them.
    np.testing.assert_allclose(window.ransac_view.error_nm, ransac.residuals)
    expected = row_means(c_design[:, 1], c_ransac.residuals, 10.0)[1]
    np.testing.assert_allclose(line_data(window.ransac_view, window.ransac_view.row_plot, "in-image corrected", "dy")[1], expected[:, 1])
    expected = row_means(d_design[:, 1], d_sem - d_design, 10.0)[1]
    np.testing.assert_allclose(line_data(window.ransac_view, window.ransac_view.raw_plot, "drift corrected", "dx")[1],
                               expected[:, 0])
    expected = row_means(c_design[:, 1], c_moving.residuals, 10.0)[1]
    np.testing.assert_allclose(line_data(window.moving_view, window.moving_view.row_plot, "in-image corrected", "dx")[1], expected[:, 0])
    assert {name for name, _ in window.moving_view.line_boxes} == {"uncorrected", "in-image corrected"}  # no drift moving window
    assert "in-image corrected: 500 contacts" in window.ransac_view.label.text()
    assert "drift corrected: 400 contacts" in window.ransac_view.label.text()
    assert list(window.pitch_view.pitch) == ["stitched, no affine", "after RANSAC affine",
                                             "in-image corrected, after RANSAC affine", "drift corrected, after RANSAC affine"]
    # Each set's own tab: its registration error, on the colour scale of the RANSAC tab.
    np.testing.assert_allclose(window.extra_views["drift corrected"].error_nm, d_ransac.residuals)
    assert window.extra_views["in-image corrected"].color_bar.levels() == window.ransac_view.color_bar.levels()
    window.close()


def test_external_reference_goes_into_every_registration_view():
    d_sem, d_design, d_ransac, _ = analysed(seed=6, n=400)
    window, sem, design, *_ = make_window(extra={"drift corrected": (d_sem, d_design, d_ransac, None)})
    sites = design[:5]
    window.add_external("tool A", sites, np.full((5, 2), 0.3))
    views = window.registration_views()
    assert len(views) == 4 and window.extra_views["drift corrected"] in views and window.tuner.view in views
    for view in views:
        line, _, ring = view.line_items["tool A", "dy"]
        np.testing.assert_allclose(line.getData()[1], 0.3)
        assert len(ring.data) == 5
    window.close()
