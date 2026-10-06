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


def line_data(plot, name):
    """(x, y) of the curve called name in plot's legend."""
    item = next(sample.item for sample, label in plot.legend.items if label.text == name)
    return item.getData()


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
    np.testing.assert_allclose(line_data(window.ransac_view.row_plot, "dy, in-image corrected")[1], expected[:, 1])
    expected = row_means(d_design[:, 1], d_sem - d_design, 10.0)[1]
    np.testing.assert_allclose(line_data(window.ransac_view.raw_plot, "dx, no affine, drift corrected")[1],
                               expected[:, 0])
    expected = row_means(c_design[:, 1], c_moving.residuals, 10.0)[1]
    np.testing.assert_allclose(line_data(window.moving_view.row_plot, "dx, in-image corrected")[1], expected[:, 0])
    moving_names = [label.text for _, label in window.moving_view.row_plot.legend.items]
    assert not any("drift corrected" in name for name in moving_names)  # no moving-window result given
    assert "in-image corrected: 500 contacts" in window.ransac_view.label.text()
    assert "drift corrected: 400 contacts" in window.ransac_view.label.text()
    assert list(window.pitch_view.pitch) == ["stitched, no affine", "after RANSAC affine",
                                             "in-image corrected, after RANSAC affine", "drift corrected, after RANSAC affine"]
    # Each set's own tab: its registration error, on the colour scale of the RANSAC tab.
    np.testing.assert_allclose(window.extra_views["drift corrected"].error_nm, d_ransac.residuals)
    assert window.extra_views["in-image corrected"].color_bar.levels() == window.ransac_view.color_bar.levels()
    window.close()
