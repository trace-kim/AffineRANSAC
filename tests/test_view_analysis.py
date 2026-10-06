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


def make_window(stitch_view=None, affine_stitching=None):
    sem, design, ransac, moving = analysed()
    pg.mkQApp()
    window = AnalysisWindow(sem, design, ransac, moving, window_um=20, step_um=5, seed=1,
                            stitch_view=stitch_view, use_opengl=False, affine_stitching=affine_stitching)
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


def test_affine_stitching_adds_lines_and_one_tab():
    affine = analysed(seed=5, n=500)  # other contacts than the translation-stitched set
    affine_sem, affine_design, affine_ransac, affine_moving = affine
    window, sem, design, ransac, moving = make_window(affine_stitching=affine)

    tabs = [window.tabText(i) for i in range(window.count())]
    assert tabs == ["RANSAC monitor", "Registration, RANSAC", "Registration, moving window",
                    "Moving-window tuner", "Row pitch", "Affine stitching"]
    # The original lines stay; the affine-stitched rows are added beside them.
    np.testing.assert_allclose(window.ransac_view.error_nm, ransac.residuals)
    expected = row_means(affine_design[:, 1], affine_ransac.residuals, 10.0)[1]
    np.testing.assert_allclose(line_data(window.ransac_view.row_plot, "dy, affine stitching")[1], expected[:, 1])
    expected = row_means(affine_design[:, 1], affine_moving.residuals, 10.0)[1]
    np.testing.assert_allclose(line_data(window.moving_view.row_plot, "dx, affine stitching")[1], expected[:, 0])
    expected = row_means(affine_design[:, 1], affine_sem - affine_design, 10.0)[1]
    np.testing.assert_allclose(line_data(window.ransac_view.raw_plot, "dx, no affine, affine stitching")[1],
                               expected[:, 0])
    assert "affine stitching: 500 contacts" in window.ransac_view.label.text()
    assert list(window.pitch_view.pitch) == ["stitched, no affine", "after RANSAC affine",
                                             "affine stitching, no affine", "affine stitching, after RANSAC affine"]
    # Its own tab: the affine-stitched registration error, on the same colour scale.
    np.testing.assert_allclose(window.affine_view.error_nm, affine_ransac.residuals)
    assert window.affine_view.color_bar.levels() == window.ransac_view.color_bar.levels()
    window.close()
