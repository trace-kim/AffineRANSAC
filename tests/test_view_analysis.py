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


def make_window(stitch_view=None):
    sem, design, *_ = synthetic(n=600)
    ransac = ransac_affine(sem, design, threshold_nm=0.5, seed=1)
    moving = moving_window_affine(sem, design, 20_000, 5_000)
    pg.mkQApp()
    window = AnalysisWindow(sem, design, ransac, moving, window_um=20, step_um=5, seed=1,
                            stitch_view=stitch_view, use_opengl=False)
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
