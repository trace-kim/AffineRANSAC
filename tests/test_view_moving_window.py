"""Tests for the moving-window tuner, run without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import numpy as np
import pyqtgraph as pg

from affine_ransac.fitting.moving_window import moving_window_affine
from affine_ransac.fitting.ransac import ransac_affine
from affine_ransac.view_moving_window import MovingWindowTuner
from test_ransac import synthetic


def make_tuner():
    sem, design, *_ = synthetic(n=600)  # 40 um field
    result = ransac_affine(sem, design, threshold_nm=0.5, seed=1)
    pg.mkQApp()
    return MovingWindowTuner(sem, design, result, window_um=20, step_um=5, use_opengl=False), sem, design


def test_tuner_shows_the_moving_window_result_and_follows_the_sliders():
    tuner, sem, design = make_tuner()
    np.testing.assert_allclose(tuner.view.error_nm, moving_window_affine(sem, design, 20_000, 5_000).residuals)

    tuner.window.slider.setValue(30 * 2)  # slider ticks are 0.5 um
    assert tuner.window.value() == 30.0
    tuner.recompute()  # the timer would do this after the slider stops
    expected = moving_window_affine(sem, design, 30_000, 5_000)
    np.testing.assert_allclose(tuner.view.error_nm, expected.residuals)
    x, _ = tuner.term_curve.getData()
    assert len(x) == len(expected.centres_nm)
    tuner.close()


def test_ransac_terms_are_compared_at_the_window_reference_points():
    tuner, sem, design = make_tuner()
    tuner.term.setCurrentText("Mx (ppm)")  # linear term: the same everywhere for the global affine
    _, ransac_mx = tuner.term_ransac.getData()
    np.testing.assert_allclose(ransac_mx, ransac_mx[0])
    tuner.term.setCurrentText("Tx (nm)")
    _, ransac_tx = tuner.term_ransac.getData()
    assert np.ptp(ransac_tx) > 0  # the shift of the global affine depends on where it is taken
    tuner.close()


def test_invalid_settings_keep_the_last_result_and_say_why():
    tuner, *_ = make_tuner()
    before = tuner.view.error_nm.copy()
    tuner.step.spin.setValue(25.0)  # larger than the 20 um window
    tuner.recompute()
    assert "Not updated" in tuner.status.text()
    np.testing.assert_array_equal(tuner.view.error_nm, before)
    tuner.close()
