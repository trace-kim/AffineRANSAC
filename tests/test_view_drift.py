"""Tests for the drift-per-stripe viewer, run without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import numpy as np
import pyqtgraph as pg

from affine_ransac.fitting.drift import stripe_drift
from affine_ransac.view_drift import StripeDriftView

ROWS_Y = np.arange(0.0, 100_000.0, 120.0)


def make_view():
    y = np.concatenate([np.repeat(ROWS_Y, 3)] * 2)
    stripe = np.repeat([0, 1], 3 * len(ROWS_Y))
    error = np.zeros((len(y), 2))
    error[:, 1] = np.where(stripe == 0, 2e-5, -2e-5) * (y - ROWS_Y.mean())  # opposite drifts
    drift = stripe_drift(y, error, stripe, window_nm=10_000)
    pg.mkQApp()
    return StripeDriftView(drift, np.array([0.0, ROWS_Y.mean()]), use_opengl=False), drift


def test_each_stripe_shows_its_row_means_and_the_subtracted_curve():
    view, drift = make_view()
    assert list(view.boxes) == [0, 1] and all(len(lines) == 4 for lines in view.lines.values())
    dy_curve = view.lines[1][3]  # stripe 2: dx thin, dx bold, dy thin, dy bold
    np.testing.assert_allclose(dy_curve.getData()[1], drift.correction(1, drift.stripes[1].row_y_nm)[:, 1], atol=1e-9)
    assert "stripe 2: dx" in view.label.text() and "Common straight line" in view.label.text()
    view.close()


def test_a_stripe_can_be_hidden_in_both_plots():
    view, _ = make_view()
    view.boxes[0].setChecked(False)
    assert not any(line.isVisible() for line in view.lines[0])
    assert all(line.isVisible() for line in view.lines[1])
    view.close()
