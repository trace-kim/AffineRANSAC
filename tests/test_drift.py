import numpy as np

from affine_ransac.fitting.drift import drift_curve, stripe_drift
from affine_ransac.registration import row_means

ROWS_Y = 3_000_000.0 + np.arange(0.0, 200_000.0, 120.0)  # 200 µm of rows, 120 nm apart, mask-scale y


def contacts(error_of_y, noise_nm=0.0, seed=0):
    """Design y (N,) and errors (N, 2) of 10 contacts per row: error_of_y(y) (N, 2) + noise."""
    rng = np.random.default_rng(seed)
    y = np.repeat(ROWS_Y, 10)
    return y, error_of_y(y) + rng.normal(0, noise_nm, (len(y), 2))


def line(y):
    return np.column_stack([3.0 + 2e-5 * (y - ROWS_Y[0]), -1.0 - 4e-5 * (y - ROWS_Y[0])])  # 20 / -40 ppm


def bump(y):
    """A drift bend: 2 nm in x, -1.5 nm in y, 15 µm wide, at 120 µm."""
    shape = np.exp(-0.5 * ((y - ROWS_Y[0] - 120_000.0) / 15_000.0) ** 2)
    return np.column_stack([2.0 * shape, -1.5 * shape])


def test_a_straight_trend_is_left_for_the_global_affine():
    y, error = contacts(line)
    drift = drift_curve(y, error, window_nm=20_000)
    np.testing.assert_allclose(drift.curve_nm, drift.line_nm, atol=1e-9)
    np.testing.assert_allclose(drift.correction(y), 0, atol=1e-9)
    # The straight line's terms: 20 / -40 ppm, and its value at the centre.
    np.testing.assert_allclose(drift.line_slope * 1e6, [20.0, -40.0])
    np.testing.assert_allclose(drift.line_intercept_nm, line(np.array([drift.line_centre_y_nm]))[0])
    assert drift.window_nm == 20_000


def test_the_correction_is_the_bend_alone():
    y, error = contacts(lambda y: line(y) + bump(y), noise_nm=0.3)
    drift = drift_curve(y, error, window_nm=10_000)

    # The bump minus its own straight-line part: that part, like the trend, is left for the global affine.
    straight = np.polyfit(ROWS_Y - ROWS_Y.mean(), bump(ROWS_Y), 1)
    expected = bump(ROWS_Y) - (np.outer(ROWS_Y - ROWS_Y.mean(), straight[0]) + straight[1])
    np.testing.assert_allclose(drift.correction(ROWS_Y), expected, atol=0.1)
    np.testing.assert_allclose(drift.row_mean_nm, row_means(y, error)[1])


def test_each_stripe_loses_its_own_drift_and_keeps_the_common_line():
    # Two stripes measured in opposite directions: a drift growing in time runs up in one and down in
    # the other (+-30 ppm in dy), on top of the field's straight trend; stripe 1 also has a bend.
    y = np.concatenate([np.repeat(ROWS_Y, 5), np.repeat(ROWS_Y, 5)])
    stripe = np.repeat([0, 1], 5 * len(ROWS_Y))
    offset = y - ROWS_Y.mean()
    drift = np.column_stack([np.zeros_like(y), np.where(stripe == 0, 3e-5, -3e-5) * offset])
    drift[stripe == 1] += bump(y[stripe == 1])
    error = line(y) + drift + np.random.default_rng(1).normal(0, 0.2, (len(y), 2))

    result = stripe_drift(y, error, stripe, window_nm=10_000)

    corrected = error - np.vstack([result.correction(s, y[stripe == s]) for s in (0, 1)])
    for s in (0, 1):  # every stripe now follows the common straight line
        rows = row_means(y[stripe == s], corrected[stripe == s])[1]
        np.testing.assert_allclose(rows, result.line(ROWS_Y), atol=0.35)
    # The common line is the field's trend (20 / -40 ppm) plus the average of the stripes' drifts.
    np.testing.assert_allclose(result.common.line_slope[1] * 1e6, -40.0 + np.polyfit(ROWS_Y, bump(ROWS_Y)[:, 1], 1)[0] * 1e6 / 2, atol=2)
    np.testing.assert_allclose(result.correction(7, ROWS_Y[:3]), 0)  # a stripe without contacts
