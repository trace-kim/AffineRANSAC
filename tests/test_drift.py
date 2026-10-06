import numpy as np

from affine_ransac.fitting.drift import drift_curve
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
