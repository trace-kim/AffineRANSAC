import numpy as np
import pytest

from affine_ransac.fitting.moving_window import moving_window_affine, window_centres
from affine_ransac.geometry.affine import apply_affine

# 200 nm pitch grid, 10 um wide, 100 um tall (nm).
DESIGN = np.array([(x, y) for x in np.arange(0, 10_000, 200.0) for y in np.arange(0, 100_000, 200.0)])


def test_window_centres_span_the_field():
    centres = window_centres(np.array([0.0, 100_000.0]), 40_000, 15_000)
    np.testing.assert_allclose(centres, [20_000, 35_000, 50_000, 65_000, 80_000])
    np.testing.assert_allclose(window_centres(np.array([0.0, 30_000.0]), 40_000, 5_000), [15_000])


def test_one_global_affine_is_removed_by_every_window():
    true = np.array([[1 + 20e-6, -5e-6, 30.0], [8e-6, 1 - 10e-6, -12.0], [0, 0, 1]])
    sem = apply_affine(np.linalg.inv(true), DESIGN)  # true maps SEM onto design

    result = moving_window_affine(sem, DESIGN, 40_000, 5_000)

    np.testing.assert_allclose(result.residuals, 0, atol=1e-6)
    np.testing.assert_allclose(result.models[:, :2, :2], np.tile(true[:2, :2], (len(result.models), 1, 1)), atol=1e-9)


def test_each_contact_uses_the_window_with_the_nearest_centre():
    result = moving_window_affine(DESIGN.copy(), DESIGN, 40_000, 5_000)
    distance = np.abs(DESIGN[:, 1, None] - result.centres_nm[None])
    np.testing.assert_array_equal(result.window, distance.argmin(axis=1))
    assert (distance[np.arange(len(DESIGN)), result.window] <= 20_000).all()  # inside its own window
    assert (result.counts > 0).all()


def test_slow_drift_along_y_is_absorbed_by_the_windows_but_not_by_one_affine():
    # A bow: x is displaced by up to 10 nm, quadratic in y. One affine for the field cannot follow it.
    y = DESIGN[:, 1]
    sem = DESIGN + np.column_stack([10.0 * ((y - 50_000) / 50_000) ** 2, np.zeros(len(y))])

    windowed = moving_window_affine(sem, DESIGN, 20_000, 2_000)
    single = moving_window_affine(sem, DESIGN, 100_000, 100_000)  # one window = the whole field

    assert len(single.centres_nm) == 1
    assert np.abs(windowed.residuals).max() < 0.2 * np.abs(single.residuals).max()


def test_bad_settings_and_empty_windows_raise():
    with pytest.raises(ValueError, match="step_nm"):
        moving_window_affine(DESIGN, DESIGN, 10_000, 20_000)
    sparse = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [0.0, 90_000.0]])
    with pytest.raises(ValueError, match="need 3"):
        moving_window_affine(sparse, sparse, 40_000, 10_000)
