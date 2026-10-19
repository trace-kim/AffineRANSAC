import numpy as np
import pytest

from affine_ransac.fitting.moving_window import moving_window_affine, moving_window_per_stripe, window_centres
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


def two_stripes(rng=None):
    """Observations of a 200 nm grid in two stripes, x 0-10 um and 9.6-19.8 um (the 3 columns in both are observed
    twice), 50 um tall: (design, stripe, groups), groups = the observations of each contact."""
    design = np.array([(x, y) for x in np.arange(0, 20_000, 200.0) for y in np.arange(0, 50_000, 200.0)])
    observations, stripe, groups = [], [], []
    for point in design:
        mine = [s for s, (low, high) in enumerate([(0, 10_000), (9_600, 19_800)]) if low <= point[0] <= high]
        groups.append(np.arange(len(observations), len(observations) + len(mine)))
        observations += [point] * len(mine)
        stripe += mine
    return np.array(observations), np.array(stripe), groups


def test_each_stripe_drifts_on_its_own_and_its_windows_remove_it():
    # The stripes were measured at different times: each has its own scale and rotation. Windows per stripe remove
    # both; windows spanning both stripes cannot.
    design, stripe, groups = two_stripes()
    affines = [np.array([[1 + 30e-6, -10e-6, 4.0], [12e-6, 1 - 25e-6, -3.0], [0, 0, 1]]),
               np.array([[1 - 20e-6, 15e-6, -2.0], [-8e-6, 1 + 40e-6, 5.0], [0, 0, 1]])]
    sem = np.empty_like(design)
    for s, true in enumerate(affines):
        sem[stripe == s] = apply_affine(np.linalg.inv(true), design[stripe == s])  # true maps SEM onto design

    result = moving_window_per_stripe(sem, design, stripe, groups, 20_000, 5_000)

    np.testing.assert_allclose(result.residuals, 0, atol=1e-6)
    assert result.residuals.shape == (len(groups), 2)
    merged_sem = np.array([sem[g].mean(axis=0) for g in groups])
    merged_design = np.array([design[g].mean(axis=0) for g in groups])
    together = moving_window_affine(merged_sem, merged_design, 20_000, 5_000)
    assert np.abs(together.residuals).max() > 0.1  # nm


def test_a_contact_seen_by_two_stripes_gets_the_mean_of_both_corrections():
    design, stripe, groups = two_stripes()
    sem = design + np.random.default_rng(3).normal(0, 0.2, design.shape)

    result = moving_window_per_stripe(sem, design, stripe, groups, 20_000, 5_000)

    per_stripe = np.empty_like(design)
    for s in (0, 1):
        per_stripe[stripe == s] = moving_window_affine(sem[stripe == s], design[stripe == s], 20_000, 5_000).residuals
    twice = [k for k, g in enumerate(groups) if len(g) == 2]
    assert len(twice) == 3 * 250  # the 3 shared columns
    np.testing.assert_allclose(result.residuals[twice], [per_stripe[groups[k]].mean(axis=0) for k in twice])
    windows = len(window_centres(design[stripe == 0, 1], 20_000, 5_000))
    np.testing.assert_array_equal(result.stripe, [0] * windows + [1] * windows)  # the windows of each stripe in turn
    assert (result.stripe[result.window] == stripe[[g[0] for g in groups]]).all()  # its first observation's stripe


def test_one_stripe_gives_the_moving_window_of_every_contact():
    sem = DESIGN + np.random.default_rng(4).normal(0, 0.2, DESIGN.shape)
    groups = [np.array([k]) for k in range(len(DESIGN))]

    result = moving_window_per_stripe(sem, DESIGN, np.zeros(len(DESIGN), int), groups, 40_000, 5_000)

    every = moving_window_affine(sem, DESIGN, 40_000, 5_000)
    np.testing.assert_array_equal(result.residuals, every.residuals)
    np.testing.assert_array_equal(result.window, every.window)
    np.testing.assert_array_equal(result.stripe, np.zeros(len(every.centres_nm)))


def test_a_stripe_too_sparse_for_a_window_is_named():
    points = np.vstack([DESIGN, [[20_000.0, 0.0], [20_100.0, 0.0], [20_000.0, 90_000.0]]])
    stripe = np.r_[np.zeros(len(DESIGN), int), [1, 1, 1]]
    with pytest.raises(ValueError, match="Stripe 2: window"):
        moving_window_per_stripe(points, points, stripe, [np.array([k]) for k in range(len(points))], 40_000, 10_000)
