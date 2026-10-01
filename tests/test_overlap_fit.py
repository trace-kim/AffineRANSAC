import numpy as np
import pytest

from affine_ransac.overlap_fit import apply_fit, fit_overlap

# One row of 16 contacts along a 2.7 um strip, like a real overlap strip.
STRIP = np.column_stack([np.arange(16) * 180.0 - 1350.0, np.full(16, -1300.0)])


def rotate(points, angle, center):
    cos, sin = np.cos(angle), np.sin(angle)
    return (points - center) @ np.array([[cos, sin], [-sin, cos]]) + center


def test_pure_translation_is_recovered_exactly():
    fit = fit_overlap(STRIP, STRIP + [-6.0, 4.0])
    np.testing.assert_allclose(fit.shift, [-6, 4])
    assert fit.rotation == 0 and fit.inliers.all() and fit.rms() < 1e-9


def test_outlier_is_flagged_and_does_not_pull_the_shift():
    rng = np.random.default_rng(0)
    b = STRIP + [-6.0, 4.0] + rng.normal(0, 0.3, STRIP.shape)
    b[3] += [2.0, -3.0]  # one contact measured badly

    fit = fit_overlap(STRIP, b)

    assert not fit.inliers[3] and fit.inliers.sum() >= 13
    np.testing.assert_allclose(fit.shift, [-6, 4], atol=0.25)
    assert fit.rms() < 0.6


def test_rotation_is_recovered_about_the_centroid():
    angle = 300e-6  # 300 urad: 0.8 nm across the 2.7 um strip
    center = STRIP.mean(axis=0)
    b = rotate(STRIP, angle, center) + [1.5, -2.0]

    rigid = fit_overlap(STRIP, b, rotation=True)
    shift_only = fit_overlap(STRIP, b)

    assert abs(rigid.rotation - angle) < 1e-9
    np.testing.assert_allclose(rigid.shift, [1.5, -2.0], atol=1e-9)
    np.testing.assert_allclose(apply_fit(rigid, STRIP), b, atol=1e-9)
    assert rigid.rms() < 1e-9 < shift_only.rms()  # translation alone cannot explain a rotation


def test_no_points_raises():
    with pytest.raises(ValueError):
        fit_overlap(np.empty((0, 2)), np.empty((0, 2)))
