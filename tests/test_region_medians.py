"""Grouped median selection must preserve contour refinement, including denoising and upscaling."""

import numpy as np
import pytest
from scipy import ndimage

from affine_ransac.features.region_medians import region_medians


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("strided", [False, True])
def test_selected_region_medians_equal_scipy_exactly_and_leave_inputs_unchanged(dtype, strided):
    rng = np.random.default_rng(85)
    image = rng.normal(-80, 30, (41, 53)).astype(dtype)
    labels = rng.integers(0, 12, image.shape, dtype=np.int32)
    if strided:
        image, labels = image[::2, ::2], labels[::2, ::2]
    before, labels_before = image.copy(), labels.copy()
    selected = np.array([9, 2, 9, 7])  # holes/cut contours may share a region; other regions are not requested
    np.testing.assert_array_equal(region_medians(image, labels, selected), ndimage.median(image, labels, selected))
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(labels, labels_before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_even_odd_single_and_background_groups_use_the_original_rounding(dtype):
    image = np.array([[0.1, 0.2, 0.4, 0.8], [-3.1, -3.2, -3.3, 4.9]], dtype=dtype)
    labels = np.array([[1, 1, 2, 2], [2, 3, 4, 0]], dtype=np.int32)
    selected = np.array([0, 1, 2, 3, 4])
    np.testing.assert_array_equal(region_medians(image, labels, selected), ndimage.median(image, labels, selected))
    assert region_medians(image, labels, np.empty(0, int)).shape == (0,)


def test_grouped_selection_does_not_globally_sort_or_copy_all_background(monkeypatch):
    import affine_ransac.features.region_medians as module

    image = np.arange(80 * 100, dtype=np.float32).reshape(80, 100)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[10:20, 10:30], labels[30:33, 50:54] = 1, 2
    expected = ndimage.median(image, labels, [2])
    original, sizes = module._collect_values, []

    def collect(image, labels, positions, values):
        sizes.append(len(values))
        return original(image, labels, positions, values)

    def forbidden(*args, **kwargs):
        raise AssertionError("No global pixel sort")

    monkeypatch.setattr(module, "_collect_values", collect)
    monkeypatch.setattr(np, "lexsort", forbidden)
    np.testing.assert_array_equal(region_medians(image, labels, np.array([2])), expected)
    assert sizes == [12]  # neither the background nor the discarded region is copied


def test_denoised_four_times_upscaled_contours_match_scipy_region_medians(monkeypatch):
    import affine_ransac.features.region_medians as module
    from affine_ransac.pipeline import find_contours
    from affine_ransac.step_log import StepLog
    from test_contours import BRIGHT, shape_image, vertical_lines

    image = shape_image(vertical_lines)
    options = dict(BRIGHT, denoise="total variation", tv_weight=0.1, upscale=4)
    log = StepLog()
    found = find_contours(image, **options, log=log)[1]
    assert log.counts["total variation rounds"] > 0 and "upscale" in log.seconds
    monkeypatch.setattr(module, "region_medians", lambda image, labels, numbers: ndimage.median(image, labels, numbers))
    reference = find_contours(image, **options)[1]
    np.testing.assert_array_equal(found.region, reference.region)
    assert len(found.points) == len(reference.points) == 4
    for actual, expected in zip(found.points, reference.points):
        np.testing.assert_array_equal(actual, expected)
