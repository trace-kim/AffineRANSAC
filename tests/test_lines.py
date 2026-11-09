"""Horizontal lines and their edges (features/lines.py, pipeline.find_lines; docs/SPEC.md S2, D153)."""

import cv2
import numpy as np
import pytest

from affine_ransac.features.edges import RULES, without_isolated
from affine_ransac.features.lines import column_noise_px, detect_lines, edge_profiles, roughness_px
from affine_ransac.pipeline import find_lines

SHAPE = (200, 300)  # rows, columns
LINES = [(30.3, 50.7), (90.0, 112.4), (150.6, 171.2)]  # (upper, lower) edge of each bright line, px (y down)


def line_image(lines=LINES, shape=SHAPE, dark=60.0, bright=180.0, blur=1.0, noise=0.0, seed=0) -> np.ndarray:
    """Bright horizontal lines on a dark background (uint8): each pixel's brightness from the share of it the lines
    cover, blurred (Gaussian, sigma blur px), with Gaussian noise. An edge is a height in px or a function of the
    column x."""
    rows, cols = shape
    x, r = np.arange(cols), np.arange(rows)[:, None]
    image = np.full(shape, dark)
    for upper, lower in lines:
        upper = upper(x) if callable(upper) else np.full(cols, upper)
        lower = lower(x) if callable(lower) else np.full(cols, lower)
        image += (bright - dark) * np.clip(np.minimum(r + 0.5, lower) - np.maximum(r - 0.5, upper), 0, 1)
    image = cv2.GaussianBlur(image, (0, 0), blur)
    image += np.random.default_rng(seed).normal(0, noise, shape)
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


BRIGHT = dict(dark_contacts=False)  # bright lines


def test_each_line_has_an_upper_and_a_lower_edge_at_its_height_in_every_column():
    detected, refined = find_lines(line_image(), **BRIGHT)  # mid-level
    upper, lower = np.array(LINES).T
    assert refined.upper.shape == refined.lower.shape == (3, SHAPE[1])  # from the top down
    np.testing.assert_allclose(refined.upper, np.repeat(upper[:, None], SHAPE[1], axis=1), atol=0.05)
    np.testing.assert_allclose(refined.lower, np.repeat(lower[:, None], SHAPE[1], axis=1), atol=0.05)
    assert np.all(np.abs(detected.upper - upper[:, None]) <= 1) and np.all(np.abs(detected.lower - lower[:, None]) <= 1)
    assert detected.left_out == 0


@pytest.mark.parametrize("rule", RULES)
def test_every_rule_finds_the_edges_within_a_pixel(rule):
    """The threshold rule puts them at the detection's level (Otsu: not half-way here, the lines cover a third of the
    image); the steepest gradient on the steepest pixel step (the profile down a column is linear between pixels)."""
    _, refined = find_lines(line_image(), refinement=rule, **BRIGHT)
    upper, lower = np.array(LINES).T
    assert np.all(np.abs(refined.upper - upper[:, None]) < 1) and np.all(np.abs(refined.lower - lower[:, None]) < 1)


def test_dark_lines_are_the_bright_ones_of_the_negative_image():
    _, bright = find_lines(line_image(), **BRIGHT)
    _, dark = find_lines(255 - line_image(), dark_contacts=True)
    np.testing.assert_allclose(dark.edges, bright.edges, atol=1e-4)


def test_a_line_cut_by_the_image_border_has_one_edge_and_other_regions_are_left_out():
    image = line_image([(-10.0, 20.4), (90.0, 112.4)])
    image[150:170, 100:140] = 180  # a bright patch that does not cross the image
    lines = detect_lines(image, dark_lines=False)
    assert len(lines.upper) == 2 and lines.left_out == 1
    assert np.isnan(lines.upper[0]).all() and np.isfinite(lines.lower[0]).all()  # the top line: its lower edge only
    _, refined = find_lines(image, **BRIGHT)
    np.testing.assert_allclose(refined.lower[0], 20.4, atol=0.05)


def test_the_profile_along_an_edge_is_its_wiggle_without_the_straight_line():
    wiggle = 1.5 * np.sin(2 * np.pi * np.arange(SHAPE[1]) / 100)  # three wiggles along the edge
    image = line_image([(lambda x: 60.2 + 0.01 * x + 1.5 * np.sin(2 * np.pi * x / 100), 85.0)])
    _, refined = find_lines(image, **BRIGHT)
    x = np.arange(SHAPE[1])
    wiggle -= np.polyval(np.polyfit(x, wiggle, 1), x)  # a straight line fitted along it takes its own trend too
    upper_profile, lower_profile = edge_profiles(refined)
    # the tilt of 0.01 px per column removed (not at the first and last columns, where line_image's blur mirrors
    # the image and bends the wiggle)
    np.testing.assert_allclose(upper_profile[3:-3], wiggle[3:-3], atol=0.05)
    np.testing.assert_allclose(lower_profile, 0, atol=0.05)
    assert roughness_px(refined) == pytest.approx(np.sqrt(np.mean(wiggle ** 2) / 2), rel=0.05)  # both edges


def test_the_noise_per_column_is_that_of_two_images_of_the_same_lines():
    """Two images of the same straight lines with their own noise: each column's edge differs between them by the
    noise of two columns. One image, from its neighbouring columns, must give the same noise (the edges are refined
    on the image blurred across the lines only, so neighbouring columns do not share noise)."""
    _, first = find_lines(line_image(noise=12.0, seed=1), **BRIGHT)
    _, second = find_lines(line_image(noise=12.0, seed=2), **BRIGHT)
    noise = np.sqrt(np.mean((first.edges - second.edges) ** 2) / 2)
    assert 0.05 < noise < 0.5  # the test means something
    assert column_noise_px(first) == pytest.approx(noise, rel=0.15)
    assert roughness_px(first) == pytest.approx(noise, rel=0.15)  # straight lines: their profiles are the noise


@pytest.mark.parametrize("factor", [2, 3])
def test_upscaled_the_edges_are_at_the_image_columns_in_its_pixels(factor):
    _, refined = find_lines(line_image(), upscale=factor, **BRIGHT)
    upper, lower = np.array(LINES).T
    assert refined.upper.shape == (3, SHAPE[1])
    np.testing.assert_allclose(refined.upper, np.repeat(upper[:, None], SHAPE[1], axis=1), atol=0.05)
    np.testing.assert_allclose(refined.lower, np.repeat(lower[:, None], SHAPE[1], axis=1), atol=0.05)


def test_an_open_edge_has_no_neighbours_past_its_ends():
    ramp = np.linspace(0.0, 10.0, 50)  # a contour closed over its ends would see 10 beside 0
    np.testing.assert_array_equal(without_isolated(ramp, closed=False), ramp)
    assert without_isolated(ramp)[0] != ramp[0]
    spike = ramp.copy()
    spike[0] = 5.0  # isolated, with neighbours on one side only
    assert without_isolated(spike, closed=False)[0] == pytest.approx(np.median(ramp[1:4]))
