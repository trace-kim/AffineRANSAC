"""The contours of any pattern (features/contours.py, pipeline.find_contours; docs/SPEC.md S2, D155)."""

import cv2
import numpy as np
import pytest

from affine_ransac.features.edges import RULES
from affine_ransac.pipeline import find_contacts, find_contours
from test_pipeline import FOV, LATTICE, SIZE, render_tile

SHAPE = (200, 300)  # rows, columns
BRIGHT = dict(dark_contacts=False)  # bright regions


def shape_image(inside, shape=SHAPE, dark=60.0, bright=180.0, blur=1.0, noise=0.0, seed=0) -> np.ndarray:
    """A bright pattern on a dark background (uint8): inside(x, y) says where the pattern is (px, y down); each
    pixel's brightness from the share of it the pattern covers (8 x 8 samples), blurred (Gaussian, sigma blur px),
    with Gaussian noise."""
    rows, cols = shape
    fine = (np.arange(8) + 0.5) / 8 - 0.5
    y = (np.arange(rows)[:, None] + fine).reshape(-1, 1)
    x = (np.arange(cols)[:, None] + fine).reshape(1, -1)
    cover = np.broadcast_to(inside(x, y), (rows * 8, cols * 8)).reshape(rows, 8, cols, 8).mean(axis=(1, 3))
    image = cv2.GaussianBlur(dark + (bright - dark) * cover, (0, 0), blur)
    image += np.random.default_rng(seed).normal(0, noise, shape)
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def vertical_lines(x, y):
    """Two lines from the top to the bottom of the image: edges at x = 60.3, 85.7 and 180.0, 201.4."""
    return ((x > 60.3) & (x < 85.7)) | ((x > 180.0) & (x < 201.4))


def diagonal_line(x, y):
    """A line 24 px wide, 30 degrees off the x axis, across the whole image: its middle through (150, 100)."""
    return np.abs(across_diagonal(x, y)) < 12


def across_diagonal(x, y):
    """Distance (px) of (x, y) from the diagonal line's middle, across the line."""
    angle = np.radians(30)
    return -(x - 150) * np.sin(angle) + (y - 100) * np.cos(angle)


def ring(x, y):
    """A ring around (150, 100): from radius 30.4 to 55.2."""
    r = np.hypot(x - 150, y - 100)
    return (r > 30.4) & (r < 55.2)


def test_vertical_lines_running_out_of_the_image_give_an_open_contour_per_edge():
    detected, refined = find_contours(shape_image(vertical_lines), **BRIGHT)
    assert len(refined.points) == 4 and not refined.closed.any()
    assert len(np.unique(refined.region)) == 2 and detected.left_out == {"small": 0, "large": 0}
    order = np.argsort([points[:, 0].mean() for points in refined.points])
    for k, x_edge, faces in zip(order, (60.3, 85.7, 180.0, 201.4), (-1, 1, -1, 1)):
        points, normals = refined.points[k], refined.normals[k]
        np.testing.assert_allclose(points[:, 0], x_edge, atol=0.05)
        np.testing.assert_allclose(normals, np.tile([faces, 0.0], (len(points), 1)), atol=1e-9)  # out of its line
        # from one image border to the other, without the outermost rows
        assert points[:, 1].min() == 1 and points[:, 1].max() == SHAPE[0] - 2
        assert np.all(np.abs(detected.points[k][:, 0] - x_edge) <= 1)


def test_a_diagonal_line_gives_its_two_edges():
    _, refined = find_contours(shape_image(diagonal_line), **BRIGHT)
    assert len(refined.points) == 2 and not refined.closed.any()
    for points, normals in zip(refined.points, refined.normals):
        away = points[5:-5]  # at the image border the blurred line is cut, and its edge bends
        distance = across_diagonal(away[:, 0], away[:, 1])
        np.testing.assert_allclose(np.abs(distance), 12, atol=0.1)
        facing = across_diagonal(150 + normals[5:-5, 0], 100 + normals[5:-5, 1])  # +1 or -1: straight out of the line
        np.testing.assert_allclose(facing, np.sign(distance), atol=0.02)


def test_a_region_with_a_hole_gives_a_closed_contour_for_each_boundary():
    _, refined = find_contours(shape_image(ring), **BRIGHT)
    assert len(refined.points) == 2 and refined.closed.all() and refined.region[0] == refined.region[1]
    radii = sorted(np.hypot(*(points - [150, 100]).T).mean() for points in refined.points)
    np.testing.assert_allclose(radii, (30.4, 55.2), atol=0.1)
    for points, normals in zip(refined.points, refined.normals):
        radial = points - [150, 100]
        radius = np.hypot(*radial.T)
        np.testing.assert_allclose(radius, radius.mean(), atol=0.15)
        outward = (normals * radial).sum(axis=1) / radius  # +1: away from the centre
        np.testing.assert_allclose(outward, -1 if radius.mean() < 40 else 1, atol=0.05)  # the hole's: into the hole


@pytest.mark.parametrize("rule", RULES)
def test_every_rule_finds_the_edges_within_a_pixel(rule):
    _, refined = find_contours(shape_image(vertical_lines), refinement=rule, **BRIGHT)
    x = np.sort([points[:, 0].mean() for points in refined.points])
    np.testing.assert_allclose(x, (60.3, 85.7, 180.0, 201.4), atol=1)


def test_dark_regions_are_the_bright_ones_of_the_negative_image():
    _, bright = find_contours(shape_image(ring), **BRIGHT)
    _, dark = find_contours(255 - shape_image(ring), dark_contacts=True)
    for a, b in zip(dark.points, bright.points):
        np.testing.assert_allclose(a, b, atol=1e-3)


def test_refinement_none_gives_the_detected_contours_and_small_regions_are_left_out():
    def with_speck(x, y):
        return vertical_lines(x, y) | (np.hypot(x - 130, y - 100) < 4)  # about 50 px
    options = dict(min_area_px=100, **BRIGHT)
    detected, refined = find_contours(shape_image(with_speck), refinement="none", **options)
    assert refined is detected and len(detected.points) == 4
    assert detected.left_out == {"small": 1, "large": 0}
    assert len(find_contours(shape_image(with_speck), **BRIGHT)[0].points) == 5  # default: 20 px, the speck is kept
    assert find_contours(shape_image(with_speck), max_area_px=1000, **options)[0].left_out == {"small": 1, "large": 2}


def test_upscaled_contours_are_in_the_pixels_of_the_image():
    _, plain = find_contours(shape_image(vertical_lines), **BRIGHT)
    _, upscaled = find_contours(shape_image(vertical_lines), upscale=2, **BRIGHT)
    assert len(upscaled.points) == 4
    x = [np.sort([points[:, 0].mean() for points in found.points]) for found in (plain, upscaled)]
    np.testing.assert_allclose(x[1], x[0], atol=0.05)
    assert max(points[:, 1].max() for points in upscaled.points) < SHAPE[0] - 1  # inside the image


def test_contacts_inside_the_image_are_the_closed_contours_and_cut_ones_are_open():
    """The contacts' own detection drops a contact cut by the image border; here it gives its edge inside the image."""
    image = render_tile(LATTICE, (0.0, 0.0), FOV, SIZE, seed=1)
    _, contacts, _ = find_contacts(image)
    detected, refined = find_contours(image)
    assert int(refined.closed.sum()) == len(contacts.centers)
    assert int((~refined.closed).sum()) >= contacts.left_out["border"] > 0
    centres = np.array([points.mean(axis=0) for points, closed in zip(refined.points, refined.closed) if closed])
    nearest = np.linalg.norm(centres[:, None] - contacts.centers[None], axis=2).min(axis=1)
    assert nearest.max() < 0.2  # the same contacts, the same edges
