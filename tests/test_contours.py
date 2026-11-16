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


def test_a_log_gets_the_time_of_every_step_and_the_counts_they_depend_on():
    """D157: find_contours fills a StepLog, for the contour diagnosis log; the contours are the same with it.
    D159: no flattening or denoising asked, so neither is in it."""
    from affine_ransac.step_log import StepLog
    log = StepLog()
    _, refined = find_contours(shape_image(ring), **BRIGHT, log=log)
    _, plain = find_contours(shape_image(ring), **BRIGHT)
    assert all(np.array_equal(a, b) for a, b in zip(refined.points, plain.points))
    assert list(log.seconds) == [
        "upscale", "threshold mask", "regions of the mask",
        "go through the regions, leave out by size", "trace every kept region in its bounding box",
        "cut the boundaries at the image border, direction of every point", "smooth the image for the edges",
        "sample the image across the edge at every point",
        "interior level of every region (sorts every pixel of the image)",
        "band level of every contour (from all edge points of its region)",
        "edge position of every point, contour by contour", "move the points to their edges",
        "back to the image's own pixels"]
    assert all(seconds >= 0 for seconds in log.seconds.values())
    assert log.counts["regions"] == log.counts["kept"] == 1 and log.counts["holes"] == 1  # the ring and its hole
    assert 110 ** 2 < log.counts["box_pixels"] < 115 ** 2  # the ring's bounding box: its outer diameter squared


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


@pytest.mark.parametrize("binning", (2, 3, 4))
def test_regions_found_in_the_binned_image_give_the_edges_of_the_image_itself(binning):
    """D161: the regions are found in the image binned (3: 200 x 300 px is no multiple of it, so its two scales differ
    a little); every edge is refined in the image itself, as without binning; the detected contours, as coarse as
    the binned pixels, are given in the image's pixels."""
    detected, refined = find_contours(shape_image(vertical_lines), detection_binning=binning, **BRIGHT)
    assert len(refined.points) == 4 and not refined.closed.any()
    order = np.argsort([points[:, 0].mean() for points in refined.points])
    for k, x_edge, faces in zip(order, (60.3, 85.7, 180.0, 201.4), (-1, 1, -1, 1)):
        points, normals = refined.points[k], refined.normals[k]
        np.testing.assert_allclose(points[:, 0], x_edge, atol=0.05)
        np.testing.assert_allclose(normals, np.tile([faces, 0.0], (len(points), 1)), atol=1e-9)
        assert np.all(np.abs(detected.points[k][:, 0] - x_edge) <= binning)
        assert 0 < detected.points[k][:, 1].min() and detected.points[k][:, 1].max() < SHAPE[0] - 1


def test_binned_detection_with_upscaled_refinement_and_on_a_ring():
    """D161 with D131: found binned, refined upscaled; a ring's two closed contours, facing the right ways, at their
    radii as without binning."""
    _, both = find_contours(shape_image(vertical_lines), detection_binning=2, upscale=2, **BRIGHT)
    x = np.sort([points[:, 0].mean() for points in both.points])
    np.testing.assert_allclose(x, (60.3, 85.7, 180.0, 201.4), atol=0.05)
    _, refined = find_contours(shape_image(ring), detection_binning=4, **BRIGHT)
    assert len(refined.points) == 2 and refined.closed.all()
    for points, normals in zip(refined.points, refined.normals):
        radial = points - [150, 100]
        radius = np.hypot(*radial.T)
        np.testing.assert_allclose(radius, 30.4 if radius.mean() < 40 else 55.2, atol=0.15)
        outward = (normals * radial).sum(axis=1) / radius
        assert np.all(np.sign(outward) == (-1 if radius.mean() < 40 else 1))  # the hole's into the hole


def test_binned_detection_logs_its_own_steps():
    from affine_ransac.step_log import StepLog
    log = StepLog()
    find_contours(shape_image(ring), detection_binning=4, **BRIGHT, log=log)
    steps = list(log.seconds)
    assert steps.index("bin the image for the detection") < steps.index("threshold mask") < steps.index(
        "the detected contours and regions in the image the edges are refined in") < steps.index(
        "sample the image across the edge at every point")
    assert 25 ** 2 < log.counts["box_pixels"] < 30 ** 2  # the ring's bounding box in binned pixels


def test_rescaled_contours_and_labels_cover_the_same_area():
    from affine_ransac.features.contours import Contours, labels_in, rescaled_contours
    edge = Contours([np.array([[0.0, 0.0], [1.0, 1.0]])], [np.full((2, 2), np.sqrt(0.5))], np.array([False]),
                    np.array([1]), 100.0)
    stretched = rescaled_contours(edge, (2.0, 1.0))
    np.testing.assert_allclose(stretched.points[0], [[0.5, 0.0], [2.5, 1.0]])  # pixel centres: (x + 0.5) 2 - 0.5
    np.testing.assert_allclose(stretched.normals[0], np.tile([1.0, 2.0], (2, 1)) / np.sqrt(5))  # at right angles still
    labels = np.array([[1, 2, 0], [3, 3, 0]])
    np.testing.assert_array_equal(labels_in(labels, (4, 6)), np.repeat(np.repeat(labels, 2, axis=0), 2, axis=1))


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
