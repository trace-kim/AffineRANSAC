"""Stitching by contours (contour_stitching, D156) on synthetic edge points: images of known stage errors over
straight lines (with and without edge roughness) and over shapes that fix the position both ways."""

import numpy as np
import pytest

from affine_ransac.contour_stitching import (EdgePoints, coarse_fit, edge_points, grid_shifts, mismatch,
                                             overlap_distances, placed_by_contours, tile_box)
from affine_ransac.features.contours import Contours

PIXEL = 2.0    # nm
FOV = 400.0    # nm: images of 200 px
PITCH = 60.0   # nm
NOISE = 0.3    # nm, per edge point


def vertical_lines(rng, roughness_nm: float = 0.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """The edges of vertical lines PITCH apart, half a pitch wide, as (points, normals) per edge, a point every
    pixel along y; roughness_nm: each edge's own smooth wiggle (rms)."""
    y = np.arange(-600.0, 600.0, PIXEL)
    edges = []
    for centre in np.arange(-600.0, 600.0, PITCH):
        for side in (-1.0, 1.0):
            wiggle = np.zeros_like(y)
            if roughness_nm:
                wiggle = np.convolve(rng.normal(size=len(y) + 20), np.hanning(21), mode="valid")
                wiggle *= roughness_nm / wiggle.std()
            x = centre + side * PITCH / 4 + wiggle
            edges.append((np.column_stack([x, y]), np.tile([side, 0.0], (len(y), 1))))
    return edges


def circles(rng) -> list[tuple[np.ndarray, np.ndarray]]:
    """The edges of circles at random places: they fix a position both ways, and no shift repeats them."""
    edges = []
    for centre in rng.uniform(-600, 600, (150, 2)):
        angle = np.arange(0, 2 * np.pi, PIXEL / 20.0)
        normal = np.column_stack([np.cos(angle), np.sin(angle)])
        edges.append((centre + 20.0 * normal, normal))
    return edges


def seen(edges, center, error, rng) -> EdgePoints:
    """What an image at `center` (stage position), really lying at center + error, reports of the edges: their
    points inside its field, with noise across the edge, in mask nm at the stage position."""
    xy, normals = np.concatenate([e[0] for e in edges]), np.concatenate([e[1] for e in edges])
    inside = (np.abs(xy - (np.asarray(center) + error)) < FOV / 2 - PIXEL).all(axis=1)
    xy, normals = xy[inside], normals[inside]
    return EdgePoints(xy - error + rng.normal(0, NOISE, (len(xy), 1)) * normals, normals, PIXEL)


def column(edges, errors, rng):
    """Images one above the other, overlapping by a quarter of the field, with these stage errors."""
    centers = np.array([[0.0, k * 0.75 * FOV] for k in range(len(errors))])
    fovs = np.full((len(errors), 2), FOV)
    return [seen(edges, c, e, rng) for c, e in zip(centers, np.asarray(errors, float))], centers, fovs


def relative(corrections, choices):
    """The corrections relative to the image that stayed (the one no choice names)."""
    stayed = (set(range(len(corrections))) - {c.tile for c in choices}).pop()
    return corrections - corrections[stayed], stayed


def test_shapes_fix_the_position_both_ways():
    rng = np.random.default_rng(1)
    errors = np.array([[0.0, 0.0], [7.3, -4.6], [-5.2, 9.1]])
    edges, centers, fovs = column(circles(rng), errors, rng)
    corrections, choices = placed_by_contours(edges, centers, fovs, search_nm=25.0, bin_nm=PIXEL)
    moved, stayed = relative(corrections, choices)
    assert np.abs(moved - (errors - errors[stayed])).max() < 0.2
    assert all(c.positions == 1 and c.directions == 2 for c in choices)
    assert all(c.rms_nm < 2 * NOISE for c in choices)
    assert all(d.rms_nm < 2 * NOISE for d in overlap_distances(edges, centers, fovs, corrections, PIXEL))
    unplaced = overlap_distances(edges, centers, fovs, np.zeros((3, 2)), PIXEL)  # at the stage positions: apart
    assert max(d.rms_nm for d in unplaced) > 5 * NOISE


def test_straight_lines_fix_the_position_across_them_and_the_stage_gives_it_along():
    rng = np.random.default_rng(2)
    errors = np.array([[0.0, 0.0], [7.3, -4.6]])
    edges, centers, fovs = column(vertical_lines(rng), errors, rng)
    corrections, choices = placed_by_contours(edges, centers, fovs, search_nm=25.0, bin_nm=PIXEL)
    moved, stayed = relative(corrections, choices)
    other = 1 - stayed
    assert moved[other, 0] == pytest.approx((errors[other] - errors[stayed])[0], abs=0.1)  # across the lines
    assert moved[other, 1] == 0.0                                                          # along: the stage
    assert choices[0].directions == 1 and choices[0].positions == 1


def test_lines_a_pitch_apart_are_tied_and_the_nearest_position_is_taken():
    """Perfect lines, an image more than half a pitch from its stage position: nothing tells the rows apart, so the
    nearest is taken, a whole pitch from the true one, and the choice says several positions fit."""
    rng = np.random.default_rng(3)
    errors = np.array([[0.0, 0.0], [40.0, 0.0]])
    edges, centers, fovs = column(vertical_lines(rng), errors, rng)
    corrections, choices = placed_by_contours(edges, centers, fovs, search_nm=70.0, bin_nm=PIXEL)
    moved, stayed = relative(corrections, choices)
    off = (moved[1 - stayed] - (errors[1 - stayed] - errors[stayed]))[0]
    assert abs(off) == pytest.approx(PITCH, abs=0.2)
    assert choices[0].positions > 1


def test_edge_roughness_tells_the_rows_apart_and_gives_the_position_along_the_lines():
    """Each edge has its own wiggle, well above the noise: the true row is the only one that fits, although another
    lies nearer the stage position, and the position along the lines comes from the edges, to the search grid."""
    rng = np.random.default_rng(4)
    errors = np.array([[0.0, 0.0], [40.0, 12.0]])
    edges, centers, fovs = column(vertical_lines(rng, roughness_nm=1.5), errors, rng)
    corrections, choices = placed_by_contours(edges, centers, fovs, search_nm=70.0, bin_nm=PIXEL)
    moved, stayed = relative(corrections, choices)
    off = moved[1 - stayed] - (errors[1 - stayed] - errors[stayed])
    assert abs(off[0]) < 0.3 and abs(off[1]) <= PIXEL
    assert choices[0].positions == 1


def test_an_image_with_no_edges_to_compare_is_not_placed():
    rng = np.random.default_rng(5)
    edges, centers, fovs = column(circles(rng), np.zeros((3, 2)), rng)
    edges[2] = EdgePoints(np.empty((0, 2)), np.empty((0, 2)), PIXEL)
    corrections, choices = placed_by_contours(edges, centers, fovs, search_nm=25.0, bin_nm=PIXEL)
    assert np.isnan(corrections[2]).all() and not np.isnan(corrections[:2]).any()
    assert len(choices) == 1


def test_mismatch_is_smallest_at_the_true_shift():
    rng = np.random.default_rng(6)
    edges, centers, fovs = column(circles(rng), [[0, 0], [6.0, -10.0]], rng)
    total, count = mismatch(edges[0], tile_box(centers[0], fovs[0]), edges[1], tile_box(centers[1], fovs[1]),
                            reach=13, bin_nm=PIXEL)
    shifts = grid_shifts(coarse_fit(total, count, PIXEL, PIXEL), PIXEL)
    assert len(shifts) and np.abs(shifts - [6.0, -10.0]).max() <= PIXEL


def test_edge_points_are_the_contours_in_nm_facing_y_up():
    contours = Contours([np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, np.nan], [3.0, 0.0]])],
                        [np.tile([0.0, -1.0], (4, 1))], np.array([False]), np.array([1]), 100.0)
    points = edge_points(contours, (5, 5), 2.0)  # the image centre is pixel (2, 2); its top row faces up
    assert np.allclose(points.xy, [[-4, 4], [-2, 4], [2, 4]])
    assert np.allclose(points.normals, [[0, 1]] * 3)
    assert points.step_nm == 2.0
