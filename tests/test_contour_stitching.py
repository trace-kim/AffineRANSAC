"""Stitching by contours (contour_stitching, D156) on synthetic edge points: images of known stage errors over
straight lines (with and without edge roughness) and over shapes that fix the position both ways."""

import numpy as np
import pytest
from scipy import signal

import affine_ransac.contour_stitching as stitching
from affine_ransac.contour_stitching import (EdgePoints, coarse_fit, edge_points, grid_shifts, mismatch,
                                             overlap_distances, placed_by_contours, tile_box)
from affine_ransac.features.contours import Contours
from affine_ransac.step_log import StepLog

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


@pytest.mark.parametrize("shape,size", [((1, 1), 1), ((7, 11), 5), ((10, 6), 4)])
@pytest.mark.parametrize("field", ["full", "partial", "empty"])
def test_covered_counts_equal_direct_correlation(shape, size, field):
    rng = np.random.default_rng(70)
    image = rng.integers(0, 4, size=shape).astype(np.float32)
    covered_x, covered_y = [np.ones(n + size - 1, bool) for n in shape[::-1]]
    if field == "partial":
        covered_x[:2], covered_y[-3:] = False, False
    elif field == "empty":
        covered_x[:] = False
    expected = signal.correlate(np.outer(covered_y, covered_x).astype(float), image, mode="valid", method="direct")
    np.testing.assert_array_equal(stitching.covered_counts(image, covered_x, covered_y, size), expected)


def direct_mismatch(theirs, their_box, mine, my_box, reach, bin_nm):
    """Small test oracle: explicitly sum distances for every point at every allowed grid shift."""
    grown, size = reach * bin_nm, 2 * reach + 1
    origin = np.maximum([my_box[0], my_box[2]], np.array([their_box[0], their_box[2]]) - grown)
    end = np.minimum([my_box[1], my_box[3]], np.array([their_box[1], their_box[3]]) + grown)
    shape = np.ceil((end - origin) / bin_nm).astype(int) + 1
    mine_grid = np.round((mine.xy - origin) / bin_nm).astype(int)
    their_grid = np.round((theirs.xy - origin) / bin_nm).astype(int) + reach
    my_sector = stitching.facing_sectors(mine.normals)[0]
    sectors, other = stitching.facing_sectors(theirs.normals)
    usable = ((their_grid >= 0) & (their_grid < shape + 2 * reach)).all(axis=1)
    total, count = np.zeros((size, size)), np.zeros((size, size))
    for row in range(size):
        for col in range(size):
            for point, sector in zip(mine_grid, my_sector):
                if not ((point >= 0) & (point < shape)).all():
                    continue
                shifted = point + [col, row]
                xy = origin + (shifted - reach) * bin_nm
                if not (their_box[0] <= xy[0] <= their_box[1] and their_box[2] <= xy[1] <= their_box[3]):
                    continue
                facing = usable & ((sectors == sector) | (other == sector))
                squared = ((their_grid[facing] - shifted) ** 2).sum(axis=1) * bin_nm ** 2
                total[row, col] += min(squared.min(initial=np.inf), (stitching.GATE_PX * bin_nm) ** 2)
                count[row, col] += 1
    return total, count


@pytest.mark.parametrize("reach,bin_nm", [(0, 1.0), (3, 1.7), (5, 0.8)])
def test_mismatch_equals_explicit_grid_distances_with_one_small_inverse_fft(monkeypatch, reach, bin_nm):
    rng = np.random.default_rng(71)
    normals = np.column_stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)])
    mine = EdgePoints(rng.uniform(-8, 8, (40, 2)), normals[np.arange(40) % 8], bin_nm)
    theirs = EdgePoints(rng.uniform(-5, 9, (25, 2)), normals[np.arange(25) % 3], bin_nm)
    my_box, their_box = (-8.1, 8.3, -8.2, 8.4), (-4.7, 9.1, -4.6, 9.2)
    calls, inverse = [], stitching.fft.irfftn

    def counted(array, shape):
        calls.append(shape)
        return inverse(array, shape)

    monkeypatch.setattr(stitching.fft, "irfftn", counted)
    total, count = mismatch(theirs, their_box, mine, my_box, reach, bin_nm)
    expected_total, expected_count = direct_mismatch(theirs, their_box, mine, my_box, reach, bin_nm)
    np.testing.assert_array_equal(count, expected_count)
    np.testing.assert_allclose(total, expected_total, rtol=2e-6, atol=2e-4)
    assert len(calls) == 1
    np.testing.assert_array_equal(coarse_fit(total, count, bin_nm, bin_nm),
                                  coarse_fit(expected_total, expected_count, bin_nm, bin_nm))


@pytest.mark.parametrize("pattern", ["shapes", "straight", "slanted", "rough"])
def test_batched_fits_match_scalar_fits_at_mask_scale(pattern):
    rng = np.random.default_rng(72)
    outline = circles(rng) if pattern == "shapes" else vertical_lines(rng, 1.5 if pattern == "rough" else 0)
    edges, centers, fovs = column(outline, [[0, 0], [7.3, -4.6]], rng)
    theirs, mine = edges
    if pattern == "slanted":
        angle = 0.47
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        theirs = EdgePoints(theirs.xy @ rotation, theirs.normals @ rotation, PIXEL)
        mine = EdgePoints(mine.xy @ rotation, mine.normals @ rotation, PIXEL)
    offset = np.array([2e7, -3e7])
    theirs, mine = theirs.moved(offset), stitching.thinned(mine.moved(offset), PIXEL)
    box = tile_box(centers[0] + offset, fovs[0])
    boxes = [box, tuple(np.array(box) + [20, 20, 30, 30])]  # overlap membership is a union
    shifts = rng.uniform(-15, 15, (stitching.FIT_BATCH * 2 + 3, 2))
    tree = stitching.facing_tree(theirs, 3 * PIXEL)
    expected = np.array([stitching.exact_fit(tree, theirs, mine, boxes, s, 3 * PIXEL) for s in shifts])
    found = stitching.exact_fits(tree, theirs, mine, boxes, shifts, 3 * PIXEL)
    np.testing.assert_array_equal(found[:, 1], expected[:, 1])
    np.testing.assert_allclose(found[:, 0], expected[:, 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(stitching.as_good(found[:, 0], np.maximum(found[:, 1], 1)),
                                  stitching.as_good(expected[:, 0], np.maximum(expected[:, 1], 1)))


@pytest.mark.parametrize("points", [0, 1, 2, 3])
@pytest.mark.parametrize("reference", [False, True])
def test_batched_fits_handle_empty_unpaired_and_small_samples(points, reference):
    xy = np.column_stack([np.arange(points, dtype=float), np.zeros(points)])
    mine = EdgePoints(xy, np.tile([0.0, 1.0], (points, 1)), 1.0)
    theirs = mine.moved([0, 0.4]) if reference else EdgePoints(np.empty((0, 2)), np.empty((0, 2)), 1.0)
    tree, boxes = stitching.facing_tree(theirs, 3.0), [(-20, 20, -20, 20)]
    shifts = np.array([[0.0, 0.0], [0, 14], [0, 15], [0, 40]])  # includes the inset border itself
    expected = np.array([stitching.exact_fit(tree, theirs, mine, boxes, s, 3.0) for s in shifts])
    np.testing.assert_allclose(stitching.exact_fits(tree, theirs, mine, boxes, shifts, 3.0), expected, atol=1e-12)
    assert stitching.exact_fits(tree, theirs, mine, boxes, np.empty((0, 2)), 3.0).shape == (0, 2)


def test_batches_bound_queries_and_keep_the_full_shift_search():
    class CountingTree:
        def __init__(self, tree):
            self.tree, self.counts = tree, []

        def query(self, points, **kwargs):
            self.counts.append(len(points))
            return self.tree.query(points, **kwargs)

    xy = np.column_stack([np.arange(600), np.zeros(600)])
    points = EdgePoints(xy, np.tile([0.0, 1.0], (600, 1)), 1.0)
    tree = CountingTree(stitching.facing_tree(points, 3.0))
    shifts = np.zeros((stitching.FIT_BATCH * 2 + 1, 2))
    fits = stitching.exact_fits(tree, points, points, [(-100, 1000, -100, 100)], shifts, 3.0)
    np.testing.assert_array_equal(fits, np.tile([0.0, 600], (len(shifts), 1)))
    assert tree.counts == [600 * stitching.FIT_BATCH, 600 * stitching.FIT_BATCH, 600]


def test_logged_placement_with_several_neighbours_equals_scalar_candidates(monkeypatch):
    rng = np.random.default_rng(73)
    centers = np.array([[0, 0], [300, 0], [0, 300], [300, 300], [1800, 1800]], dtype=float)
    errors = rng.uniform(-8, 8, centers.shape)
    outline = circles(rng)
    edges = [seen(outline, c, e, rng) for c, e in zip(centers, errors)]
    fovs, log = np.full(centers.shape, FOV), StepLog()
    visits = []
    actual, choices = placed_by_contours(edges, centers, fovs, 25.0, PIXEL,
                                         lambda done, total: visits.append((done, total)), log=log)
    monkeypatch.setattr(stitching, "exact_fits", lambda tree, theirs, mine, boxes, shifts, gate:
                        np.array([stitching.exact_fit(tree, theirs, mine, boxes, s, gate) for s in shifts]))
    expected, old_choices = placed_by_contours(edges, centers, fovs, 25.0, PIXEL)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert [(c.tile, c.pairs, c.positions, c.directions) for c in choices] == [
        (c.tile, c.pairs, c.positions, c.directions) for c in old_choices]
    assert visits == [(2, 5), (3, 5), (4, 5)] and np.isnan(actual[4]).all()
    assert log.counts["images"] == 5 and log.counts["exact_shifts"] > 0
    assert len(log.seconds) == 5 and all(t >= 0 for t in log.seconds.values())
