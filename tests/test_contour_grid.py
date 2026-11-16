"""Correctness of D165's exact sparse-grid operations; no timing benchmarks."""

import numpy as np
import pytest
from scipy import ndimage, signal

from affine_ransac.contour_grid import capped_cost, compact_runs, correlate_runs, point_coverage
from affine_ransac.contour_stitching import EdgePoints, mismatch, tile_box
from affine_ransac.step_log import StepLog


@pytest.mark.parametrize("bin_nm", [0.13, 1.0, 1.7, 2.5])
@pytest.mark.parametrize("kind", ["empty", "corners", "duplicates", "dense"])
def test_local_distance_painting_equals_capped_edt(bin_nm, kind):
    shape = (19, 23)
    if kind == "empty":
        rows, columns = np.empty(0, int), np.empty(0, int)
    elif kind == "corners":
        rows, columns = np.array([0, 18, 0, 18]), np.array([0, 0, 22, 22])
    else:
        rng = np.random.default_rng(81)
        rows, columns = rng.integers(0, shape[0], 900), rng.integers(0, shape[1], 900)
        if kind == "duplicates":
            rows, columns = rows[:20].repeat(4), columns[:20].repeat(4)
    edges = np.zeros(shape, bool)
    edges[rows, columns] = True
    gate = 3 * bin_nm
    distances = ndimage.distance_transform_edt(~edges) * bin_nm if edges.any() else np.full(shape, gate)
    expected = (np.minimum(distances, gate) ** 2).astype(np.float32)
    np.testing.assert_array_equal(capped_cost(rows, columns, shape, bin_nm, 3), expected)


@pytest.mark.parametrize("transpose", [False, True])
def test_run_correlation_counts_duplicates_and_separated_runs(transpose):
    rng = np.random.default_rng(82)
    image = np.zeros((9, 31), np.float32)
    image[0, :11], image[3, 7:26], image[8, 14:] = 2, 1, 4
    image[5, 4] = 1
    if transpose:
        image = image.T.copy()
    row, col = np.nonzero(image)
    row, col = np.repeat(row, image[row, col].astype(int)), np.repeat(col, image[row, col].astype(int))
    order = rng.permutation(len(row))
    row, col = row[order], col[order]
    size = 7
    cost = rng.uniform(0, 9, np.array(image.shape) + size - 1).astype(np.float32)
    runs, weights, transposed = compact_runs(row, col, image.shape)
    assert transposed == transpose and len(runs) == 4
    actual = correlate_runs(cost.T if transposed else cost, runs, weights, size)
    if transposed:
        actual = actual.T
    expected = signal.correlate(cost.astype(float), image, mode="valid", method="direct")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)


@pytest.mark.parametrize("size", [1, 4, 13])
@pytest.mark.parametrize("field", ["full", "partial", "empty"])
def test_sparse_coverage_rectangles_equal_point_image_correlation(size, field):
    rng = np.random.default_rng(83)
    shape = (9, 17)
    rows, columns = rng.integers(0, shape[0], 200), rng.integers(0, shape[1], 200)
    image = np.zeros(shape)
    np.add.at(image, (rows, columns), 1)
    cx, cy = np.ones(shape[1] + size - 1, bool), np.ones(shape[0] + size - 1, bool)
    if field == "partial":
        cx[:4], cy[-3:] = False, False
    elif field == "empty":
        cy[:] = False
    expected = signal.correlate(np.outer(cy, cx).astype(float), image, mode="valid", method="direct")
    np.testing.assert_array_equal(point_coverage(rows, columns, cx, cy, size), expected)


@pytest.mark.parametrize("transpose", [False, True])
def test_compact_edges_need_neither_fft_nor_distance_transform(monkeypatch, transpose):
    def forbidden(*args, **kwargs):
        raise AssertionError("Compact contour runs must bypass whole-image EDT and FFT")

    import affine_ransac.contour_stitching as stitching
    monkeypatch.setattr(stitching.fft, "rfftn", forbidden)
    monkeypatch.setattr(ndimage, "distance_transform_edt", forbidden)
    xy = np.array([(x, y) for y in (-15, 15) for x in range(-60, 61)], float)
    normals = np.tile([0.0, 1.0], (len(xy), 1))
    if transpose:
        xy, normals = xy[:, ::-1].copy(), normals[:, ::-1].copy()
    edges = EdgePoints(xy, normals, 1.0)
    box, log = tile_box([0, 0], [130, 130]), StepLog()
    total, count = mismatch(edges, box, edges, box, 8, 1.0, log=log)
    assert total[8, 8] == 0 and count[8, 8] == len(xy)
    assert log.counts == {"run_sum_sectors": 1}


def test_fragmented_sector_uses_fft_and_matches_direct_grid_distances():
    from test_contour_stitching import direct_mismatch

    rng = np.random.default_rng(84)
    normals = np.tile([0.0, 1.0], (160, 1))
    points = EdgePoints(rng.uniform(-14, 14, (160, 2)), normals, 1.0)
    box, log = tile_box([0, 0], [30, 30]), StepLog()
    total, count = mismatch(points, box, points, box, 12, 1.0, log=log)
    expected, expected_count = direct_mismatch(points, box, points, box, 12, 1.0)
    np.testing.assert_array_equal(count, expected_count)
    np.testing.assert_allclose(total, expected, atol=3e-4, rtol=2e-6)
    assert log.counts == {"fft_sectors": 1}


def test_mixed_run_and_fft_sectors_add_to_the_same_complete_grid():
    from test_contour_stitching import direct_mismatch

    rng = np.random.default_rng(87)
    line = np.column_stack([np.arange(-12, 13), np.full(25, 11)])
    xy = np.vstack([line, rng.uniform(-14, 14, (120, 2))])
    normals = np.vstack([np.tile([0.0, 1.0], (25, 1)), np.tile([1.0, 0.0], (120, 1))])
    points = EdgePoints(xy, normals, 1.0)
    box, log = tile_box([0, 0], [30, 30]), StepLog()
    total, count = mismatch(points, box, points, box, 10, 1.0, log=log)
    expected, expected_count = direct_mismatch(points, box, points, box, 10, 1.0)
    np.testing.assert_array_equal(count, expected_count)
    np.testing.assert_allclose(total, expected, atol=3e-4, rtol=2e-6)
    assert log.counts == {"fft_sectors": 1, "run_sum_sectors": 1}
