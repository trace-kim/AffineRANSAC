import numpy as np

from affine_ransac.stitching import connected_groups, fix_tile, solve_tile_shifts


def test_recovers_tile_errors_up_to_their_mean():
    # True per-tile corrections; measured pair shifts d_ij = t_i - t_j.
    true = np.array([[1.0, -2.0], [3.0, 0.5], [-1.0, 1.0], [0.0, 0.0]])
    pairs = [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3)]
    shifts = np.array([true[i] - true[j] for i, j in pairs])

    corrections = solve_tile_shifts(4, pairs, shifts)

    np.testing.assert_allclose(corrections, true - true.mean(axis=0), atol=1e-9)
    np.testing.assert_allclose(corrections.mean(axis=0), 0, atol=1e-12)


def test_weights_favour_reliable_pairs():
    # A triangle with one inconsistent pair: the heavily weighted pairs win.
    pairs = [(0, 1), (1, 2), (0, 2)]
    shifts = np.array([[1.0, 0.0], [1.0, 0.0], [5.0, 0.0]])  # consistent would be 2.0
    c = solve_tile_shifts(3, pairs, shifts, weights=[100, 100, 1])
    assert abs((c[0] - c[1])[0] - 1.0) < 0.05 and abs((c[1] - c[2])[0] - 1.0) < 0.05


def test_tiles_outside_the_largest_group_get_nan_never_zero():
    # Tiles 0-1-2 are linked; 3-4 are linked to each other only; 5 has no pair at all.
    true = np.array([[1.0, 0.0], [2.0, 1.0], [0.0, -1.0], [5.0, 5.0], [6.0, 4.0], [9.0, 9.0]])
    pairs = [(0, 1), (1, 2), (3, 4)]
    shifts = np.array([true[i] - true[j] for i, j in pairs])

    c = solve_tile_shifts(6, pairs, shifts)

    group = true[:3]
    np.testing.assert_allclose(c[:3], group - group.mean(axis=0), atol=1e-9)  # mean zero over the group
    assert np.isnan(c[3:]).all()


def test_connected_groups_largest_first():
    assert connected_groups(6, [(4, 5), (0, 2), (2, 3)]) == [[0, 2, 3], [4, 5], [1]]


def test_fix_tile_keeps_that_tile_and_the_differences():
    c = np.array([[1.0, -1.0], [3.0, 2.0], [-4.0, -1.0]])
    fixed = fix_tile(c, 0)
    np.testing.assert_allclose(fixed[0], 0)
    np.testing.assert_allclose(fixed[1] - fixed[2], c[1] - c[2])
