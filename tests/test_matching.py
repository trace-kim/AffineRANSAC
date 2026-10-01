import numpy as np

from affine_ransac.matching import match_points


def test_pairs_shifted_points_in_any_order():
    a = np.array([[0, 0], [100, 0], [0, 100]], dtype=float)
    b = a[[2, 0, 1]] + [3, -2]  # same points, shuffled and slightly shifted

    ia, ib = match_points(a, b, max_distance=10)

    np.testing.assert_allclose(b[ib] - a[ia], [[3, -2]] * 3)
    assert sorted(ia) == [0, 1, 2]


def test_points_without_partner_are_dropped():
    design = np.array([[0, 0], [100, 0], [200, 0]], dtype=float)
    sem = np.array([[101, 1], [199, 0]], dtype=float)  # first contact missing in SEM

    ia, ib = match_points(design, sem, max_distance=10)

    assert list(ia) == [1, 2] and list(ib) == [0, 1]


def test_too_far_is_not_a_match():
    ia, ib = match_points(np.array([[0.0, 0.0]]), np.array([[50.0, 0.0]]), max_distance=10)
    assert len(ia) == len(ib) == 0


def test_one_to_one_when_two_points_compete():
    # Two SEM points near one design point: only the closer one is paired.
    design = np.array([[0.0, 0.0]])
    sem = np.array([[2.0, 0.0], [5.0, 0.0]])

    ia, ib = match_points(design, sem, max_distance=10)

    assert list(ia) == [0] and list(ib) == [0]


def test_empty_input():
    ia, ib = match_points(np.empty((0, 2)), np.array([[1.0, 2.0]]), max_distance=10)
    assert len(ia) == len(ib) == 0
