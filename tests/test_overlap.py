import numpy as np

from affine_ransac.overlap import match_overlap, overlap_box, points_in_box


def test_overlap_box_vertical_neighbours():
    # Tile A spans y -500..500; tile B, 900 nm lower, spans y -1400..-400.
    # Overlap: the full width in x, and a 100 nm strip y -500..-400.
    box = overlap_box((0, 0), (1000, 1000), (0, -900), (1000, 1000))
    assert box == (-500, 500, -500, -400)


def test_overlap_box_corner_and_none():
    assert overlap_box((0, 0), (1000, 1000), (900, 900), (1000, 1000)) == (400, 500, 400, 500)
    assert overlap_box((0, 0), (1000, 1000), (1000, 0), (1000, 1000)) is None  # edges just touch
    assert overlap_box((0, 0), (1000, 1000), (0, 5000), (1000, 1000)) is None


def test_points_in_box_with_margin():
    points = np.array([[0, 0], [10, 0], [12, 0]], dtype=float)
    assert list(points_in_box(points, (-1, 10, -1, 1))) == [0, 1]
    assert list(points_in_box(points, (-1, 10, -1, 1), margin=3)) == [0, 1, 2]


def test_match_overlap_pairs_only_the_shared_contacts():
    # One contact array (pitch 100 nm). Tile A covers y >= -450, tile B covers y <= -350,
    # so the overlap strip holds the row at y = -400. Tile B measures everything 3 nm off.
    grid = np.array([(x, y) for x in range(-400, 401, 100) for y in range(-1200, 401, 100)], dtype=float)
    a = grid[grid[:, 1] >= -450]
    b = grid[grid[:, 1] <= -350] + [3, -2]
    box = overlap_box((0, 0), (1000, 900), (0, -800), (1000, 900))  # y strip -350..-450

    ia, ib = match_overlap(a, b, box, max_distance=20)

    assert len(ia) == 9  # the 9 contacts of the row y = -400
    np.testing.assert_allclose(a[ia][:, 1], -400)
    np.testing.assert_allclose(b[ib] - a[ia], [[3, -2]] * 9)
