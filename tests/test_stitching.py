import numpy as np
import pytest

from affine_ransac.geometry.affine import apply_affine
from affine_ransac.stitching import (apply_correction, connected_groups, first_stitched_tile, fix_tile,
                                    placement_corrections, solve_tile_affines, solve_tile_rigid, solve_tile_shifts)
from test_affine import known_affine

ORIGIN = np.array([5_000_000.0, -2_000_000.0])  # mask-scale coordinates
FOV, STEP = 1000.0, 800.0  # 200 nm overlap strips
LATTICE = ORIGIN + np.array([(x, y) for x in np.arange(-700, 2300, 60.0) for y in np.arange(-700, 2300, 60.0)])


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


def test_placement_corrections():
    sem = np.array([[np.nan, np.nan], [1.0, 2.0], [3.0, -2.0]])
    design = np.array([[0.5, 0.0], [-1.0, 1.0], [1.0, -1.0]])

    placements = placement_corrections(sem, design)

    np.testing.assert_allclose(placements["nominal"][0], 0)
    np.testing.assert_allclose(placements["mean"][1], design)
    assert first_stitched_tile(sem, design) == 1  # tile 0 has no SEM correction
    first_sem, first_design = placements["first"]
    np.testing.assert_allclose(first_sem[1], 0) and np.testing.assert_allclose(first_design[1], 0)
    np.testing.assert_allclose(first_sem[2] - first_sem[1], sem[2] - sem[1])
    assert "first" not in placement_corrections(np.full((2, 2), np.nan), np.zeros((2, 2)))


def about(matrix, center):
    """matrix (written about the origin) acting about center instead."""
    to_center = np.eye(3)
    to_center[:2, 2] = center
    return to_center @ matrix @ np.linalg.inv(to_center)


def rotation(theta_rad, t=(0.0, 0.0)):
    """An exact rotation by theta about the origin, then the shift t (3, 3)."""
    m = np.eye(3)
    m[:2, :2] = [[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]]
    m[:2, 2] = t
    return m


def grid_ties(truth, centers):
    """Pairs and ties of tiles whose true placement is truth[k] (nominal -> true): tile k reports a
    contact p of its FOV at truth[k]^-1(p)."""
    seen = [np.all(np.abs(LATTICE - c) < FOV / 2 - 5, axis=1) for c in centers]
    pairs, ties = [], []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            both = seen[i] & seen[j]
            if both.sum() >= 5:
                pairs.append((i, j))
                ties.append((apply_affine(np.linalg.inv(truth[i]), LATTICE[both]),
                             apply_affine(np.linalg.inv(truth[j]), LATTICE[both])))
    return pairs, ties


def grid(n=3):
    return ORIGIN + np.array([(x, y) for x in range(n) for y in range(n)], float) * STEP


def test_tile_affines_are_recovered_up_to_one_common_affine():
    centers = grid()
    rng = np.random.default_rng(0)
    truth = np.array([about(known_affine(*rng.normal(0, 200, 4), t=rng.normal(0, 5, 2)), c) for c in centers])
    pairs, ties = grid_ties(truth, centers)

    affines = solve_tile_affines(len(centers), pairs, ties, centers)

    # Stitched = H(true) with the same H for every tile (the gauge freedom).
    common = [a @ np.linalg.inv(t) for a, t in zip(affines, truth)]
    for h in common[1:]:  # 0.01 ppm: the tiny pull of D towards 0 biases it by ~1 ppb
        np.testing.assert_allclose(h[:2, :2], common[0][:2, :2], atol=1e-8)
        np.testing.assert_allclose(apply_affine(h, centers[:1]), apply_affine(common[0], centers[:1]), atol=1e-4)
    # Gauge: mean linear part = identity, mean shift at the tile centres = 0.
    np.testing.assert_allclose(affines[:, :2, :2].mean(axis=0), np.eye(2), atol=1e-12)
    shifts = np.array([apply_affine(a, c[None])[0] - c for a, c in zip(affines, centers)])
    np.testing.assert_allclose(shifts.mean(axis=0), 0, atol=1e-6)


def test_pure_shifts_give_the_translation_solution():
    centers = grid()
    true_shifts = np.random.default_rng(1).normal(0, 5, (len(centers), 2))
    truth = np.array([about(known_affine(t=t), c) for t, c in zip(true_shifts, centers)])
    pairs, ties = grid_ties(truth, centers)

    affines = solve_tile_affines(len(centers), pairs, ties, centers)
    shifts = solve_tile_shifts(len(centers), pairs, [(b - a).mean(axis=0) for a, b in ties],
                               weights=[len(a) for a, _ in ties])

    np.testing.assert_allclose(affines[:, :2, :2], np.tile(np.eye(2), (len(centers), 1, 1)), atol=1e-12)
    np.testing.assert_allclose(affines[:, :2, 2], shifts, atol=1e-6)


def test_affine_tiles_outside_the_largest_group_get_nan():
    centers = np.vstack([grid(2), ORIGIN + [50_000.0, 0.0]])  # tile 4 far away: no pair
    truth = np.array([about(known_affine(), c) for c in centers])
    pairs, ties = grid_ties(truth, centers)

    affines = solve_tile_affines(len(centers), pairs, ties, centers)

    assert np.isnan(affines[4]).all() and not np.isnan(affines[:4]).any()


def test_tile_with_ties_on_one_line_is_warned_and_kept_finite():
    # Two tiles sharing one column of contacts: the scale along x is not measured.
    centers = ORIGIN + np.array([[0.0, 0.0], [930.0, 0.0]])
    truth = np.array([about(known_affine(t=(1.0, 2.0)), centers[0]), about(known_affine(), centers[1])])
    pairs, ties = grid_ties(truth, centers)
    assert np.ptp(ties[0][0][:, 0]) < 1e-6  # one column

    with pytest.warns(UserWarning, match="nearly. on one line"):
        affines = solve_tile_affines(2, pairs, ties, centers)

    assert np.isfinite(affines).all()
    np.testing.assert_allclose(affines[:, 0, 0], 1.0, atol=1e-9)  # unmeasured x scale stays nominal


def test_apply_correction_takes_a_shift_or_an_affine():
    points = np.array([[1.0, 2.0], [3.0, -1.0]])
    np.testing.assert_allclose(apply_correction(points, np.array([0.5, -1.0])), points + [0.5, -1.0])
    affine = known_affine(theta_urad=1000, t=(2.0, 0.0))
    np.testing.assert_allclose(apply_correction(points, affine), apply_affine(affine, points))


def test_tile_rotations_are_recovered_up_to_one_common_rotation_and_shift():
    centers = grid()
    rng = np.random.default_rng(2)
    truth = np.array([about(rotation(rng.normal(0, 300e-6), rng.normal(0, 5, 2)), c) for c in centers])
    pairs, ties = grid_ties(truth, centers)

    rigid = solve_tile_rigid(len(centers), pairs, ties, centers)

    common = [r @ np.linalg.inv(t) for r, t in zip(rigid, truth)]  # the same for every tile (gauge)
    for h in common[1:]:  # first-order rotation: differences of order θ² ~ 1e-7
        np.testing.assert_allclose(h[:2, :2], common[0][:2, :2], atol=3e-7)
        np.testing.assert_allclose(apply_affine(h, centers[:1]), apply_affine(common[0], centers[:1]), atol=1e-3)
    angles = np.arctan2(rigid[:, 1, 0], rigid[:, 0, 0])
    np.testing.assert_allclose(angles.mean(), 0, atol=1e-12)  # gauge: mean rotation 0 ...
    shifts = np.array([apply_affine(r, c[None])[0] - c for r, c in zip(rigid, centers)])
    np.testing.assert_allclose(shifts.mean(axis=0), 0, atol=1e-6)  # ... and mean shift 0
    np.testing.assert_allclose(np.linalg.det(rigid[:, :2, :2]), 1.0, atol=1e-12)  # pure rotations


def test_rigid_with_pure_shifts_gives_the_translation_solution():
    centers = grid()
    true_shifts = np.random.default_rng(3).normal(0, 5, (len(centers), 2))
    truth = np.array([about(known_affine(t=t), c) for t, c in zip(true_shifts, centers)])
    pairs, ties = grid_ties(truth, centers)

    rigid = solve_tile_rigid(len(centers), pairs, ties, centers)
    shifts = solve_tile_shifts(len(centers), pairs, [(b - a).mean(axis=0) for a, b in ties],
                               weights=[len(a) for a, _ in ties])

    np.testing.assert_allclose(rigid[:, :2, :2], np.tile(np.eye(2), (len(centers), 1, 1)), atol=1e-9)
    np.testing.assert_allclose(rigid[:, :2, 2], shifts, atol=1e-6)


def test_rigid_tiles_outside_the_largest_group_get_nan():
    centers = np.vstack([grid(2), ORIGIN + [50_000.0, 0.0]])
    truth = np.array([about(known_affine(), c) for c in centers])
    pairs, ties = grid_ties(truth, centers)
    rigid = solve_tile_rigid(len(centers), pairs, ties, centers)
    assert np.isnan(rigid[4]).all() and not np.isnan(rigid[:4]).any()
