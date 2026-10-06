import numpy as np

from affine_ransac.matching import choose_shift, match_points, match_with_shift


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


# --- match_with_shift ------------------------------------------------------------------------

PITCH = 130.0


def lattice(n=12, pitch=PITCH):
    return np.array([(x, y) for x in np.arange(n) * pitch for y in np.arange(n) * pitch])


def sem_of(design, shift, noise_nm=1.0, seed=0, drop=0.05):
    """SEM view of the design: shifted, noisy, a few contacts missing, rows shuffled.
    Returns the SEM points and, for each, the index of its true design contact."""
    rng = np.random.default_rng(seed)
    keep = rng.random(len(design)) > drop
    sem = design[keep] + shift + rng.normal(0, noise_nm, (keep.sum(), 2))
    order = rng.permutation(len(sem))
    return sem[order], np.flatnonzero(keep)[order]


def assert_pairs_correct(match, truth):
    assert len(match.ia) > 0
    np.testing.assert_array_equal(match.ia, truth[match.ib])


def test_large_offset_below_half_pitch_is_found_and_paired_correctly():
    design = lattice()
    sem, truth = sem_of(design, shift=(55.0, -48.0))  # beyond a 40 nm gate, below P / 2 = 65 nm

    match = match_with_shift(design, sem, search_nm=150, tolerance_nm=10)

    np.testing.assert_allclose(match.shift, (55.0, -48.0), atol=0.3)
    assert_pairs_correct(match, truth)
    assert match.score == len(sem)
    # Option (a): the pairs keep the full difference; the shift only chose them.
    np.testing.assert_allclose((sem[match.ib] - design[match.ia]).mean(axis=0), match.shift, atol=1e-9)
    # On a periodic array the one-pitch shifts also match most points: decided by the tie-break.
    assert match.ambiguous


def test_large_gate_pairs_wrong_neighbours_the_shift_search_does_not():
    # The failure being fixed: offset near P / 2 plus noise; with a 100 nm gate the neighbour
    # is nearer for some contacts.
    design = lattice()
    sem, truth = sem_of(design, shift=(62.0, 0.0), noise_nm=3.0, seed=2)
    ia, ib = match_points(design, sem, max_distance=100)
    assert (ia != truth[ib]).any()
    assert_pairs_correct(match_with_shift(design, sem, search_nm=150, tolerance_nm=10), truth)


def test_offset_beyond_half_pitch_takes_the_smallest_shift_and_is_flagged():
    # Documents the tie-break's limit: a true offset of 0.6 P is taken as -0.4 P (one pitch off).
    design = lattice()
    sem, _ = sem_of(design, shift=(0.6 * PITCH, 0.0))
    match = match_with_shift(design, sem, search_nm=200, tolerance_nm=10)
    np.testing.assert_allclose(match.shift, (-0.4 * PITCH, 0.0), atol=0.5)
    assert match.ambiguous


def test_aperiodic_points_have_one_clear_shift():
    design = np.random.default_rng(5).uniform(0, 2000, (150, 2))
    sem, truth = sem_of(design, shift=(-70.0, 33.0), seed=6)
    match = match_with_shift(design, sem, search_nm=150, tolerance_nm=10)
    np.testing.assert_allclose(match.shift, (-70.0, 33.0), atol=0.3)
    assert_pairs_correct(match, truth)
    assert not match.ambiguous


def test_nothing_within_the_search_radius():
    match = match_with_shift(np.array([[0.0, 0.0]]), np.array([[500.0, 0.0]]), search_nm=100, tolerance_nm=10)
    assert match.score == 0 and len(match.ia) == 0 and np.isnan(match.shift).all()
    assert match_with_shift(np.empty((0, 2)), np.array([[1.0, 1.0]]), 100, 10).score == 0


def test_choose_shift_best_score_unless_near_tie_then_smallest():
    shifts = np.array([[60.0, 0.0], [-70.0, 0.0], [5.0, 0.0]])
    assert choose_shift(shifts, np.array([100, 50, 20])) == (0, False)
    assert choose_shift(shifts, np.array([100, 95, 20])) == (0, True)   # 60 < 70
    assert choose_shift(shifts, np.array([100, 95, 92])) == (2, True)   # within 10 %: smallest wins


# --- consistent_choice -----------------------------------------------------------------------
from affine_ransac.matching import consistent_choice, scored_shifts  # noqa: E402


def mosaic(shift_x, shift_y=-20.0, n=4, seed=0):
    """n x n tiles of 12 x 12 contacts (pitch 130 nm), neighbours along x and y. Tile k's SEM
    contacts are its design contacts + (shift_x[k], shift_y) + 1 nm noise, rows shuffled."""
    rng = np.random.default_rng(seed)
    designs, sems, truths, neighbours = [], [], [], []
    for k in range(n * n):
        row, col = divmod(k, n)
        design = lattice() + (col * 1300.0, row * 1300.0)
        sem = design + (shift_x[k], shift_y) + rng.normal(0, 1.0, design.shape)
        order = rng.permutation(len(sem))
        designs.append(design)
        sems.append(sem[order])
        truths.append(order)  # SEM row r is design contact order[r]
        if col + 1 < n:
            neighbours.append((k, k + 1))
        if row + 1 < n:
            neighbours.append((k, k + n))
    return designs, sems, truths, neighbours


def chosen_shifts(designs, sems, neighbours):
    candidates = [scored_shifts(d, s, search_nm=200, tolerance_nm=10) for d, s in zip(designs, sems)]
    choice = consistent_choice(candidates, neighbours)
    return candidates, choice, np.array([c[k].shift for c, k in zip(candidates, choice)])


def test_shift_drifting_across_half_pitch_stays_on_one_row():
    shift_x = np.linspace(45.0, 85.0, 16)  # crosses P / 2 = 65 nm halfway through the mosaic
    designs, sems, truths, neighbours = mosaic(shift_x)

    # The per-tile smallest-shift rule jumps one row where the shift passes P / 2 (the reported failure).
    per_tile = np.array([match_with_shift(d, s, 200, 10).shift[0] for d, s in zip(designs, sems)])
    assert (np.abs(per_tile - shift_x) > 100).any()

    candidates, choice, shifts = chosen_shifts(designs, sems, neighbours)
    np.testing.assert_allclose(shifts[:, 0], shift_x, atol=0.5)
    for c, k, truth in zip(candidates, choice, truths):
        assert_pairs_correct(c[k], truth)


def test_whole_field_beyond_half_pitch_is_decided_by_the_total_score():
    # Every tile 80 nm off (> P / 2): each tile alone takes -50 nm; summed over the field, the true
    # offset matches one more column per tile and wins.
    shift_x = np.full(16, 80.0)
    designs, sems, _, neighbours = mosaic(shift_x, seed=1)
    _, _, shifts = chosen_shifts(designs, sems, neighbours)
    np.testing.assert_allclose(shifts[:, 0], 80.0, atol=0.5)


def test_tiles_without_candidates_or_neighbours():
    designs, sems, _, _ = mosaic(np.full(16, 30.0))
    candidates = [scored_shifts(d, s, 200, 10) for d, s in zip(designs[:2], sems[:2])] + [[]]
    choice = consistent_choice(candidates, neighbours=[(0, 2)])  # tile 2 has nothing to match
    assert choice[2] == -1
    for t in (0, 1):  # no usable neighbours: each keeps its best-scoring candidate
        assert choice[t] == 0
        np.testing.assert_allclose(candidates[t][0].shift, (30.0, -20.0), atol=0.5)
