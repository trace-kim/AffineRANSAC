import numpy as np

from affine_ransac.matching import (ShiftMatch, choose_shift, consistent_choice, match_points, match_with_shift,
                                    near_ties, scored_shifts)


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


def lattice(n=12, pitch=PITCH, holes=0.0, seed=0):
    """n x n contacts; a fraction `holes` of the sites left empty (the same sites in design and
    SEM: the pattern itself has no contact there)."""
    sites = np.array([(x, y) for x in np.arange(n) * pitch for y in np.arange(n) * pitch])
    return sites[np.random.default_rng(seed).random(len(sites)) >= holes]


def sem_of(design, shift, noise_nm=1.0, seed=0, drop=0.0, same_area=False):
    """SEM view of the design: shifted, noisy, optionally a few failed detections (drop), rows
    shuffled. same_area: the SEM image covers the SAME area as the design clip (the pattern is
    offset inside it), so contacts shifted out of that area are not seen.
    Returns the SEM points and, for each, the index of its true design contact."""
    rng = np.random.default_rng(seed)
    sem = design + shift + rng.normal(0, noise_nm, design.shape)
    keep = rng.random(len(design)) >= drop
    if same_area:
        keep &= np.all((sem >= design.min(axis=0)) & (sem <= design.max(axis=0)), axis=1)
    index = np.flatnonzero(keep)
    order = rng.permutation(len(index))
    return sem[index][order], index[order]


def assert_pairs_correct(match, truth):
    assert len(match.ia) > 0
    np.testing.assert_array_equal(match.ia, truth[match.ib])


def test_large_offset_below_half_pitch_is_found_and_paired_correctly():
    design = lattice()
    sem, truth = sem_of(design, shift=(55.0, -48.0), drop=0.05)  # beyond a 40 nm gate, below P / 2

    match, tied = match_with_shift(design, sem, search_nm=150, tolerance_nm=10)

    np.testing.assert_allclose(match.shift, (55.0, -48.0), atol=0.3)
    assert_pairs_correct(match, truth)
    # Option (a): the pairs keep the full difference; the shift only chose them.
    np.testing.assert_allclose((sem[match.ib] - design[match.ia]).mean(axis=0), match.shift, atol=1e-9)
    assert tied  # a fully periodic array: one row off has no extra mismatches, the smallest shift decides


def test_large_gate_pairs_wrong_neighbours_the_shift_search_does_not():
    design = lattice()
    sem, truth = sem_of(design, shift=(62.0, 0.0), noise_nm=3.0, seed=2)
    ia, ib = match_points(design, sem, max_distance=100)
    assert (ia != truth[ib]).any()
    assert_pairs_correct(match_with_shift(design, sem, search_nm=150, tolerance_nm=10)[0], truth)


def test_missing_contacts_decide_even_beyond_half_pitch():
    # The real-data case: a pattern with missing contacts, offset 80 nm (> P / 2) inside the same
    # image area. One row off (-50 nm) loses only a few % of the pairs: the former rules (a tie
    # within 10 %, then the smallest shift) took it. Its mismatches show it clearly.
    design = lattice(holes=0.08, seed=3)
    sem, truth = sem_of(design, shift=(80.0, -20.0), seed=4, same_area=True)

    candidates = scored_shifts(design, sem, search_nm=150, tolerance_nm=10)
    match, tied = match_with_shift(design, sem, search_nm=150, tolerance_nm=10)

    np.testing.assert_allclose(match.shift, (80.0, -20.0), atol=0.3)
    assert_pairs_correct(match, truth)
    assert not tied and match.mismatches == 0
    one_row_off = min(candidates, key=lambda m: np.linalg.norm(m.shift - (-50.0, -20.0)))
    assert one_row_off.score >= 0.9 * match.score  # a 'tie' for the former 10 % rule
    assert one_row_off.mismatches > 5


def test_failed_detections_do_not_hide_the_true_shift():
    design = lattice(holes=0.08, seed=5)
    sem, truth = sem_of(design, shift=(-40.0, 75.0), seed=6, drop=0.03, same_area=True)
    match, tied = match_with_shift(design, sem, search_nm=150, tolerance_nm=10)
    np.testing.assert_allclose(match.shift, (-40.0, 75.0), atol=0.3)
    assert_pairs_correct(match, truth)
    assert not tied


def test_fully_periodic_beyond_half_pitch_takes_the_smallest_shift_and_is_tied():
    # The documented limit: no missing contact, nothing to tell the rows apart.
    design = lattice()
    sem, _ = sem_of(design, shift=(0.6 * PITCH, 0.0), same_area=True)
    match, tied = match_with_shift(design, sem, search_nm=200, tolerance_nm=10)
    np.testing.assert_allclose(match.shift, (-0.4 * PITCH, 0.0), atol=0.5)
    assert tied


def test_aperiodic_points_have_one_clear_shift():
    design = np.random.default_rng(5).uniform(0, 2000, (150, 2))
    sem, truth = sem_of(design, shift=(-70.0, 33.0), seed=6)
    match, tied = match_with_shift(design, sem, search_nm=150, tolerance_nm=10)
    np.testing.assert_allclose(match.shift, (-70.0, 33.0), atol=0.3)
    assert_pairs_correct(match, truth)
    assert not tied


def test_nothing_within_the_search_radius():
    match, tied = match_with_shift(np.array([[0.0, 0.0]]), np.array([[500.0, 0.0]]), search_nm=100, tolerance_nm=10)
    assert match.score == 0 and len(match.ia) == 0 and np.isnan(match.shift).all() and not tied
    assert match_with_shift(np.empty((0, 2)), np.array([[1.0, 1.0]]), 100, 10)[0].score == 0


def fake(shift, mismatches, compared=100):
    return ShiftMatch(np.empty(0, int), np.empty(0, int), np.array(shift, float), 50, mismatches, compared)


def test_near_ties_and_choose_shift():
    # Best first, as scored_shifts returns them.
    clear = [fake((60, 0), 0), fake((-70, 0), 5), fake((5, 0), 9)]
    assert near_ties(clear) == [0] and choose_shift(clear) == (0, False)
    tied = [fake((60, 0), 0), fake((-70, 0), 1), fake((5, 0), 9)]
    assert near_ties(tied) == [0, 1] and choose_shift(tied) == (0, True)    # 60 < 70
    tied3 = [fake((60, 0), 0), fake((-70, 0), 1), fake((5, 0), 1)]
    assert choose_shift(tied3) == (2, True)                                  # all tied: smallest


# --- consistent_choice -----------------------------------------------------------------------


def mosaic(shift_x, shift_y=-20.0, n=4, seed=0, holes=None):
    """n x n tiles of 12 x 12 sites (pitch 130 nm), neighbours along x and y. Tile k: its own
    pattern (holes[k] = fraction of empty sites, default none), its SEM = design + (shift_x[k],
    shift_y) + 1 nm noise inside the same area, rows shuffled."""
    holes = np.zeros(n * n) if holes is None else holes
    designs, sems, truths, neighbours = [], [], [], []
    for k in range(n * n):
        row, col = divmod(k, n)
        design = lattice(holes=holes[k], seed=seed + k) + (col * 1300.0, row * 1300.0)
        sem, truth = sem_of(design, (shift_x[k], shift_y), seed=seed + 100 + k, same_area=True)
        designs.append(design)
        sems.append(sem)
        truths.append(truth)
        if col + 1 < n:
            neighbours.append((k, k + 1))
        if row + 1 < n:
            neighbours.append((k, k + n))
    return designs, sems, truths, neighbours


def chosen_shifts(designs, sems, neighbours):
    candidates = [scored_shifts(d, s, search_nm=200, tolerance_nm=10) for d, s in zip(designs, sems)]
    choice = consistent_choice(candidates, neighbours)
    return candidates, choice, np.array([c[k].shift for c, k in zip(candidates, choice)])


def test_tiles_with_missing_contacts_decide_themselves_neighbours_cannot_overrule():
    # Independent shifts per tile, some beyond P / 2, neighbours differing by up to ~140 nm:
    # following the neighbours would be wrong; every tile's own missing contacts decide.
    shift_x = np.random.default_rng(7).uniform(-85, 85, 16)
    designs, sems, truths, neighbours = mosaic(shift_x, holes=np.full(16, 0.08))
    candidates, choice, shifts = chosen_shifts(designs, sems, neighbours)
    np.testing.assert_allclose(shifts[:, 0], shift_x, atol=0.5)
    for c, k, truth in zip(candidates, choice, truths):
        assert_pairs_correct(c[k], truth)


def test_periodic_tiles_follow_their_decided_neighbours():
    # Every tile 80 nm off (> P / 2). Only the tiles of the left column have missing contacts;
    # the fully periodic tiles alone would take -50 nm, but follow the decided ones.
    holes = np.array([0.08 if k % 4 == 0 else 0.0 for k in range(16)])
    designs, sems, truths, neighbours = mosaic(np.full(16, 80.0), holes=holes)
    _, _, shifts = chosen_shifts(designs, sems, neighbours)
    np.testing.assert_allclose(shifts[:, 0], 80.0, atol=0.5)


def test_fully_periodic_drift_across_half_pitch_stays_on_one_row():
    # No missing contacts anywhere: the neighbours keep the field on one row, and the group's
    # offset is the smallest median shift (40..80 nm: median 60 nm vs -70 nm one row off).
    shift_x = np.linspace(40.0, 80.0, 16)
    designs, sems, truths, neighbours = mosaic(shift_x)
    per_tile = np.array([match_with_shift(d, s, 200, 10)[0].shift[0] for d, s in zip(designs, sems)])
    assert (np.abs(per_tile - shift_x) > 100).any()  # alone, tiles beyond 65 nm jump one row
    _, _, shifts = chosen_shifts(designs, sems, neighbours)
    np.testing.assert_allclose(shifts[:, 0], shift_x, atol=0.5)


def test_tiles_without_candidates_or_neighbours():
    designs, sems, _, _ = mosaic(np.full(16, 30.0))
    candidates = [scored_shifts(d, s, 200, 10) for d, s in zip(designs[:2], sems[:2])] + [[]]
    choice = consistent_choice(candidates, neighbours=[(0, 2)])  # tile 2 has nothing to match
    assert choice[2] == -1
    for t in (0, 1):  # no usable neighbours: choose_shift on its own
        assert choice[t] == choose_shift(candidates[t])[0]
        np.testing.assert_allclose(candidates[t][choice[t]].shift, (30.0, -20.0), atol=0.5)
