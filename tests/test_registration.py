import numpy as np
import pytest

from affine_ransac.registration import (design_errors, error_summary, group_rows, merge_observations, paired_errors,
                                        row_means, row_pitch)

DESIGN = np.array([(x, y) for x in np.arange(0, 500, 100.0) for y in np.arange(0, 300, 100.0)])


def test_raw_error_is_stitched_sem_minus_design():
    # Tile 0 is measured 3 nm too far right; its stitching correction brings back 1 nm of that.
    sem = [DESIGN + (3.0, 0.0), DESIGN + (0.0, -2.0)]
    corrections = np.array([[-1.0, 0.0], [0.0, 0.0]])

    errors = design_errors(sem, [DESIGN, DESIGN], corrections, np.zeros((2, 2)), design_ok=[True, True])

    assert errors.skipped == {}
    assert len(errors.tile) == 2 * len(DESIGN)
    np.testing.assert_allclose(errors.error_nm[errors.tile == 0], np.tile([2.0, 0.0], (len(DESIGN), 1)))
    np.testing.assert_allclose(errors.error_nm[errors.tile == 1], np.tile([0.0, -2.0], (len(DESIGN), 1)))
    np.testing.assert_allclose(errors.sem_nm - errors.design_nm, errors.error_nm)


def test_design_errors_take_affine_corrections():
    # Tile 0 measured 0.2 % too small about the origin; its affine correction scales it back.
    sem = [DESIGN / 1.002, DESIGN + (0.0, -2.0)]
    corrections = np.array([np.diag([1.002, 1.002, 1.0]), np.eye(3)])

    errors = design_errors(sem, [DESIGN, DESIGN], corrections, np.zeros((2, 2)), design_ok=[True, True])

    np.testing.assert_allclose(errors.error_nm[errors.tile == 0], 0, atol=1e-9)
    np.testing.assert_allclose(errors.error_nm[errors.tile == 1], np.tile([0.0, -2.0], (len(DESIGN), 1)))


def test_large_tile_offset_is_matched_and_kept_in_the_error():
    # Tile 1 is 45 nm off (beyond the former 40 nm gate, below half the 100 nm pitch along x):
    # the shift search pairs it correctly and the error keeps the full 45 nm (option a).
    sem = [DESIGN + (0.5, 0.0), DESIGN + (45.0, 3.0)]
    errors = design_errors(sem, [DESIGN, DESIGN], np.zeros((2, 2)), np.zeros((2, 2)), [True, True])

    assert errors.skipped == {}
    np.testing.assert_allclose(errors.error_nm[errors.tile == 1], np.tile([45.0, 3.0], (len(DESIGN), 1)))
    np.testing.assert_allclose(errors.match_shift_nm, [[0.5, 0.0], [45.0, 3.0]])
    assert errors.ambiguous.dtype == bool and len(errors.ambiguous) == 2


def test_design_corrections_move_the_design():
    # The design file of tile 0 is drawn 2 nm too far left; its design correction fixes that.
    errors = design_errors([DESIGN], [DESIGN - (2.0, 0.0)], np.zeros((1, 2)), np.array([[2.0, 0.0]]), [True])
    np.testing.assert_allclose(errors.error_nm, 0, atol=1e-12)
    np.testing.assert_allclose(errors.design_nm, errors.sem_nm)


def test_tiles_not_measured_are_listed_with_the_reason_and_warned():
    sem = [DESIGN] * 4 + [DESIGN + 1000.0]
    sem_corrections = np.array([[0.0, 0.0], [np.nan, np.nan], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    design_corrections = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [np.nan, np.nan], [0.0, 0.0]])

    with pytest.warns(UserWarning, match="4 tile.s. not measured"):
        errors = design_errors(sem, [DESIGN] * 5, sem_corrections, design_corrections,
                               design_ok=[True, True, False, True, True])

    assert errors.skipped == {1: "SEM tile not stitched", 2: "design tone not matched (flagged)",
                              3: "design tile not stitched", 4: "no SEM contact matched the design"}
    assert set(errors.tile) == {0}


def test_paired_errors_take_affine_corrections():
    # Tile 0 is corrected by a 0.1 % scale about (200, 100) and a shift; tile 1 by nothing.
    affine = np.array([[1.001, 0.0, -0.2 + 1.0], [0.0, 1.001, -0.1], [0.0, 0.0, 1.0]])
    identity_nan = np.full((3, 3), np.nan)

    with pytest.warns(UserWarning, match="1 tile.s. not measured"):
        errors = paired_errors([DESIGN, DESIGN], [DESIGN, DESIGN], np.array([affine, identity_nan]), np.zeros((2, 2)))

    assert errors.skipped == {1: "SEM tile not stitched"}
    np.testing.assert_allclose(errors.sem_nm, DESIGN @ affine[:2, :2].T + affine[:2, 2])


def test_paired_errors_keep_the_row_pairing_without_matching():
    # Row k of SEM and design is the same contact, even where another design contact is nearer.
    design = DESIGN
    sem = DESIGN[::-1] + 0.5  # every SEM row far from its own design row
    sem_corrections = np.array([[0.0, 0.0], [np.nan, np.nan], [1.0, -1.0], [0.0, 0.0]])
    design_corrections = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 2.0], [np.nan, np.nan]])

    with pytest.warns(UserWarning, match="3 tile.s. not measured"):
        errors = paired_errors([sem] * 4 + [np.empty((0, 2))],
                               [design] * 4 + [np.empty((0, 2))],
                               np.vstack([sem_corrections, [0.0, 0.0]]), np.vstack([design_corrections, [0.0, 0.0]]))

    assert errors.skipped == {1: "SEM tile not stitched", 3: "design tile not stitched", 4: "no contacts"}
    np.testing.assert_allclose(errors.error_nm[errors.tile == 0], sem - design)
    np.testing.assert_allclose(errors.error_nm[errors.tile == 2], (sem + [1.0, -1.0]) - (design + [0.0, 2.0]))
    np.testing.assert_allclose(errors.design_nominal_nm[errors.tile == 2], design)


def test_error_summary():
    error = np.array([[1.0, -1.0], [3.0, 1.0]])
    summary = error_summary(error)
    assert summary["count"] == 2
    assert summary["mean_x_nm"] == 2.0 and summary["mean_y_nm"] == 0.0
    assert summary["3sigma_x_nm"] == pytest.approx(3.0) and summary["max_nm"] == pytest.approx(np.sqrt(10))
    assert error_summary(np.empty((0, 2))) == {"count": 0}


def overlap_errors():
    """Tiles 0 and 1 overlap in the contact at x = 400 (both see it); tile 1 also sees x = 500.
    Tile 1's design file is drawn 3 nm too far left at nominal placement; its correction is +3."""
    design_0 = np.array([[300.0, 0.0], [400.0, 0.0]])
    design_1 = np.array([[400.0, 0.0], [500.0, 0.0]]) - (3.0, 0.0)
    sem = [design_0 + (1.0, 0.0), design_1 + (3.0, 0.0) + (2.0, 0.5)]
    design_corrections = np.array([[0.0, 0.0], [3.0, 0.0]])
    errors = design_errors(sem, [design_0, design_1], np.zeros((2, 2)), design_corrections, [True, True])
    return errors, design_corrections


def test_merge_averages_the_observations_of_one_contact():
    errors, design_corrections = overlap_errors()

    merged = merge_observations(errors, design_corrections)

    assert len(errors.tile) == 4 and len(merged.count) == 3
    shared = np.flatnonzero(merged.count == 2)[0]
    np.testing.assert_allclose(merged.design_nm[shared], [400.0, 0.0])
    np.testing.assert_allclose(merged.error_nm[shared], [1.5, 0.25])  # mean of (1, 0) and (2, 0.5)
    assert merged.spread_nm[shared] == pytest.approx(np.hypot(1.0, 0.5))
    assert sorted(errors.tile[merged.members[shared]].tolist()) == [0, 1]
    assert (merged.spread_nm[merged.count == 1] == 0).all()


def test_merge_groups_by_stitched_design_in_every_placement():
    # At nominal placement tile 1's design is still 3 nm off, but grouping uses the stitched design.
    errors, design_corrections = overlap_errors()
    sem = [errors.sem_nm[errors.tile == 0], errors.sem_nm[errors.tile == 1]]
    designs = [errors.design_nominal_nm[errors.tile == k] for k in (0, 1)]
    nominal = design_errors(sem, designs, np.zeros((2, 2)), np.zeros((2, 2)), [True, True])

    merged = merge_observations(nominal, design_corrections)

    assert sorted(merged.count.tolist()) == [1, 1, 2]


def test_merge_never_joins_two_contacts_of_one_tile():
    design = np.array([[0.0, 0.0], [3.0, 0.0]])  # 3 nm apart in the same tile: not the same contact
    errors = design_errors([design], [design], np.zeros((1, 2)), np.zeros((1, 2)), [True], tolerance_nm=1.0)
    assert len(merge_observations(errors, np.zeros((1, 2))).count) == 2



def test_group_rows_by_design_y():
    # Two rows 100 nm apart; the second row's contacts from another tile sit 0.4 nm higher.
    y = np.array([0.0, 0.0, 100.0, 100.4, 0.0, 100.4])
    np.testing.assert_array_equal(group_rows(y, gap_nm=10), [0, 0, 1, 1, 0, 1])


def test_row_means_average_the_contacts_of_each_row():
    y = np.array([100.0, 0.0, 0.2, 100.1, 200.0])
    error = np.array([[1.0, -1.0], [2.0, 0.0], [4.0, 2.0], [3.0, 1.0], [5.0, 5.0]])
    row_y, mean, count = row_means(y, error, gap_nm=10)
    np.testing.assert_allclose(row_y, [0.1, 100.05, 200.0])
    np.testing.assert_allclose(mean, [[3.0, 1.0], [2.0, 0.0], [5.0, 5.0]])
    np.testing.assert_array_equal(count, [2, 2, 1])


def test_row_pitch_is_the_spacing_of_neighbouring_row_means():
    design_y = np.array([0.0, 0.0, 100.0, 100.0, 200.0])
    sem_y = np.array([0.5, -0.5, 101.0, 103.0, 199.0])  # rows at 0, 102, 199
    mid_y, pitch = row_pitch(design_y, sem_y, gap_nm=10)
    np.testing.assert_allclose(mid_y, [50.0, 150.0])
    np.testing.assert_allclose(pitch, [102.0, 97.0])
    np.testing.assert_allclose(row_pitch(design_y, design_y, gap_nm=10)[1], [100.0, 100.0])


def test_neighbours_keep_every_tile_on_the_same_row():
    # Every tile 80 nm off (> P / 2 = 65 nm); only the left column of tiles has missing contacts.
    # Alone, the fully periodic tiles take -50 nm (one row off, error -50 instead of +80); with
    # neighbours they follow the tiles their missing contacts decided, and the error keeps 80 nm.
    from test_matching import mosaic
    from affine_ransac.registration import neighbour_jumps
    holes = np.array([0.08 if k % 4 == 0 else 0.0 for k in range(16)])
    designs, sems, _, neighbours = mosaic(np.full(16, 80.0), holes=holes)
    zero = np.zeros((16, 2))

    alone = design_errors(sems, designs, zero, zero, [True] * 16, search_nm=200)
    together = design_errors(sems, designs, zero, zero, [True] * 16, search_nm=200, neighbours=neighbours)

    np.testing.assert_allclose(alone.match_shift_nm[holes == 0, 0], -50.0, atol=0.5)
    np.testing.assert_allclose(together.match_shift_nm[:, 0], 80.0, atol=0.5)
    for k in range(16):
        np.testing.assert_allclose(together.error_nm[together.tile == k, 0].mean(), 80.0, atol=0.5)
    assert np.nanmax(together.neighbour_jump_nm) < 5
    assert np.nanmax(neighbour_jumps(alone.match_shift_nm, neighbours)) > 100
    assert np.isnan(alone.neighbour_jump_nm).all()  # no neighbours given
    np.testing.assert_array_equal(together.ambiguous, holes == 0)  # tied = no missing contact decided
    assert np.all(together.match_mismatches == 0)  # every tile matched its design exactly
    # The largest difference to any neighbour; tiles without a shift give NaN and are not neighbours.
    shifts = np.array([[0.0, 0.0], [3.0, 4.0], [np.nan, np.nan], [0.0, 10.0]])
    np.testing.assert_allclose(neighbour_jumps(shifts, [(0, 1), (1, 2), (0, 3)]), [10.0, 5.0, np.nan, 10.0])
