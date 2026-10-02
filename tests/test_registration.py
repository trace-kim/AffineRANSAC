import numpy as np
import pytest

from affine_ransac.registration import binned_mean_2d, design_errors, error_summary, merge_observations, profile

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
    errors = design_errors([design], [design], np.zeros((1, 2)), np.zeros((1, 2)), [True], max_match_nm=1.0)
    assert len(merge_observations(errors, np.zeros((1, 2))).count) == 2


def test_binned_mean_2d():
    points = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 5.0], [150.0, 150.0]])
    values = np.array([1.0, 3.0, 5.0, 7.0])
    grid, x_edges, y_edges = binned_mean_2d(points, values, bin_nm=100.0)
    np.testing.assert_allclose(x_edges, [0, 100, 200]) and np.testing.assert_allclose(y_edges, [0, 100, 200])
    assert grid[0, 0] == 3.0 and grid[1, 1] == 7.0  # (1 + 3 + 5) / 3 in the lower-left bin
    assert np.isnan(grid[0, 1]) and np.isnan(grid[1, 0])


def test_profile_averages_along_one_axis():
    y = np.array([0.0, 10.0, 20.0, 120.0, 130.0, 350.0])
    dx = np.array([1.0, 2.0, 3.0, -1.0, -3.0, 4.0])
    centers, mean, std, count = profile(y, dx, bin_nm=100.0)
    np.testing.assert_allclose(centers, [50, 150, 350])  # the empty bin 200-300 is left out
    np.testing.assert_allclose(mean, [2.0, -2.0, 4.0])
    np.testing.assert_allclose(std, [np.std([1, 2, 3]), 1.0, 0.0])
    np.testing.assert_array_equal(count, [3, 2, 1])
