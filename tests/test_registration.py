import numpy as np
import pytest

from affine_ransac.registration import design_errors, error_summary

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
