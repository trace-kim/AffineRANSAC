import numpy as np
import pytest

from affine_ransac.registration import design_errors, error_summary

DESIGN = np.array([(x, y) for x in np.arange(0, 500, 100.0) for y in np.arange(0, 300, 100.0)])


def test_raw_error_is_stitched_sem_minus_design():
    # Tile 0 is measured 3 nm too far right; its stitching correction brings back 1 nm of that.
    sem = [DESIGN + (3.0, 0.0), DESIGN + (0.0, -2.0)]
    corrections = np.array([[-1.0, 0.0], [0.0, 0.0]])

    errors = design_errors(sem, [DESIGN, DESIGN], corrections, design_ok=[True, True])

    assert errors.skipped == {}
    assert len(errors.tile) == 2 * len(DESIGN)
    np.testing.assert_allclose(errors.error_nm[errors.tile == 0], np.tile([2.0, 0.0], (len(DESIGN), 1)))
    np.testing.assert_allclose(errors.error_nm[errors.tile == 1], np.tile([0.0, -2.0], (len(DESIGN), 1)))
    np.testing.assert_allclose(errors.sem_nm - errors.design_nm, errors.error_nm)


def test_tiles_without_errors_are_listed_and_warned():
    sem = [DESIGN, DESIGN, DESIGN, DESIGN + 1000.0]
    corrections = np.array([[0.0, 0.0], [np.nan, np.nan], [0.0, 0.0], [0.0, 0.0]])

    with pytest.warns(UserWarning, match="No design errors for 3 tile"):
        errors = design_errors(sem, [DESIGN] * 4, corrections, design_ok=[True, True, False, True])

    assert set(errors.skipped) == {1, 2, 3}
    assert "not stitched" in errors.skipped[1] and "flagged" in errors.skipped[2] and "matched" in errors.skipped[3]
    assert set(errors.tile) == {0}


def test_error_summary():
    error = np.array([[1.0, -1.0], [3.0, 1.0]])
    summary = error_summary(error)
    assert summary["count"] == 2
    assert summary["mean_x_nm"] == 2.0 and summary["mean_y_nm"] == 0.0
    assert summary["3sigma_x_nm"] == pytest.approx(3.0) and summary["max_nm"] == pytest.approx(np.sqrt(10))
    assert error_summary(np.empty((0, 2))) == {"count": 0}
