import numpy as np
import pytest

from affine_ransac.fitting.ransac import needed_iterations, ransac_affine, ransac_affine_steps
from affine_ransac.geometry.affine import apply_affine, decompose
from test_affine import known_affine

REFERENCE = np.array([5_000_000.0, 2_000_000.0])  # mask-scale coordinates


def synthetic(n=400, noise_nm=0.1, defects=40, seed=0):
    """Design contacts over a 40 um field; SEM = affine(design) + noise, plus defects 5-20 nm off.
    Returns (sem, design, true affine about the design centroid, defect mask)."""
    rng = np.random.default_rng(seed)
    design = REFERENCE + rng.uniform(-20_000, 20_000, (n, 2))
    truth = known_affine(theta_urad=40, mx_ppm=-25, my_ppm=10, skew_urad=8, t=(3.0, -1.5))
    center = design.mean(axis=0)
    # truth maps SEM -> design about the centroid, so SEM = truth^-1(design).
    sem = apply_affine(np.linalg.inv(truth), design - center) + center + rng.normal(0, noise_nm, (n, 2))
    defect = np.zeros(n, bool)
    defect[rng.choice(n, defects, replace=False)] = True
    angle = rng.uniform(0, 2 * np.pi, defects)
    sem[defect] += rng.uniform(5, 20, (defects, 1)) * np.column_stack([np.cos(angle), np.sin(angle)])
    return sem, design, truth, defect


def test_recovers_the_affine_and_flags_the_defects():
    sem, design, truth, defect = synthetic()

    result = ransac_affine(sem, design, threshold_nm=0.5, seed=1)

    np.testing.assert_allclose(result.reference, design.mean(axis=0))
    fitted, expected = decompose(result.model), decompose(truth)
    for term in ("Tx_nm", "Ty_nm"):
        assert fitted[term] == pytest.approx(expected[term], abs=0.05)
    for term in ("Mx_ppm", "My_ppm", "rotation_urad", "orthogonality_urad"):
        assert fitted[term] == pytest.approx(expected[term], abs=1.0)
    np.testing.assert_array_equal(result.inliers, ~defect)
    # Residuals are reported for every point, defects included (large).
    assert len(result.residuals) == len(sem)
    assert np.linalg.norm(result.residuals[defect], axis=1).min() > 4


def test_same_seed_same_result():
    sem, design, *_ = synthetic()
    a = ransac_affine(sem, design, threshold_nm=0.5, seed=7)
    b = ransac_affine(sem, design, threshold_nm=0.5, seed=7)
    np.testing.assert_array_equal(a.best_sample, b.best_sample)
    np.testing.assert_array_equal(a.model, b.model)


def test_steps_search_then_refit_and_best_never_gets_worse():
    sem, design, *_ = synthetic()
    steps = list(ransac_affine_steps(sem, design, threshold_nm=0.5, seed=1))
    search = [s for s in steps if s.stage == "search"]
    refit = [s for s in steps if s.stage == "refit"]
    assert search and refit and steps.index(refit[0]) == len(search)  # refits come after the search
    best_counts = [s.best_inliers.sum() for s in search]
    assert best_counts == sorted(best_counts)
    assert len(search) >= search[-1].needed_iterations  # stopped because N was reached
    assert len(search) - 1 < search[-2].needed_iterations  # ... and not earlier
    assert all(s.sample is not None and len(s.sample) == 3 for s in search)


def test_degenerate_samples_are_rejected():
    # Collinear points: every sample is degenerate, so no model can be found.
    line = np.column_stack([np.linspace(0, 1e4, 20), np.zeros(20)])
    steps = list(ransac_affine_steps(line, line, threshold_nm=0.5, max_iters=50, seed=0))
    assert all(s.degenerate for s in steps)
    with pytest.raises(ValueError, match="no model"):
        ransac_affine(line, line, threshold_nm=0.5, max_iters=50)


def test_low_inlier_ratio_warns():
    sem, design, *_ = synthetic(noise_nm=1.0)  # noise far above the threshold
    with pytest.warns(UserWarning, match="inlier ratio"):
        ransac_affine(sem, design, threshold_nm=0.5, max_iters=300, seed=0)


def test_needed_iterations():
    assert needed_iterations(1.0, 0.999, 10_000) == 1
    assert needed_iterations(0.0, 0.999, 10_000) == 10_000
    assert needed_iterations(0.5, 0.999, 10_000) == 52  # log(0.001) / log(1 - 0.125)
