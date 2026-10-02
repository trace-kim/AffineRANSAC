import numpy as np
import pytest

from affine_ransac.geometry.affine import apply_affine, decompose, fit_affine, report_terms, triangle_area


def known_affine(theta_urad=0.0, mx_ppm=0.0, my_ppm=0.0, skew_urad=0.0, t=(0.0, 0.0)):
    """Small-angle affine with the SPEC §8.2 terms: Rx = θ + ω/2 (= c), Ry = θ − ω/2 (= −b)."""
    rx, ry = (theta_urad + skew_urad / 2) * 1e-6, (theta_urad - skew_urad / 2) * 1e-6
    return np.array([[1 + mx_ppm * 1e-6, -ry, t[0]], [rx, 1 + my_ppm * 1e-6, t[1]], [0, 0, 1]])


def test_exact_fit_from_three_points():
    m = known_affine(theta_urad=50, mx_ppm=20, t=(3.0, -2.0))
    src = np.array([[0.0, 0.0], [10_000.0, 0.0], [0.0, 8_000.0]])
    np.testing.assert_allclose(fit_affine(src, apply_affine(m, src)), m, atol=1e-12)


def test_least_squares_fit_recovers_the_affine_through_noise():
    rng = np.random.default_rng(0)
    m = known_affine(theta_urad=-30, my_ppm=15, skew_urad=10, t=(1.0, 4.0))
    # Slope precision ~ noise / (field std * sqrt(N)) = 0.3 nm / (11.5 um * 141) ~ 0.2 ppm.
    src = rng.uniform(-20_000, 20_000, (20_000, 2))
    dst = apply_affine(m, src) + rng.normal(0, 0.3, src.shape)
    fitted = decompose(fit_affine(src, dst))
    assert fitted["rotation_urad"] == pytest.approx(-30, abs=1)
    assert fitted["My_ppm"] == pytest.approx(15, abs=1)
    assert fitted["orthogonality_urad"] == pytest.approx(10, abs=1)
    assert fitted["Tx_nm"] == pytest.approx(1.0, abs=0.05)


def test_decomposition_signs():
    # A pure counter-clockwise rotation moves a point on +x toward +y: rotation > 0.
    rotation = decompose(known_affine(theta_urad=100))
    assert rotation["rotation_urad"] == pytest.approx(100) and rotation["orthogonality_urad"] == pytest.approx(0)
    point = apply_affine(known_affine(theta_urad=100), np.array([[1e6, 0.0]]))[0]
    assert point[1] > 0
    scale = decompose(known_affine(mx_ppm=5, my_ppm=-3))
    assert scale["Mx_ppm"] == pytest.approx(5) and scale["My_ppm"] == pytest.approx(-3)


def test_triangle_area():
    assert triangle_area([0, 0], [4, 0], [0, 3]) == 6
    assert triangle_area([0, 0], [1, 1], [2, 2]) == 0


def test_report_terms_degrees_and_shift_at_the_field_edge():
    points = np.array([[-10_000.0, 0.0], [10_000.0, 0.0], [0.0, 5_000.0]])  # furthest point 10 um away
    rows = {label: (value, edge) for label, value, edge in report_terms(known_affine(theta_urad=100, mx_ppm=50), points)}

    value, edge = rows["rotation (°)"]
    assert value == pytest.approx(np.degrees(100e-6)) and edge == pytest.approx(1.0)  # 100 urad * 10 um = 1 nm
    value, edge = rows["Mx (ppm)"]
    assert value == pytest.approx(50) and edge == pytest.approx(0.5)  # 50 ppm * 10 um = 0.5 nm
    # The edge shift of each term matches moving the furthest point with that term alone.
    moved = apply_affine(known_affine(theta_urad=100), points) - points
    assert np.linalg.norm(moved, axis=1).max() == pytest.approx(1.0, rel=1e-3)
