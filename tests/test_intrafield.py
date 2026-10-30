import numpy as np
import pytest

from affine_ransac.geometry.affine import apply_affine, fit_affine
from affine_ransac.intrafield import IntrafieldMap, correct_points, estimate_formula, map_shift, overlap_slopes
from affine_ransac.pipeline import stitch_tiles

ORIGIN = np.array([5_000_000.0, -2_000_000.0])  # mask-scale coordinates
FOV, STEP, PITCH = 2000.0, 1600.0, 100.0         # 4 x 4 images, 400 nm overlaps
KX, KY = 2e-7, -1.5e-7                            # the shared bow, nm per nm²: about 0.2 nm at the image edge
LATTICE = ORIGIN + np.array([(x, y) for x in np.arange(-1500, 6500, PITCH) for y in np.arange(-1500, 6500, PITCH)])


def bow(u):
    return np.column_stack([KX * u[:, 0] ** 2, KY * u[:, 1] ** 2])


def barrel(u):
    return 1e-9 * u * (u ** 2).sum(axis=1, keepdims=True)  # 3rd order, radial: about 2.7 nm at the image corners


def images(scale_ppm=0.0, noise_nm=0.02, seed=0, distortion=bow):
    """Image centres (16, 2) and per image (design_local, sem_local): the contacts fully inside its FOV.
    Every image sees u + distortion(u), then its own affine (scale/rotation ~scale_ppm, shift ~3 nm) + noise."""
    rng = np.random.default_rng(seed)
    centers = ORIGIN + np.array([(i, j) for i in range(4) for j in range(4)], float) * STEP
    data = []
    for c in centers:
        u = LATTICE[np.all(np.abs(LATTICE - c) < FOV / 2 - 20, axis=1)] - c
        own = np.eye(3)
        own[:2, :2] += rng.normal(0, scale_ppm * 1e-6, (2, 2))
        own[:2, 2] = rng.normal(0, 3.0, 2)
        data.append((u, apply_affine(own, u + distortion(u)) + rng.normal(0, noise_nm, u.shape)))
    return centers, data


def not_affine_nm(design, sem):
    """RMS (nm) of what no affine of the design explains in one image's SEM points."""
    return np.sqrt(np.mean((apply_affine(fit_affine(sem, design), sem) - design) ** 2))


def test_formula_is_the_shared_distortion_whatever_each_image_affine():
    """D145: with the formula removed, every image's points are an affine image of its design again (the 0.02 nm noise
    is left), a partial image too: fitted with every image's own affine at once, it takes part without bias."""
    centers, data = images(scale_ppm=300, noise_nm=0.02, distortion=barrel)
    u, sem = data[0]
    data[0] = (u[: len(u) // 2], sem[: len(u) // 2])

    m = estimate_formula([u for u, _ in data], [s for _, s in data], order=3)

    assert m.images == 16 and m.count.sum() == sum(len(u) for u, _ in data)
    assert min(not_affine_nm(u, sem) for u, sem in data) > 0.15  # the barrel
    assert max(not_affine_nm(u, sem - map_shift(m, sem)) for u, sem in data) < 0.025


def test_formula_is_left_out_where_the_contacts_cannot_determine_it():
    """Every image's contacts in three rows: y³ looks like y there, which each image's affine takes (D145)."""
    centers, data = images(distortion=barrel)
    rows = [(u[k], sem[k]) for u, sem in data for k in [np.isin(u[:, 1], (-500.0, 0.0, 500.0))]]

    assert estimate_formula([u for u, _ in rows], [s for _, s in rows], order=3) is None
    assert estimate_formula([u for u, _ in rows], [s for _, s in rows], order=2) is not None


def test_map_shift_is_bilinear_between_nodes_and_extrapolated_outside():
    shift = np.zeros((2, 2, 2))
    shift[:, 1, 0] = 1.0  # dx = 1 nm at x = 100 (both rows), 0 at x = 0
    m = IntrafieldMap(np.array([0.0, 100.0]), np.array([0.0, 100.0]), shift, np.ones((2, 2), int), 1)

    np.testing.assert_allclose(map_shift(m, np.array([[50.0, 20.0], [150.0, 0.0]])), [[0.5, 0.0], [1.5, 0.0]])
    np.testing.assert_allclose(correct_points(np.array([[1050.0, 0.0]]), (1000.0, 0.0), m), [[1049.5, 0.0]])


def test_overlaps_show_the_distortion_and_agree_after_the_correction():
    centers, data = images()  # the shared bow and a stage shift per image only
    points = [c + sem for c, (_, sem) in zip(centers, data)]
    fovs = np.full((len(centers), 2), FOV)
    before = overlap_slopes(points, stitch_tiles(points, centers, fovs), centers)

    m = estimate_formula([u for u, _ in data], [s for _, s in data])
    corrected = [correct_points(p, c, m) for p, c in zip(points, centers)]
    after = overlap_slopes(corrected, stitch_tiles(corrected, centers, fovs), centers)

    # A vertical overlap is the top of the lower image and the bottom of the upper one, STEP apart in
    # each image's own frame: (b - a)_y = KY ((u - STEP)² - u²), so d(dy)/dy = -2 KY STEP; likewise in x.
    assert len(before["vertical"]) == len(before["horizontal"]) == 12
    assert np.median(before["vertical"][:, 1, 1]) == pytest.approx(-2 * KY * STEP * 1e6, abs=30)
    assert np.median(before["horizontal"][:, 0, 0]) == pytest.approx(-2 * KX * STEP * 1e6, abs=30)
    # Left after the correction: the noise.
    assert abs(np.median(after["vertical"][:, 1, 1])) < 60
    assert abs(np.median(after["horizontal"][:, 0, 0])) < 60
