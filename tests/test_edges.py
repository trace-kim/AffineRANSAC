import cv2
import numpy as np
import pytest
from scipy.spatial import cKDTree

from affine_ransac.features.contact import detect_contacts
from affine_ransac.features.edges import (RULES, crossings, edge_offsets, outward_normals, refine_edges, sample_profiles,
                                          without_isolated)
from affine_ransac.io.sem_image import load_sem_image
from sample_sem import write_sem_like


def test_outward_normals_point_outwards_for_either_order():
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    circle = np.column_stack([10 + 5 * np.cos(angles), 20 + 5 * np.sin(angles)])
    for contour in (circle, circle[::-1]):
        normals = outward_normals(contour, span=1)
        radial = (contour - [10, 20]) / 5
        np.testing.assert_allclose((normals * radial).sum(axis=1), 1, atol=0.01)



@pytest.mark.parametrize("rule", RULES)
def test_spur_tip_is_kept_in_place_not_nan(rule):
    # A dark disk with a 1 px wide spur: at the spur tip the contour points k - 3 and k + 3
    # coincide, so there is no tangent. That point must stay put, not turn the centre into NaN.
    image = np.full((60, 60), 200, np.uint8)
    yy, xx = np.mgrid[:60, :60]
    image[(xx - 30) ** 2 + (yy - 30) ** 2 <= 64] = 40
    image[30, 38:46] = 40
    found = detect_contacts(image, blur_sigma=0)
    refined = refine_edges(image, found, rule=rule)
    assert np.isfinite(refined.centers).all()
    assert all(np.isfinite(c).all() for c in refined.contours)


def test_sample_profiles_follows_each_direction():
    image = np.zeros((50, 50), dtype=np.uint8)
    image[20, 30] = 100  # 10 px right of (20, 20)
    points = np.array([[20.0, 20.0], [20.0, 20.0]])
    directions = np.array([[1.0, 0.0], [0.0, 1.0]])  # right, and down (y points down)
    profile = sample_profiles(image, points, directions, np.array([0.0, 10.0]))
    np.testing.assert_allclose(profile, [[0, 100], [0, 0]])


def test_edge_offsets_find_steepest_rise_between_samples():
    offsets = np.arange(-5, 5, 0.5)
    steps = np.array([1.3, -2.2, 0.0])  # one profile per step position
    profiles = 40 + 160 / (1 + np.exp(-(offsets[None, :] - steps[:, None]) * 2))
    np.testing.assert_allclose(edge_offsets(profiles, offsets, rising=True), steps, atol=0.05)
    np.testing.assert_allclose(edge_offsets(-profiles, offsets, rising=False), steps, atol=0.05)


def exact_ellipse_image(center, semi_axes, size=80, dark=40, bright=180, blur=2.0, oversample=16):
    """Blurred dark ellipse with exact sub-pixel area coverage (cv2's anti-aliased drawing is
    slightly asymmetric, which would bias the test by ~0.1 px)."""
    fine_y, fine_x = (np.indices((size * oversample, size * oversample)) + 0.5) / oversample - 0.5
    inside = ((fine_x - center[0]) / semi_axes[0]) ** 2 + ((fine_y - center[1]) / semi_axes[1]) ** 2 < 1
    coverage = inside.reshape(size, oversample, size, oversample).mean(axis=(1, 3))
    image = (bright - (bright - dark) * coverage).astype(np.float32)
    return cv2.GaussianBlur(image, (0, 0), blur).round().astype(np.uint8)


def distance_to_ellipse(points, center, semi_axes):
    """Distance (px) from each point to the nearest point of the ellipse outline."""
    t = np.linspace(0, 2 * np.pi, 20000)
    outline = np.column_stack([center[0] + semi_axes[0] * np.cos(t), center[1] + semi_axes[1] * np.sin(t)])
    return np.array([np.linalg.norm(outline - p, axis=1).min() for p in points])


@pytest.mark.parametrize("rule", RULES)
def test_refined_contour_follows_an_ellipse(rule):
    # Elongated contact: semi-axes 24 x 10 px, a plain blurred step (no band). Its steepest rise and its half-way
    # level both lie on the true boundary, all the way around, so the refined contour should follow the ellipse.
    center, axes = (40.3, 37.6), (24, 10)
    image = exact_ellipse_image(center, axes)

    refined = refine_edges(image, detect_contacts(image, blur_sigma=0), blur_sigma=0, rule=rule)

    np.testing.assert_allclose(refined.centers, [center], atol=0.05)
    distance = distance_to_ellipse(refined.contours[0], center, axes)
    assert distance.mean() < 0.3
    # At the sharp tips of the long axis (curvature radius ~4 px) the blur pulls the edge
    # slightly inward; that is a property of the blurred image, not of the search.
    assert distance.max() < 1.5


@pytest.mark.parametrize("rule", RULES)
def test_search_window_is_a_fixed_pixel_distance(rule):
    image = exact_ellipse_image((40, 40), (20, 12))
    otsu = detect_contacts(image, blur_sigma=0)

    refined = refine_edges(image, otsu, search_px=0.5, blur_sigma=0, rule=rule)

    moved = np.linalg.norm(refined.contours[0] - otsu.contours[0], axis=1)
    assert moved.max() <= 0.5 + 1e-9


@pytest.mark.parametrize("rule", RULES)
def test_refine_keeps_centres_accurate_on_sem_like_image(tmp_path, rule):
    path = tmp_path / "tile.jpg"
    truth = write_sem_like(path, databar_px=0)
    image = load_sem_image(path)
    found = detect_contacts(image)

    refined = refine_edges(image, found, rule=rule)

    assert len(refined.centers) == len(found.centers) == len(truth)
    errors = np.linalg.norm(refined.centers[:, None] - truth[None], axis=2).min(axis=1)
    assert errors.max() < 0.2  # pixels
    # The refined edge sits further out than the Otsu edge (on the rise towards the bright rim).
    assert refined.areas.mean() > found.areas.mean()


def test_sample_profiles_handles_more_points_than_one_remap_call():
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    points = np.tile([[3.0, 4.0]], (70000, 1))  # more than cv2.remap's 32767-row limit
    profiles = sample_profiles(image, points, np.tile([[1.0, 0.0]], (70000, 1)), np.array([0.0, 2.0]))
    assert profiles.shape == (70000, 2)
    np.testing.assert_allclose(profiles[[0, -1]], [[43, 45], [43, 45]])


def banded_image(centres, radii, size, shading=0.0, noise=0.0, seed=0, oversample=4):
    """Contacts as an SEM shows them (D122): a dark interior (radius px, sub-pixel coverage) and a bright band just
    outside its edge, the band's brightness x (1 + shading cos(angle)) (detector shading), on a grey background;
    blurred, Gaussian noise added."""
    fine_y, fine_x = (np.indices((size * oversample, size * oversample)) + 0.5) / oversample - 0.5
    signal = np.full(fine_x.shape, 110.0)
    for (cx, cy), radius in zip(centres, radii):
        dx, dy = fine_x - cx, fine_y - cy
        edge = np.hypot(dx, dy) - radius  # signed distance from the edge, px
        band = np.exp(-0.5 * ((edge - 2.0) / 1.5) ** 2) * (1 + shading * np.cos(np.arctan2(dy, dx)))
        signal += -50.0 * (edge < 0) + 80.0 * band
    image = cv2.GaussianBlur(signal.reshape(size, oversample, size, oversample).mean(axis=(1, 3)), (0, 0), 1.2)
    image += np.random.default_rng(seed).normal(0, noise, image.shape)
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def centre_errors(found: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """(N, 2) found - true centre of every true contact (its nearest found centre)."""
    _, nearest = cKDTree(found).query(truth)
    return found[nearest] - truth


def test_mid_level_follows_noisy_edges_better_than_the_steepest_gradient():
    # Strong noise: the steepest rise jumps to noise point by point, the half-way level does not (D122). Each
    # rule's common shift is left out: a shift of every contact is harmless.
    rng = np.random.default_rng(1)
    centres = np.array([(x, y) for x in range(24, 240, 48) for y in range(24, 240, 48)]) + rng.uniform(-0.5, 0.5, (25, 2))
    image = banded_image(centres, np.full(25, 12.0), 240, noise=25, seed=2)
    found = detect_contacts(image, method="band")
    assert len(found.centers) == 25

    scatter = {}
    for rule in RULES:
        errors = centre_errors(refine_edges(image, found, rule=rule).centers, centres)
        scatter[rule] = np.sqrt(((errors - errors.mean(axis=0)) ** 2).mean())
    assert scatter["mid-level"] < 0.75 * scatter["steepest gradient"]  # 0.54-0.65 over noise seeds 2-6


def test_mid_level_centre_does_not_depend_on_contact_size_under_one_sided_shading():
    # One side of the band brighter (detector shading) shifts every contact's edge the same way; it must not shift
    # contacts of different size differently (as fitting a model of the mean contact would, D122).
    centres, radii = np.array([[30.3, 30.6], [90.7, 30.2], [30.5, 90.4], [90.2, 90.8]]), np.array([9.0, 12.0, 15.0, 18.0])
    image = banded_image(centres, radii, 120, shading=0.4)
    found = detect_contacts(image, method="band")

    errors = centre_errors(refine_edges(image, found).centers, centres)

    assert len(found.centers) == 4
    # about the same shift (here ~0.5 px) whatever the size: it changes by under 1 % of the radius change (a
    # fitted mean contact moved by about the whole radius change)
    assert np.ptp(errors, axis=0).max() < 0.01 * np.ptp(radii)


def test_isolated_points_take_their_neighbours_median_and_a_real_bump_stays():
    shifts = np.zeros(60)
    shifts[10] = 5.0  # one point jumped to noise
    shifts[20] = np.nan  # one point found no crossing
    shifts[40:49] = 1.5  # nine neighbouring points out: a real bump of the edge

    cleaned = without_isolated(shifts)

    assert cleaned[10] == 0 and cleaned[20] == 0
    np.testing.assert_array_equal(cleaned[41:48], 1.5)
    assert cleaned.sum() == pytest.approx(13.5)  # the bump's steps are rounded, its area kept


def test_bright_contacts_get_the_contour_of_the_negative_image():
    centres = np.array([[30.3, 30.6], [90.7, 30.2]])
    image = banded_image(centres, [12.0, 12.0], 120, noise=5)
    found = detect_contacts(image, method="band")

    dark = refine_edges(image, found)
    bright = refine_edges(255 - image, found, dark_contacts=False)

    np.testing.assert_allclose(bright.centers, dark.centers, atol=1e-4)


def no_band_image(centres, radius, shape=(80, 140), noise=0.0, seed=0, oversample=8) -> np.ndarray:
    """Dark disks (60, exact sub-pixel coverage) on a grey background (130), no bright band (wafer ADI images, D130);
    beam blur 1 px, Gaussian noise."""
    fine_y, fine_x = (np.indices((shape[0] * oversample, shape[1] * oversample)) + 0.5) / oversample - 0.5
    inside = np.zeros(fine_x.shape, bool)
    for x, y in centres:
        inside |= np.hypot(fine_x - x, fine_y - y) < radius
    coverage = inside.reshape(shape[0], oversample, shape[1], oversample).mean(axis=(1, 3))
    image = cv2.GaussianBlur((130 - 70 * coverage).astype(np.float32), (0, 0), 1.0)
    image += np.random.default_rng(seed).normal(0, noise, shape)
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def test_crossings_find_where_each_profile_rises_through_the_level():
    offsets = np.arange(-2.0, 2.5, 0.5)
    profiles = np.array([100 + 10 * offsets, np.full(len(offsets), 50.0)])  # through 105 at +0.5; never
    np.testing.assert_allclose(crossings(profiles, offsets, 105.0), [0.5, np.nan])


def test_threshold_rule_puts_the_contour_on_the_detection_threshold():
    """D130: each refined point lies where the smoothed image crosses the detection's threshold, one level for every
    contact; the centre is the disk's."""
    centres = np.array([[40.3, 39.6], [100.8, 40.2]])
    image = no_band_image(centres, 12.0)
    found = detect_contacts(image)
    refined = refine_edges(image, found, rule="threshold")

    smoothed = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.0)  # refine_edges' own smoothing
    points = np.concatenate(refined.contours)
    values = sample_profiles(smoothed, points, np.zeros_like(points), np.zeros(1))[:, 0]
    assert np.abs(values - found.threshold).max() < 0.5
    np.testing.assert_allclose(refined.centers, centres, atol=0.01)  # found: 0.002 px


def test_threshold_rule_on_bright_contacts_is_that_of_the_negative_image():
    image = no_band_image([[40.3, 39.6], [100.8, 40.2]], 12.0, noise=5)
    dark = refine_edges(image, detect_contacts(image), rule="threshold")
    negative = 255 - image
    bright = refine_edges(negative, detect_contacts(negative, dark_contacts=False), dark_contacts=False,
                          rule="threshold")
    np.testing.assert_allclose(bright.centers, dark.centers, atol=0.02)


def test_unknown_rule_is_refused():
    image = exact_ellipse_image((40, 40), (20, 12))
    with pytest.raises(ValueError, match="rule"):
        refine_edges(image, detect_contacts(image, blur_sigma=0), rule="steepest")
