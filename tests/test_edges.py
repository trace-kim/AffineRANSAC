import cv2
import numpy as np

from affine_ransac.features.contact import detect_contacts
from affine_ransac.features.edges import edge_radius, refine_edges, sample_rays
from affine_ransac.io.sem_image import load_sem_image
from sample_sem import write_sem_like


def test_sample_rays_follows_the_ray():
    image = np.zeros((50, 50), dtype=np.uint8)
    image[20, 30] = 100  # 10 px right of (20, 20)
    angles = np.array([0.0, np.pi / 2])  # right, and down (y points down)
    profile = sample_rays(image, (20, 20), angles, np.array([0.0, 10.0]))
    np.testing.assert_allclose(profile, [[0, 100], [0, 0]])


def test_edge_radius_finds_steepest_rise_between_samples():
    radii = np.arange(0, 20, 1.0)
    profile = 40 + 160 / (1 + np.exp(-(radii - 10.3) * 2))  # smooth step centred at r = 10.3
    assert abs(edge_radius(profile, radii, rising=True) - 10.3) < 0.1
    assert abs(edge_radius(-profile, radii, rising=False) - 10.3) < 0.1


def exact_disc_image(center, radius, size=80, dark=40, bright=180, blur=2.0, oversample=16):
    """Blurred dark disc with exact sub-pixel area coverage (cv2.circle's anti-aliasing is
    slightly asymmetric, which would bias the test by ~0.1 px)."""
    fine = (np.indices((size * oversample, size * oversample)) + 0.5) / oversample - 0.5
    inside = (fine[1] - center[0]) ** 2 + (fine[0] - center[1]) ** 2 < radius ** 2
    coverage = inside.reshape(size, oversample, size, oversample).mean(axis=(1, 3))
    image = (bright - (bright - dark) * coverage).astype(np.float32)
    return cv2.GaussianBlur(image, (0, 0), blur).round().astype(np.uint8)


def test_refined_edge_is_at_steepest_rise_of_a_blurred_disc():
    # Dark disc of radius 15 px on a bright background, blurred: the steepest rise is at r ~ 15.
    image = exact_disc_image((40, 37), 15)

    refined = refine_edges(image, detect_contacts(image, blur_sigma=0), blur_sigma=0)

    np.testing.assert_allclose(refined.centers, [[40, 37]], atol=0.05)
    radii = np.linalg.norm(refined.contours[0] - [40, 37], axis=1)
    assert abs(radii.mean() - 15) < 0.5 and radii.std() < 0.2


def test_refine_keeps_centres_accurate_on_sem_like_image(tmp_path):
    path = tmp_path / "tile.jpg"
    truth = write_sem_like(path, databar_px=0)
    image = load_sem_image(path)
    found = detect_contacts(image)

    refined = refine_edges(image, found)

    assert len(refined.centers) == len(found.centers) == len(truth)
    errors = np.linalg.norm(refined.centers[:, None] - truth[None], axis=2).min(axis=1)
    assert errors.max() < 0.2  # pixels
    # The refined edge sits further out than the Otsu edge (on the rise towards the bright rim).
    assert refined.areas.mean() > found.areas.mean()
