import cv2
import numpy as np
import pytest

from affine_ransac.features.contact import detect_contacts
from affine_ransac.io.sem_image import crop_databar, load_sem_image
from sample_sem import write_sem_like


def nearest_distances(found, truth):
    """Distance from each found point to its nearest true point."""
    return np.linalg.norm(found[:, None, :] - truth[None, :, :], axis=2).min(axis=1)


def test_finds_all_contacts_accurately(tmp_path):
    path = tmp_path / "tile.jpg"
    truth = write_sem_like(path, databar_px=64)
    image = crop_databar(load_sem_image(path), 64)

    found = detect_contacts(image)

    assert len(found.centers) == len(truth)
    assert nearest_distances(found.centers, truth).max() < 0.2  # pixels
    assert len(found.contours) == len(truth)
    assert found.areas.min() > 300  # radius-14 holes, ~460 px after blur + threshold


def test_contact_cut_by_image_border_is_dropped():
    image = np.full((100, 100), 120, dtype=np.uint8)
    cv2.circle(image, (50, 50), 10, 30, thickness=-1)  # fully inside
    cv2.circle(image, (0, 50), 10, 30, thickness=-1)  # cut by the left border

    found = detect_contacts(image, blur_sigma=0)

    np.testing.assert_allclose(found.centers, [[50, 50]], atol=1e-9)


def test_small_regions_are_dropped():
    image = np.full((100, 100), 120, dtype=np.uint8)
    cv2.circle(image, (30, 30), 10, 30, thickness=-1)
    image[70:72, 70:72] = 30  # 4-pixel speck

    found = detect_contacts(image, blur_sigma=0, min_area_px=20)

    assert len(found.centers) == 1


def test_bright_contacts():
    image = np.full((100, 100), 60, dtype=np.uint8)
    cv2.circle(image, (40, 60), 10, 220, thickness=-1)

    assert len(detect_contacts(image, blur_sigma=0).centers) == 0  # wrong polarity
    found = detect_contacts(image, dark_contacts=False, blur_sigma=0)
    np.testing.assert_allclose(found.centers, [[40, 60]], atol=1e-9)


def test_contour_surrounds_centre():
    image = np.full((60, 60), 120, dtype=np.uint8)
    cv2.circle(image, (25, 35), 8, 30, thickness=-1)

    contour = detect_contacts(image, blur_sigma=0).contours[0]

    radii = np.linalg.norm(contour - [25, 35], axis=1)
    assert 7 <= radii.min() and radii.max() <= 9


def test_crop_databar_keeps_coordinates():
    image = np.arange(50, dtype=np.uint8).reshape(10, 5)
    cropped = crop_databar(image, 3)
    assert cropped.shape == (7, 5)
    assert cropped[2, 4] == image[2, 4]  # same (x, y) -> same pixel
    assert crop_databar(image, 0) is image


def wide_band_image(band_px, pitch_px=64, radius_px=14, seed=0):
    """Dark holes inside bright bands band_px wide on a grey background; returns (image, centres)."""
    image = np.full((512, 512), 110, dtype=np.uint8)
    centers = np.array([(x, y) for y in range(40, 480, pitch_px) for x in range(40, 480, pitch_px)])
    for x, y in centers:
        cv2.circle(image, (int(x), int(y)), radius_px + band_px // 2 + 1, 200, thickness=band_px)
        cv2.circle(image, (int(x), int(y)), radius_px, 40, thickness=-1)
    image = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
    noise = np.random.default_rng(seed).normal(0, 6, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8), centers.astype(float)


def test_three_classes_find_the_holes_when_bands_are_wide():
    # Bands 16 px wide nearly touch: plain Otsu splits background | band and picks up the
    # background pinched off between the bands; the three-class threshold (hole | background) does not.
    image, truth = wide_band_image(band_px=16)
    assert len(detect_contacts(image).centers) > len(truth)

    found = detect_contacts(image, classes=3)

    assert len(found.centers) == len(truth)
    assert nearest_distances(found.centers, truth).max() < 0.2


def test_three_classes_agree_with_plain_otsu_on_narrow_bands():
    image, truth = wide_band_image(band_px=3)
    plain, three = detect_contacts(image), detect_contacts(image, classes=3)
    assert len(plain.centers) == len(three.centers) == len(truth)
    assert nearest_distances(three.centers, truth).max() < 0.2


def test_three_classes_bright_contacts_and_bad_class_count():
    image, truth = wide_band_image(band_px=16)
    found = detect_contacts(255 - image, dark_contacts=False, classes=3)  # bright holes, dark bands
    assert len(found.centers) == len(truth)
    with pytest.raises(ValueError, match="2 or 3"):
        detect_contacts(image, classes=4)
