import cv2
import numpy as np
import pytest

from affine_ransac.features.contact import detect_contacts, flatten_background
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


def test_regions_left_out_are_counted_per_reason():
    """D128: each region not taken as a contact is counted under the first reason that applies."""
    image = np.full((100, 100), 120, dtype=np.uint8)
    cv2.circle(image, (30, 30), 6, 30, thickness=-1)  # a contact, ~113 px
    cv2.circle(image, (70, 70), 12, 30, thickness=-1)  # larger, ~450 px
    cv2.circle(image, (0, 50), 6, 30, thickness=-1)  # cut by the border
    image[80:82, 20:22] = 30  # 4-pixel speck
    image[10:12, 60:62] = 30  # another

    found = detect_contacts(image, blur_sigma=0, min_area_px=20, max_area_px=300)

    np.testing.assert_allclose(found.centers, [[30, 30]], atol=1e-9)
    assert found.left_out == {"border": 1, "small": 2, "large": 1, "not solid": 0}


def ramp_image():
    """Dark disks (radius 7 px) 50 grey levels below their surroundings, on a background that brightens from 60 to 200
    across the image (charging, shading); and the disks' centres."""
    yy, xx = np.mgrid[:200, :300]
    centres = np.array([(x, y) for x in range(30, 300, 40) for y in range(30, 200, 40)], float)
    holes = np.zeros(xx.shape, bool)
    for x, y in centres:
        holes |= (xx - x) ** 2 + (yy - y) ** 2 <= 49
    return np.clip(60 + 140 * xx / 299 - 50 * holes, 0, 255).astype(np.uint8), centres


def test_flattening_the_background_lets_one_threshold_find_every_contact():
    """D130: one threshold cannot fit a background that brightens across the image; with the brightness around the
    contacts subtracted, every contact is found where it is, and the grey levels stay in the image's range."""
    image, centres = ramp_image()
    assert len(detect_contacts(image, blur_sigma=0).centers) < len(centres) / 2

    flat = flatten_background(image, 20)
    found = detect_contacts(flat, blur_sigma=0)
    assert len(found.centers) == len(centres)
    np.testing.assert_allclose(found.centers[np.lexsort(found.centers.T[::-1])],
                               centres[np.lexsort(centres.T[::-1])], atol=0.05)
    assert abs(float(flat.mean()) - float(image.mean())) < 5


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


def wide_band_image(band_px, pitch_px=64, radius_px=14, hole_grey=40, seed=0):
    """Holes (grey level hole_grey) inside bright bands band_px wide on a grey (110) background;
    returns (image, centres)."""
    image = np.full((512, 512), 110, dtype=np.uint8)
    centers = np.array([(x, y) for y in range(40, 480, pitch_px) for x in range(40, 480, pitch_px)])
    for x, y in centers:
        cv2.circle(image, (int(x), int(y)), radius_px + band_px // 2 + 1, 200, thickness=band_px)
        cv2.circle(image, (int(x), int(y)), radius_px, hole_grey, thickness=-1)
    image = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
    noise = np.random.default_rng(seed).normal(0, 6, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8), centers.astype(float)


def test_three_classes_find_the_holes_when_bands_are_wide():
    # Bands 16 px wide nearly touch: plain Otsu splits background | band and picks up the
    # background pinched off between the bands; the three-class threshold (hole | background) does not.
    image, truth = wide_band_image(band_px=16)
    assert len(detect_contacts(image).centers) > len(truth)

    found = detect_contacts(image, method="otsu3")

    assert len(found.centers) == len(truth)
    assert nearest_distances(found.centers, truth).max() < 0.2


def test_three_classes_agree_with_plain_otsu_on_narrow_bands():
    image, truth = wide_band_image(band_px=3)
    plain, three = detect_contacts(image), detect_contacts(image, method="otsu3")
    assert len(plain.centers) == len(three.centers) == len(truth)
    assert nearest_distances(three.centers, truth).max() < 0.2


def test_three_classes_bright_contacts_and_bad_method():
    image, truth = wide_band_image(band_px=16)
    found = detect_contacts(255 - image, dark_contacts=False, method="otsu3")  # bright holes, dark bands
    assert len(found.centers) == len(truth)
    with pytest.raises(ValueError, match="method must be one of"):
        detect_contacts(image, method="nope")


def test_band_method_finds_the_regions_enclosed_by_bands():
    for band_px in (3, 16):  # narrow, and wide bands that nearly touch (background pinched off)
        image, truth = wide_band_image(band_px)
        found = detect_contacts(image, method="band")
        assert len(found.centers) == len(truth), band_px
        assert nearest_distances(found.centers, truth).max() < 0.2


def test_band_method_does_not_need_holes_darker_than_the_background():
    image, truth = wide_band_image(band_px=16, hole_grey=110)  # interior = background grey
    found = detect_contacts(image, method="band")
    assert len(found.centers) == len(truth)
    assert nearest_distances(found.centers, truth).max() < 0.2


def test_solidity_filter_drops_the_pinched_off_background():
    image, truth = wide_band_image(band_px=16)
    unfiltered = detect_contacts(image, method="band", min_solidity=0.0)
    assert len(unfiltered.centers) > len(truth)  # the concave background pieces between the bands


def test_contact_contrast_tells_contacts_from_regions_cut_out_of_noise():
    from sample_sem import render_tile
    from affine_ransac.features.contact import contact_contrast, keep_contacts
    lattice = np.array([(x, y) for x in np.arange(-1530, 1530, 90.0) for y in np.arange(-1530, 1530, 90.0)])
    pattern = render_tile(lattice, (0.0, 0.0), seed=1)
    noise = cv2.GaussianBlur(np.random.default_rng(0).normal(0, 12, pattern.shape), (0, 0), 1.0)  # correlated
    blank = np.clip(110 + noise, 0, 255).astype(np.uint8)

    real, false = (detect_contacts(image, method="otsu") for image in (pattern, blank))

    assert len(real.centers) > 20 and len(false.centers) > 20  # Otsu always cuts something out of noise
    assert contact_contrast(pattern, real).min() > 10
    assert contact_contrast(blank, false).max() < 5
    kept = keep_contacts(real, np.arange(len(real.centers)) % 2 == 0)
    assert len(kept.contours) == len(kept.centers) == len(kept.areas) == (len(real.centers) + 1) // 2
