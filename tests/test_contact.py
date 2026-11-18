import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from skimage.restoration import denoise_tv_chambolle

from affine_ransac.features.contact import (DetectedContacts, contact_mask, denoised_image, detect_contacts,
                                            equalization_lut, flatten_background, in_image_pixels, upscaled_image)
from affine_ransac.features.total_variation import total_variation
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


def test_an_upscaled_image_covers_the_same_area_and_its_contacts_map_back():
    """D131: pixel (x, y) of the image upscaled f times lies at ((x + 0.5) / f - 0.5, (y + 0.5) / f - 0.5) of the
    image; a spot's centroid in the upscaled image maps back onto its pixel, and areas shrink by f squared."""
    image = np.zeros((9, 11), np.uint8)
    image[4, 6] = 200
    assert upscaled_image(image, 1) is image
    for factor in (2, 3, 4):
        big = upscaled_image(image, factor)
        assert big.shape == (9 * factor, 11 * factor)
        rows, cols = np.indices(big.shape)
        weight = big.astype(float)
        centre = np.array([[(weight * cols).sum(), (weight * rows).sum()]]) / weight.sum()
        found = in_image_pixels(DetectedContacts(centre, np.array([16.0 * factor ** 2]), [centre], 100.0), factor)
        np.testing.assert_allclose(found.centers, [[6, 4]], atol=1e-6)
        np.testing.assert_allclose(found.contours[0], [[6, 4]], atol=1e-6)
        assert found.areas[0] == 16 and found.threshold == 100.0


def test_total_variation_gives_scikit_images_numbers():
    """D159: the compiled loop returns exactly what scikit-image's does, whether it ends by its cost (weights 0.05 to
    0.5: after different numbers of rounds) or by the limit of rounds; and how many rounds it took."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:61, :90]
    disks = 120 + 60 * (((xx % 20) - 10) ** 2 + ((yy % 20) - 10) ** 2 > 30)
    image = np.clip(disks + rng.normal(0, 25, disks.shape), 0, 255) / 255.0
    for weight in (0.05, 0.1, 0.5):
        flat, rounds = total_variation(image, weight)
        np.testing.assert_array_equal(flat, denoise_tv_chambolle(image, weight=weight))
        assert 1 < rounds < 200
        assert not np.array_equal(flat, denoise_tv_chambolle(image, weight=weight, max_num_iter=rounds - 1))
    flat, rounds = total_variation(image, 0.5, max_rounds=3)
    np.testing.assert_array_equal(flat, denoise_tv_chambolle(image, weight=0.5, max_num_iter=3))
    assert rounds == 3


def test_total_variation_warmup_runs_once_for_concurrent_callers(monkeypatch):
    from affine_ransac.features import total_variation as tv

    monkeypatch.setattr(tv, "_loops_ready", False)
    entered, finish = threading.Event(), threading.Event()
    start = threading.Barrier(4)
    calls = []

    def warmup(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert finish.wait(5)

    def prepare():
        start.wait(timeout=5)
        tv.compile_loops()

    monkeypatch.setattr(tv, "total_variation", warmup)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(prepare) for _ in range(4)]
        try:
            assert entered.wait(5)
            assert not any(task.done() for task in pending)
        finally:
            finish.set()
        for task in pending:
            task.result(timeout=5)
    tv.compile_loops()
    assert len(calls) == 1


def test_total_variation_warmup_retries_after_failure(monkeypatch):
    from affine_ransac.features import total_variation as tv

    monkeypatch.setattr(tv, "_loops_ready", False)
    calls = []

    def warmup(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("compilation failed")

    monkeypatch.setattr(tv, "total_variation", warmup)
    with pytest.raises(RuntimeError, match="compilation failed"):
        tv.compile_loops()
    tv.compile_loops()
    tv.compile_loops()
    assert len(calls) == 2


def test_each_denoising_method_is_the_librarys_own():
    """D132: OpenCV's median and non-local means (patch 7 px, search 21 px), scikit-image's total variation
    (Chambolle) on the grey levels / 255; none gives the image itself."""
    image = np.random.default_rng(0).integers(0, 256, (40, 50)).astype(np.uint8)
    assert denoised_image(image) is image
    np.testing.assert_array_equal(denoised_image(image, "median", median_px=5), cv2.medianBlur(image, 5))
    np.testing.assert_array_equal(denoised_image(image, "non-local means", nlm_h=12.0),
                                  cv2.fastNlMeansDenoising(image, None, 12.0, 7, 21))
    flat = denoise_tv_chambolle(image / 255.0, weight=0.2)
    np.testing.assert_array_equal(denoised_image(image, "total variation", tv_weight=0.2),
                                  np.clip(np.round(flat * 255), 0, 255).astype(np.uint8))
    with pytest.raises(ValueError, match="denoise must be one of"):
        denoised_image(image, "wavelet")


def test_the_median_removes_isolated_pixels_and_keeps_an_edge_where_it_is():
    edge = np.full((20, 20), 60, np.uint8)
    edge[:, 10:] = 130
    spotted = edge.copy()
    spotted[5, 4], spotted[12, 15] = 200, 0  # a bright pixel in the dark side, a dark one in the bright side
    np.testing.assert_array_equal(denoised_image(spotted, "median"), edge)


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


def test_the_equalization_table_is_opencvs_histogram_equalization():
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (37, 501)), np.clip(rng.normal(150, 9, (300, 200)), 0, 255),
              rng.choice([3, 90, 91, 250], (64, 64)), np.full((10, 10), 77)]  # wide, narrow, few levels, one level
    for image in (np.asarray(i, np.uint8) for i in images):
        np.testing.assert_array_equal(cv2.LUT(image, equalization_lut(image)), cv2.equalizeHist(image))
        assert (np.diff(equalization_lut(image).astype(int)) >= 0).all()  # keeps the order of the grey levels


def test_global_equalization_cuts_the_image_at_one_grey_level():
    """D167: the threshold is chosen on the equalized grey levels, and it is that grey level of the blurred image:
    the mask is the blurred image cut there, for every method and either tone."""
    image, _ = wide_band_image(band_px=16)  # three grey levels: otsu3 and band have their classes
    blurred = cv2.GaussianBlur(image, (0, 0), 1.5)
    for method in ("otsu", "otsu3", "band"):
        for dark in (True, False):
            mask, threshold = contact_mask(image, dark, 1.5, method, equalize="global")
            np.testing.assert_array_equal(mask, (blurred <= threshold) if dark else (blurred > threshold))
            on_equalized, _ = contact_mask(cv2.equalizeHist(blurred), dark, 0, method)
            np.testing.assert_array_equal(mask, on_equalized)
    assert contact_mask(image, equalize="global")[1] != contact_mask(image)[1]  # Otsu chose on other grey levels


def test_adaptive_equalization_thresholds_the_clahe_image():
    image, _ = wide_band_image(band_px=3)
    blurred = cv2.GaussianBlur(image, (0, 0), 1.5)
    equalized = cv2.createCLAHE(3.0, (4, 4)).apply(blurred)
    threshold, expected = cv2.threshold(equalized, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask, found = contact_mask(image, True, 1.5, "otsu", equalize="adaptive", clahe_clip=3.0, clahe_tiles=4)
    np.testing.assert_array_equal(mask, expected)
    assert found == threshold
    with pytest.raises(ValueError, match="equalize must be one of"):
        contact_mask(image, equalize="local")


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
