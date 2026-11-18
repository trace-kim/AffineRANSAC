"""The temporary horizontal L/S release path, on synthetic SEM images and OASIS clips."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import cv2
import klayout.db as kdb
import numpy as np
import pytest

from affine_ransac.analysis import HORIZONTAL_LS
from affine_ransac.release import app as release_app
from affine_ransac.release.agent import review
from affine_ransac.release.analysis import (DEFAULTS, AnalysisFailed, analyse_folder, read_contours,
                                           read_design_images, read_display_images, tile_affines)
from affine_ransac.release.cache import load_cache
from affine_ransac.release.contour_dialog import diagnose
from affine_ransac.release.horizontal_ls import find_bands
from test_release import make_window, wait


SETTINGS = replace(DEFAULTS, pattern=HORIZONTAL_LS, detection_blur_px=0, edge_blur_px=0, max_match_nm=7,
                   design_max_match_nm=7, design_tolerance_nm=3, group_radius_nm=2, min_matched=2,
                   moving_window_um=0.24, moving_step_um=0.08)


def test_white_separators_give_both_interiors_and_close_at_the_sides():
    image = np.full((60, 80), 50, np.uint8)
    image[22:34] = 52  # nearly the same tone as the space on either side
    image[[7, 21, 35, 49]] = 225
    s = replace(SETTINGS, detection="otsu3", dark_contacts=False, upscale=2)
    d = diagnose(image, s)
    for found in (d.detected, d.refined):
        np.testing.assert_allclose(found.centers, [[39.5, 14], [39.5, 28], [39.5, 42]], atol=0.15)
        for c in found.contours:
            # The upper and lower edges meet via vertical segments on the right and left.
            np.testing.assert_array_equal(c[[0, 79, 80, -1], 0], [0, 79, 79, 0])
    assert d.mask[28, 80] and not d.mask[42, 80]
    assert len(find_bands(np.full_like(image, 50), s)[1].centers) == 0


def test_a_thin_white_band_can_step_diagonally_without_joining_its_two_interiors():
    image = np.full((60, 80), 60, np.uint8)
    x = np.arange(80)
    for y in [8, 20, 32, 44]:
        image[y + x // 20, x] = 220
    detected, _ = find_bands(image, replace(SETTINGS, refinement="none"))
    np.testing.assert_allclose(detected.centers[:, 1], [15.5, 27.5, 39.5])
    np.testing.assert_array_equal(detected.contours[0][:80, 1], 9 + x // 20)


def mosaic(folder):
    """Three stripes of three tiles. The SEM pixels are 1 × 2 nm. Alternate clips draw the opposite tone."""
    records = []
    clips = folder / "ContourCAD"
    clips.mkdir()
    sem_offsets = np.array([-2, 0, 2, 2, -2, 0, 0, 2, -2])
    design_offsets = np.array([1, -1, 0, -1, 0, 1, 0, 1, -1])
    edges = np.arange(-255, 580, 16)
    for k in range(9):
        x, y = (k // 3) * 96, (k % 3) * 176
        image = np.full((128, 128), 70, np.uint8)
        # A common SEM offset and a repeating 2 nm edge error survive both tile stitchings.
        heights = edges + 2 + 2 * (np.arange(len(edges)) % 4 == 0) - y - sem_offsets[k]
        rows = (127 - heights) // 2
        image[rows[(rows >= 0) & (rows < len(image))]] = 230
        image_path, oas_path = folder / f"CD{k}.png", clips / f"SA{k}.oas"
        cv2.imencode(".png", image)[1].tofile(str(image_path))

        layout = kdb.Layout()
        layout.dbu = 0.001
        shapes = layout.create_cell("TOP").shapes(layout.layer(4, 0))
        for low, high in zip(edges[k % 2::2], edges[1 + k % 2::2]):
            # Fractured lines extend past the sides of the clip.
            bottom, top = int(low - y - design_offsets[k]), int(high - y - design_offsets[k])
            for left, right in ((-70, 0), (0, 70)):
                shapes.insert(kdb.Box(left, bottom, right, top))
        layout.write(str(oas_path))
        records.append(SimpleNamespace(tile_id=f"T{k}", image_path=image_path, oas_path=oas_path,
                                       fov_x_nm=128, fov_y_nm=256, center_x_nm=1e6 + x, center_y_nm=2e6 + y))
    return records, sem_offsets, design_offsets


@pytest.mark.parametrize("per_stripe", [False, True])
def test_release_stitches_both_sets_in_y_and_calibrates_the_measured_heights(tmp_path, monkeypatch, per_stripe):
    records, sem_offsets, design_offsets = mosaic(tmp_path)
    messages = []
    with ThreadPoolExecutor(2) as pool:
        if per_stripe:
            run = analyse_folder(tmp_path, "images", replace(SETTINGS, moving_window_per_stripe=True),
                                 records, log=messages.append, keep_tiles=True, pool=pool)
        else:
            window = make_window(tmp_path)
            monkeypatch.setattr(release_app, "image_records", lambda folder: records)
            monkeypatch.setattr(window, "settings", lambda: SETTINGS)
            monkeypatch.setattr(window, "kept_helpers", lambda: pool)
            window.open_folder(str(tmp_path))
            wait(window)
            run = window.result
        for stitch, known in ((run.stitch, sem_offsets), (run.design_stitch, design_offsets)):
            assert len(stitch.pairs) == 20  # includes horizontal and corner overlaps outside both centroid columns
            np.testing.assert_array_equal(stitch.corrections[:, 0], 0)
            np.testing.assert_allclose(stitch.corrections[:, 1], known - known.mean(), atol=1e-8)
        np.testing.assert_array_equal(run.sem_nm[:, 0], run.design_nm[:, 0])
        height_error = run.sem_nm[:, 1] - run.design_nm[:, 1]
        np.testing.assert_allclose(np.unique(height_error.round(6)), [2, 3])
        assert len(run.moving.counts) > 1
        assert np.isnan(run.moving.residuals[:, 0]).all()
        assert 0.1 < np.abs(run.moving.residuals[:, 1]).max() < 1  # y offset removed, local errors retained
        np.testing.assert_array_equal(tile_affines(run)[:, 0], np.tile([1, 0, 0], (9, 1)))
        assert len(np.unique(run.moving.stripe)) == (3 if per_stripe else 0)
        assert review(run)["measured_axes"] == ["y"]

        if not per_stripe:
            reopened, _ = load_cache(tmp_path, window.cache_dir)
            np.testing.assert_equal(reopened.moving.residuals, run.moving.residuals)
            c = read_contours(reopened, log=messages.append, pool=pool)
            assert not any("differ from the centres" in line for line in messages)
            np.testing.assert_array_equal(c.refined[c.start[:-1]], c.refined[c.start[1:] - 2])
            assert all(mask.min() == 0 and mask.max() == 1 for mask in read_design_images(reopened))
            window.stitching_view.set_contours(c)
            window.stitching_view.set_images(read_display_images(run.image_paths))
            assert window.stitching_view.image_bases[0][1, 1] == 2.0
            view = window.registration_view
            assert "X is unmeasured" in view.summary.text()
            assert [panel.name for panel, _, _ in view.maps] == ["dy"]
            view.values.setCurrentIndex(1)  # grid average and its exports also contain only measured errors
            assert "dx" not in view.profiles_text(",") and "mean dy" in view.profiles_text(",")
            window.close()


def test_uniform_sem_stops_with_inspectable_empty_centres(tmp_path):
    records, _, _ = mosaic(tmp_path)
    cv2.imencode(".png", np.full((128, 128), 90, np.uint8))[1].tofile(str(records[0].image_path))
    with pytest.warns(UserWarning), pytest.raises(AnalysisFailed) as failed:
        analyse_folder(tmp_path, "images", SETTINGS, records[:1], log=lambda text: None)
    assert failed.value.result.sem_points[0].shape == (0, 2)
    assert len(failed.value.result.design_points[0]) and failed.value.result.moving is None


def test_missing_and_blank_clips_leave_the_other_rows_measurable(tmp_path):
    records, _, _ = mosaic(tmp_path)
    empty = kdb.Layout()
    empty.create_cell("TOP")
    empty.write(str(records[0].oas_path))
    records[1].oas_path = None
    with ThreadPoolExecutor(2) as pool, pytest.warns(UserWarning):
        run = analyse_folder(tmp_path, "images", SETTINGS, records, log=lambda text: None, pool=pool)
    np.testing.assert_array_equal(run.design_stitch.unplaced, [0, 1])
    assert not len(run.stitch.unplaced) and np.isfinite(run.moving.residuals[:, 1]).all()
    masks = read_design_images(run)
    assert not masks[0].any() and masks[1] is None
