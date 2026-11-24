"""The temporary horizontal L/S release path, on synthetic SEM images and OASIS clips."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import cv2
import klayout.db as kdb
import numpy as np
import pytest

from affine_ransac.analysis import HORIZONTAL_LS
from affine_ransac.fitting.moving_window import moving_window_affine
from affine_ransac.io.design import horizontal_band_outlines
from affine_ransac.release import app as release_app
from affine_ransac.release import horizontal_ls
from affine_ransac.release.agent import review
from affine_ransac.release.analysis import (DEFAULTS, AnalysisFailed, analyse_folder, read_contours,
                                           read_design_images, read_display_images, tile_affines)
from affine_ransac.release.cache import load_cache, save_cache
from affine_ransac.release.contour_dialog import count_lines, diagnose
from affine_ransac.release.horizontal_ls import find_bands, paired_rows, stitch_bands
from affine_ransac.release.views import NO_LS_CD, ContactSizeTab, LineSpaceCDView, matrix_of
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
        expected = [[x, y] for x in np.arange(3.5, 80, 8) for y in (14, 28, 42)]
        np.testing.assert_allclose(found.centers, expected, atol=0.15)
        for c, (x, _) in zip(found.contours, expected):
            # Vertical closures at adjoining pixel faces, including both outer image edges.
            np.testing.assert_array_equal(c[[0, len(c) // 2 - 1, len(c) // 2, -1], 0],
                                          [x - 4, x + 4, x + 4, x - 4])
    assert "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3]" in count_lines(d, s)[0]
    assert d.mask[28, 80] and not d.mask[42, 80]
    assert len(find_bands(np.full_like(image, 50), s)[1].centers) == 0


def test_a_thin_white_band_can_step_diagonally_without_joining_its_two_interiors():
    image = np.full((60, 80), 60, np.uint8)
    x = np.arange(80)
    for y in [8, 20, 32, 44]:
        image[y + x // 20, x] = 220
    detected, _, _ = find_bands(image, replace(SETTINGS, refinement="none"))
    centres = detected.centers.reshape(10, 3, 2)
    np.testing.assert_allclose(centres[:, :, 1], (14 + (x // 20).reshape(10, 8).mean(axis=1))[:, None]
                               + [0, 12, 24])
    for k, c in enumerate(detected.contours[::3]):
        np.testing.assert_array_equal(c[1:9, 1], 8.5 + x[k * 8:k * 8 + 8] // 20)


def test_different_interior_levels_and_a_broken_separator_only_cost_local_slices():
    image = np.full((77, 103), 40, np.uint8)
    image[21:41] = 140  # a two-class threshold would put this interior with the white separators
    image[61:] = 140
    image[[1, 21, 41, 61]] = 240
    image[41, 36] = 80  # joins two regions in slice 4
    image[52, 85] = 240  # an isolated bright speck inside a different slice
    s = replace(SETTINGS, refinement="none", min_area_px=1500, max_area_px=2200)
    d = diagnose(image, s)
    assert d.mask[30, 5] and not d.mask[41, 5]
    for k in range(10):
        x = (k + 0.5) * 10.3 - 0.5
        y = d.refined.centers[np.isclose(d.refined.centers[:, 0], x), 1]
        np.testing.assert_allclose(y, [11] if k == 3 else [11, 31, 51])
    assert d.detected.left_out["not crossing or joined"] >= 1


def test_a_dark_speck_in_a_white_band_leaves_its_two_regions_without_that_neighbour():
    image = np.full((60, 80), 60, np.uint8)
    image[[10, 50]] = 225
    image[28:33] = 225
    image[30, 5] = 60  # in the first of the ten slices
    _, _, edges = find_bands(image, replace(SETTINGS, refinement="none"))
    edges = edges.reshape(10, 2, 4)  # per slice its two complete regions, from the top down: top, bottom, above, below
    np.testing.assert_allclose(edges[1:, 0], np.tile([10.5, 27.5, 9.5, 32.5], (9, 1)))  # across the cut ones too
    np.testing.assert_allclose(edges[1:, 1], np.tile([32.5, 49.5, 27.5, 50.5], (9, 1)))
    np.testing.assert_allclose(edges[0, 0, :3], [10.5, 27.5, 9.5])
    assert np.isnan(edges[0, 0, 3]) and np.isnan(edges[0, 1, 2])


def test_crossed_refinement_keeps_the_detected_edges_at_that_column(monkeypatch):
    image = np.full((45, 80), 60, np.uint8)
    image[[5, 20, 35]] = 235

    def crossed(image, lines, *args):
        upper = lines.upper.copy() - 0.25
        upper[:, 3] = lines.lower[:, 3] + 1
        return replace(lines, upper=upper)

    monkeypatch.setattr(horizontal_ls, "refine_line_edges", crossed)
    found, measured, _ = find_bands(image, SETTINGS)
    assert len(found.contours) == len(measured.contours) == 20
    for a, b in zip(found.contours, measured.contours):
        assert a[4, 1] == b[4, 1]  # column 3, after the slice's left cut
        assert b[3, 1] == a[3, 1] + 0.25


def test_design_is_clipped_before_checking_for_complete_bands():
    layout = kdb.Layout()
    layout.dbu = 0.001
    shapes = layout.create_cell("TOP").shapes(layout.layer(4, 0))
    shapes.insert(kdb.Box(-50, -20, 50, -10))
    shapes.insert(kdb.Box(-50, 0, 10, 10))  # line and adjacent space both end within the field
    outlines, drawn = horizontal_band_outlines(layout, 4, 0, (100, 100), 10)
    xs = [(c[:, 0].min(), c[:, 0].max()) for c in outlines]
    for left in range(-50, 50, 10):
        assert xs.count((left, left + 10)) == (3 if left < 10 else 1)
    # Drawn bands: the line in every slice, the box ending at x = 10 in the first six; clear: the space between.
    assert drawn.tolist() == [True] * 16 + [False] * 6


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
                                 records, log=messages.append, keep_tiles=True, pool=pool, detailed_log=True)
            assert any("T0 L/S centres per x slice" in line and "design [" in line for line in messages)
        else:
            window = make_window(tmp_path)
            monkeypatch.setattr(release_app, "image_records", lambda folder: records)
            monkeypatch.setattr(window, "settings", lambda: SETTINGS)
            monkeypatch.setattr(window, "kept_helpers", lambda: pool)
            window.open_folder(str(tmp_path))
            wait(window)
            run = window.result
        for stitch, known in ((run.stitch, sem_offsets), (run.design_stitch, design_offsets)):
            assert len(stitch.pairs) == 20  # includes side and corner overlaps on staggered slice grids
            np.testing.assert_array_equal(stitch.corrections[:, 0], 0)
            np.testing.assert_allclose(stitch.corrections[:, 1], known - known.mean(), atol=1e-8)
            for pair in stitch.pairs:
                assert len(np.unique(pair.ia)) == len(np.unique(pair.ib)) == len(pair.ia)
        assert all(len(np.unique(p[:, 0].round(5))) == 10 for p in run.sem_points + run.design_points)
        np.testing.assert_array_equal(run.sem_nm[:, 0], run.design_nm[:, 0])
        height_error = run.sem_nm[:, 1] - run.design_nm[:, 1]
        np.testing.assert_allclose(np.unique(height_error.round(6)), [2, 3])
        assert len(run.moving.counts) > 1
        assert np.isnan(run.moving.residuals[:, 0]).all()
        assert 0.1 < np.abs(run.moving.residuals[:, 1]).max() < 1  # y offset removed, local errors retained
        np.testing.assert_array_equal(tile_affines(run)[:, 0], np.tile([1, 0, 0], (9, 1)))
        assert len(np.unique(run.moving.stripe)) == (3 if per_stripe else 0)
        assert review(run)["measured_axes"] == ["y"]
        cd = run.line_space_cd  # neighbouring clips draw opposite tones: a band seen by both is neither
        unknown = cd.band == horizontal_ls.NOT_KNOWN
        assert len(cd.band) == len(run.design_nm) and unknown.any() and not unknown.all()
        assert np.isnan(cd.design[unknown]).all() and np.isfinite(cd.design[~unknown]).all()

        if not per_stripe:
            reopened, _ = load_cache(tmp_path, window.cache_dir)
            np.testing.assert_equal(reopened.moving.residuals, run.moving.residuals)
            c = read_contours(reopened, log=messages.append, pool=pool)
            assert not any("differ from the centres" in line for line in messages)
            np.testing.assert_array_equal(c.refined[c.start[:-1]], c.refined[c.start[1:] - 2])
            assert all(mask.min() == 0 and mask.max() == 1 for mask in read_design_images(reopened))
            stitching = window.stitching_view
            stitching.set_contours(c)
            stitching.set_images(read_display_images(run.image_paths))
            stitching.placement.setCurrentText("Stitched + affine")
            for affine, item in zip(tile_affines(run), stitching.image_items):
                np.testing.assert_allclose(matrix_of(item.transform())[:2, :2], affine[:2, :2] @ np.diag([1, 2]))
            view = window.registration_view
            assert "X is unmeasured" in view.summary.text()
            assert [panel.name for panel, _, _ in view.maps] == ["dy"]
            view.values.setCurrentIndex(1)  # grid average and its exports also contain only measured errors
            assert "dx" not in view.profiles_text(",") and "mean dy" in view.profiles_text(",")
            old_points = [np.column_stack((np.full(len(p) // 10, center[0]), p[:len(p) // 10, 1]))
                          for p, center in zip(run.sem_points, run.centers)]
            old = read_contours(replace(reopened, analysis=9, sem_points=old_points), log=messages.append, pool=pool)
            assert len(old.refined_centres) * 10 == len(c.refined_centres)
            np.testing.assert_allclose(old.refined_centres[:, 0], 0, atol=1e-8)
            assert not any("differ from the centres" in line for line in messages)
            window.close()


def test_tilted_rows_in_a_narrow_overlap_match_at_a_common_x():
    centers = np.array([[0., 0.], [96., 0.]])
    sizes = np.full((2, 2), 100.)
    points = []
    for cx, dy in ((0, 0), (96, 4)):
        points.append(np.array([[x, y + 0.2 * x + dy] for x in cx + np.arange(-45, 50, 10)
                                for y in (-30, -10, 10, 30)]))
    slopes = [np.full(len(p), 0.2) for p in points]
    stitch = stitch_bands(points, slopes, centers, sizes, 5, SETTINGS, "SEM")
    p, = stitch.pairs
    assert np.all(points[0][p.ia, 0] == 45) and np.all(points[1][p.ib, 0] == 51)
    np.testing.assert_allclose(p.fit.shift, [0, 4])  # comparing the two raw heights here would give 5.2
    assert len(p.ia) == 4


def test_design_pairs_keep_their_slice_when_other_columns_have_the_same_heights():
    sem = [np.array([[-45., 2.], [-45., 22.], [-25., 42.]])]
    design = [np.array([[-45., 0.], [-45., 20.], [-15., 40.]])]
    zero = np.zeros((1, 2))
    errors = paired_rows(sem, design, [np.zeros(3)], zero, zero, zero, np.full((1, 2), 100.), [], SETTINGS)
    np.testing.assert_array_equal(errors.sem_index, [0, 1])
    np.testing.assert_array_equal(errors.design_nm[:, 0], [-45, -45])
    np.testing.assert_allclose(errors.error_nm, [[0, 2], [0, 2]])


def test_fitting_partial_slices_keeps_only_observable_affine_terms():
    # Stage scatter within one sampled column must not turn a small height error into a large tilt.
    x = np.tile([0, 0.001], 5)
    design = np.column_stack((x, np.repeat(np.arange(5) * 15., 2)))
    sem = design.copy()
    sem[:, 1] += 1 + 200 * x
    fit = moving_window_affine(sem, design, 200, 100, y_only=True, min_span_nm=(5, 2))
    assert fit.models[0, 1, 0] == 0
    np.testing.assert_allclose(fit.models[0, 1, 1], np.polyfit(sem[:, 1], design[:, 1], 1)[0])
    # Ten samples of one slightly sloping design row constrain tilt, but cannot constrain y scale.
    x = np.arange(10) * 12.
    design = np.column_stack((x, 0.05 * x))
    sem = np.column_stack((x, 8 + 0.15 * x + np.tile([0.1, -0.1], 5)))
    fit = moving_window_affine(sem, design, 200, 100, y_only=True, min_span_nm=(5, 2))
    assert fit.models[0, 1, 1] == 1
    assert fit.models[0, 1, 0] == pytest.approx(-0.1, abs=0.001)


def test_one_tilted_sem_tile_calibrates_from_its_ten_columns(tmp_path):
    records, _, _ = mosaic(tmp_path)
    tile = records[0]
    tile.fov_x_nm = tile.fov_y_nm = 120
    image = np.full((120, 120), 55, np.uint8)
    x = np.arange(120)
    edges = [15, 35, 55, 75, 95]
    for row in edges:
        image[row + x // 12 - 5, x] = 225
    cv2.imencode(".png", image)[1].tofile(str(tile.image_path))
    layout = kdb.Layout()
    layout.dbu = 0.0005
    shapes = layout.create_cell("TOP").shapes(layout.layer(4, 0))
    for top, bottom in zip(edges[::2], edges[1::2]):
        shapes.insert(kdb.Box(-120, 119 - 2 * bottom, 120, 119 - 2 * top))
    shapes.insert(kdb.Box(-120, -120, 120, 119 - 2 * edges[-1]))
    layout.write(str(tile.oas_path))
    s = replace(SETTINGS, refinement="none", design_tolerance_nm=7)
    with ThreadPoolExecutor(1) as pool:
        run = analyse_folder(tmp_path, "images", s, [tile], log=lambda line: None, pool=pool)
    assert len(run.design_nm) == 40
    np.testing.assert_allclose(tile_affines(run)[0, :2, :2], [[1, 0], [1 / 12, 1]], atol=1e-9)
    np.testing.assert_allclose(run.moving.residuals[:, 1], 0, atol=1e-8)


def line_space_tile(tmp_path):
    """One 120 x 120 px tile, pixels 1 x 2 nm, 1 px white rows between interiors of the same tone: lines 19 px (38 nm)
    wide, spaces 9 px (18 nm). Its clip draws the spaces at their interiors, so each design line also holds its two
    white rows (42 nm). The interiors at the top and bottom of the field are cut by its border."""
    records, _, _ = mosaic(tmp_path)
    tile = records[0]
    tile.fov_x_nm, tile.fov_y_nm = 120, 240
    image = np.full((120, 120), 60, np.uint8)
    image[[10, 30, 40, 60, 70, 90, 100]] = 225
    cv2.imencode(".png", image)[1].tofile(str(tile.image_path))
    layout = kdb.Layout()
    layout.dbu = 0.001
    shapes = layout.create_cell("TOP").shapes(layout.layer(4, 0))
    for bottom, top in ((100, 120), (40, 58), (-20, -2), (-80, -62)):  # nm, y up from the field's centre
        shapes.insert(kdb.Box(-70, bottom, 70, top))
    layout.write(str(tile.oas_path))
    return tile


def test_line_and_space_cds_follow_the_design_tone_drawn_as_spaces(tmp_path):
    tile = line_space_tile(tmp_path)
    with ThreadPoolExecutor(1) as pool:
        run = analyse_folder(tmp_path, "images", replace(SETTINGS, design_tolerance_nm=7), [tile],
                             log=lambda line: None, pool=pool)
    cd = run.line_space_cd
    assert len(cd.band) == 60 and (cd.band == horizontal_ls.LINE).sum() == (cd.band == horizontal_ls.SPACE).sum() == 30
    line, space = cd.band == horizontal_ls.LINE, cd.band == horizontal_ls.SPACE
    # Own outline, band middle, space side. The top line takes its space-side edge from the space cut by the field's
    # top, the bottom space its band middle from the line cut by its bottom.
    np.testing.assert_allclose(cd.sem[line], np.tile([38, 40, 42], (30, 1)), atol=1e-9)
    np.testing.assert_allclose(cd.sem[space], np.tile([18, 20, 18], (30, 1)), atol=1e-9)
    np.testing.assert_allclose(cd.design[line], 42, atol=1e-9)
    np.testing.assert_allclose(cd.design[space], 18, atol=1e-9)
    np.testing.assert_allclose(np.unique(run.design_nm[line, 1] - tile.center_y_nm), [-41, 19, 79])  # 3 lines

    save_cache(run, tmp_path / "user")
    reopened, _ = load_cache(tmp_path, tmp_path / "user")
    for name in ("sem", "design", "band"):
        np.testing.assert_array_equal(getattr(reopened.line_space_cd, name), getattr(cd, name))


def test_the_l_s_cd_tab_shows_lines_and_spaces_or_one_of_them(tmp_path, monkeypatch):
    tile = line_space_tile(tmp_path)
    window = make_window(tmp_path)
    window.size_action.setChecked(True)
    monkeypatch.setattr(release_app, "image_records", lambda folder: [tile])
    monkeypatch.setattr(window, "settings", lambda: replace(SETTINGS, design_tolerance_nm=7))
    with ThreadPoolExecutor(1) as pool:
        monkeypatch.setattr(window, "kept_helpers", lambda: pool)
        window.open_folder(str(tmp_path))
        wait(window)
    tabs, tab = window.centralWidget(), window.size_tab
    assert tabs.tabText(2) == "L/S CD" and tab.run_button.isHidden() and tab.roi.isHidden()
    view = tab.view
    assert isinstance(view, LineSpaceCDView) and tab.note.text() == ""
    assert view.shown().all() and "Line and space CD, SEM (nm)" in view.plot.titleLabel.text
    assert "30 on lines and 30 on spaces" in view.summary.text()
    assert "Line CD (nm): SEM mean 38.00" in view.summary.text()
    view.show_band.setCurrentText("Spaces")
    np.testing.assert_array_equal(view.shown(), window.result.line_space_cd.band == horizontal_ls.SPACE)
    assert "Space CD, SEM (nm)" in view.plot.titleLabel.text
    view.difference_button.setChecked(True)
    view.show_band.setCurrentText("Lines")
    np.testing.assert_allclose(view.values()[view.shown()], -4)
    assert "Line CD, SEM − design (nm), own outline" in view.plot.titleLabel.text
    view.edge.setCurrentText("Space side")
    np.testing.assert_allclose(view.values()[view.shown()], 0, atol=1e-9)
    assert "space side" in view.plot.titleLabel.text and "Line CD (nm): SEM mean 42.00" in view.summary.text()
    view.edge.setCurrentText("Band middle")
    np.testing.assert_allclose(view.values()[view.shown()], -2, atol=1e-9)
    view.draw_map()
    assert (view.image.image[..., :3] == 255).all(axis=-1).any()  # white between the centres
    window.close()

    saved = replace(window.result, line_space_cd=None)  # a run saved before D182
    assert ContactSizeTab(saved, use_opengl=False).note.text() == NO_LS_CD


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
    records[1].oas_path = records[2].oas_path = None  # an entire stripe has no design measurements
    with ThreadPoolExecutor(2) as pool, pytest.warns(UserWarning):
        run = analyse_folder(tmp_path, "images", replace(SETTINGS, moving_window_per_stripe=True),
                             records, log=lambda text: None, pool=pool)
    np.testing.assert_array_equal(run.design_stitch.unplaced, [0, 1, 2])
    np.testing.assert_array_equal(tile_affines(run)[:3], np.tile(np.eye(3), (3, 1, 1)))
    assert not len(run.stitch.unplaced) and np.isfinite(run.moving.residuals[:, 1]).all()
    masks = read_design_images(run)
    assert not masks[0].any() and masks[1] is None
