"""End-to-end analysis of synthetic folders (analysis.py) and the window of its result, without a display."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest

from affine_ransac import analysis
from affine_ransac.analysis import (Settings, analyse_contour_folder, analyse_images, check_image_records,
                                    check_settings, measure_contact_sizes, restitch, settings_for)
from affine_ransac.pipeline import residual_summary
from affine_ransac.view_analysis import analysis_window
from affine_ransac.view_contact_size import ContactSizeTab
from affine_ransac.view_points import PointsStitchView
from affine_ransac.view_stitch import StitchViewer

# Contour CSVs: 2 stripes x 3 tiles of 2.88 um, 2.6 um apart (~280 nm overlaps); contact pitch 136 nm.
FOV, PITCH = 2880.0, 136.0
NOMINAL = np.array([(x, y) for x in (0.0, 2600.0) for y in (0.0, 2600.0, 5200.0)])
STAGE = np.array([[1.2, -0.7], [-3.1, 2.4], [0.5, 1.5], [2.0, -2.0], [-1.0, 0.8], [0.3, -1.6]])  # SEM tile errors, nm
FILES = np.array([[0.0, 0.0], [4.0, -3.0], [-2.0, 5.0], [1.0, 1.0], [-3.0, 2.0], [2.5, -0.5]])   # design file offsets
BOTH = "in-image + stripe drift corrected, translation + rotation"  # the set placed with translation + rotation
SETS = ["uncorrected", "in-image corrected", "drift corrected", "stripe drift corrected", BOTH]
TABS = ["Stitching", "RANSAC monitor", "Registration, RANSAC", "Registration, moving window", "Moving-window tuner",
        "Row pitch", "In-image corrected", "Drift corrected", "Stripe drift corrected", BOTH[0].upper() + BOTH[1:],
        "Stripe drift curves", "Stitching residuals", "Contact size"]


def write_contour_folder(folder, local_columns=True, noise_nm=0.1, seed=0, sem_error=None):
    """One CSV per tile (real column names): the contacts inside its FOV, design as stored in its file,
    SEM as measured (the tile sits at nominal + STAGE[k]); DesignX, DesignY relative to the image centre.
    sem_error(k, local_nm, mask_nm): optional (N, 2) nm added to tile k's SEM centres, from the contacts'
    positions relative to the image centre and on the mask. Plus a summary table of the measuring tool,
    which must be left out."""
    rng = np.random.default_rng(seed)
    lattice = np.array([(x, y) for x in np.arange(-1500, 4200, PITCH) for y in np.arange(-1500, 6800, PITCH)])
    for k, nominal in enumerate(NOMINAL):
        contacts = lattice[np.all(np.abs(lattice - nominal) < FOV / 2 - 30, axis=1)]
        design = contacts - FILES[k]
        sem = contacts - STAGE[k] + rng.normal(0, noise_nm, contacts.shape)
        if sem_error is not None:
            sem += sem_error(k, contacts - nominal, contacts)
        header = "DesignX,DesignY,DesignX_Add,DesignY_Add,SEMX_Add,SEMY_Add" if local_columns else \
            "DesignX_Add,DesignY_Add,SEMX_Add,SEMY_Add"
        rows = [",".join(f"{float(v)!r}" for v in ((*(d - nominal), *d, *m) if local_columns else (*d, *m)))
                for d, m in zip(design, sem)]
        (folder / f"CD{k:06d}.csv").write_text("\n".join([header] + rows) + "\n", encoding="utf-8")
    (folder / "AffineCoefficients_Summary.csv").write_text("FileName,DA,DB\nall,1,2\n", encoding="utf-8")


def test_contour_folder_runs_the_notebook_flow(tmp_path):
    write_contour_folder(tmp_path)
    lines = []
    with pytest.warns(UserWarning, match="left out: .'AffineCoefficients_Summary.csv'"):
        result = analyse_contour_folder(tmp_path, Settings(), log=lines.append)

    assert result.kind == "contour CSV" and result.tile_ids == [f"CD{k:06d}" for k in range(6)]
    assert list(result.sets) == SETS
    assert list(result.stitchings) == ["raw", "in-image corrected", "stripe drift corrected",
                                       "in-image + stripe drift corrected"]
    np.testing.assert_allclose(result.stitchings["raw"].stitch.corrections, STAGE - STAGE.mean(axis=0), atol=0.1)
    np.testing.assert_allclose(result.design_stitch.corrections, FILES - FILES.mean(axis=0), atol=1e-6)
    main = result.sets["uncorrected"]
    assert main.ransac.inliers.mean() > 0.9 and np.abs(main.ransac.residuals).max() < 0.5  # only noise left
    assert result.sets["drift corrected"].moving is None and main.moving is not None
    assert sorted(result.stripe_drift.stripes) == [0, 1]
    assert result.stitchings["raw"].rigid.shape == (6, 3, 3)
    assert lines == result.log and lines[0] == "6 of 6 files used, " + lines[0].split(", ", 1)[1]
    assert any(line.startswith("RANSAC, stripe drift corrected") for line in lines)
    assert result.sets[BOTH].moving is not None and any(line.startswith(f"RANSAC, {BOTH}") for line in lines)

    pg.mkQApp()
    window = analysis_window(result, use_opengl=False)
    assert [window.tabText(i) for i in range(window.count())] == TABS
    assert isinstance(window.widget(0).view, PointsStitchView)
    assert ("drift correction applied", "dx") in window.extra_views["drift corrected"].line_boxes
    assert result.contact_sizes is None and window.widget(window.count() - 1).text().startswith("N/A")
    window.close()


def test_without_image_centres_the_in_image_correction_is_left_out(tmp_path):
    write_contour_folder(tmp_path, local_columns=False)
    lines = []
    with pytest.warns(UserWarning, match="AffineCoefficients_Summary.csv"):
        result = analyse_contour_folder(tmp_path, Settings(), log=lines.append)
    assert list(result.sets) == ["uncorrected", "drift corrected", "stripe drift corrected"]
    assert list(result.stitchings) == ["raw", "stripe drift corrected"]
    assert any("in-image correction is left out" in line for line in lines)
    pg.mkQApp()
    window = analysis_window(result, use_opengl=False)
    assert "In-image corrected" not in [window.tabText(i) for i in range(window.count())]
    window.close()


def test_in_image_and_stripe_drift_corrections_together_leave_only_the_noise_in_the_overlaps(tmp_path):
    """A distortion shared by all images and a drift along y that differs between the two stripes (both
    exaggerated): each correction removes its own part of the tie residuals, both together nearly all."""
    def sem_error(k, local, mask):
        twist = 1.2e-6 * local[:, 0] * local[:, 1]  # up to 2 nm at the image corners; not affine: in the map
        drift = (1 if k < 3 else -1) * 1e-3 * (mask[:, 1] - 2600)  # dy, opposite in the two stripes
        return np.column_stack([twist, twist + drift])

    write_contour_folder(tmp_path, sem_error=sem_error)
    with pytest.warns(UserWarning):  # the summary table left out; few RANSAC inliers with these errors
        result = analyse_contour_folder(tmp_path, Settings(), log=lambda line: None)
    rms = {name: residual_summary(st.points, st.stitch, st.stitch.corrections, result.centers)["all overlaps RMS (nm)"]
           for name, st in result.stitchings.items()}
    both = rms["in-image + stripe drift corrected"]
    assert both < 0.3 and both < 0.25 * min(rms["in-image corrected"], rms["stripe drift corrected"]), rms
    assert max(rms["in-image corrected"], rms["stripe drift corrected"]) < rms["raw"], rms


def test_the_translation_and_rotation_set_removes_the_rotation_of_each_tile(tmp_path):
    """Every tile sees the mask rotated by its own angle (about 1 mrad): the sets placed with
    translation keep each tile's rotation in their registration error, the set placed with
    translation + rotation removes it (the mean rotation goes into the global affine)."""
    angles = np.random.default_rng(3).normal(0, 1e-3, len(NOMINAL))

    def sem_error(k, local, mask):
        return angles[k] * np.column_stack([-local[:, 1], local[:, 0]])

    write_contour_folder(tmp_path, sem_error=sem_error)
    with pytest.warns(UserWarning):  # the summary table left out; few RANSAC inliers with these errors
        result = analyse_contour_folder(tmp_path, Settings(), log=lambda line: None)
    rms = {name: np.sqrt((c.ransac.residuals ** 2).mean()) for name, c in result.sets.items()}
    assert rms[BOTH] < 0.15 and rms[BOTH] < 0.25 * rms["stripe drift corrected"], rms
    assert np.sqrt((result.sets[BOTH].moving.residuals ** 2).mean()) < 0.15


def test_settings_for_each_kind_of_folder():
    assert "tile_margin_nm" in settings_for("contour CSV") and "detection" not in settings_for("contour CSV")
    assert "detection" in settings_for("images") and "tile_margin_nm" not in settings_for("images")
    assert "max_match_nm" in settings_for("contour CSV") and "max_match_nm" in settings_for("images")


def write_image_folder(folder, design=None):
    """2 x 2 synthetic SEM tiles with their .oas files (test_pipeline), and their records as the metadata reader
    returns them. design(k, nominal): optional tile-local design centres of tile k instead of the lattice."""
    from test_pipeline import ERRORS, FOV as IMAGE_FOV, LATTICE, NOMINAL as IMAGE_NOMINAL, visible, write_design, write_tile
    records = []
    for k, (nominal, error) in enumerate(zip(IMAGE_NOMINAL, ERRORS)):
        write_tile(folder / f"{k}.jpg", nominal + error, seed=k)
        centres = visible(LATTICE - nominal, margin=0) if design is None else design(k, nominal)
        write_design(folder / f"{k}.oas", centres)
        records.append(SimpleNamespace(tile_id=f"T{k}", image_path=folder / f"{k}.jpg", oas_path=folder / f"{k}.oas",
                                       center_x_nm=nominal[0], center_y_nm=nominal[1], fov_x_nm=IMAGE_FOV,
                                       fov_y_nm=IMAGE_FOV))
    return records


IMAGE_SETTINGS = Settings(layer=1, datatype=0, detection="otsu")


def test_image_folder_runs_the_notebook_flow(tmp_path):
    from test_pipeline import ERRORS
    records = write_image_folder(tmp_path)
    lines = []
    result = analyse_images(tmp_path, records, IMAGE_SETTINGS, log=lines.append)

    assert result.kind == "images" and result.tile_ids == ["T0", "T1", "T2", "T3"]
    np.testing.assert_allclose(result.stitchings["raw"].stitch.corrections, ERRORS - ERRORS.mean(axis=0), atol=0.3)
    assert list(result.sets) == SETS
    assert list(result.stitchings) == ["raw", "in-image corrected", "stripe drift corrected",
                                       "in-image + stripe drift corrected"]
    assert len(result.images.tiles) == 4 and set(result.images.merged) == {"nominal", "mean", "first"}
    assert lines[0] == "Sample tiles matched to their design: 3 of 3"
    assert "4 of 4 tiles loaded and detected" in lines
    assert result.contact_sizes is None  # measured only on request

    pg.mkQApp()
    window = analysis_window(result, use_opengl=False)
    assert isinstance(window.widget(0).view, StitchViewer)
    assert isinstance(window.widget(window.count() - 1), ContactSizeTab)
    result.images = None  # e.g. a saved result: the points view instead
    other = analysis_window(result, use_opengl=False)
    assert isinstance(other.widget(0).view, PointsStitchView)
    window.close()
    other.close()


def test_restitch_computes_the_stitching_and_every_result_after_it_again(tmp_path):
    folder = tmp_path / "csv"
    folder.mkdir()
    write_contour_folder(folder)
    first = analyse_contour_folder(folder, Settings(), log=lambda line: None)
    lines = []
    second = restitch(first, "measurement order first", log=lines.append)
    assert second.settings.stitch_method == "measurement order first" and first.settings.stitch_method == "all overlaps"
    assert second.log[:len(first.log)] == first.log and lines[0].startswith("=== Stitching method: measurement")
    assert list(second.sets) == list(first.sets)
    # No drift here: both methods find the same stage errors.
    np.testing.assert_allclose(second.stitchings["raw"].stitch.corrections, first.stitchings["raw"].stitch.corrections,
                               atol=0.05)
    with pytest.raises(ValueError, match="must be one of"):
        restitch(first, "fastest")

    images = analyse_images(tmp_path, write_image_folder(tmp_path), IMAGE_SETTINGS, log=lambda line: None)
    images.contact_sizes = measure_contact_sizes(images)
    again = restitch(images, "measurement order first", log=lambda line: None)
    assert again.images.tiles is images.images.tiles and again.contact_sizes is None  # tiles reused, sizes measured anew
    images.images = None
    with pytest.raises(ValueError, match="needs its tiles in memory"):
        restitch(images, "all overlaps")


def test_contact_sizes_are_measured_on_request(tmp_path):
    result = analyse_images(tmp_path, write_image_folder(tmp_path), IMAGE_SETTINGS, log=lambda line: None)
    steps = []

    sizes = measure_contact_sizes(result, roi_width_nm=0.0, progress=lambda done, total: steps.append((done, total)))

    # One row per contact of the uncorrected set: SEM discs of 40 nm, design squares of 40 nm.
    assert steps == [(1, 4), (2, 4), (3, 4), (4, 4)] and sizes.roi_width_nm == 0.0
    assert sizes.sem.shape == sizes.design.shape == (len(result.sets["uncorrected"].design_nm), 3)
    np.testing.assert_allclose(sizes.design, np.tile([1600.0, 40.0, 40.0], (len(sizes.design), 1)))
    np.testing.assert_allclose(np.median(sizes.sem, axis=0), [np.pi * 20 ** 2, 40, 40], rtol=0.1)

    pg.mkQApp()
    tab = ContactSizeTab(result, use_opengl=False)  # the window's tab: Run measurement in the background
    assert tab.view is None
    tab.roi.setValue(5.0)
    tab.run()
    deadline = time.time() + 60
    while tab.view is None and time.time() < deadline:
        pg.QtWidgets.QApplication.processEvents()
        time.sleep(0.01)
    assert tab.view is not None and result.contact_sizes.roi_width_nm == 5.0
    assert tab.status.text().startswith(f"Measured {len(sizes.sem)} contacts")

    # Behind the map: each tile's SEM image placed like its contacts; the map transparent between them.
    view = tab.view
    assert view.sem_box.isEnabled() and view.design_box.isEnabled() and not view.sem_box.isChecked()
    view.sem_box.setChecked(True)
    view.design_box.setChecked(True)
    images = view.background_items[view.sem_box]
    assert len(images) == 3 and all(len(items) == 4 for items in images)  # per map, one per tile
    assert len(view.background_items[view.design_box][0]) == 4
    tiles = [tile for tile, _ in view.backgrounds[view.sem_box][0]]
    corner = images[0][0].mapToParent(pg.QtCore.QPointF(*(tiles[0].refined.centers[0] + 0.5)))
    distance_nm = np.linalg.norm(view.position_um - [corner.x(), corner.y()], axis=1).min() * 1000
    assert distance_nm < 2  # an SEM contact sits on its dot of the map
    view.draw_maps()
    assert view.maps[0][2].image.shape[-1] == 4  # RGBA
    view.sem_box.setChecked(False)
    assert not images[0][0].isVisible()
    tab.close()

    result.images = None  # a reopened result: no images to measure on
    with pytest.raises(ValueError, match="analyse the image folder again"):
        measure_contact_sizes(result)
    reopened = ContactSizeTab(result, use_opengl=False)
    assert not reopened.run_button.isEnabled()
    assert not reopened.view.sem_box.isEnabled() and not reopened.view.design_box.isEnabled()


def test_invalid_settings_fail_before_any_work(tmp_path):
    with pytest.raises(ValueError, match=r"max_match_nm must be > 0.*detection must be one of"):
        analyse_images(tmp_path, [], Settings(max_match_nm=0.0, detection="otsu4"))
    with pytest.raises(ValueError, match="min_match_fraction must be in"):
        check_settings(Settings(min_match_fraction=1.5), "images")
    check_settings(Settings(min_match_fraction=1.5), "contour CSV")  # not used for contour CSVs
    with pytest.raises(ValueError, match="tile_margin_nm must be >= 0"):
        analyse_contour_folder(tmp_path, Settings(tile_margin_nm=-1.0))


def test_a_folder_that_cannot_be_analysed_fails_before_loading_the_tiles(tmp_path):
    records = write_image_folder(tmp_path)
    with pytest.raises(ValueError, match="lists no images"):
        check_image_records([], (1, 0))
    with pytest.raises(ValueError, match="No image has a paired .oas"):
        check_image_records([SimpleNamespace(**{**vars(r), "oas_path": None}) for r in records], (1, 0))
    with pytest.raises(ValueError, match="1 image file.s. not found"):
        check_image_records([SimpleNamespace(**{**vars(records[0]), "image_path": tmp_path / "x.jpg"})] + records[1:],
                            (1, 0))
    with pytest.raises(ValueError, match=r"Layer 5/0 is not in 0.oas; its layers: \[\(1, 0\)\]"):
        analyse_images(tmp_path, records, Settings(layer=5, datatype=0))


def test_tiles_loaded_in_parallel_equal_tiles_loaded_one_by_one(tmp_path):
    records = write_image_folder(tmp_path)
    s = IMAGE_SETTINGS
    tile_ids, tiles = analysis.load_image_tiles(records, s, log=lambda line: None)
    assert tile_ids == [r.tile_id for r in records]
    for r, tile in zip(records, tiles):
        alone = analysis.process_tile(r.image_path, r.oas_path, (r.center_x_nm, r.center_y_nm),
                                      (r.fov_x_nm, r.fov_y_nm), (s.layer, s.datatype), s.search_px,
                                      s.design_search_nm, s.min_match_fraction, s.detection, s.design_tolerance_nm)
        np.testing.assert_array_equal(tile.sem_points_nm(), alone.sem_points_nm())
        np.testing.assert_array_equal(tile.design_points_nm(), alone.design_points_nm())
        assert tile.design.ok == alone.design.ok


def test_worker_count_is_half_the_logical_cores_at_least_one(monkeypatch):
    monkeypatch.setattr(analysis.os, "cpu_count", lambda: 16)
    assert analysis.worker_count() == 8
    monkeypatch.setattr(analysis.os, "cpu_count", lambda: 1)
    assert analysis.worker_count() == 1
    monkeypatch.setattr(analysis.os, "cpu_count", lambda: None)
    assert analysis.worker_count() == 1


def test_no_design_match_in_the_sample_tiles_stops_after_them(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    records = write_image_folder(tmp_path, design=lambda k, nominal: rng.uniform(-300, 300, (60, 2)))  # no lattice
    loaded = []
    original = analysis.process_tile
    monkeypatch.setattr(analysis, "process_tile", lambda *args, **kwargs: loaded.append(1) or original(*args, **kwargs))

    with pytest.raises(ValueError, match="No design tone matched the SEM in the sample tiles"):
        analyse_images(tmp_path, records, IMAGE_SETTINGS)
    assert len(loaded) == 3  # the sample tiles only, not the whole folder


def test_tile_loading_logs_its_timing(tmp_path):
    records = write_image_folder(tmp_path)
    lines = []
    _, tiles = analysis.load_image_tiles(records, IMAGE_SETTINGS, log=lines.append)
    assert set(tiles[0].seconds) == {"read", "detect", "refine", "oas", "contacts", "shift", "reversed"}
    timing = [line for line in lines if line.startswith("Timing:")]
    assert len(timing) == 1 and "shift" in timing[0] and "reversed tone was tried" in timing[0]
