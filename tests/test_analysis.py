"""End-to-end analysis of synthetic folders (analysis.py) and the window of its result, without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest

from affine_ransac.analysis import Settings, analyse_contour_folder, analyse_images, settings_for
from affine_ransac.pipeline import residual_summary
from affine_ransac.view_analysis import analysis_window
from affine_ransac.view_points import PointsStitchView
from affine_ransac.view_stitch import StitchViewer

# Contour CSVs: 2 stripes x 3 tiles of 2.88 um, 2.6 um apart (~280 nm overlaps); contact pitch 136 nm.
FOV, PITCH = 2880.0, 136.0
NOMINAL = np.array([(x, y) for x in (0.0, 2600.0) for y in (0.0, 2600.0, 5200.0)])
STAGE = np.array([[1.2, -0.7], [-3.1, 2.4], [0.5, 1.5], [2.0, -2.0], [-1.0, 0.8], [0.3, -1.6]])  # SEM tile errors, nm
FILES = np.array([[0.0, 0.0], [4.0, -3.0], [-2.0, 5.0], [1.0, 1.0], [-3.0, 2.0], [2.5, -0.5]])   # design file offsets
TABS = ["Stitching", "RANSAC monitor", "Registration, RANSAC", "Registration, moving window", "Moving-window tuner",
        "Row pitch", "In-image corrected", "Drift corrected", "Stripe drift corrected", "Stripe drift curves",
        "Stitching residuals"]


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
    assert list(result.sets) == ["uncorrected", "in-image corrected", "drift corrected", "stripe drift corrected"]
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

    pg.mkQApp()
    window = analysis_window(result, use_opengl=False)
    assert [window.tabText(i) for i in range(window.count())] == TABS
    assert isinstance(window.widget(0), PointsStitchView)
    assert ("drift correction applied", "dx") in window.extra_views["drift corrected"].line_boxes
    window.close()


def test_without_image_centres_the_in_image_correction_is_left_out(tmp_path):
    write_contour_folder(tmp_path, local_columns=False)
    lines = []
    with pytest.warns(UserWarning, match="AffineCoefficients_Summary.csv"):
        result = analyse_contour_folder(tmp_path, Settings(), log=lines.append)
    assert "in-image corrected" not in result.sets
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


def test_settings_for_each_kind_of_folder():
    assert "tile_margin_nm" in settings_for("contour CSV") and "detection" not in settings_for("contour CSV")
    assert "detection" in settings_for("images") and "tile_margin_nm" not in settings_for("images")
    assert "max_match_nm" in settings_for("contour CSV") and "max_match_nm" in settings_for("images")


def test_image_folder_runs_the_notebook_flow(tmp_path):
    from test_pipeline import ERRORS, FOV as IMAGE_FOV, LATTICE, NOMINAL as IMAGE_NOMINAL, visible, write_design, write_tile
    records = []
    for k, (nominal, error) in enumerate(zip(IMAGE_NOMINAL, ERRORS)):
        write_tile(tmp_path / f"{k}.jpg", nominal + error, seed=k)
        write_design(tmp_path / f"{k}.oas", visible(LATTICE - nominal, margin=0))
        records.append(SimpleNamespace(tile_id=f"T{k}", image_path=tmp_path / f"{k}.jpg", oas_path=tmp_path / f"{k}.oas",
                                       center_x_nm=nominal[0], center_y_nm=nominal[1], fov_x_nm=IMAGE_FOV,
                                       fov_y_nm=IMAGE_FOV))
    lines = []
    result = analyse_images(tmp_path, records, Settings(layer=1, datatype=0, detection="otsu"), log=lines.append)

    assert result.kind == "images" and result.tile_ids == ["T0", "T1", "T2", "T3"]
    np.testing.assert_allclose(result.stitchings["raw"].stitch.corrections, ERRORS - ERRORS.mean(axis=0), atol=0.3)
    assert list(result.sets) == ["uncorrected", "in-image corrected", "drift corrected", "stripe drift corrected"]
    assert list(result.stitchings) == ["raw", "in-image corrected", "stripe drift corrected",
                                       "in-image + stripe drift corrected"]
    assert len(result.images.tiles) == 4 and set(result.images.merged) == {"nominal", "mean", "first"}
    assert lines[3] == "4 of 4 tiles loaded and detected"

    pg.mkQApp()
    window = analysis_window(result, use_opengl=False)
    assert isinstance(window.widget(0), StitchViewer)
    result.images = None  # e.g. a saved result: the points view instead
    other = analysis_window(result, use_opengl=False)
    assert isinstance(other.widget(0), PointsStitchView)
    window.close()
    other.close()
