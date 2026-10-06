"""Tests for the stitching residual viewer, run without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import numpy as np
import pyqtgraph as pg

from affine_ransac.geometry.affine import apply_affine
from affine_ransac.pipeline import residual_summary, stitch_tiles, stitch_tiles_rigid, tie_residuals
from affine_ransac.view_stitch_residuals import StitchResidualView
from test_pipeline import ERRORS, FOV, LATTICE, NOMINAL
from test_stitching import about, rotation


def rotated_tiles():
    """2 x 2 tiles, each seeing the lattice with its own small rotation and stage error."""
    rng = np.random.default_rng(5)
    truth = [about(rotation(rng.normal(0, 3000e-6), e), c) for e, c in zip(ERRORS, NOMINAL)]
    points = []
    for t, c in zip(truth, NOMINAL):
        inside = np.all(np.abs(apply_affine(np.linalg.inv(t), LATTICE) - c) < FOV / 2 - 10, axis=1)
        points.append(apply_affine(np.linalg.inv(t), LATTICE[inside]))
    return points


def make_view():
    points = rotated_tiles()
    stitch = stitch_tiles(points, NOMINAL, np.full((4, 2), FOV))
    stitchings = {"translation": (points, stitch, stitch.corrections),
                  "translation + rotation": (points, stitch, stitch_tiles_rigid(points, NOMINAL, stitch))}
    pg.mkQApp()
    return StitchResidualView(stitchings, NOMINAL, NOMINAL.mean(axis=0), use_opengl=False), stitchings


def test_summary_per_stitching_and_one_point_per_overlap():
    view, stitchings = make_view()
    points, stitch, corrections = stitchings["translation"]
    all_rms = np.sqrt((tie_residuals(points, stitch, corrections) ** 2).sum(axis=1).mean())
    assert view.summary["translation"]["all overlaps RMS (nm)"] == residual_summary(points, stitch, corrections, NOMINAL)[
        "all overlaps RMS (nm)"]
    np.testing.assert_allclose(view.summary["translation"]["all overlaps RMS (nm)"], all_rms)
    assert view.summary["translation + rotation"]["all overlaps RMS (nm)"] < 0.1 * all_rms  # the rotations are removed
    within, between = view.items["translation"]
    assert len(within.getData()[0]) == view.summary["translation"]["overlaps within stripes"] == 2
    assert len(between.getData()[0]) == view.summary["translation"]["overlaps between stripes"] == 2
    assert "translation + rotation: within stripes" in view.label.text()
    view.close()


def test_a_stitching_can_be_hidden():
    view, _ = make_view()
    view.boxes["translation"].setChecked(False)
    assert not any(item.isVisible() for item in view.items["translation"])
    assert all(item.isVisible() for item in view.items["translation + rotation"])
    view.close()
