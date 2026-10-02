"""Smoke test: every plot in notebooks/overlap_plots.py runs without error (no display)."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window; must be set before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from affine_ransac.overlap_fit import fit_overlap
from affine_ransac.stitching import solve_tile_shifts

sys.path.insert(0, str(Path(__file__).parents[1] / "notebooks"))
import overlap_plots  # noqa: E402


@pytest.fixture(autouse=True)
def no_display(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)  # nothing to show without a display
    yield
    plt.close("all")


def strip_points():
    rng = np.random.default_rng(0)
    a = np.column_stack([np.arange(12) * 180.0 - 1000, np.full(12, -1300.0)])
    b = a + [-5.0, 4.0] + rng.normal(0, 0.3, a.shape)
    b[4] += [2.0, -2.0]
    return a, b


def test_pair_plots():
    a, b = strip_points()
    fit_t, fit_r = fit_overlap(a, b), fit_overlap(a, b, rotation=True)
    overlap_plots.plot_translation_fit(a, b, fit_t, 50, 500, 3.0)
    overlap_plots.plot_rotation_fit(a, fit_t, fit_r, 500)

    crop = np.random.default_rng(1).random((40, 200)).astype(np.float32)
    box = (-1100.0, 1100.0, -1400.0, -1200.0)
    overlap_plots.plot_image_alignment(crop, crop, np.array([0.3, -0.2]), box, b - a, fit_t.inliers,
                                       fit_t.shift, fit_t.shift + 0.1)


def test_all_pairs_and_stitching_plots():
    a, b = strip_points()
    fit = fit_overlap(a, b)
    summary = pd.DataFrame([{
        "matched": len(a), "outliers": int((~fit.inliers).sum()), "rms_translation_nm": 0.4, "rms_rigid_nm": 0.2,
        "rotation_urad": 100.0 * k, "image_minus_contact_x_nm": 0.1, "image_minus_contact_y_nm": -0.1,
    } for k in range(3)])
    overlap_plots.plot_all_pairs(summary)

    centers = np.array([[0.0, 0.0], [0.0, -2600.0]])
    fovs = np.full((2, 2), 2880.0)
    corrections = solve_tile_shifts(2, [(0, 1)], [fit.shift])
    overlap_plots.plot_stitch_residuals(centers, fovs, ["t0", "t1"], [(0, 1, a, b, fit.inliers)], corrections, 50, 500)

    fig, ax = plt.subplots()
    contour = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    overlap_plots.plot_mosaic(ax, np.zeros((10, 10)), (-5, 5, -5, 5), [[contour]], [contour], [(-1, 1, -1, 1)], "mosaic")
