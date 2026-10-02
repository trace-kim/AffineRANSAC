"""Tests for the registration error viewer, run without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import numpy as np
import pyqtgraph as pg

from affine_ransac.fitting.ransac import ransac_affine
from affine_ransac.view_registration import RegistrationView
from test_ransac import synthetic


def make_view():
    sem, design, *_ = synthetic()
    result = ransac_affine(sem, design, threshold_nm=0.5, seed=1)
    pg.mkQApp()
    view = RegistrationView(design, result.residuals, result.inliers, result.reference,
                            heatmap_bin_nm=5000, profile_bin_nm=5000, use_opengl=False)
    return view, design, result


def test_heatmaps_hold_the_binned_mean_error():
    view, design, result = make_view()
    (plot, image, grid), _ = view.heatmaps
    assert image.image.shape == grid.shape and np.isfinite(grid).any()
    # The image spans the binned area, in µm relative to the reference point.
    rect = image.mapRectToParent(image.boundingRect())
    position = (design - result.reference) / 1000
    assert rect.left() <= position[:, 0].min() and rect.right() >= position[:, 0].max()
    assert rect.top() <= position[:, 1].min() and rect.bottom() >= position[:, 1].max()  # QRectF: top = smaller y
    low, high = view.color_bar.levels()
    assert low == -high and high > 0  # symmetric scale
    view.close()


def test_profiles_average_over_x_along_y():
    view, design, result = make_view()
    centers, mean, std, count = view.profiles["dy"]
    assert count.sum() == len(design)
    # The x-averaged mean of all contacts includes the outliers; inliers alone stay near 0.
    assert np.abs(result.residuals[result.inliers, 1]).max() < 0.5
    assert len(centers) == len(mean) == len(std)
    view.close()


def test_summary_reports_all_contacts_and_inliers():
    view, _, result = make_view()
    text = view.findChild(pg.QtWidgets.QLabel).text()
    assert f"all contacts: {len(result.inliers)}" in text and f"inliers: {result.inliers.sum()}" in text
    view.close()


def test_exclude_outliers_rebins_the_heatmaps_from_the_inliers():
    view, design, result = make_view()
    high_all = view.color_bar.levels()[1]
    view.exclude_box.setChecked(True)
    high_inliers = view.color_bar.levels()[1]
    assert high_inliers < high_all  # defects no longer set the colour scale
    assert high_inliers <= np.abs(result.residuals[result.inliers]).max() + 1e-9
    view.close()
