"""Tests for the registration error viewer, run without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import numpy as np
import pyqtgraph as pg

from affine_ransac.fitting.ransac import ransac_affine
from affine_ransac.view_registration import RegistrationView, colorize, rasterize
from test_ransac import synthetic


def test_rasterize_one_pixel_per_point_and_mean_where_points_share_a_pixel():
    points = np.array([[0.5, 0.5], [0.6, 0.6], [3.5, 1.5]])
    grid = rasterize(points, np.array([1.0, 3.0, 5.0]), rect=(0, 0, 4, 2), shape=(2, 4), radius_px=0)
    assert grid[0, 0] == 2.0  # two points in the bottom-left pixel: their mean
    assert grid[1, 3] == 5.0  # row 0 is the bottom, so y = 1.5 is row 1
    assert np.isnan(grid).sum() == 6


def test_rasterize_draws_disks_when_zoomed_in():
    grid = rasterize(np.array([[5.5, 5.5]]), np.array([2.0]), rect=(0, 0, 11, 11), shape=(11, 11), radius_px=2)
    assert np.nansum(grid > 0) == 13  # a radius-2 disk: 13 pixels
    assert grid[5, 7] == 2.0 and np.isnan(grid[5, 8])


def make_view():
    sem, design, *_ = synthetic(n=600)
    result = ransac_affine(sem, design, threshold_nm=0.5, seed=1)
    pg.mkQApp()
    view = RegistrationView(design, result.residuals, result.reference, use_opengl=False)
    return view, design, result


def test_maps_show_every_contact_and_share_the_view():
    view, design, result = make_view()
    view.show()
    pg.mkQApp().processEvents()
    (plot_x, image_x, _), (plot_y, _, _) = view.maps
    shown = (image_x.image.max(axis=2) > 0).sum()  # coloured (not background) pixels
    assert shown >= len(design) * 0.5  # zoomed out: about one pixel per contact (some share a pixel)
    plot_x.vb.setRange(xRange=(-1, 1), yRange=(-1, 1), padding=0)
    assert np.allclose(plot_y.vb.viewRect().getRect(), plot_x.vb.viewRect().getRect())
    view.rasterize_maps()  # the timer would do this after a zoom
    low, high = view.color_bar.levels()
    assert low == -high and high > 0
    view.close()


def test_row_profile_is_the_mean_error_of_each_row():
    sem, design, *_ = synthetic(n=600)
    # Put the contacts on 5 rows; add a row-dependent dx to the error.
    rows_y = design[:, 1].min() + 4000.0 * np.arange(5)
    design[:, 1] = rows_y[np.arange(len(design)) % 5]
    error = np.zeros_like(design)
    error[:, 0] = np.arange(len(design)) % 5 * 0.1  # row k: +0.1 k nm in dx
    pg.mkQApp()
    view = RegistrationView(design, error, design.mean(axis=0), use_opengl=False)
    assert len(view.row_y) == 5 and view.row_count.sum() == len(design)
    np.testing.assert_allclose(view.row_mean[:, 0], 0.1 * np.arange(5), atol=1e-9)
    view.close()


def test_colorize_jet_and_background_where_empty():
    rgba = colorize(np.array([[-1.0, 0.0, 1.0, np.nan]]), levels=(-1, 1))
    assert rgba.shape == (1, 4, 3) and rgba.dtype == np.uint8
    np.testing.assert_array_equal(rgba[0, 0, :3], [0, 0, 128])    # low end: dark blue
    np.testing.assert_array_equal(rgba[0, 2, :3], [128, 0, 0])    # high end: dark red
    assert (rgba[0, 3] == 0).all()                                # NaN: black background
