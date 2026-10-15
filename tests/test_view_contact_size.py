"""Tests for the contact size maps, run without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import numpy as np
import pyqtgraph as pg

from affine_ransac.view_contact_size import ContactSizeView, value_levels

DESIGN = np.array([(x, y) for x in np.arange(0, 20_000, 100.0) for y in np.arange(0, 10_000, 100.0)])


def make_view():
    rng = np.random.default_rng(0)
    design_sizes = np.tile([1600.0, 40.0, 40.0], (len(DESIGN), 1))
    sem_sizes = design_sizes + rng.normal(0, [30, 1, 1], design_sizes.shape) + [100, 2, -1]
    sem_sizes[0, 1] = np.nan  # no chord crossed this contour
    pg.mkQApp()
    view = ContactSizeView(DESIGN, sem_sizes, design_sizes, DESIGN.mean(axis=0), use_opengl=False)
    view.resize(1500, 700)
    view.show()
    return view, sem_sizes, design_sizes


def test_area_map_at_start_and_any_combination_of_maps():
    view, *_ = make_view()
    assert [w.isVisible() for w, *_ in view.maps] == [True, False, False]
    for box in view.show_boxes:
        box.setChecked(True)
    assert [w.isVisible() for w, *_ in view.maps] == [True, True, True]
    view.show_boxes[0].setChecked(False)
    assert [w.isVisible() for w, *_ in view.maps] == [False, True, True]
    view.close()


def test_sem_value_or_sem_minus_design_with_its_own_colour_scale():
    view, sem_sizes, design_sizes = make_view()
    np.testing.assert_allclose(view.values(1), sem_sizes[:, 1])
    low, high = view.maps[0][3].levels()
    assert 1600 < low < 1700 < high  # SEM area about 1700 nm²
    view.difference_button.setChecked(True)
    np.testing.assert_allclose(view.values(1), sem_sizes[:, 1] - design_sizes[:, 1])
    low, high = view.maps[0][3].levels()
    assert 0 < low < 100 < high  # around the 100 nm² bias, not centred on 0
    assert "SEM − design" in view.maps[2][1].titleLabel.text
    view.close()


def test_maps_share_zoom_and_summary_counts_the_contacts_without_a_value():
    view, *_ = make_view()
    view.maps[0][1].vb.setRange(xRange=(-1, 1), yRange=(-1, 1), padding=0)
    view.show_boxes[2].setChecked(True)  # shown later: opens on the same area
    pg.QtWidgets.QApplication.processEvents()
    np.testing.assert_allclose(view.maps[2][1].vb.viewRange()[0], view.maps[0][1].vb.viewRange()[0], atol=0.05)
    view.maps[2][1].vb.setRange(xRange=(3, 5), yRange=(0, 2), padding=0)  # zoom on one map: all follow
    np.testing.assert_allclose(view.maps[0][1].vb.viewRange(), view.maps[2][1].vb.viewRange(), atol=1e-9)
    assert "xCD (nm): SEM mean 42.0" in view.label.text() and "1 contact(s) without a value" in view.label.text()
    view.close()


def test_value_levels():
    assert value_levels(np.array([np.nan])) == (0.0, 1.0)
    assert value_levels(np.array([5.0, 5.0])) == (4.0, 6.0)
    np.testing.assert_allclose(value_levels(np.array([*np.arange(101.0), np.nan])), (1, 99))
