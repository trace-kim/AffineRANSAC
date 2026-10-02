"""Tests for the RANSAC monitor, run without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import numpy as np
import pyqtgraph as pg
import pytest

from affine_ransac.fitting.ransac import ransac_affine
from affine_ransac.geometry.affine import decompose, report_terms
from affine_ransac.view_ransac import RansacMonitor
from test_ransac import synthetic


def make_monitor(**kwargs):
    sem, design, truth, defect = synthetic()
    pg.mkQApp()
    return RansacMonitor(sem, design, threshold_nm=0.5, seed=1, use_opengl=False, **kwargs), sem, design


def test_step_draws_the_sample_and_its_inliers():
    monitor, sem, _ = make_monitor()
    assert monitor.step()
    step = monitor.last_step
    assert step.stage == "search" and len(step.sample) == 3
    x, _ = monitor.sample_triangle.getData()
    assert len(x) == 4  # closed triangle
    shown = sum(len(item.data) for item in monitor.map_points)
    assert shown == len(sem)
    assert "Search iteration 1" in monitor.status.text()
    monitor.close()


def test_run_to_end_shows_the_final_affine():
    monitor, sem, design = make_monitor()
    monitor.run_to_end()
    assert monitor.done and monitor.last_step.stage == "refit"

    result = ransac_affine(sem, design, threshold_nm=0.5, seed=1)
    rows = report_terms(result.model, design - result.reference)
    best = [float(monitor.table.item(row, 2).text()) for row in range(len(rows))]
    best_edge = [float(monitor.table.item(row, 3).text()) for row in range(len(rows))]
    np.testing.assert_allclose(best, [value for _, value, _ in rows], atol=1e-3)
    np.testing.assert_allclose(best_edge, [edge for *_, edge in rows], atol=1e-3)
    assert monitor.table.verticalHeaderItem(4).text() == "rotation (°)"
    assert "Finished" in monitor.status.text()
    assert not monitor.step()  # nothing left
    monitor.close()


def test_linear_fit_line_slopes_are_the_model_terms():
    monitor, *_ = make_monitor()
    monitor.run_to_end()
    terms = decompose(monitor.last_step.best_model)
    # The model maps SEM -> design; SEM − design along x grows with x by −Mx (in nm per nm).
    (_, line) = monitor.linear[0, 0]
    x_um, dx_nm = line.getData()
    slope = (dx_nm[1] - dx_nm[0]) / ((x_um[1] - x_um[0]) * 1000)
    assert slope * 1e6 == pytest.approx(-terms["Mx_ppm"], rel=1e-6)
    monitor.close()


def test_restart_uses_the_panel_threshold():
    monitor, *_ = make_monitor()
    monitor.run_to_end()
    monitor.threshold_box.setValue(2.0)
    monitor.restart()
    assert not monitor.done and monitor.threshold_nm == 2.0 and monitor.history == []
    monitor.step()
    assert monitor.last_step.iteration == 1
    monitor.close()
