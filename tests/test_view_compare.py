"""Tests for the stitching comparison window, run without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from affine_ransac.fitting.moving_window import moving_window_affine
from affine_ransac.fitting.ransac import ransac_affine
from affine_ransac.registration import MergedErrors, row_means, stitching_summary
from affine_ransac.view_compare import StitchingComparison
from test_ransac import synthetic


def run(seed):
    sem, design, *_ = synthetic(n=600, seed=seed)
    contacts = MergedErrors(design, sem, sem - design, np.ones(len(sem), int), np.zeros(len(sem)),
                            [np.array([k]) for k in range(len(sem))], {})
    return contacts, ransac_affine(sem, design, threshold_nm=0.5, seed=1), moving_window_affine(sem, design, 20_000, 5_000)


def make_window(stitch_view=None):
    runs = {"translation": run(0), "affine": run(1)}
    summary = {name: stitching_summary(*r, ties_nm=np.zeros((3, 2))) for name, r in runs.items()}
    pg.mkQApp()
    window = StitchingComparison(runs, summary, window_um=20, step_um=5, seed=1, stitch_view=stitch_view,
                                 use_opengl=False)
    return window, runs, summary


def test_comparison_tab_then_one_analysis_per_stitching():
    pg.mkQApp()  # before any widget
    window, runs, _ = make_window(QtWidgets.QLabel("stitch"))
    tabs = [window.tabText(i) for i in range(window.count())]
    assert tabs == ["Stitching", "Comparison", "Analysis, translation stitching", "Analysis, affine stitching"]
    np.testing.assert_allclose(window.analyses["affine"].ransac_view.error_nm, runs["affine"][1].residuals)
    window.close()


def test_row_plots_have_one_line_per_stitching():
    window, runs, _ = make_window()
    plot = window.comparison.plots["after the RANSAC affine", "dy"]
    curves = plot.listDataItems()
    assert len(curves) == 2
    contacts, ransac, _ = runs["affine"]
    expected = row_means(contacts.design_nm[:, 1], ransac.residuals, 10.0)[1][:, 1]
    np.testing.assert_allclose(curves[1].getData()[1], expected)
    window.close()


def test_table_lists_the_summary_per_stitching():
    window, _, summary = make_window()
    table = window.comparison.table
    assert [table.horizontalHeaderItem(c).text() for c in range(2)] == ["translation", "affine"]
    row = list(summary["affine"]).index("RANSAC 3σy (nm)")
    assert table.item(row, 1).text() == f"{summary['affine']['RANSAC 3σy (nm)']:.3f}"
    window.close()
