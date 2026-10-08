"""Tests for the standalone analyzer (app.py), run without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import importlib.util
import time

import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from affine_ransac.analysis import Settings
from affine_ransac.app import AnalyzerWindow, SettingsDialog, folder_kind
from affine_ransac.view_analysis import AnalysisWindow
from test_analysis import write_contour_folder


def make_window(tmp_path):
    pg.mkQApp()
    preferences = QtCore.QSettings(str(tmp_path / "preferences.ini"), QtCore.QSettings.Format.IniFormat)
    window = AnalyzerWindow(preferences)  # not the user's own preferences
    window.opengl_action.setChecked(False)
    return window


def wait(window, seconds=120):
    """Let the background job finish (its signals arrive through the event loop)."""
    end = time.time() + seconds
    while window.busy and time.time() < end:
        pg.mkQApp().processEvents()
        time.sleep(0.01)
    assert not window.busy, "the job did not finish"


def test_folder_kind(tmp_path):
    write_contour_folder(tmp_path)
    assert folder_kind(tmp_path) == "contour CSV"
    (tmp_path / "ContourCAD").mkdir()
    assert folder_kind(tmp_path) == "images"


def test_settings_dialog_shows_the_settings_of_the_folder_kind():
    pg.mkQApp()
    dialog = SettingsDialog(Settings(), "contour CSV")
    assert "tile_margin_nm" in dialog.editors and "detection" not in dialog.editors
    assert not dialog.editors["ransac_threshold_nm"].isEnabled()  # fixed
    dialog.editors["max_match_nm"].setValue(40.0)
    dialog.editors["first_stripe_upward"].setChecked(False)
    dialog.editors["ransac_placement"].setCurrentText("first")
    values = dialog.values()
    assert (values.max_match_nm, values.first_stripe_upward, values.ransac_placement) == (40.0, False, "first")
    assert values.detection == Settings().detection  # the other kind's settings are kept
    images = SettingsDialog(Settings(), "images")
    assert "detection" in images.editors and "tile_margin_nm" not in images.editors


def test_analyse_a_folder_save_reopen_and_add_an_external_measurement(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    write_contour_folder(folder)
    window = make_window(tmp_path)
    assert not window.actions["save_results"].isEnabled() and not window.actions["external"].isEnabled()

    window.analyse(str(folder), "contour CSV", Settings())
    assert window.busy and not window.actions["open_folder"].isEnabled()
    wait(window)
    assert isinstance(window.centralWidget(), AnalysisWindow) and window.result.kind == "contour CSV"
    log = window.log_view.toPlainText()
    assert "6 of 6 files used" in log and "Warning: 1 .csv file(s)" in log  # the summary table, left out
    assert window.actions["save_results"].isEnabled()

    window.save_results(str(tmp_path / "result.npz"))
    wait(window)
    assert (tmp_path / "result.npz").is_file() and "Saved" in window.log_view.toPlainText()
    analysed = window.result
    window.open_results(str(tmp_path / "result.npz"))
    wait(window)
    assert window.result is not analysed and window.result.tile_ids == analysed.tile_ids
    assert "Log of that run" in window.log_view.toPlainText()

    sites_um = window.result.sets["uncorrected"].design_nm[:3] / 1000
    (tmp_path / "tool.txt").write_text("".join(f"{x} {y} 0.5 -0.5\n" for x, y in sites_um), encoding="utf-8")
    window.add_external(str(tmp_path / "tool.txt"), flip_sign=True)
    assert ("tool (sign flipped)", "dx") in window.analysis.ransac_view.line_boxes
    assert "3 sites, 3 inside the analysed field" in window.log_view.toPlainText()
    window.close()


def test_a_failed_job_is_reported_and_the_window_stays_usable(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda parent, title, text: shown.append(text))
    window = make_window(tmp_path)
    window.analyse(str(tmp_path / "missing"), "contour CSV", Settings())
    wait(window)
    assert shown and "No contour .csv files" in shown[0]
    assert window.result is None and window.actions["open_folder"].isEnabled()
    assert "Traceback" in window.log_view.toPlainText()
    window.close()


def test_image_folders_need_the_metadata_reader(tmp_path, monkeypatch):
    if importlib.util.find_spec("affine_ransac.io.metadata") is not None:
        pytest.skip("the metadata reader exists on this machine")
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda parent, title, text: shown.append(text))
    window = make_window(tmp_path)
    window.analyse(str(tmp_path), "images", Settings())
    wait(window)
    assert shown and "affine_ransac.io.metadata" in shown[0]
    window.close()
