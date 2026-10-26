"""The release's contour diagnosis window (release/contour_dialog.py, docs/SPEC.md D129), without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import time
from dataclasses import replace

import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtTest

from affine_ransac.analysis import CONTOUR_SETTINGS, contour_options
from affine_ransac.features.contact import contact_mask
from affine_ransac.io.sem_image import load_sem_image
from affine_ransac.pipeline import find_contacts
from affine_ransac.release import style
from affine_ransac.release.analysis import DEFAULTS
from affine_ransac.release.contour_dialog import GROUPS, ContourDialog
from test_pipeline import FOV, LATTICE, SIZE, render_tile
from test_release import make_window

SETTINGS = replace(DEFAULTS, detection="otsu")  # the synthetic tiles are found with otsu (test_release_versions)


def open_dialog(tmp_path, settings=SETTINGS):
    """The window with a synthetic tile loaded and its contacts found; and the tile's path."""
    style.setup()
    pg.mkQApp()
    path = tmp_path / "tile.png"
    cv2.imencode(".png", render_tile(LATTICE, (0.0, 0.0), FOV, SIZE, seed=1))[1].tofile(str(path))
    dialog = ContourDialog(settings, str(tmp_path))
    dialog.load_image(str(path))
    settle(dialog)
    return dialog, path


def settle(dialog, seconds=30):
    """Wait until the window has shown the contacts of the settings in it."""
    end = time.time() + seconds
    while (dialog.running or dialog.timer.isActive()) and time.time() < end:
        pg.mkQApp().processEvents()
        time.sleep(0.01)
    assert not dialog.running and not dialog.timer.isActive(), "the search did not finish"


def test_the_window_edits_every_contour_setting_in_two_groups():
    assert tuple(name for _, names in GROUPS for name in names) == CONTOUR_SETTINGS


def test_the_window_shows_the_contacts_a_run_finds(tmp_path):
    dialog, path = open_dialog(tmp_path)
    image = load_sem_image(path)
    detected, refined, _ = find_contacts(image, **contour_options(SETTINGS))

    for name, contacts in (("Detected centres", detected), ("Refined centres", refined)):
        x, y = dialog.layers[name].getData()
        np.testing.assert_array_equal(np.column_stack([x, y]), contacts.centers)
    np.testing.assert_array_equal(dialog.layers["Threshold mask"].image, contact_mask(image, method="otsu")[0])
    assert dialog.summary.text().startswith(f"<b>{len(refined.centers)} contacts found</b> (otsu, threshold at grey "
                                            f"level {detected.threshold:.0f})")
    assert "touching the image border" in dialog.summary.text()  # the contacts cut by the image's edges
    # the mask covers the image: off at start; the contours and centres on (D76, D113)
    assert [name for name, box in dialog.boxes.items() if box.isChecked()] == \
        ["Detected contours", "Detected centres", "Refined contours", "Refined centres"]
    assert not dialog.layers["Threshold mask"].isVisible() and dialog.layers["Refined contours"].isVisible()


def test_a_change_finds_the_contacts_again_with_the_newest_settings(tmp_path):
    dialog, _ = open_dialog(tmp_path)
    dialog.editors["max_area_px"].setValue(1)
    settle(dialog)
    assert dialog.summary.text().startswith("<b>0 contacts found</b>") and "larger than 1 px" in dialog.summary.text()

    # changes made while a search runs: only the newest settings are shown
    dialog.editors["max_area_px"].setValue(0)
    dialog.search()
    dialog.editors["min_area_px"].setValue(10 ** 6)
    dialog.search()
    settle(dialog)
    assert dialog.summary.text().startswith("<b>0 contacts found</b>") and "smaller than 1000000 px" in \
        dialog.summary.text()


def test_invalid_settings_are_named_and_cannot_be_applied(tmp_path):
    dialog, _ = open_dialog(tmp_path)
    dialog.editors["normal_span"].setValue(0)
    settle(dialog)
    assert "normal_span must be &gt;= 1" in dialog.summary.text()
    assert not dialog.apply_button.isEnabled()
    dialog.editors["normal_span"].setValue(3)
    settle(dialog)
    assert dialog.apply_button.isEnabled() and "contacts found" in dialog.summary.text()


def test_settings_that_change_nothing_are_greyed_out(tmp_path):
    dialog, _ = open_dialog(tmp_path)
    assert not dialog.editors["min_solidity"].isEnabled()  # band only
    dialog.editors["detection"].setCurrentText("band")
    assert dialog.editors["min_solidity"].isEnabled()
    dialog.editors["refinement"].setCurrentText("none")
    assert [name for name, widget in dialog.editors.items() if not widget.isEnabled()] == \
        ["search_px", "edge_step_px", "edge_blur_px", "normal_span"]
    assert not dialog.boxes["Refined contours"].isEnabled() and not dialog.layers["Refined contours"].isVisible()


def test_enter_in_a_setting_does_not_close_the_window(tmp_path):
    dialog, _ = open_dialog(tmp_path)
    closed = []
    dialog.finished.connect(closed.append)
    QtTest.QTest.keyClick(dialog.editors["min_area_px"], QtCore.Qt.Key.Key_Return)
    assert closed == []


def test_apply_keeps_the_settings_for_the_next_run_and_cancel_does_not(tmp_path, monkeypatch):
    window = make_window(tmp_path)
    assert window.actions["contours"].isEnabled()

    def change_and(answer):
        def exec_(dialog):
            dialog.editors["min_area_px"].setValue(50)
            dialog.editors["edge_blur_px"].setValue(0.5)
            return answer
        return exec_
    monkeypatch.setattr(ContourDialog, "exec", change_and(0))  # Cancel
    window.diagnose_contours()
    assert window.settings().min_area_px == 20

    monkeypatch.setattr(ContourDialog, "exec", change_and(1))  # Apply to parameters
    window.diagnose_contours()
    assert (window.settings().min_area_px, window.settings().edge_blur_px) == (50, 0.5)
    window.close()
