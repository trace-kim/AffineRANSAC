"""Saved parameters in MaskReg's Parameters dialog (release/parameters_dialog.py, docs/SPEC.md D136), without a
display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import json
from dataclasses import replace

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from affine_ransac.analysis import CONTOUR_SETTINGS
from affine_ransac.release.analysis import DEFAULTS, PARAMETERS
from affine_ransac.release.parameters_dialog import SECTIONS, USER_PARAMETERS, ParametersDialog, read_sets
from test_release import make_window


def preferences(tmp_path):
    pg.mkQApp()
    return QtCore.QSettings(str(tmp_path / "preferences.ini"), QtCore.QSettings.Format.IniFormat)


def answer(monkeypatch, name, yes=True):
    """The user types `name` when asked for a name, and answers Yes (or No) when asked to replace or delete."""
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args, **kwargs: (name, True))
    button = QtWidgets.QMessageBox.StandardButton.Yes if yes else QtWidgets.QMessageBox.StandardButton.No
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: button)


def test_parameters_saved_under_a_name_load_again_with_those_of_the_other_kind_of_folder(tmp_path, monkeypatch):
    saved_in = preferences(tmp_path)
    settings = replace(DEFAULTS, max_match_nm=40.0, detection="otsu", tile_margin_nm=7.0)
    dialog = ParametersDialog(settings, PARAMETERS["images"], saved_in)
    assert not dialog.load_button.isEnabled() and not dialog.delete_button.isEnabled()  # none saved yet
    dialog.editors["moving_window_um"].setValue(3.0)
    answer(monkeypatch, "  ADI wafer ")
    dialog.save_button.click()
    saved = read_sets(saved_in)["ADI wafer"]
    assert sorted(saved) == sorted(USER_PARAMETERS)
    assert saved["moving_window_um"] == 3.0 and saved["tile_margin_nm"] == 7.0  # not shown: contour CSV only

    # Later, a contour CSV folder open and other parameters in use (one users cannot change: layer).
    other = ParametersDialog(replace(DEFAULTS, layer=3), PARAMETERS["contour CSV"], saved_in)
    assert other.saved.currentText() == "ADI wafer" and other.note.text() == ""  # not the parameters in use
    other.load_button.click()
    assert other.editors["max_match_nm"].value() == 40.0 and other.editors["tile_margin_nm"].value() == 7.0
    assert other.values() == replace(settings, moving_window_um=3.0, layer=3)  # also the image parameters


def test_a_set_saved_before_a_parameter_existed_gives_that_parameter_its_default(tmp_path):
    saved_in = preferences(tmp_path)
    saved_in.setValue("parameter_sets", json.dumps({"old": {"max_match_nm": 30.0, "stitch_refined": False,
                                                            "gone": 1}}))  # stitch_refined: refinement none (D124)
    dialog = ParametersDialog(replace(DEFAULTS, overlap_tolerance_nm=5.0), PARAMETERS["images"], saved_in)
    dialog.load_button.click()
    assert dialog.values() == replace(DEFAULTS, max_match_nm=30.0, refinement="none")


def test_a_name_is_replaced_only_if_the_user_agrees_and_sets_can_be_deleted(tmp_path, monkeypatch):
    saved_in = preferences(tmp_path)
    dialog = ParametersDialog(DEFAULTS, PARAMETERS["images"], saved_in)
    answer(monkeypatch, "b")
    dialog.save_button.click()
    dialog.editors["max_match_nm"].setValue(50.0)
    answer(monkeypatch, "b", yes=False)
    dialog.save_button.click()
    assert read_sets(saved_in)["b"]["max_match_nm"] == DEFAULTS.max_match_nm
    answer(monkeypatch, "b")
    dialog.save_button.click()
    assert read_sets(saved_in)["b"]["max_match_nm"] == 50.0
    answer(monkeypatch, "A")
    dialog.save_button.click()
    assert [dialog.saved.itemText(k) for k in range(dialog.saved.count())] == ["A", "b"]  # by name, any case

    # The dialog opens on the set the parameters in use come from.
    reopened = ParametersDialog(replace(DEFAULTS, max_match_nm=50.0), PARAMETERS["images"], saved_in)
    assert reopened.saved.currentText() == "A" and "in use are 'A'" in reopened.note.text()

    reopened.saved.setCurrentText("b")
    answer(monkeypatch, "", yes=False)
    reopened.delete_button.click()
    assert sorted(read_sets(saved_in)) == ["A", "b"]
    answer(monkeypatch, "")
    reopened.delete_button.click()
    assert sorted(read_sets(saved_in)) == ["A"] and reopened.saved.currentText() == "A"


def test_a_set_loaded_in_maskreg_is_used_for_the_next_run(tmp_path, monkeypatch):
    window = make_window(tmp_path)
    window.preferences.setValue("parameter_sets", json.dumps({"ADI": {"max_match_nm": 30.0}}))

    def load_and_ok(dialog):
        dialog.load_button.click()
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(ParametersDialog, "exec", load_and_ok)
    window.edit_parameters()
    assert window.settings() == replace(DEFAULTS, max_match_nm=30.0)
    window.close()


def test_the_parameters_are_shown_in_sections_the_rarely_changed_ones_folded(tmp_path):
    """D147: every parameter in one section; Contour detection and Advanced folded when the dialog opens, unless one of
    their parameters differs from its default; a heading unfolds its section."""
    in_sections = [name for _, names, _ in SECTIONS for name in names]
    assert sorted(in_sections) == sorted(set(in_sections)) == sorted(USER_PARAMETERS)

    dialog = ParametersDialog(DEFAULTS, PARAMETERS["images"], preferences(tmp_path))
    assert [heading.text() for heading in dialog.headings.values()] == [
        "Stitching (2)", "In-image correction (1)", "Registration (3)", "Contour detection (18)", "Advanced (11)"]
    advanced = set(SECTIONS[-1][1]) - {"tile_margin_nm"}  # contour CSV only
    assert {name for name, widget in dialog.editors.items() if widget.isHidden()} == set(CONTOUR_SETTINGS) | advanced
    dialog.headings["Advanced"].click()
    assert not dialog.editors["max_match_nm"].isHidden() and dialog.editors["upscale"].isHidden()
    assert dialog.values() == DEFAULTS  # folded parameters keep their values

    changed = ParametersDialog(replace(DEFAULTS, row_gap_nm=7.0), PARAMETERS["contour CSV"], preferences(tmp_path))
    assert list(changed.headings) == ["Stitching", "In-image correction", "Registration", "Advanced"]
    assert not changed.editors["row_gap_nm"].isHidden()  # Advanced open: a changed value is never hidden
