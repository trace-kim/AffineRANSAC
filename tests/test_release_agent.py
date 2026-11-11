"""MaskReg from the command line for an AI agent (affine_ransac.release.agent, D154), on synthetic folders."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # the review imports the views' module; before Qt starts

import pytest

from affine_ransac import app
from affine_ransac.release import agent
from affine_ransac.release.analysis import AnalysisFailed
from affine_ransac.release.cache import BACKUPS, CACHE_NAME
from test_analysis import write_contour_folder, write_image_folder
from test_release import result_with_failures


def call(capsys, *arguments) -> tuple[int, dict]:
    """The exit code and the JSON printed by one command."""
    code = agent.main([str(a) for a in arguments])
    return code, json.loads(capsys.readouterr().out)


def test_run_saves_a_run_of_its_own_and_prints_its_review(tmp_path, capsys):
    data, out = tmp_path / "data", tmp_path / "runs" / "first.npz"
    data.mkdir()
    write_contour_folder(data)
    with pytest.warns(UserWarning):  # the measuring tool's summary table, left out
        code, found = call(capsys, "run", data, "--out", out)

    assert code == 0 and found["run_file"] == str(out) and out.is_file()
    assert not (data / CACHE_NAME).exists() and not (data / BACKUPS).exists()  # the folder's own run is the user's
    assert found["kind"] == "contour CSV" and found["tiles"] == 6 and found["parameters_changed_from_default"] == {}
    assert found["stitching"]["verdict"] == ["Stitching OK. All 6 tiles stitched."]
    assert found["stitching"]["measured"]["tiles_not_stitched"] == {"count": 0, "names": []}
    errors = found["registration_error_after_moving_window_nm"]
    assert errors["count"] == found["contacts_measured"] > 0 and errors["max_nm"] < 1.0  # only the 0.1 nm noise
    assert call(capsys, "review", out) == (0, found)  # the review of the file is the one the run printed


def test_parameters_are_set_by_name_and_only_those_of_the_parameters_dialog(tmp_path, capsys):
    write_contour_folder(tmp_path)
    with pytest.warns(UserWarning):
        code, found = call(capsys, "run", tmp_path, "--out", tmp_path / "run.npz", "--set", "min_matched=3",
                           "--set", "moving_window_per_stripe=true")
    assert code == 0 and found["parameters_changed_from_default"] == {"min_matched": 3,
                                                                      "moving_window_per_stripe": True}

    for pair, message in (("ransac_seed=1", "is not a parameter of contour CSV folders. They are: max_match_nm"),
                          ("min_matched=many", "min_matched must be a int (is 'many')"),
                          ("moving_window_per_stripe=1", "moving_window_per_stripe must be true or false"),
                          ("min_matched=0", "min_matched must be >= 1")):  # the last: the analysis's own check
        code, found = call(capsys, "run", tmp_path, "--out", tmp_path / "bad.npz", "--set", pair)
        assert code == 1 and message in found["error"]
    assert not (tmp_path / "bad.npz").exists()


def test_an_image_folder_is_run_from_its_images(tmp_path, capsys, monkeypatch):
    records = write_image_folder(tmp_path)
    monkeypatch.setattr(app, "image_records", lambda folder: records)  # the metadata reader: only on the data machine

    code, found = call(capsys, "run", tmp_path, "--out", tmp_path / "run.npz", "--set", "detection=otsu")

    assert code == 0 and found["kind"] == "images" and found["tiles"] == 4
    assert found["parameters_changed_from_default"] == {"detection": "otsu"}
    assert found["contacts_measured"] > 0 and found["stitching"]["measured"]["overlaps_used"] > 0


def test_the_review_names_what_the_stitching_left_out(tmp_path):
    found = agent.review(result_with_failures(tmp_path / "data"))

    assert found["stitching"]["verdict"][0].startswith("Warning: 1 of 6 tiles not stitched.")
    assert found["stitching"]["measured"]["overlaps_not_matched"]["count"] == 2
    assert found["stitching"]["design"]["overlaps_not_matched"]["count"] == 1
    assert found["stitching"]["design"]["tiles_not_stitched"] == {"count": 1, "names": ["CD000005"]}
    json.dumps(found, default=agent.plain)  # every value can be printed


def test_a_run_that_stops_after_the_stitching_prints_what_it_found_and_saves_nothing(tmp_path, capsys, monkeypatch):
    partial = result_with_failures(tmp_path / "data")
    partial.moving = None

    def fail(*arguments):
        raise AnalysisFailed("Only 0 measured contact(s)", partial, {5: "not stitched"})
    monkeypatch.setattr(agent, "analyse_folder", fail)

    code, found = call(capsys, "run", tmp_path / "data", "--out", tmp_path / "run.npz")

    assert code == 1 and found["failed"] == "Only 0 measured contact(s)"
    assert found["tiles_not_measured"] == {"CD000005": "not stitched"}
    assert "registration_error_after_moving_window_nm" not in found and not (tmp_path / "run.npz").exists()


def test_a_folder_or_a_file_that_is_not_there_is_an_error_in_words(tmp_path, capsys):
    assert call(capsys, "run", tmp_path / "none") == (1, {"error": f"{tmp_path / 'none'} is not a folder"})
    assert call(capsys, "review", tmp_path / "none.npz") == (1, {"error": f"{tmp_path / 'none.npz'} is not a file"})
