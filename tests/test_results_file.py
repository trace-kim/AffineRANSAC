"""Saving and loading an analysis result (results_file.py), run without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import dataclasses
import json
import warnings

import numpy as np
import pyqtgraph as pg
import pytest

from affine_ransac.analysis import Settings, analyse_contour_folder
from affine_ransac.results_file import FORMAT, load_result, save_result
from affine_ransac.view_analysis import analysis_window
from test_analysis import TABS, write_contour_folder


def analysed(folder):
    write_contour_folder(folder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the folder's summary table is left out with a warning
        return analyse_contour_folder(folder, Settings(max_match_nm=30.0), log=lambda line: None)


def assert_same(a, b, where="result"):
    """a and b hold the same values: dataclasses field by field, lists, dicts, tuples, arrays, numbers."""
    if dataclasses.is_dataclass(a):
        assert type(a) is type(b), where
        for f in dataclasses.fields(a):
            assert_same(getattr(a, f.name), getattr(b, f.name), f"{where}.{f.name}")
    elif isinstance(a, dict):
        assert list(a) == list(b), where
        for key in a:
            assert_same(a[key], b[key], f"{where}[{key!r}]")
    elif isinstance(a, (list, tuple)) and not (a and isinstance(a[0], (int, float))):
        assert len(a) == len(b), where
        for k, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{where}[{k}]")
    elif a is None or isinstance(a, str):
        assert a == b, where
    else:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=where)
        assert np.asarray(a).dtype.kind == np.asarray(b).dtype.kind, where


def test_a_saved_result_loads_back_the_same_and_opens_the_same_window(tmp_path):
    result = analysed(tmp_path)
    save_result(result, tmp_path / "result.npz")
    loaded = load_result(tmp_path / "result.npz")

    assert_same(result, loaded)
    assert loaded.settings.max_match_nm == 30.0 and loaded.log == result.log
    pg.mkQApp()
    window = analysis_window(loaded, use_opengl=False)
    assert [window.tabText(i) for i in range(window.count())] == TABS
    window.close()


def test_the_file_is_plain_arrays(tmp_path):
    result = analysed(tmp_path)
    save_result(result, tmp_path / "result.npz")
    with np.load(tmp_path / "result.npz", allow_pickle=False) as arrays:  # no pickled objects inside
        info = json.loads(str(arrays["info"]))
        np.testing.assert_array_equal(arrays["sets/uncorrected/ransac/residuals"],
                                      result.sets["uncorrected"].ransac.residuals)
    assert info["format"] == FORMAT and info["kind"] == "contour CSV" and info["tile_ids"] == result.tile_ids


def test_another_format_is_refused_and_unknown_settings_are_ignored(tmp_path):
    result = analysed(tmp_path)
    save_result(result, tmp_path / "result.npz")
    with np.load(tmp_path / "result.npz", allow_pickle=False) as file:
        arrays = dict(file)
    info = json.loads(str(arrays["info"]))
    info["settings"]["setting_of_a_later_version"] = 1
    del info["settings"]["drift_window_um"]  # saved before this setting existed: its default
    arrays["info"] = np.array(json.dumps(info))
    np.savez(tmp_path / "edited.npz", **arrays)
    assert load_result(tmp_path / "edited.npz").settings.drift_window_um == Settings().drift_window_um

    info["format"] = FORMAT + 1
    arrays["info"] = np.array(json.dumps(info))
    np.savez(tmp_path / "newer.npz", **arrays)
    with pytest.raises(ValueError, match="Analyse the folder again"):
        load_result(tmp_path / "newer.npz")
