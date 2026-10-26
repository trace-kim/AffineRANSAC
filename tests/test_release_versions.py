"""The rules for MaskReg's versions and run files (docs/SPEC.md, "Versions and run files", D99).

The first two tests stop a change that would break the rules silently:
- RESULTS: what the release computes on the synthetic folders, per ANALYSIS number (release/analysis.py). If a change
  moves these numbers, the same folder now gives other results: raise ANALYSIS by one, record the new numbers under
  it (print them with  pytest tests/test_release_versions.py -k results -s), add a CHANGELOG.md line. Never edit the
  numbers of an earlier ANALYSIS.
- LAYOUT: what a run file holds, per FORMAT (release/cache.py). If a change alters it and a file saved before can no
  longer be read, raise FORMAT and record the new layout under it; if old files still load (the new entry is
  optional, with a default), add it to the current layout. Either way, add a CHANGELOG.md line.
"""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

from dataclasses import replace

import cv2
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from affine_ransac import __version__
from affine_ransac.release.analysis import ANALYSIS, DEFAULTS, analyse_folder
from affine_ransac.release.cache import (BACKUPS, CACHE_NAME, FORMAT, NewerRunFile, load_cache, made_by,
                                         save_cache)
from test_analysis import write_contour_folder, write_image_folder
from test_pipeline import ERRORS, FOV, LATTICE, NOMINAL, SIZE, render_tile
from test_release import make_window, wait

# Per ANALYSIS: [contacts, rms dx, rms dy, max |dx|, max |dy|, dx, dy of the first contact, sum |SEM correction| x, y,
# sum |design correction| x, y]; residuals and corrections in nm.
RESULTS = {
    1: {"contour CSV": [2360, 0.097645195, 0.097127712, 0.387878965, 0.376539609, 0.022559669, -0.052001588,
                        8.149617877, 8.95389255, 12.5, 11.5],
        "contour CSV, 3 um window": [2360, 0.09756319, 0.096903189, 0.38584456, 0.384917053, 0.024032299,
                                     -0.05281951, 8.149617877, 8.95389255, 12.5, 11.5],
        "images": [196, 0.054032238, 0.049703496, 0.184937517, 0.140998837, 0.017220172, 0.013214268,
                   6.451748645, 6.84837345, 0.0, 0.0]},
    # 2 (D104): realignment searches offsets of any size and stops when none is clearly best; these folders do not
    # realign, so their numbers are those of 1.
    2: {"contour CSV": [2360, 0.097645195, 0.097127712, 0.387878965, 0.376539609, 0.022559669, -0.052001588,
                        8.149617877, 8.95389255, 12.5, 11.5],
        "contour CSV, 3 um window": [2360, 0.09756319, 0.096903189, 0.38584456, 0.384917053, 0.024032299,
                                     -0.05281951, 8.149617877, 8.95389255, 12.5, 11.5],
        "images": [196, 0.054032238, 0.049703496, 0.184937517, 0.140998837, 0.017220172, 0.013214268,
                   6.451748645, 6.84837345, 0.0, 0.0]},
    # 3 (D105): of the realignment offsets that fit equally (a repeating pattern), the one pairing the most
    # contacts, no stop; these folders do not realign, so their numbers are those of 1.
    3: {"contour CSV": [2360, 0.097645195, 0.097127712, 0.387878965, 0.376539609, 0.022559669, -0.052001588,
                        8.149617877, 8.95389255, 12.5, 11.5],
        "contour CSV, 3 um window": [2360, 0.09756319, 0.096903189, 0.38584456, 0.384917053, 0.024032299,
                                     -0.05281951, 8.149617877, 8.95389255, 12.5, 11.5],
        "images": [196, 0.054032238, 0.049703496, 0.184937517, 0.140998837, 0.017220172, 0.013214268,
                   6.451748645, 6.84837345, 0.0, 0.0]},
    # 4 (D122): edge refinement at the mid-level (half-way between a contact's interior and its band), not the
    # steepest gradient; contour CSV folders have no images, so their numbers are those of 1.
    4: {"contour CSV": [2360, 0.097645195, 0.097127712, 0.387878965, 0.376539609, 0.022559669, -0.052001588,
                        8.149617877, 8.95389255, 12.5, 11.5],
        "contour CSV, 3 um window": [2360, 0.09756319, 0.096903189, 0.38584456, 0.384917053, 0.024032299,
                                     -0.05281951, 8.149617877, 8.95389255, 12.5, 11.5],
        "images": [196, 0.018585826, 0.020004241, 0.046613292, 0.06696727, -0.015755086, -0.001440986,
                   6.441456148, 6.821792166, 0.0, 0.0]},
    # 5 (D124): refinement none (was stitch_refined off) chooses the design tone with the detected centres too;
    # these folders are refined, so their numbers are those of 4.
    5: {"contour CSV": [2360, 0.097645195, 0.097127712, 0.387878965, 0.376539609, 0.022559669, -0.052001588,
                        8.149617877, 8.95389255, 12.5, 11.5],
        "contour CSV, 3 um window": [2360, 0.09756319, 0.096903189, 0.38584456, 0.384917053, 0.024032299,
                                     -0.05281951, 8.149617877, 8.95389255, 12.5, 11.5],
        "images": [196, 0.018585826, 0.020004241, 0.046613292, 0.06696727, -0.015755086, -0.001440986,
                   6.441456148, 6.821792166, 0.0, 0.0]},
    # 6 (D138): image by image places by agreement; these folders are stitched over all overlaps, so their numbers are
    # those of 5.
    6: {"contour CSV": [2360, 0.097645195, 0.097127712, 0.387878965, 0.376539609, 0.022559669, -0.052001588,
                        8.149617877, 8.95389255, 12.5, 11.5],
        "contour CSV, 3 um window": [2360, 0.09756319, 0.096903189, 0.38584456, 0.384917053, 0.024032299,
                                     -0.05281951, 8.149617877, 8.95389255, 12.5, 11.5],
        "images": [196, 0.018585826, 0.020004241, 0.046613292, 0.06696727, -0.015755086, -0.001440986,
                   6.441456148, 6.821792166, 0.0, 0.0]},
}

# Per FORMAT: the arrays (without the optional contact_sizes/*) and the info keys of a run file.
STITCH = ["corrections", "pairs/box", "pairs/center", "pairs/ia/data", "pairs/ia/lengths", "pairs/ib/data",
          "pairs/ib/lengths", "pairs/ij", "pairs/inliers/data", "pairs/inliers/lengths", "pairs/residuals/data",
          "pairs/residuals/lengths", "pairs/rotation", "pairs/shift", "rejected/box", "rejected/counts",
          "rejected/failed", "rejected/ij"]
LAYOUT = {
    3: {"arrays": sorted(["centers", "design_nm", "design_points/data", "design_points/lengths", "info",
                          "moving/centres_nm", "moving/counts", "moving/models", "moving/references", "moving/residuals",
                          "moving/stripe", "moving/window", "oas_centers", "sem_nm", "sem_points/data", "sem_points/lengths", "sizes",
                          "slipped", "stripe"] + [f"stitch/{name}" for name in STITCH] + [f"design_stitch/{name}" for name in STITCH]),
        "info": ["analysis", "folder", "format", "image_paths", "kind", "log", "oas_paths", "saved", "settings",
                 "tile_ids", "version"]},
}


def fingerprint(result) -> list[float]:
    r = result.moving.residuals
    return [len(result.design_nm), *np.sqrt(np.mean(r ** 2, axis=0)), *np.abs(r).max(axis=0), *r[0],
            *np.nansum(np.abs(result.stitch.corrections), axis=0),
            *np.nansum(np.abs(result.design_stitch.corrections), axis=0)]


def write_lossless_image_folder(folder):
    """The synthetic image folder of test_analysis with its tiles saved as PNG: JPEG encoders differ between OpenCV
    builds, PNG is exact."""
    records = write_image_folder(folder)
    for k, (record, nominal, error) in enumerate(zip(records, NOMINAL, ERRORS)):
        record.image_path = folder / f"{k}.png"
        cv2.imencode(".png", render_tile(LATTICE, nominal + error, FOV, SIZE, seed=k))[1].tofile(str(record.image_path))
    return records


def analyse_contours(folder, settings=DEFAULTS):
    """The synthetic contour folder of test_analysis, analysed (by default with the release's defaults)."""
    write_contour_folder(folder)
    with pytest.warns(UserWarning):
        return analyse_folder(folder, "contour CSV", settings, log=lambda line: None)


def test_the_results_are_those_recorded_for_their_analysis_number(tmp_path):
    for name in ("contours", "short window", "images"):
        (tmp_path / name).mkdir()
    found = {"contour CSV": fingerprint(analyse_contours(tmp_path / "contours")),
             # the field fits in one window of the default height: a short one makes the window move
             "contour CSV, 3 um window": fingerprint(analyse_contours(
                 tmp_path / "short window", replace(DEFAULTS, moving_window_um=3.0, moving_step_um=1.0))),
             "images": fingerprint(analyse_folder(tmp_path / "images", "images",
                                                  replace(DEFAULTS, detection="otsu"),  # the tiles have no bright band
                                                  write_lossless_image_folder(tmp_path / "images"),
                                                  log=lambda line: None))}
    print({kind: [round(float(v), 9) for v in values] for kind, values in found.items()})
    assert ANALYSIS in RESULTS, f"record the results of analysis {ANALYSIS} in RESULTS"
    for kind, expected in RESULTS[ANALYSIS].items():
        assert found[kind] == pytest.approx(expected, abs=1e-6), (
            f"{kind}: the results changed. If that is intended, raise ANALYSIS (release/analysis.py) to "
            f"{ANALYSIS + 1}, record these numbers under it and add a CHANGELOG.md line")


def test_a_run_file_holds_the_layout_recorded_for_its_format(tmp_path):
    path = save_cache(analyse_contours(tmp_path), tmp_path / "user")
    with np.load(path) as arrays:
        found = {"arrays": sorted(arrays.files), "info": sorted(json.loads(str(arrays["info"])))}
    assert FORMAT in LAYOUT, f"record the layout of format {FORMAT} in LAYOUT"
    assert found == LAYOUT[FORMAT], (
        "The run file's layout changed. If a file saved before can no longer be read, raise FORMAT "
        "(release/cache.py) and record this layout under it; if old files still load, add the change to the current "
        "layout. Either way, add a CHANGELOG.md line")


def rewrite_info(path, **changes):
    """Edit a saved run's info, as if another MaskReg had written it; None removes a key."""
    with np.load(path) as arrays:
        content = {name: arrays[name] for name in arrays.files}
    info = json.loads(str(content["info"]))
    info.update(changes)
    content["info"] = np.array(json.dumps({key: value for key, value in info.items() if value is not None}))
    with open(path, "wb") as f:
        np.savez_compressed(f, **content)


def test_opening_checks_the_format_and_the_analysis_of_a_saved_run(tmp_path):
    data, user = tmp_path / "data", tmp_path / "user"
    data.mkdir()
    path = save_cache(analyse_contours(data), user)
    run, _ = load_cache(data, user)
    assert (run.version, run.analysis) == (__version__, ANALYSIS)

    lines = []
    rewrite_info(path, version=None, analysis=None)  # saved before versions were
    run, _ = load_cache(data, user, lines.append)
    assert (run.version, run.analysis, made_by(run.version)) == ("", 0, "MaskReg before 1.0.0") and lines == []

    rewrite_info(path, format=FORMAT - 1)  # older format: not read, analysed again
    assert load_cache(data, user, lines.append) is None and "older file format" in lines[-1]

    rewrite_info(path, format=FORMAT + 1, version="9.0.0")  # newer format: never replaced silently
    with pytest.raises(NewerRunFile, match="made by MaskReg 9.0.0, in a newer file format"):
        load_cache(data, user, lines.append)

    path.write_bytes(b"damaged")  # e.g. an interrupted copy
    assert load_cache(data, user, lines.append) is None and "cannot be read" in lines[-1]


def test_a_run_saved_before_a_setting_existed_says_what_it_did(tmp_path):
    # Runs saved before D122 refined their edges at the steepest gradient and have no `refinement`: they read as
    # that, not as today's default, so that what is shown (and Load contours) matches how they were made.
    data, user = tmp_path / "data", tmp_path / "user"
    data.mkdir()
    path = save_cache(analyse_contours(data), user)
    assert load_cache(data, user)[0].settings.refinement == DEFAULTS.refinement == "mid-level"

    with np.load(path) as arrays:
        settings = json.loads(str(arrays["info"]))["settings"]
    rewrite_info(path, settings={name: value for name, value in settings.items() if name != "refinement"})
    assert load_cache(data, user)[0].settings.refinement == "steepest gradient"
    # until D124 the centres without refinement were a check box, stitch_refined: unticked reads as refinement none
    rewrite_info(path, settings={**settings, "stitch_refined": False})
    assert load_cache(data, user)[0].settings.refinement == "none"
    rewrite_info(path, settings={**settings, "stitch_refined": True})
    assert load_cache(data, user)[0].settings.refinement == "mid-level"


def test_a_run_file_without_an_optional_array_added_since_still_opens(tmp_path):
    # moving/stripe (D123) was added to format 3: a file saved before has none and opens with an empty one.
    data, user = tmp_path / "data", tmp_path / "user"
    data.mkdir()
    path = save_cache(analyse_contours(data), user)
    with np.load(path) as arrays:
        content = {name: arrays[name] for name in arrays.files if name != "moving/stripe"}
    with open(path, "wb") as f:
        np.savez_compressed(f, **content)

    run, _ = load_cache(data, user)
    assert run.moving.stripe.size == 0 and len(run.moving.residuals) == len(run.design_nm)


def test_an_outdated_run_is_shown_flagged_and_analysed_again_if_the_user_agrees(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    save_cache(replace(analyse_contours(data), analysis=ANALYSIS - 1, version="0.9.0"), str(tmp_path / "user cache"))
    asked, answers = [], [QtWidgets.QMessageBox.StandardButton.No, QtWidgets.QMessageBox.StandardButton.Yes]
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda parent, title, text: asked.append(text) or answers.pop(0))
    window = make_window(tmp_path)

    window.open_folder(str(data))  # shown as it is, flagged; the user says no
    wait(window)
    assert window.result.version == "0.9.0" and "OUTDATED: made by MaskReg 0.9.0" in window.status.idle.text()
    assert len(asked) == 1 and "made by MaskReg 0.9.0" in asked[0]

    window.open_folder(str(data))  # asked again; yes: analysed again (wait covers the restart), old run kept as a backup
    wait(window)
    assert len(asked) == 2 and window.result.analysis == ANALYSIS and "OUTDATED" not in window.status.idle.text()
    assert len(list((data / BACKUPS).iterdir())) == 1 and load_cache(data, tmp_path / "user cache")[0].version == __version__


def test_a_run_of_a_newer_format_is_not_opened_nor_replaced(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    path = save_cache(analyse_contours(data), str(tmp_path / "user cache"))
    rewrite_info(path, format=FORMAT + 1, version="9.0.0")
    before = path.read_bytes()
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda parent, title, text: shown.append(text))
    window = make_window(tmp_path)

    window.open_folder(str(data))
    wait(window)
    assert window.result is None and shown[0].startswith("The saved analysis of this folder was made by MaskReg 9.0.0")
    assert path.read_bytes() == before and window.actions["restart"].isEnabled()  # the user may still restart
    assert (data / CACHE_NAME) == path
