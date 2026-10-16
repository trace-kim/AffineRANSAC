"""The released analyzer (affine_ransac.release), on synthetic folders, without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import time
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from affine_ransac.analysis import ContactSizes, Settings
from affine_ransac.registration import row_means
from affine_ransac.release import app as release_app, style
from affine_ransac.release.analysis import PARAMETERS, analyse_folder, measure_contact_sizes, read_display_images
from affine_ransac.release.app import MaskRegWindow
from affine_ransac.release.cache import BACKUPS, CACHE_NAME, cache_paths, load_cache, save_cache
from affine_ransac.release.views import (NEEDS_TILES, NOT_IMAGES, ContactSizeTab, ContactSizeView, RegistrationView,
                                         StitchingView)
from test_analysis import IMAGE_SETTINGS, STAGE, write_contour_folder, write_image_folder


def test_contour_folder_gives_the_stitching_and_the_moving_window_error(tmp_path):
    write_contour_folder(tmp_path)
    steps, lines = [], []
    with pytest.warns(UserWarning):  # the measuring tool's summary table, left out
        result = analyse_folder(tmp_path, "contour CSV", Settings(), log=lines.append,
                                progress=lambda fraction, stage: steps.append(fraction))

    # The SEM tiles sit at nominal + STAGE: stitching takes it out (up to the mean, the gauge).
    np.testing.assert_allclose(result.stitch.corrections, STAGE - STAGE.mean(axis=0), atol=0.2)
    assert result.moving.residuals.shape == result.design_nm.shape
    assert np.abs(result.moving.residuals).max() < 1.0  # only the 0.1 nm noise is left
    assert steps == sorted(steps) and steps[-1] == 1.0
    assert lines[0] == "6 files, " + lines[0].split(", ")[1] and result.image_paths == []


def test_the_error_is_measured_minus_design(tmp_path):
    """A contact measured +5 nm in x of where the design puts it has dx = +5 nm (no affine: raw error)."""
    def bump(k, local, mask):
        error = np.zeros_like(mask)
        error[np.all(np.abs(mask - [1300.0, 2600.0]) < 70, axis=1), 0] = 5.0  # a few contacts, mid-field
        return error
    write_contour_folder(tmp_path, sem_error=bump)
    with pytest.warns(UserWarning):
        result = analyse_folder(tmp_path, "contour CSV", Settings(), log=lambda line: None)
    bumped = np.all(np.abs(result.design_nm - [1300.0, 2600.0]) < 70, axis=1)
    assert bumped.any()
    assert np.all(result.moving.residuals[bumped, 0] > 3.0)


def test_image_folder_reports_progress_per_tile_and_keeps_the_image_paths(tmp_path):
    records = write_image_folder(tmp_path)
    stages = []
    result = analyse_folder(tmp_path, "images", IMAGE_SETTINGS, records, log=lambda line: None,
                            progress=lambda fraction, stage: stages.append(stage))
    assert [s for s in stages if s.startswith("Loading tile")] == [f"Loading tile {k} of 4" for k in range(1, 5)]
    assert result.image_paths == [str(r.image_path) for r in records]
    images = read_display_images(result.image_paths)
    assert len(images) == 4 and images[0][0].shape == images[0][1]  # small enough: full size


def test_the_cache_is_saved_in_the_folder_or_else_in_the_user_cache(tmp_path):
    data, user = tmp_path / "data", tmp_path / "user"
    data.mkdir()
    write_contour_folder(data)
    with pytest.warns(UserWarning):
        result = analyse_folder(data, "contour CSV", Settings(), log=lambda line: None)
    assert load_cache(data, user) is None

    assert save_cache(result, user) == data / CACHE_NAME
    loaded, saved = load_cache(data, user)
    np.testing.assert_array_equal(loaded.moving.residuals, result.moving.residuals)
    np.testing.assert_array_equal(loaded.stitch.corrections, result.stitch.corrections)
    assert loaded.tile_ids == result.tile_ids and loaded.settings == result.settings and len(saved) == 16
    assert loaded.contact_sizes is None  # none measured

    (data / CACHE_NAME).unlink()
    (data / CACHE_NAME).mkdir()  # the data folder's file cannot be written
    assert save_cache(result, user) == cache_paths(data, user)[1]
    assert load_cache(data, user) is not None


def test_a_new_run_keeps_the_last_five_runs_it_replaced_as_backups(tmp_path, monkeypatch):
    data, user = tmp_path / "data", tmp_path / "user"
    data.mkdir()
    write_contour_folder(data)
    with pytest.warns(UserWarning):
        result = analyse_folder(data, "contour CSV", Settings(), log=lambda line: None)
    current, backups = data / CACHE_NAME, data / BACKUPS
    start = time.mktime((2026, 10, 16, 9, 0, 0, 0, 0, -1))  # local time, as the names
    for k in range(7):
        save_cache(replace(result, log=[f"run {k}"]), user)
        os.utime(current, (start + 60 * k, start + 60 * k))  # run k saved at 09:0k
    stamps = lambda: [p.name.removeprefix("MaskReg_analysis_") for p in sorted(backups.iterdir())]
    assert stamps() == [f"20261016-09{k:02d}00.npz" for k in range(1, 6)]  # runs 1-5; run 0, the oldest, deleted
    assert load_cache(data, user)[0].log == ["run 6"]
    with np.load(backups / "MaskReg_analysis_20261016-090500.npz") as arrays:
        assert "run 5" in str(arrays["info"])

    os.utime(current, (start + 300, start + 300))  # run 6 saved in the same second as run 5
    save_cache(replace(result, log=["run 7"]), user)
    assert stamps()[-2:] == ["20261016-090500.npz", "20261016-090501.npz"] and len(stamps()) == 5

    save_cache(replace(result, log=["run 7, sizes added"]), user, backup=False)  # the same run: no backup
    assert len(stamps()) == 5 and load_cache(data, user)[0].log == ["run 7, sizes added"]

    def disk_full(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(np, "savez_compressed", disk_full)
    with pytest.raises(OSError, match="could not be saved"):
        save_cache(replace(result, log=["run 8"]), user)
    assert load_cache(data, user)[0].log == ["run 7, sizes added"] and len(stamps()) == 5
    assert not list(tmp_path.rglob("*.part"))


def test_figures_are_white_and_label_mask_coordinates(tmp_path):
    style.setup()
    pg.mkQApp()
    axis = style.MaskAxis("bottom", origin_nm=12000.0)  # data in nm relative to x = 12 µm
    assert axis.tickStrings([0.0, 500.0], axis.scale, 500.0) == ["12.0", "12.5"]
    write_contour_folder(tmp_path)
    with pytest.warns(UserWarning):
        result = analyse_folder(tmp_path, "contour CSV", Settings(), log=lambda line: None)
    view = RegistrationView(result, use_opengl=False)
    view.resize(1200, 900)
    view.show()
    QtWidgets.QApplication.processEvents()
    image = style.render_image(view.figure)
    assert image.width() == 2 * view.figure.viewport().width()
    assert image.pixelColor(2, 2).name() == "#ffffff"
    table = view.profiles()
    assert table[0][0] == "row y (µm)" and table[0][3] == "row mean dr (nm)" and table[0][4] == "column x (µm)"
    assert len(table) == max(len(view.row_y), len(view.column_x)) + 1

    # The profiles beside a map follow it: the row profile its y, the column profile its x.
    panel = view.maps[0][0]
    panel.map.vb.setRange(xRange=(0.2, 0.8), yRange=(1.0, 3.0), padding=0)
    QtWidgets.QApplication.processEvents()
    assert np.allclose(panel.rows.vb.viewRange()[1], panel.map.vb.viewRange()[1])
    assert np.allclose(panel.columns.vb.viewRange()[0], panel.map.vb.viewRange()[0])
    rows_line = panel.rows.listDataItems()[0].getData()  # rotated: x = mean error, y = row position
    np.testing.assert_allclose(rows_line[0], view.row_mean[:, 0])
    (ax, ay), (mx, my) = view.arrow_panel.map.vb.viewRange(), panel.map.vb.viewRange()
    assert np.allclose(ay, my) and np.isclose(np.mean(ax), np.mean(mx))  # same area (x as wide as each map's shape)
    dr_line = view.arrow_panel.rows.listDataItems()[0].getData()  # beside the arrows: the mean |error| per row
    _, dr_mean, _ = row_means(view.design_nm[:, 1], np.linalg.norm(view.error_nm, axis=1), result.settings.row_gap_nm)
    np.testing.assert_allclose(dr_line[0], dr_mean)

    # Arrows: one shaft and two head strokes per contact, scaled, coloured by their length.
    assert arrow_points(view) == 6 * len(view.error_nm)
    assert sum(len(item.getData()[0]) > 0 for item in view.arrows) > 1  # several colours
    view.dr_bar.setLevels((0.0, 1e6))  # every arrow in the first colour, in contact order
    view.arrow_scale.setValue(1000)
    x, y = view.arrows[0].getData()
    np.testing.assert_allclose([x[1] - x[0], y[1] - y[0]], view.error_nm[0] * 1000 / 1000)  # nm x 1000 on the map, in µm
    stitching = StitchingView(result, use_opengl=False)
    assert not stitching.load_button.isEnabled()  # contour CSVs: no images
    view.close()


def arrow_points(view) -> int:
    """The points of all arrows drawn, over every colour."""
    return sum(len(item.getData()[0]) for item in view.arrows)


def make_window(tmp_path):
    style.setup()
    pg.mkQApp()
    preferences = QtCore.QSettings(str(tmp_path / "preferences.ini"), QtCore.QSettings.Format.IniFormat)
    preferences.setValue("use_opengl", "false")
    return MaskRegWindow(preferences, cache_dir=str(tmp_path / "user cache"))


def test_realignment_is_a_parameter_of_image_folders():
    assert "realign" in PARAMETERS["images"]
    assert "realign" not in PARAMETERS["contour CSV"]


def wait(window, seconds=120):
    end = time.time() + seconds
    while window.busy and time.time() < end:
        pg.mkQApp().processEvents()
        time.sleep(0.01)
    assert not window.busy, "the job did not finish"


def test_opening_a_folder_analyses_it_once_then_opens_the_saved_result(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    write_contour_folder(data)
    window = make_window(tmp_path)
    assert not window.actions["restart"].isEnabled()

    window.open_folder(str(data))
    assert window.busy and window.status.bar.isVisible() is not None
    wait(window)
    assert (data / CACHE_NAME).is_file() and "No saved analysis" in window.log_view.toPlainText()
    tabs = window.centralWidget()
    assert [tabs.tabText(k) for k in range(tabs.count())] == ["Stitching", "Registration"]
    assert "contacts" in window.status.idle.text() and window.actions["restart"].isEnabled()

    analysed = window.result
    window.open_folder(str(data))
    wait(window)
    assert "Saved analysis of" in window.log_view.toPlainText() and window.result is not analysed

    # Parameters changed: the restart uses them and saves over the old result, which is kept as a backup.
    window.preferences.setValue("settings", '{"moving_window_um": 3.0, "moving_step_um": 1.0}')
    window.restart()
    wait(window)
    assert window.result.settings.moving_window_um == 3.0
    reopened, _ = load_cache(data, tmp_path / "user cache")
    assert reopened.settings.moving_window_um == 3.0
    assert len(list((data / BACKUPS).iterdir())) == 1 and "Previous run kept as" in window.log_view.toPlainText()


def test_folders_shown_are_listed_in_open_recent_newest_first(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    write_contour_folder(data)
    window = make_window(tmp_path)
    for k in range(12):
        (tmp_path / f"old{k}").mkdir()
        window.remember_folder(str(tmp_path / f"old{k}"))
    (tmp_path / "old11").rmdir()  # gone from the disk

    window.open_folder(data.as_posix())  # forward slashes, as the folder dialog gives them
    wait(window)
    recent = window.recent_folders()
    assert len(recent) == release_app.RECENT and recent[0] == str(data) and recent[1] == str(tmp_path / "old11")
    window.remember_folder(str(tmp_path / "old10"))
    window.remember_folder(str(data))  # the same folder moves up, once
    assert window.recent_folders()[:2] == [str(data), str(tmp_path / "old10")]

    window.fill_recent_menu()
    actions = [a for a in window.recent_menu.actions() if not a.isSeparator()]
    assert len(actions) == release_app.RECENT + 1 and actions[-1].text() == "Clear list"
    gone = next(a for a in actions if "old11" in a.text())
    assert "not found" in gone.text() and not gone.isEnabled()
    opened = []
    monkeypatch.setattr(window, "open_folder", opened.append)
    actions[1].trigger()
    assert opened == [str(tmp_path / "old10")] and window.preferences.value("last_folder") == opened[0]
    actions[-1].trigger()
    assert window.recent_folders() == []


def test_sem_images_are_loaded_on_request(tmp_path, monkeypatch):
    records = write_image_folder(tmp_path)
    monkeypatch.setattr(release_app, "image_records", lambda folder: records)
    window = make_window(tmp_path)
    window.preferences.setValue("settings", '{"layer": 1, "datatype": 0, "detection": "otsu"}')
    window.open_folder(str(tmp_path))
    wait(window)
    view = window.stitching_view
    assert view.load_button.isEnabled() and not view.images_box.isEnabled()
    window.load_images()
    wait(window)
    assert len(view.image_items) == 4 and view.images_box.isChecked() and not view.load_button.isEnabled()
    view.opacity.setValue(40)
    assert view.image_items[0].opacity() == 0.4 and view.opacity_label.text() == "40 %"
    window.gpu_action.setChecked(False)
    window.set_gpu(False)  # rebuilt: the images loaded are kept
    assert len(window.stitching_view.image_items) == 4


def test_the_design_pattern_is_loaded_on_request_in_stripe_colours(tmp_path, monkeypatch):
    records = write_image_folder(tmp_path)
    monkeypatch.setattr(release_app, "image_records", lambda folder: records)
    window = make_window(tmp_path)
    window.preferences.setValue("settings", '{"layer": 1, "datatype": 0, "detection": "otsu"}')
    window.open_folder(str(tmp_path))
    wait(window)
    view = window.stitching_view
    assert view.design_button.isEnabled() and not view.design_box.isEnabled()
    window.load_design()
    wait(window)
    r = window.result
    assert [k for k, _ in view.design_items] == [0, 1, 2, 3] and view.design_box.isChecked()
    assert all(mask.max() == 1 and mask.mean() > 0.05 for mask in window.design)  # the contacts are filled
    for k, item in view.design_items:  # each at its file's position, in its stripe's colour
        rect = item.mapRectToParent(item.boundingRect())
        np.testing.assert_allclose([rect.center().x(), rect.center().y()],
                                   r.oas_centers[k] - view.origin + view.shifts["stitched"][1][k], atol=1)
        color = QtGui.QColor(style.STRIPES[int(r.stripe[k]) % len(style.STRIPES)])
        assert tuple(item.lut[1][:3]) == (color.red(), color.green(), color.blue())
    reopened, _ = load_cache(tmp_path, str(tmp_path / "user cache"))
    assert reopened.oas_paths == r.oas_paths and np.allclose(reopened.oas_centers, r.oas_centers)


def write_layers(path, layers, empty=()):
    """An .oas with one box on each (layer, datatype) of `layers`, and the layers `empty` defined without shapes."""
    import klayout.db as kdb
    layout = kdb.Layout()
    top = layout.create_cell("TOP")
    for layer, datatype in layers:
        top.shapes(layout.layer(layer, datatype)).insert(kdb.Box(0, 0, 40, 40))
    for layer, datatype in empty:
        layout.layer(layer, datatype)
    layout.write(str(path))


def test_the_contact_layer_is_detected_in_the_oas_files(tmp_path):
    from types import SimpleNamespace
    from affine_ransac.analysis import detect_layer
    def records(*paths):
        return [SimpleNamespace(oas_path=p) for p in paths] + [SimpleNamespace(oas_path=None)]
    write_layers(tmp_path / "a.oas", [(7, 0)], empty=[(1, 0)])  # an empty layer does not count
    write_layers(tmp_path / "b.oas", [(7, 0)])
    assert detect_layer(records(tmp_path / "a.oas", tmp_path / "b.oas")) == (7, 0)

    write_layers(tmp_path / "c.oas", [(7, 0), (8, 1)])
    with pytest.raises(ValueError, match=r"2 layers.*7/0 \(e.g. a.oas\), 8/1 \(e.g. c.oas\)"):
        detect_layer(records(tmp_path / "a.oas", tmp_path / "c.oas"))
    write_layers(tmp_path / "d.oas", [], empty=[(1, 0)])
    with pytest.raises(ValueError, match="No .oas file holds any shape"):
        detect_layer(records(tmp_path / "d.oas"))


def test_an_image_run_uses_the_detected_layer_whatever_the_settings_say(tmp_path):
    records = write_image_folder(tmp_path)  # contacts on layer 1/0
    lines = []
    result = analyse_folder(tmp_path, "images", Settings(layer=100, datatype=5, detection="otsu"), records,
                            log=lines.append)
    assert (result.settings.layer, result.settings.datatype) == (1, 0)
    assert lines[0].startswith("Contact layer 1/0")


def test_release_defaults(tmp_path):
    window = make_window(tmp_path)
    assert window.settings().max_match_nm == 100.0
    assert "layer" not in PARAMETERS["images"] and "datatype" not in PARAMETERS["images"]  # detected


def test_the_scan_direction_comes_from_the_file_numbers():
    from affine_ransac.pipeline import tile_stripes
    from affine_ransac.release.analysis import file_numbers, scan_direction
    # 2 stripes x 3 images, serpentine: stripe 1 measured downward (CD1 at the top), stripe 2 upward.
    centers = np.array([(0, 2), (0, 1), (0, 0), (1, 0), (1, 1), (1, 2)], dtype=float) * 2600
    names = [f"CD{k:06d}" for k in range(1, 7)]
    lines = []
    assert scan_direction(names, centers, tile_stripes(centers), True, lines.append) is False
    assert lines == ["Scan: first stripe measured downward (file numbers)"]

    lines = []  # every stripe upward: a raster, not a serpentine
    raster = np.array([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], dtype=float) * 2600
    assert scan_direction(names, raster, tile_stripes(raster), False, lines.append) is True
    assert lines[1].startswith("!! Stripe(s) [2] do not alternate")

    assert file_numbers(["CD7", "CD10"]).tolist() == [7, 10]
    assert file_numbers(["CD7", "tile"]) is None and file_numbers(["A1", "B1"]) is None
    lines = []
    assert scan_direction(["a", "b", "c", "d", "e", "f"], centers, tile_stripes(centers), True,
                          lines.append) is True  # fallback
    assert lines[0].startswith("!! Scan direction not known")


def test_a_run_stores_the_detected_direction(tmp_path):
    write_contour_folder(tmp_path)  # CD000000..: each stripe from its bottom up
    lines = []
    with pytest.warns(UserWarning):
        result = analyse_folder(tmp_path, "contour CSV", Settings(first_stripe_upward=False), log=lines.append)
    assert result.settings.first_stripe_upward is True
    assert "Scan: first stripe measured upward (file numbers)" in lines
    assert "first_stripe_upward" not in PARAMETERS["images"] + PARAMETERS["contour CSV"]


def test_grid_means_average_the_points_of_each_cell():
    from affine_ransac.registration import grid_means
    points = np.array([[0.0, 0.0], [5.0, 2.0], [12.0, 1.0], [1.0, 25.0]])
    values = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0], [7.0, 40.0]])
    corner, mean, count = grid_means(points, values, 10.0)
    assert corner.tolist() == [0.0, 0.0] and mean.shape == (3, 2, 2)
    assert count.tolist() == [[2, 1], [0, 0], [1, 0]]
    np.testing.assert_allclose(mean[0, 0], [2.0, 15.0])
    np.testing.assert_allclose(mean[0, 1], [5.0, 30.0])
    assert np.isnan(mean[1, 0]).all()


def test_grid_average_grid_lines_and_one_graph_images(tmp_path):
    from affine_ransac.registration import grid_means
    style.setup()
    pg.mkQApp()
    write_contour_folder(tmp_path)
    with pytest.warns(UserWarning):
        result = analyse_folder(tmp_path, "contour CSV", Settings(), log=lambda line: None)
    view = RegistrationView(result, use_opengl=False)
    view.resize(1400, 900)
    view.show()
    QtWidgets.QApplication.processEvents()

    view.cell.setValue(0.5)
    view.values.setCurrentIndex(1)  # grid average
    _, mean, count = grid_means(result.design_nm, result.moving.residuals, 500.0)
    assert view.maps[0][1].image.shape[:2] == mean.shape[:2]  # one image pixel per cell
    assert arrow_points(view) == 6 * (count > 0).sum()  # one arrow per cell with contacts
    view.values.setCurrentIndex(0)
    assert arrow_points(view) == 6 * len(result.design_nm)
    assert not view.cell.isEnabled()

    assert view.grid_lines.isChecked()  # on at start, on every map
    assert all(panel.map.getAxis("bottom").grid for panel in view.panels)
    view.grid_lines.setChecked(False)
    assert not view.arrow_panel.map.getAxis("bottom").grid

    figure, graphs = view.shown_graphs()
    names = [name for name, _ in graphs]
    assert figure is view.figure and names[:6] == ["arrow map", "dr mean per row", "dr mean per column",
                                                   "dx map", "dx mean per row", "dx mean per column"]
    assert dict(graphs)["arrow map"][-1] is view.dr_bar
    map_items = dict(graphs)["dx map"]
    assert map_items[-1] is view.color_bar  # a map comes with its colour bar
    image = style.render_graph(figure, map_items)
    on_screen = sum(figure.mapFromScene(item.sceneBoundingRect()).boundingRect().width() for item in map_items)
    assert image.width() == 2 * on_screen
    menu = [a.text() for a in view.maps[0][0].rows.vb.menu.actions()]
    assert "Copy this graph" in menu and "Save this graph..." in menu
    view.close()


def test_contacts_belong_to_the_stripes_whose_x_range_holds_them():
    from affine_ransac.pipeline import contact_stripes
    centers = np.array([[0.0, 0.0], [0.0, 100.0], [90.0, 0.0], [90.0, 100.0]])  # two stripes, 10 nm overlap
    sizes = np.full((4, 2), 100.0)
    x = np.array([-40.0, 42.0, 45.0, 100.0, 200.0])  # stripe 0, overlap, overlap, stripe 1, outside (nearest: 1)
    assert contact_stripes(x, centers, sizes, np.array([0, 0, 1, 1])).tolist() == [[True, False], [True, True], [True, True],
                                                           [False, True], [False, True]]


def test_the_registration_tab_shows_the_stripes_ticked(tmp_path):
    from affine_ransac.pipeline import contact_stripes
    style.setup()
    pg.mkQApp()
    write_contour_folder(tmp_path)
    with pytest.warns(UserWarning):
        result = analyse_folder(tmp_path, "contour CSV", Settings(), log=lambda line: None)
    view = RegistrationView(result, use_opengl=False)
    stripes = contact_stripes(result.design_nm[:, 0], result.centers, result.sizes, result.stripe)
    assert stripes.shape[1] == 2 and result.stripe.tolist() == [0, 0, 0, 1, 1, 1] and view.stripes.text() == "Stripes: all 2"
    assert len(view.error_nm) == len(result.design_nm)

    view.stripes.set_selected([1])
    assert view.stripes.text() == "Stripes: 1 of 2"
    np.testing.assert_array_equal(view.design_nm, result.design_nm[stripes[:, 1]])
    np.testing.assert_array_equal(view.error_nm, result.moving.residuals[stripes[:, 1]])  # not fitted again
    assert arrow_points(view) == 6 * stripes[:, 1].sum()

    view.stripes.list.item(1).setCheckState(QtCore.Qt.CheckState.Unchecked)  # the last one stays ticked
    assert view.stripes.selected() == [1]
    view.stripes.set_selected(range(2))
    assert len(view.error_nm) == len(result.design_nm)
    view.close()


def test_contact_sizes_are_measured_on_the_tiles_kept_and_saved_with_the_result(tmp_path):
    records = write_image_folder(tmp_path)
    result = analyse_folder(tmp_path, "images", IMAGE_SETTINGS, records, log=lambda line: None)
    assert result.tiles is None  # not kept unless asked
    with pytest.raises(ValueError, match="Contact size tab on"):
        measure_contact_sizes(result)

    result = analyse_folder(tmp_path, "images", IMAGE_SETTINGS, records, log=lambda line: None, keep_tiles=True)
    steps = []
    sizes = measure_contact_sizes(result, 0.0, lambda done, total: steps.append((done, total)))

    # One row per contact: SEM discs of 40 nm, design squares of 40 nm (as in the research app's test).
    assert steps == [(k, 4) for k in range(1, 5)]
    assert sizes.sem.shape == sizes.design.shape == (len(result.design_nm), 3)
    np.testing.assert_allclose(sizes.design, np.tile([1600.0, 40.0, 40.0], (len(sizes.design), 1)))
    np.testing.assert_allclose(np.median(sizes.sem, axis=0), [np.pi * 20 ** 2, 40, 40], rtol=0.1)

    save_cache(replace(result, contact_sizes=sizes), tmp_path / "user")
    loaded, _ = load_cache(tmp_path, tmp_path / "user")
    np.testing.assert_array_equal(loaded.contact_sizes.sem, sizes.sem)
    np.testing.assert_array_equal(loaded.contact_sizes.design, sizes.design)
    assert loaded.contact_sizes.roi_width_nm == 0.0 and loaded.tiles is None  # the tiles are never saved


def test_the_contact_size_maps_show_any_of_area_xcd_ycd_side_by_side(tmp_path):
    style.setup()
    pg.mkQApp()
    design = np.array([(x, y) for x in np.arange(0, 20_000, 100.0) for y in np.arange(0, 10_000, 100.0)])
    design_sizes = np.tile([1600.0, 40.0, 40.0], (len(design), 1))
    sem_sizes = design_sizes + [100.0, 2.0, -1.0]
    view = ContactSizeView(design, ContactSizes(sem_sizes, design_sizes, 5.0), use_opengl=False)
    view.resize(1500, 700)
    view.show()
    QtWidgets.QApplication.processEvents()
    plots = [plot for plot, _, _ in view.maps]
    assert [plot.isVisible() for plot in plots] == [True, False, False]  # the area at start, alone
    alone = plots[0].width()
    y_span = np.ptp(plots[0].vb.viewRange()[1])
    assert y_span < 1.2 * np.ptp(view.position_um[:, 1])  # on the contacts

    for _ in range(3):  # the maps keep their scale, however often they are shown and hidden
        view.show_boxes[1].setChecked(True)
        QtWidgets.QApplication.processEvents()
        view.show_boxes[1].setChecked(False)
        QtWidgets.QApplication.processEvents()
    view.show_boxes[1].setChecked(True)
    QtWidgets.QApplication.processEvents()
    assert [plot.isVisible() for plot in plots] == [True, True, False]
    assert plots[0].width() < 0.6 * alone  # side by side, no room left for the hidden one
    for plot in plots[:2]:
        assert np.ptp(plot.vb.viewRange()[1]) < 1.1 * y_span
    np.testing.assert_allclose(view.values(1), sem_sizes[:, 1])
    view.draw_maps()
    image = view.maps[0][1].image
    assert (image[..., :3] == 255).all(axis=-1).any()  # white between the contacts

    view.difference_button.setChecked(True)
    np.testing.assert_allclose(view.values(0), 100.0)
    assert "Area, SEM − design" in plots[0].titleLabel.text
    figure, graphs = view.shown_graphs()
    assert figure is view.figure and [name for name, _ in graphs] == ["Area map", "xCD map"]
    assert dict(graphs)["xCD map"] == [plots[1], view.maps[1][2]]  # a map comes with its colour bar
    assert "a 5 nm band of chords" in view.summary.text()
    view.close()

    write_contour_folder(tmp_path)
    with pytest.warns(UserWarning):
        result = analyse_folder(tmp_path, "contour CSV", Settings(), log=lambda line: None)
    tab = ContactSizeTab(result, use_opengl=False)
    assert tab.note.text() == NOT_IMAGES and not tab.run_button.isEnabled()


def test_the_contact_size_tab_is_hidden_until_turned_on_and_measures_on_the_tiles_of_a_run(tmp_path, monkeypatch):
    records = write_image_folder(tmp_path)
    monkeypatch.setattr(release_app, "image_records", lambda folder: records)
    window = make_window(tmp_path)
    window.preferences.setValue("settings", '{"layer": 1, "datatype": 0, "detection": "otsu"}')
    window.open_folder(str(tmp_path))
    wait(window)
    assert not window.size_action.isChecked() and window.centralWidget().count() == 2  # hidden by default
    assert window.result.tiles is None  # so the run did not keep its tiles

    window.size_action.setChecked(True)
    tabs, tab = window.centralWidget(), window.size_tab
    assert tabs.tabText(2) == "Contact size" and tabs.currentWidget() is tab
    assert not tab.run_button.isEnabled() and tab.note.text() == NEEDS_TILES
    assert MaskRegWindow(window.preferences, cache_dir=str(tmp_path / "user cache")).size_action.isChecked()  # kept

    window.restart()  # with the tab on, the run keeps its tiles
    wait(window)
    tab = window.size_tab
    assert window.result.tiles is not None and tab.run_button.isEnabled()
    tab.roi.setValue(5.0)
    tab.run_button.click()
    assert window.busy and not window.size_action.isEnabled()
    wait(window)
    assert tab.view is not None and window.result.contact_sizes.roi_width_nm == 5.0
    reopened, _ = load_cache(tmp_path, str(tmp_path / "user cache"))
    np.testing.assert_array_equal(reopened.contact_sizes.sem, window.result.contact_sizes.sem)

    measured = window.result
    window.open_folder(str(tmp_path))  # the saved result, with its sizes; the last run's tiles let go first
    assert measured.tiles is None
    wait(window)
    tab = window.size_tab
    assert tab.view is not None and tab.roi.value() == 5.0 and not tab.run_button.isEnabled()

    window.restart()
    wait(window)
    window.size_action.setChecked(False)  # hidden again: the tab goes, and the tiles with it
    assert window.size_tab is None and window.centralWidget().count() == 2 and window.result.tiles is None
