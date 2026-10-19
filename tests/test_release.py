"""The released analyzer (affine_ransac.release), on synthetic folders, without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import time
from dataclasses import replace

import cv2
import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from affine_ransac.analysis import ContactSizes, Settings, contrast_summary
from affine_ransac.io.sem_image import load_sem_image
from affine_ransac.pipeline import process_tile
from affine_ransac.registration import row_means
from affine_ransac.release import app as release_app, style
from affine_ransac.release.analysis import (DEFAULTS, PARAMETERS, AnalysisFailed, analyse_folder, measure_contact_sizes,
                                           read_contours, read_design_images, read_display_images)
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
    panel.map.vb.sigRangeChangedManually.emit([True, True])  # as a zoom with the mouse: the other maps follow
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


def settle():
    for _ in range(5):
        QtWidgets.QApplication.processEvents()


def assert_fitted(view):
    """Every map shows the full height of its contacts (extent) and no more, centred on them in x (D117)."""
    low, high = view.extent()
    for panel in view.panels:
        (x0, x1), (y0, y1) = panel.map.vb.viewRange()
        pixel = (y1 - y0) / panel.map.vb.height()
        assert y0 == pytest.approx(low[1], abs=pixel) and y1 == pytest.approx(high[1], abs=pixel)
        assert (x0 + x1) / 2 == pytest.approx((low[0] + high[0]) / 2, abs=pixel)


def test_view_all_fits_the_maps_to_the_height_of_their_contacts(tmp_path):
    """At start, after View All in any map and after a resize, every map shows the full height of its contacts,
    centred on them in x and as wide as the map is: a map's width does not depend on the contacts' shape. A zoom
    with the mouse is kept when the window is resized (D117)."""
    style.setup()
    pg.mkQApp()
    write_contour_folder(tmp_path)
    with pytest.warns(UserWarning):
        result = analyse_folder(tmp_path, "contour CSV", Settings(), log=lambda line: None)
    widths = []
    for design in (result.design_nm, result.design_nm[:, ::-1]):  # 2 stripes x 3 tiles: taller than wide; and wide
        view = RegistrationView(replace(result, design_nm=design), use_opengl=False)
        view.resize(1400, 900)
        view.show()
        settle()
        assert_fitted(view)
        widths.append(view.maps[0][0].map.vb.width())
        low, high = view.extent()
        assert np.ptp(view.position_um, axis=0) == pytest.approx(high - low - view.pitch_nm / 1000)  # half a pitch around
        panel = view.maps[0][0]

        panel.map.vb.scaleBy((0.5, 0.5))  # a zoom with the mouse
        panel.map.vb.sigRangeChangedManually.emit([True, True])
        zoomed = panel.map.vb.viewRange()
        view.resize(1300, 850)
        settle()
        assert not view.fitted and np.mean(panel.map.vb.viewRange()[1]) == pytest.approx(np.mean(zoomed[1]))
        panel.map.vb.menu.viewAll.trigger()  # View All in a colour map: every map fitted again
        settle()
        assert_fitted(view)
        view.resize(1600, 1000)
        settle()
        assert_fitted(view)
        view.close()
    assert widths[0] == pytest.approx(widths[1], rel=0.1)  # not shaped like the contacts (2 : 3 and 3 : 2)


def test_each_mean_per_row_takes_a_range_of_its_own(tmp_path):
    """Boxes just below each mean per row (D119): Auto at start, fitted to its values, the boxes showing its range;
    a range typed in sets that plot alone, kept when the stripes change; a zoom with the mouse there shows its range
    in its boxes; View All there: Auto again. The boxes lie over the figure, not in its images."""
    style.setup()
    pg.mkQApp()
    write_contour_folder(tmp_path)
    with pytest.warns(UserWarning):
        result = analyse_folder(tmp_path, "contour CSV", Settings(), log=lambda line: None)
    view = RegistrationView(result, use_opengl=False)
    view.resize(1400, 900)
    view.show()
    settle()
    plots = [panel.rows for panel in view.panels]  # dr, dx, dy
    x_range = lambda plot: list(plot.vb.viewRange()[0])
    shown = lambda boxes: [box.value() for box in boxes.boxes]
    image = style.render_image(view.figure)
    for plot, boxes in zip(plots, view.row_ranges):
        assert boxes.auto.isChecked() and shown(boxes) == pytest.approx(x_range(plot), abs=0.01)
        under = view.figure.mapFromScene(plot.sceneBoundingRect()).boundingRect()
        assert abs(boxes.y() - under.bottom()) <= 4 and under.left() < boxes.geometry().center().x() < under.right()
        area = boxes.geometry().translated(-view.figure.viewport().pos())  # in the figure's image, at 2x: blank
        area = area.intersected(QtCore.QRect(0, 0, image.width() // 2, image.height() // 2))
        assert all(image.pixelColor(2 * x, 2 * y).name() == "#ffffff"
                   for x in range(area.left(), area.right(), 4) for y in range(area.top(), area.bottom(), 4))

    before = [x_range(plot) for plot in plots]
    dx = view.row_ranges[1]
    dx.boxes[0].setValue(-3.0)
    dx.boxes[1].setValue(3.0)
    assert x_range(plots[1]) == pytest.approx([-3.0, 3.0]) and not dx.auto.isChecked()
    assert x_range(plots[0]) == pytest.approx(before[0]) and x_range(plots[2]) == pytest.approx(before[2])
    view.stripes.set_selected([1])
    assert x_range(plots[1]) == pytest.approx([-3.0, 3.0])

    dy = view.row_ranges[2]
    plots[2].vb.scaleBy((0.5, 1.0))  # a zoom with the mouse
    plots[2].vb.sigRangeChangedManually.emit([True, False])
    assert not dy.auto.isChecked() and shown(dy) == pytest.approx(x_range(plots[2]), abs=0.01)
    assert x_range(plots[1]) == pytest.approx([-3.0, 3.0])
    plots[2].vb.menu.viewAll.trigger()
    assert dy.auto.isChecked() and plots[2].vb.autoRangeEnabled()[0] and not dx.auto.isChecked()
    view.close()


def result_with_failures(folder):
    """The contour test folder analysed, with failures made up for the Stitching tab to draw (the stitching itself is
    left as it was): two measured overlaps and one design overlap not matched, one tile not stitched (design)."""
    from affine_ransac.pipeline import RejectedPair
    folder.mkdir(exist_ok=True)
    write_contour_folder(folder)
    with pytest.warns(UserWarning):
        result = analyse_folder(folder, "contour CSV", Settings(), log=lambda line: None)
    for p in result.stitch.pairs[:2]:
        result.stitch.rejected.append(RejectedPair(p.i, p.j, p.box, 9, 1, True))
    p = result.design_stitch.pairs[1]
    result.design_stitch.rejected.append(RejectedPair(p.i, p.j, p.box, 9, 0, True))
    result.design_stitch.corrections[5] = np.nan
    return result


def test_overlaps_not_matched_and_tiles_not_stitched_are_layers_of_their_own(tmp_path):
    """D107: on an OpenGL viewport pyqtgraph draws plot lines in their colour and width only, so the dashed overlaps
    not matched looked like the tiles not stitched. Now filled boxes in the colour of their stitching, and thick red
    outlines, drawn by QPainter; each its own check box, off at start."""
    pg.mkQApp()
    view = StitchingView(result_with_failures(tmp_path), use_opengl=False)
    measured, design = view.layers["Overlaps not matched"]
    tiles = view.layers["Tiles not stitched"]
    assert [len(item.opts["x0"]) for item in (measured, design, tiles)] == [2, 1, 1]
    for item, color in ((measured, style.SEM), (design, style.DESIGN)):
        assert isinstance(item, pg.BarGraphItem) and item.opts["brush"].color().name() == color
        assert not item.isVisible()
    assert isinstance(tiles, pg.BarGraphItem) and tiles.opts["pen"].widthF() == 3 and not tiles.isVisible()
    view.boxes["Overlaps not matched"].setChecked(True)
    assert measured.isVisible() and design.isVisible() and not tiles.isVisible()
    view.boxes["Tiles not stitched"].setChecked(True)
    assert tiles.isVisible()
    label = view.names[0]  # D108: dark text on a white label, readable over the SEM images
    assert label.color.name() == style.TEXT and label.fill.color().alpha() > 200
    view.close()


def test_the_stitching_status_says_how_serious_it_is(tmp_path):
    """D109: "Stitching incomplete" did not say whether the results could be trusted. Now, worst first: an error
    (overlaps paired one row off), a warning (tiles not stitched: not measured), or OK, with a note when overlaps were
    not matched (every tile still stitched)."""
    pg.mkQApp()

    def status(result):
        view = StitchingView(result, use_opengl=False)
        text = view.status.text()
        view.close()
        return text
    write_contour_folder(tmp_path)
    with pytest.warns(UserWarning):
        clean = analyse_folder(tmp_path, "contour CSV", Settings(), log=lambda line: None)
    text = status(clean)
    assert clean.slipped == [] and "Stitching OK.</b>" in text and "All 6 tiles stitched." in text

    result = result_with_failures(tmp_path / "failures")
    result.design_stitch.corrections[5] = 0.0  # every tile stitched: overlaps not matched only
    text = status(result)
    assert "Stitching OK, with overlaps not matched." in text and "2 measured and 1 design overlap(s)" in text
    assert "Warning" not in text and "Tick <i>Overlaps not matched</i> to see where" in text

    result.design_stitch.corrections[5] = np.nan
    text = status(result)
    assert "Warning: 1 of 6 tiles not stitched." in text and "Stitching OK" not in text
    assert "Tick <i>Overlaps not matched</i> and <i>Tiles not stitched</i>" in text

    result.slipped = [(0, 3)]
    text = status(result)
    assert text.index("Error: 1 overlap(s) paired one contact row off?") < text.index("Warning")
    assert "(CD000000 / CD000003)" in text

    result.slipped = None  # a run saved before the check was kept
    assert "saved before MaskReg kept the check" in status(result)


def corner_folder_view(tmp_path):
    """The image test folder analysed with min_matched 2, which leaves one design corner overlap not matched (tiles 1
    and 2: each .oas file has 2 contacts in it, they share 1), in a Stitching tab laid out offscreen."""
    records = write_image_folder(tmp_path)
    with pytest.warns(UserWarning, match="design stitching: every tile stitched; 1 overlapping pair"):
        result = analyse_folder(tmp_path, "images", replace(IMAGE_SETTINGS, min_matched=2), records,
                                log=lambda line: None)
    assert [(f.i, f.j) for f in result.design_stitch.rejected if f.failed] == [(1, 2)]
    view = StitchingView(result, use_opengl=False)
    view.resize(1200, 800)
    view.show()
    pg.mkQApp().processEvents()
    return result, view


def click(view, x_nm, y_nm):
    """A left click on the plot at mask (x, y) nm, as the scene delivers it."""
    from types import SimpleNamespace
    scene = view.plot.vb.mapViewToScene(QtCore.QPointF(x_nm - view.origin[0], y_nm - view.origin[1]))
    view._clicked(SimpleNamespace(button=lambda: QtCore.Qt.MouseButton.LeftButton, double=lambda: False,
                                  scenePos=lambda: scene))


def test_a_click_on_a_tile_shows_which_image_it_is(tmp_path):
    """D112: a click shows the tile there alone (outline and name), its SEM image in front of the others, and beside
    the plot what is wrong with it in words; a click beside every tile clears it. A click on a legend entry unticks its
    layer."""
    pg.mkQApp()
    result, view = corner_folder_view(tmp_path)
    view.set_images(read_display_images(result.image_paths))
    assert view.selection is None and view.details.isHidden()

    click(view, 0.0, -700.0)  # tile 1 only
    assert view.selection == ("tile", 1) and not view.details.isHidden()
    boxes = [item for item in view.selection_items if isinstance(item, pg.BarGraphItem)]
    labels = [item.toPlainText() for item in view.selection_items if isinstance(item, pg.TextItem)]
    assert len(boxes) == 2 and all(len(b.opts["x0"]) == 1 for b in boxes) and labels == ["T1"]  # its outline only
    assert [item.zValue() for item in view.image_items] == [-10, -9, -10, -10]  # its image in front
    text = view.details.toPlainText()
    assert text.startswith("T1\nstripe 1, centre (0.000, -0.600) µm")
    assert "Not matched, so not used to stitch" in text and "T2 (upper right), design: show why" in text
    assert "Used: T0 (above), T3 (right)" in text
    click(view, 330.0, -100.0)  # in tiles 0 and 2: tile 2's centre is the nearer
    assert view.selection == ("tile", 2) and [item.zValue() for item in view.image_items] == [-10, -10, -9, -10]
    click(view, -200.0, 200.0)  # tile 0: its corner with tile 3 has too few contacts, which is not a fault
    text = view.details.toPlainText()
    assert "Nothing wrong: every overlap with enough contacts was matched." in text
    assert "Used: T1 (below), T2 (right)" in text and "Skipped, too few contacts (fewer than min_matched): T3" in text

    (x0, x1), (y0, y1) = view.plot.vb.viewRange()  # inside the plot, at its left edge: beside every tile
    beside = np.array([x0 + 0.01 * (x1 - x0), (y0 + y1) / 2]) + view.origin
    assert not np.any(np.all(np.abs(result.centers - beside) <= result.sizes / 2, axis=1))
    click(view, *beside)
    assert view.selection is None and view.details.isHidden() and view.selection_items == []
    assert all(item.zValue() == -10 for item in view.image_items)

    sample, label = view.plot.legend.items[0]
    assert label.text.startswith("Tile outlines")
    from types import SimpleNamespace
    sample.mouseClickEvent(SimpleNamespace(button=lambda: QtCore.Qt.MouseButton.LeftButton, accept=lambda: None))
    pg.mkQApp().processEvents()
    assert not view.boxes["Tile outlines"].isChecked() and not view.layers["Tile outlines"].isVisible()
    view.close()


def test_a_box_not_matched_shows_why(tmp_path):
    """D112: with Overlaps not matched shown, a click in its box (or "show why" beside a tile) shows only its two tiles
    and each one's contacts in it, filled for the first and hollow for the second, named in the legend, and explains
    the verdict in words: here each file has 2 contacts in the corner but they share 1, and 2 pairs are needed."""
    pg.mkQApp()
    result, view = corner_folder_view(tmp_path)
    assert "(all at corners, where two images share only a few contacts)" in view.status.text()
    click(view, 300.0, -300.0)  # the corner box, while the layer is off: a tile
    assert view.selection[0] == "tile"
    view.boxes["Overlaps not matched"].setChecked(True)
    click(view, 300.0, -300.0)
    assert view.selection == ("overlap", 1, 2)

    marks = {name: np.column_stack(item.getData()) + view.origin for item, name in view.selection_legend}
    assert sorted(marks) == ["T1: design contacts in the overlap", "T2: design contacts in the overlap"]
    np.testing.assert_allclose(sorted(marks["T1: design contacts in the overlap"].tolist()),
                               [[270.0, -360.0], [270.0, -270.0]], atol=1e-6)  # T1's file only has the first
    np.testing.assert_allclose(sorted(marks["T2: design contacts in the overlap"].tolist()),
                               [[270.0, -270.0], [360.0, -270.0]], atol=1e-6)  # T2's file only has the second
    legend = [label.text for _, label in view.plot.legend.items]
    assert "T1: design contacts in the overlap" in legend and "T2: design contacts in the overlap" in legend
    text = view.details.toPlainText()
    assert text.startswith("Overlap T1 – T2") and "a corner overlap" in text
    assert "Design: not matched. 1 pair(s) found, 2 needed (min_matched)" in text
    assert "T1 has 2 contact(s) in the overlap (■), T2 has 2 (□)" in text and "Measured: skipped" in text

    view._link(QtCore.QUrl("tile:2"))  # the links beside the plot
    assert view.selection == ("tile", 2) and view.selection_legend == []
    view.boxes["Overlaps not matched"].setChecked(False)
    view._link(QtCore.QUrl("overlap:2:1"))  # "show why": the overlap, shown whole even with the layer off
    assert view.selection == ("overlap", 2, 1)
    (x0, x1), (y0, y1) = view.plot.vb.viewRange()
    corner = np.array([300.0, -300.0]) - view.origin
    assert x0 < corner[0] < x1 and y0 < corner[1] < y1 and y1 - y0 < 1000.0
    view.close()


def test_the_contours_are_found_again_as_the_run_found_them(tmp_path):
    """D113: read_contours detects and refines every image again with the run's settings: the same contacts per tile,
    and the centres the run used (refined) are exactly the run's. An image changed since the run is named with !!."""
    records = write_image_folder(tmp_path)
    result = analyse_folder(tmp_path, "images", IMAGE_SETTINGS, records, log=lambda line: None)
    lines = []
    c = read_contours(result, log=lines.append)

    assert lines == [] and np.bincount(c.tile).tolist() == [len(p) for p in result.sem_points]
    for k in range(4):
        np.testing.assert_array_equal(c.refined_centres[c.tile == k] + result.centers[k], result.sem_points[k])
    shift = np.linalg.norm(c.refined_centres - c.detected_centres, axis=1)
    assert 0 < np.median(shift) < 1.0  # the two methods differ, a little on these clean images
    first = c.detected[c.start[0]:c.start[1]]
    assert np.array_equal(first[0], first[-2]) and np.isnan(first[-1]).all()  # closed, then a NaN row
    assert c.start[-1] == len(c.detected) == len(c.refined) and int(np.isnan(c.detected[:, 0]).sum()) == len(c.tile)

    cv2.imencode(".jpg", np.full((512, 512), 110, np.uint8))[1].tofile(str(records[2].image_path))  # blank now
    lines = []
    read_contours(result, log=lines.append)
    assert lines == ["!! The contours of 1 tile(s) do not give the centres of the run (an image changed since the "
                     "run?): ['T2']"]


def test_detected_and_refined_contours_and_centres_are_layers(tmp_path, monkeypatch):
    """D113: once loaded, four layers like the others (ticked then): the contours and centres from the detection and
    after refinement, at the placement shown. Contours are drawn for the contacts in view only, none if more than
    MAX_CONTOURS are, with a note to zoom in."""
    from affine_ransac.release import views
    pg.mkQApp()
    records = write_image_folder(tmp_path)
    result = analyse_folder(tmp_path, "images", IMAGE_SETTINGS, records, log=lambda line: None)
    view = StitchingView(result, use_opengl=False, on_load_contours=lambda: None)
    view.resize(1200, 800)
    view.show()
    assert view.contours_button.isEnabled()
    assert not any(view.boxes[name].isEnabled() for name in views.CONTOURS) and "Detected contours" not in view.layers

    view.set_contours(read_contours(result))
    assert all(view.boxes[name].isEnabled() and view.boxes[name].isChecked() for name in views.CONTOURS)
    assert not view.contours_button.isEnabled()
    legend = [label.text for _, label in view.plot.legend.items]
    assert legend[-4:] == ["Detected contours (otsu)", "Refined contours", "Detected centres (otsu)", "Refined centres"]
    assert "Measured (SEM) centres, refined" in legend
    x, _ = view.layers["Detected contours"].getData()
    assert int(np.isnan(x).sum()) == len(view.contours.tile)  # all 225 contacts in view: each contour drawn
    refined = np.column_stack(view.layers["Refined centres"].getData())
    np.testing.assert_allclose(refined + view.origin,
                               np.concatenate(result.sem_points) + view.shifts["stitched"][0][view.contours.tile])

    view.placement.setCurrentIndex(1)  # nominal: the centres are the run's, as measured
    np.testing.assert_allclose(np.column_stack(view.layers["Refined centres"].getData()) + view.origin,
                               np.concatenate(result.sem_points))
    monkeypatch.setattr(views, "MAX_CONTOURS", 100)
    view.update_contours()
    x, _ = view.layers["Detected contours"].getData()
    assert x is None or len(x) == 0  # pyqtgraph gives None for no data
    assert view.contours_note.text() == "Zoom in to see the contours"
    view.plot.vb.setRange(xRange=(-100.0, 100.0), yRange=(-100.0, 100.0), padding=0)  # a few contacts
    view.update_contours()
    x, _ = view.layers["Refined contours"].getData()
    assert 0 < int(np.isnan(x).sum()) <= 100 and view.contours_note.text() == ""
    view.boxes["Refined contours"].setChecked(False)
    assert not view.layers["Refined contours"].isVisible() and view.layers["Detected contours"].isVisible()
    view.close()


def test_contours_are_loaded_on_request_and_kept(tmp_path, monkeypatch):
    records = write_image_folder(tmp_path)
    monkeypatch.setattr(release_app, "image_records", lambda folder: records)
    window = make_window(tmp_path)
    window.preferences.setValue("settings", '{"layer": 1, "datatype": 0, "detection": "otsu"}')
    window.open_folder(str(tmp_path))
    wait(window)
    assert window.stitching_view.contours_button.isEnabled()
    window.load_contours()
    wait(window)
    assert window.contours is not None and window.stitching_view.boxes["Refined contours"].isChecked()
    window.set_gpu(False)  # drawn again: the contours loaded are kept
    assert window.stitching_view.contours is window.contours
    assert not window.stitching_view.contours_button.isEnabled()


def test_the_overlaps_paired_one_row_off_are_saved_with_the_run(tmp_path):
    write_contour_folder(tmp_path)
    with pytest.warns(UserWarning):
        result = analyse_folder(tmp_path, "contour CSV", Settings(), log=lambda line: None)
    result.slipped = [(0, 3), (1, 4)]
    save_cache(result, tmp_path / "user")
    assert load_cache(tmp_path, tmp_path / "user")[0].slipped == [(0, 3), (1, 4)]
    result.slipped = None  # a run saved before D109: not known
    save_cache(result, tmp_path / "user", backup=False)
    assert load_cache(tmp_path, tmp_path / "user")[0].slipped is None


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


def test_tone_reversed_design_files_show_their_contacts_not_a_block(tmp_path):
    """D106: a tone-reversed .oas draws the area around the holes as one shape; filling it showed a solid block.
    Load design now shows the holes, the contacts the analysis read."""
    from test_pipeline import LATTICE, NOMINAL, visible, write_design
    records = write_image_folder(tmp_path)
    normal = [str(r.oas_path) for r in records]
    for k, r in enumerate(records):  # the same contacts, drawn as the frame minus the holes
        r.oas_path = tmp_path / f"{k}_reversed.oas"
        write_design(r.oas_path, visible(LATTICE - NOMINAL[k], margin=0), reversed=True)
    result = analyse_folder(tmp_path, "images", IMAGE_SETTINGS, records, log=lambda line: None)
    assert len(result.design_nm) > 150  # the analysis reads them tone reversed

    masks = read_design_images(result)
    for drawn, expected in zip(masks, read_design_images(replace(result, oas_paths=normal))):
        assert 0.1 < drawn.mean() < 0.3  # contacts, not a block (it was 1.0)
        np.testing.assert_array_equal(drawn, expected)


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
    assert figure is view.figure and names[1:7] == ["arrow map", "dr mean per row", "dx map", "dx mean per row",
                                                    "dy map", "dy mean per row"]  # no mean per column: hidden
    # First (after Whole figure): the graphs of dx and dy as laid out, their colour bar, nothing of dr (D116).
    assert names[0] == "dx and dy: maps, profiles, colour bar"
    (area,) = dict(graphs)[names[0]]
    on_screen = lambda item: figure.mapFromScene(item.sceneBoundingRect()).boundingRect()
    image = style.render_graph(figure, [area])
    assert (image.width(), image.height()) == (2 * on_screen(area).width(), 2 * on_screen(area).height())
    assert all(on_screen(area).contains(on_screen(plot)) for plot in area.items)
    assert on_screen(area).contains(on_screen(view.color_bar))
    assert not any(on_screen(area).intersects(on_screen(item)) for item in
                   (view.arrow_panel.map, view.arrow_panel.rows, view.dr_bar))
    assert dict(graphs)["arrow map"][-1] is view.dr_bar
    map_items = dict(graphs)["dx map"]
    assert map_items[-1] is view.color_bar  # a map comes with its colour bar
    image = style.render_graph(figure, map_items)
    on_screen = sum(figure.mapFromScene(item.sceneBoundingRect()).boundingRect().width() for item in map_items)
    assert image.width() == 2 * on_screen
    menu = [a.text() for a in view.maps[0][0].rows.vb.menu.actions()]
    assert "Copy this graph" in menu and "Save this graph..." in menu
    view.close()


def test_every_graph_has_a_copy_button_over_its_corner(tmp_path):
    """A Copy button over the top-left corner of every map and profile (D120), shown while its graph is: a map's
    copies it with its colour bar, a profile's the profile alone. Over the figure, not in its images."""
    style.setup()
    pg.mkQApp()
    write_contour_folder(tmp_path)
    with pytest.warns(UserWarning):
        result = analyse_folder(tmp_path, "contour CSV", Settings(), log=lambda line: None)
    view = RegistrationView(result, use_opengl=False)
    view.resize(1400, 900)
    view.show()
    settle()
    buttons = {button.plot: button for button in view.copy_buttons}
    graphs = [(plot, items) for panel in view.panels for plot, items, _ in view.graphs_of(panel)]
    assert len(buttons) == len(graphs) == 9
    image = style.render_image(view.figure)
    for plot, items in graphs:
        button = buttons[plot]
        assert button.isVisible() == plot.isVisible() == (plot not in [panel.columns for panel in view.panels])
        if not plot.isVisible():
            continue
        corner = view.figure.mapFromScene(plot.sceneBoundingRect()).boundingRect().topLeft()
        assert (button.geometry().topLeft() - corner).manhattanLength() <= 6  # over its top-left corner
        area = button.geometry().translated(-view.figure.viewport().pos())
        assert all(image.pixelColor(2 * x, 2 * y).name() == "#ffffff"  # not in the figure's image
                   for x in range(area.left(), area.right(), 3) for y in range(area.top(), area.bottom(), 3))
        button.click()
        copied = QtWidgets.QApplication.clipboard().image()
        assert copied.size() == style.render_graph(view.figure, items).size()
    assert buttons[view.maps[0][0].map].toolTip().startswith("Copy the dx map")
    assert view.graphs_of(view.maps[0][0])[0][1] == [view.maps[0][0].map, view.color_bar]  # with its colour bar

    view.columns_box.setChecked(True)
    settle()
    assert all(buttons[panel.columns].isVisible() for panel in view.panels)
    view.close()


def test_symbol_size_and_line_width(tmp_path):
    """Symbol size: Auto (a disk of 0.8 x the pitch) or every contact that many pixels across at every zoom, off in
    grid average; line width: the lines of every profile (D121)."""
    from affine_ransac.view_registration import rasterize
    one = rasterize(np.array([[5.5, 5.5]]), np.array([1.0]), (0, 0, 11, 11), (11, 11), 1.5)
    assert np.isfinite(one).sum() == 9  # a radius of 1.5 px: 3 x 3 pixels (a symbol 4 px across)
    style.setup()
    pg.mkQApp()
    write_contour_folder(tmp_path)
    with pytest.warns(UserWarning):
        result = analyse_folder(tmp_path, "contour CSV", Settings(), log=lambda line: None)
    view = RegistrationView(result, use_opengl=False)
    view.resize(1400, 900)
    view.show()
    settle()
    assert view.symbol_size.value() == 0 and view.symbol_size.text() == "Auto"
    vb = view.maps[0][0].map.vb
    centre = view.position_um.mean(axis=0)
    vb.setRange(xRange=(centre[0] - 0.5, centre[0] + 0.5), yRange=(centre[1] - 0.5, centre[1] + 0.5), padding=0)
    covered = []
    for size in (0, 3, 9):  # Auto, then 3 and 9 px
        view.symbol_size.setValue(size)
        view.draw_maps()
        covered.append(np.mean(~(view.maps[0][1].image[..., :3] == 255).all(axis=-1)))
    assert covered[1] < covered[2] < covered[0]  # zoomed in: 3 px dots < 9 px dots < disks of 0.8 x the pitch
    view.values.setCurrentIndex(1)
    assert not view.symbol_size.isEnabled()

    view.line_width.setValue(5)
    assert [line.opts["pen"].width() for lines in view.map_profiles + view.arrow_profiles for line in lines] == [5] * 6
    view.close()


def test_the_means_per_column_are_hidden_unless_ticked(tmp_path):
    """Hidden at start: out of the layout, so that the maps take their height, and out of the image menus. Ticked:
    below their maps again, aligned with them (D118)."""
    style.setup()
    pg.mkQApp()
    write_contour_folder(tmp_path)
    with pytest.warns(UserWarning):
        result = analyse_folder(tmp_path, "contour CSV", Settings(), log=lambda line: None)
    view = RegistrationView(result, use_opengl=False)
    view.resize(1400, 900)
    view.show()
    settle()
    panel = view.maps[0][0]
    names = lambda: [name for name, _ in view.shown_graphs()[1]]
    assert not view.columns_box.isChecked() and not panel.columns.isVisible()
    assert not any("per column" in name for name in names())
    tall = panel.map.vb.height()

    view.columns_box.setChecked(True)
    settle()
    assert panel.columns.isVisible() and "dx mean per column" in names()
    assert panel.map.vb.height() < 0.9 * tall  # room for the means per column
    on_screen = lambda plot: view.figure.mapFromScene(plot.vb.sceneBoundingRect()).boundingRect()
    assert abs(on_screen(panel.columns).left() - on_screen(panel.map).left()) <= 1
    assert panel.columns.vb.viewRange()[0] == pytest.approx(panel.map.vb.viewRange()[0])  # the map's x

    view.columns_box.setChecked(False)
    settle()
    assert panel.map.vb.height() == pytest.approx(tall)
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


def write_folder_measuring_nothing(folder):
    """The image folder of write_image_folder with its tiles three times as far apart (no overlaps, so each stitching
    places only its first tile) and no .oas for the first tile (flagged): no tile can be measured."""
    records = write_image_folder(folder)
    for r in records:
        r.center_x_nm, r.center_y_nm = 3 * r.center_x_nm, 3 * r.center_y_nm
    records[0].oas_path = None
    return records


NOT_MEASURED = {0: "design tone not matched (flagged)", 1: "SEM tile not stitched", 2: "SEM tile not stitched",
                3: "SEM tile not stitched"}


def test_a_run_that_measures_no_contact_fails_with_what_it_found(tmp_path):
    records = write_folder_measuring_nothing(tmp_path)
    with pytest.warns(UserWarning), pytest.raises(AnalysisFailed, match="Only 0 measured contact") as failed:
        analyse_folder(tmp_path, "images", IMAGE_SETTINGS, records, log=lambda line: None)
    error = failed.value
    assert error.not_measured == NOT_MEASURED
    r = error.result
    assert r.moving is None and len(r.design_nm) == 0 and r.image_paths == [str(rec.image_path) for rec in records]
    assert all(len(points) > 40 for points in r.sem_points) and len(r.stitch.unplaced) == 3


def test_a_failed_run_shows_its_tiles_on_the_stitching_tab_and_saves_nothing(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    records = write_folder_measuring_nothing(data)
    monkeypatch.setattr(release_app, "image_records", lambda folder: records)
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args: shown.append(args[2]))
    window = make_window(tmp_path)
    window.preferences.setValue("settings", '{"layer": 1, "datatype": 0, "detection": "otsu"}')
    window.open_folder(str(data))
    wait(window)

    tabs = window.centralWidget()
    assert [tabs.tabText(k) for k in range(tabs.count())] == ["Stitching"]  # nothing measured, nothing else to show
    assert shown[0].startswith("Only 0 measured contact(s). The Stitching tab shows")
    assert "ANALYSIS FAILED, nothing saved" in window.status.idle.text() and window.actions["restart"].isEnabled()
    assert not (data / CACHE_NAME).exists() and window.recent_folders() == []

    view = window.stitching_view
    layer, box = view.layers["Tiles not measured"], view.boxes["Tiles not measured"]
    assert not box.isChecked() and not layer.isVisible()  # off at start, like the other failure markers
    box.setChecked(True)
    assert layer.isVisible() and len(layer.opts["x0"]) == 4  # four hatched boxes
    assert "4 of 4 tiles not measured" in view.status.text() and "SEM tile not stitched (3)" in view.status.text()
    assert view.tiles_at(window.result.centers[1]) == "T1 (not measured: SEM tile not stitched)"

    window.load_images()  # the SEM images and the design load as for any result
    wait(window)
    assert len(view.image_items) == 4
    window.load_design()
    wait(window)
    assert [k for k, _ in view.design_items] == [1, 2, 3]  # the first tile has no .oas

    window.set_gpu(False)  # drawn again: still the failed run, with its images and design
    assert window.centralWidget().count() == 1 and len(window.stitching_view.image_items) == 4
    assert "Tiles not measured" in window.stitching_view.boxes
    window.size_action.setChecked(True)  # no Contact size tab: no contact was measured
    assert window.centralWidget().count() == 1 and window.size_tab is None


def add_noise(records, sigma: float, seed: int = 0):
    """Gaussian noise of sigma grey levels added to the folder's images (saved again as JPEG), like a real SEM's."""
    rng = np.random.default_rng(seed)
    for r in records:
        image = load_sem_image(r.image_path).astype(float)
        noisy = np.clip(image + rng.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)
        cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(r.image_path))


def test_noisy_images_keep_their_contacts_by_default(tmp_path):
    """D103: on noisy images real contacts stand out by only a few noise sigma, so the contrast check at its old
    default (5) dropped every one. The check is off unless asked for, and says so (!!) when it empties tiles."""
    records = write_image_folder(tmp_path)
    add_noise(records, 30.0)  # the contacts stay plain to see
    settings = replace(DEFAULTS, layer=1, datatype=0, detection="otsu")
    assert settings.min_contrast == 0

    result = analyse_folder(tmp_path, "images", settings, records, log=lambda line: None)
    assert all(len(points) > 40 for points in result.sem_points) and len(result.design_nm) > 150

    with pytest.raises(ValueError, match="No design tone matched the SEM in the sample tiles"):
        analyse_folder(tmp_path, "images", replace(settings, min_contrast=5.0), records, log=lambda line: None)
    r = records[0]
    tile = process_tile(r.image_path, r.oas_path, (r.center_x_nm, r.center_y_nm), (r.fov_x_nm, r.fov_y_nm), (1, 0),
                        detection="otsu", min_contrast=5.0)
    assert tile.low_contrast > 40 and len(tile.refined.centers) == 0
    assert contrast_summary([tile], ["T0"], 5.0).startswith("!! Contrast check")
