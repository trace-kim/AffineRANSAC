"""The release's two tabs (docs/SPEC.md D77), pyqtgraph, presentation style (style.py).

StitchingView: tile outlines, overlaps used, design and measured (SEM) centres, at the nominal or the stitched
placement; stitching failures (off at start, D76) and tile names on request; the SEM images on request
(Load SEM images: the app reads them in the background and calls set_images).

RegistrationView: the error after the moving-window affine, measured (SEM) − design, of every contact: dx and
dy maps (jet, shared colour scale, rasterized to the screen like the research view; draw_map), the mean per
row after the affine and, as a reference, of the stitched contacts without any affine. Copy rows / Export CSV
give the row profile as a table.

Everything is drawn relative to an origin in whole µm (OpenGL works in float32); the axes show mask µm.
"""

import csv
import io

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from scipy.spatial import cKDTree

from affine_ransac.registration import error_summary, row_means
from affine_ransac.release import style
from affine_ransac.release.analysis import ReleaseResult
from affine_ransac.stitching import placement_corrections
from affine_ransac.view_points import box_outlines, tile_rectangles
from affine_ransac.view_registration import JET, draw_map

PLACEMENTS = {"stitched": "Stitched", "nominal": "Nominal"}


def mask_plot(origin_nm: np.ndarray, title: str = "", data_unit_nm: float = 1.0) -> pg.PlotItem:
    """A plot of data relative to origin_nm (data_unit_nm: see MaskAxis), axes in mask µm, equal x and y scales."""
    plot = pg.PlotItem(axisItems={"bottom": style.MaskAxis("bottom", origin_nm[0], data_unit_nm),
                                  "left": style.MaskAxis("left", origin_nm[1], data_unit_nm)})
    plot.setAspectLocked(True)
    style.style_plot(plot, title, "x (µm)", "y (µm)", grid=False)
    return plot


def figure(use_opengl: bool) -> pg.GraphicsLayoutWidget:
    widget = pg.GraphicsLayoutWidget()
    if use_opengl:
        widget.useOpenGL(True)
    widget.setBackground("w")
    return widget


def tile_image_item(image: np.ndarray, center_nm, fov_x_nm: float, full_shape) -> pg.ImageItem:
    """A (possibly downsampled) SEM image covering its tile: pixel size = FOV / image width of the full image
    (full_shape, as the pipeline), image centre at center_nm, image rows downwards (y up on the plot)."""
    pixel_nm = fov_x_nm / full_shape[1]
    width, height = full_shape[1] * pixel_nm, full_shape[0] * pixel_nm
    item = pg.ImageItem(image[::-1], axisOrder="row-major", levels=(0, 255))  # row 0 at the top
    item.setRect(QtCore.QRectF(center_nm[0] - width / 2, center_nm[1] - height / 2, width, height))
    item.setZValue(-10)
    return item


class StitchingView(QtWidgets.QWidget):
    def __init__(self, result: ReleaseResult, use_opengl: bool = True, on_load_images=None):
        """on_load_images(): called by Load SEM images (None, or a result without images: button disabled)."""
        super().__init__()
        self.result = result
        r = result
        self.origin = np.round(np.concatenate(r.design_points).mean(axis=0), -3)  # whole µm
        self.shifts = {"nominal": (np.zeros_like(r.centers), np.zeros_like(r.centers)),
                       "stitched": tuple(np.nan_to_num(c) for c in
                                         placement_corrections(r.stitch.corrections, r.design_stitch.corrections)["mean"])}
        self.image_items = []

        self.graphics = figure(use_opengl)
        self.plot = mask_plot(self.origin)
        self.graphics.addItem(self.plot)
        style.legend(self.plot)
        rectangles = [(x0 - self.origin[0], x1 - self.origin[0], y0 - self.origin[1], y1 - self.origin[1])
                      for x0, x1, y0, y1 in tile_rectangles(r.centers, r.sizes)]
        self.layers = {}
        self.layers["Tile outlines"] = self._boxes(rectangles, pg.mkPen(style.TILES, width=1), "Tile outlines")
        self.layers["Overlaps used"] = self._boxes([self._local(p.box) for p in r.stitch.pairs],
                                                   pg.mkPen(style.OVERLAPS, width=1.5), "Overlaps used")
        self.layers["Design centres"] = self.plot.plot(pen=None, symbol="o", symbolSize=4, symbolPen=None,
                                                       symbolBrush=style.DESIGN, name="Design centres")
        self.layers["Measured centres"] = self.plot.plot(pen=None, symbol="o", symbolSize=3, symbolPen=None,
                                                         symbolBrush=style.SEM, name="Measured (SEM) centres")
        failed = [self._local(f.box) for f in r.stitch.rejected + r.design_stitch.rejected if f.failed]
        unplaced = sorted(set(r.stitch.unplaced.tolist()) | set(r.design_stitch.unplaced.tolist()))
        dash = QtCore.Qt.PenStyle.DashLine
        failures = [self._boxes(failed, pg.mkPen(style.FAILURE, width=2, style=dash), "Overlaps not matched"),
                    self._boxes([rectangles[k] for k in unplaced], pg.mkPen(style.FAILURE, width=3), "Tiles not stitched")]
        self.layers["Stitching failures"] = failures
        self.names = []
        for name, center in zip(r.tile_ids, r.centers):
            text = pg.TextItem(name, color=style.MUTED, anchor=(0.5, 0.5))
            text.setPos(*(center - self.origin))
            self.plot.addItem(text)
            self.names.append(text)
        self.layers["Tile names"] = self.names

        self.boxes = {}
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Placement:"))
        self.placement = QtWidgets.QComboBox()
        self.placement.addItems(PLACEMENTS.values())
        self.placement.currentIndexChanged.connect(lambda _: self.update_positions())
        controls.addWidget(self.placement)
        controls.addSpacing(16)
        hidden = {"Stitching failures", "Tile names"}  # off at start (D76)
        for name in self.layers:
            box = QtWidgets.QCheckBox(name)
            box.setChecked(name not in hidden)
            box.toggled.connect(self.update_visibility)
            controls.addWidget(box)
            self.boxes[name] = box
        self.images_box = QtWidgets.QCheckBox("SEM images")
        self.images_box.setEnabled(False)
        self.images_box.toggled.connect(self.update_visibility)
        controls.addWidget(self.images_box)
        self.load_button = QtWidgets.QPushButton("Load SEM images")
        can_load = on_load_images is not None and bool(r.image_paths)
        self.load_button.setEnabled(can_load)
        self.load_button.setToolTip("Read the SEM images and show them under the centres" if can_load else
                                    "No SEM images: this folder holds contour CSVs" if not r.image_paths else
                                    "Not available")
        if can_load:
            self.load_button.clicked.connect(on_load_images)
        controls.addWidget(self.load_button)
        controls.addStretch(1)

        self.status = QtWidgets.QLabel(self.status_text())
        self.status.setWordWrap(True)
        self.cursor_label = QtWidgets.QLabel(" ")
        self.cursor_label.setStyleSheet(f"color: {style.MUTED}")
        self.plot.scene().sigMouseMoved.connect(self._show_cursor)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(style.figure_bar("Stitching", style.image_buttons(self.graphics, "stitching")))
        layout.addWidget(self.status)
        layout.addLayout(controls)
        layout.addWidget(self.graphics, stretch=1)
        layout.addWidget(self.cursor_label)
        self.update_positions()
        self.update_visibility()
        self.plot.autoRange()

    def _local(self, box):
        x0, x1, y0, y1 = box
        return x0 - self.origin[0], x1 - self.origin[0], y0 - self.origin[1], y1 - self.origin[1]

    def _boxes(self, boxes, pen, name):
        x, y = box_outlines(boxes)
        return self.plot.plot(x, y, pen=pen, connect="finite", name=f"{name} ({len(boxes)})" if boxes else None)

    def status_text(self) -> str:
        r, parts = self.result, []
        n = len(r.tile_ids)
        for name, stitch in (("Measured", r.stitch), ("Design", r.design_stitch)):
            failed = sum(f.failed for f in stitch.rejected)
            parts.append(f"{name}: {n - len(stitch.unplaced)} of {n} tiles stitched, {len(stitch.pairs)} overlaps used, "
                         f"{failed} not matched")
        text = "; ".join(parts) + f". Method: {r.settings.stitch_method}."
        incomplete = len(r.stitch.unplaced) or len(r.design_stitch.unplaced) or any(
            f.failed for f in r.stitch.rejected + r.design_stitch.rejected)
        if incomplete:
            return f"<span style='color:{style.FAILURE}'><b>Stitching incomplete.</b></span> {text} " \
                   f"Tick <i>Stitching failures</i> to see where."
        return text

    def mode(self) -> str:
        return list(PLACEMENTS)[self.placement.currentIndex()]

    def update_positions(self):
        """Put the centres (and images) at the chosen placement."""
        r = self.result
        sem_shift, design_shift = self.shifts[self.mode()]
        sem = np.concatenate([p + sem_shift[k] for k, p in enumerate(r.sem_points)]) - self.origin
        design = np.concatenate([p + design_shift[k] for k, p in enumerate(r.design_points)]) - self.origin
        self.layers["Measured centres"].setData(sem[:, 0], sem[:, 1])
        self.layers["Design centres"].setData(design[:, 0], design[:, 1])
        for k, item in enumerate(self.image_items):
            item.setPos(*sem_shift[k])

    def update_visibility(self, *_):
        for name, items in self.layers.items():
            for item in items if isinstance(items, list) else [items]:
                item.setVisible(self.boxes[name].isChecked())
        for item in self.image_items:
            item.setVisible(self.images_box.isChecked())

    def set_images(self, images: list[tuple[np.ndarray, tuple]]):
        """Show the SEM images: per tile (image, shape of the full image), from the app's loader."""
        for item in self.image_items:
            self.plot.removeItem(item)
        self.image_items = []
        for (image, full_shape), center, size in zip(images, self.result.centers, self.result.sizes):
            item = tile_image_item(image, center - self.origin, size[0], full_shape)
            self.plot.addItem(item)
            self.image_items.append(item)
        self.images_box.setEnabled(True)
        self.images_box.setChecked(True)
        self.load_button.setEnabled(False)
        self.update_positions()
        self.update_visibility()

    def _show_cursor(self, scene_pos):
        pos = self.plot.vb.mapSceneToView(scene_pos)
        x, y = (np.array([pos.x(), pos.y()]) + self.origin) / 1000
        self.cursor_label.setText(f"x = {x:.4f} µm, y = {y:.4f} µm (mask)")


def summary_text(error_nm: np.ndarray) -> str:
    s = error_summary(error_nm)
    return (f"mean dx {s['mean_x_nm']:+.2f} nm, dy {s['mean_y_nm']:+.2f} nm; 3σ dx {s['3sigma_x_nm']:.2f} nm, "
            f"dy {s['3sigma_y_nm']:.2f} nm; max |error| {s['max_nm']:.2f} nm")


class RegistrationView(QtWidgets.QWidget):
    def __init__(self, result: ReleaseResult, use_opengl: bool = True):
        super().__init__()
        r, s = result, result.settings
        self.result = r
        self.error_nm = r.moving.residuals
        self.origin = np.round(r.design_nm.mean(axis=0), -3)  # whole µm
        self.position_um = (r.design_nm - self.origin) / 1000
        distance, _ = cKDTree(r.design_nm).query(r.design_nm, k=2)
        self.dot_um = 0.4 * np.median(distance[:, 1]) / 1000  # a disk of ~0.4 x the pitch when zoomed in

        # Maps: dx and dy with one colour bar.
        self.maps_figure = figure(use_opengl)
        self.maps = []
        for component, name in enumerate(("Δx", "Δy")):
            plot = mask_plot(self.origin, f"{name}: measured − design (nm)", data_unit_nm=1000)  # maps in µm
            image = pg.ImageItem(axisOrder="row-major")
            plot.addItem(image)
            self.maps_figure.addItem(plot, row=0, col=component)
            self.maps.append((plot, image, component))
        limit = float(np.percentile(np.abs(self.error_nm), 99)) or 1.0
        self.color_bar = pg.ColorBarItem(values=(-limit, limit), colorMap=JET, width=18)
        axis = self.color_bar.getAxis("right")
        axis.setTextPen(pg.mkPen(style.TEXT))
        axis.setStyle(tickFont=QtGui.QFont(style.FONT, style.TICK_PT))
        axis.enableAutoSIPrefix(False)
        axis.setLabel("error (nm)", color=style.TEXT, **{"font-size": f"{style.LABEL_PT}pt"})
        self.color_bar.sigLevelsChanged.connect(lambda *_: self.draw_maps())
        self.maps_figure.addItem(self.color_bar, row=0, col=2)
        self._link_maps()
        low, high = self.position_um.min(axis=0), self.position_um.max(axis=0)
        self.maps[0][0].vb.setRange(xRange=(low[0], high[0]), yRange=(low[1], high[1]))
        self.redraw = QtCore.QTimer(self, singleShot=True, interval=50)
        self.redraw.timeout.connect(self.draw_maps)
        for plot, _, _ in self.maps:
            plot.vb.sigRangeChanged.connect(lambda *_: self.redraw.start())
            plot.vb.sigResized.connect(lambda *_: self.redraw.start())

        # Row profiles: after the moving-window affine, and stitched without any affine (reference).
        self.profile_figure = figure(use_opengl)
        y = r.design_nm[:, 1]
        self.row_y, self.row_mean, _ = row_means(y, self.error_nm, s.row_gap_nm)
        _, self.raw_mean, _ = row_means(y, r.sem_nm - r.design_nm, s.row_gap_nm)
        row_um = self.row_y / 1000
        self.row_plot = self.profile_figure.addPlot(row=0, col=0)
        self.raw_plot = self.profile_figure.addPlot(row=1, col=0)
        for plot, mean, title in ((self.row_plot, self.row_mean, "Mean error per row, after the moving-window affine"),
                                  (self.raw_plot, self.raw_mean, "Reference: mean error per row, stitched, no affine")):
            style.style_plot(plot, title, "row y (µm)", "error (nm)")
            style.legend(plot)
            style.zero_line(plot)
            for component, (name, color) in enumerate((("Δx", style.DX), ("Δy", style.DY))):
                plot.plot(row_um, mean[:, component], pen=pg.mkPen(color, width=2), name=name)
        self.raw_plot.setXLink(self.row_plot)

        window = f"moving-window affine: window {s.moving_window_um:g} µm, step {s.moving_step_um:g} µm"
        summary = QtWidgets.QLabel(f"<b>{len(self.error_nm)} contacts</b>, {len(self.row_y)} rows. Error = measured "
                                   f"(SEM) − design after the {window}: {summary_text(self.error_nm)}.")
        summary.setWordWrap(True)
        copy_rows = QtWidgets.QPushButton("Copy rows")
        copy_rows.setToolTip("Copy the row profile (y µm; dx, dy nm) to the clipboard, tab-separated (pastes into Excel)")
        copy_rows.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.rows_text("\t")))
        export_rows = QtWidgets.QPushButton("Export rows CSV...")
        export_rows.clicked.connect(self.export_rows)

        maps_panel, profile_panel = QtWidgets.QWidget(), QtWidgets.QWidget()
        maps_layout = QtWidgets.QVBoxLayout(maps_panel)
        maps_layout.addLayout(style.figure_bar("Error maps", style.image_buttons(self.maps_figure, "error maps")))
        maps_layout.addWidget(self.maps_figure)
        profile_layout = QtWidgets.QVBoxLayout(profile_panel)
        profile_layout.addLayout(style.figure_bar("Row profile", [copy_rows, export_rows] + style.image_buttons(
            self.profile_figure, "row profile")))
        profile_layout.addWidget(self.profile_figure)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(maps_panel)
        splitter.addWidget(profile_panel)
        splitter.setSizes([600, 500])
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(summary)
        layout.addWidget(splitter, stretch=1)
        self.draw_maps()

    def draw_maps(self):
        for plot, image, component in self.maps:
            draw_map(plot, image, self.position_um, self.error_nm[:, component], self.color_bar.levels(), self.dot_um,
                     self.devicePixelRatioF(), background=255)

    def _link_maps(self):
        """Both maps show the same area."""
        (a, _, _), (b, _, _) = self.maps
        busy = [False]

        def follow(source, target):
            def update(*_):
                if not busy[0]:
                    busy[0] = True
                    target.vb.setRange(rect=source.vb.viewRect(), padding=0)
                    busy[0] = False
            source.vb.sigRangeChanged.connect(update)
        follow(a, b)
        follow(b, a)

    def rows(self) -> list[list[str]]:
        """The row profile as a table: header, then per row its y (mask µm) and mean dx, dy (nm) after the
        moving-window affine and stitched without affine."""
        header = ["row y (µm)", "dx after moving window (nm)", "dy after moving window (nm)",
                  "dx no affine (nm)", "dy no affine (nm)"]
        return [header] + [[f"{y / 1000:.4f}", *(f"{v:.4f}" for v in (*after, *raw))]
                           for y, after, raw in zip(self.row_y, self.row_mean, self.raw_mean)]

    def rows_text(self, delimiter: str) -> str:
        out = io.StringIO()
        csv.writer(out, delimiter=delimiter, lineterminator="\n").writerows(self.rows())
        return out.getvalue()

    def export_rows(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export row profile", "row_profile.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel reads the µ
                f.write(self.rows_text(","))
