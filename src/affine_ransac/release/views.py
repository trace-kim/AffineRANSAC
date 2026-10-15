"""The release's two tabs (docs/SPEC.md D77), pyqtgraph, presentation style (style.py).

StitchingView: tile outlines, overlaps used, design and measured (SEM) centres, at the nominal or the stitched
placement; stitching failures (off at start, D76) and tile names on request; the SEM images on request
(Load SEM images: the app reads them in the background and calls set_images), with an opacity slider.

RegistrationView: the error after the moving-window affine, measured (SEM) − design, of every contact, as colour
maps of dx and dy (jet, shared colour scale, rasterized to the screen like the research view; draw_map) or as
one arrow map; values of every contact, or their mean per cell of a square grid (grid_means); grid lines on request. Every map has its profiles beside it (MarginalMap): the mean per row on its right, rotated so
that y runs down the side like the map's y, and the mean per column below it; both follow the map's zoom and
pan. Copy profiles / Export profiles CSV give both profiles as a table; Copy / Save image the whole figure or one
graph (also in each graph's right-click menu).

Everything is drawn relative to an origin in whole µm (OpenGL works in float32); the axes show mask µm.
"""

import csv
import io

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from scipy.spatial import cKDTree

from affine_ransac.registration import error_summary, grid_means, row_means
from affine_ransac.release import style
from affine_ransac.release.analysis import ReleaseResult
from affine_ransac.stitching import placement_corrections
from affine_ransac.view_points import box_outlines, tile_rectangles
from affine_ransac.view_registration import JET, colorize, draw_map

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
        self.opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, minimum=0, maximum=100, value=100)
        self.opacity.setFixedWidth(120)
        self.opacity.setEnabled(False)
        self.opacity.setToolTip("Opacity of the SEM images")
        self.opacity.valueChanged.connect(self.set_opacity)
        self.opacity_label = QtWidgets.QLabel("100 %")
        controls.addWidget(self.opacity)
        controls.addWidget(self.opacity_label)
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
        layout.addLayout(style.figure_bar("Stitching", style.image_buttons(lambda: self.graphics, "stitching")))
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
        self.opacity.setEnabled(True)
        self.set_opacity(self.opacity.value())
        self.images_box.setChecked(True)
        self.load_button.setEnabled(False)
        self.update_positions()
        self.update_visibility()

    def set_opacity(self, percent: int):
        """Opacity of the SEM images, 0 (transparent) to 100 %."""
        for item in self.image_items:
            item.setOpacity(percent / 100)
        self.opacity_label.setText(f"{percent} %")

    def _show_cursor(self, scene_pos):
        pos = self.plot.vb.mapSceneToView(scene_pos)
        x, y = (np.array([pos.x(), pos.y()]) + self.origin) / 1000
        self.cursor_label.setText(f"x = {x:.4f} µm, y = {y:.4f} µm (mask)")


def summary_text(error_nm: np.ndarray) -> str:
    s = error_summary(error_nm)
    return (f"mean dx {s['mean_x_nm']:+.2f} nm, dy {s['mean_y_nm']:+.2f} nm; 3σ dx {s['3sigma_x_nm']:.2f} nm, "
            f"dy {s['3sigma_y_nm']:.2f} nm; max |error| {s['max_nm']:.2f} nm")


def nice_number(value: float) -> float:
    """The 1, 2 or 5 x 10^k nearest below value (value > 0)."""
    exponent = np.floor(np.log10(value))
    return max(m for m in (1, 2, 5) if m * 10 ** exponent <= value) * 10 ** exponent


def arrow_segments(start: np.ndarray, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x, y of the line segments of arrows start -> start + vector: per arrow its shaft and two head strokes (30 %
    of its length, 25° off the shaft), for PlotCurveItem(connect="pairs")."""
    end = start + vector
    angle = np.arctan2(vector[:, 1], vector[:, 0])
    head = 0.3 * np.linalg.norm(vector, axis=1)
    strokes = [start, end]
    for turn in (np.pi - np.radians(25), np.pi + np.radians(25)):
        tip = end + head[:, None] * np.column_stack([np.cos(angle + turn), np.sin(angle + turn)])
        strokes += [end, tip]
    points = np.stack(strokes, axis=1).reshape(-1, 2)
    return points[:, 0], points[:, 1]


class MarginalMap:
    """A map with the mean per row on its right (rotated: the row's y runs down the side, aligned with the map's
    y) and the mean per column below it (aligned with the map's x), in a GraphicsLayout at (row, col): the map
    takes (row, col), the row profile (row, col + 1), the column profile (row + 1, col). The profiles follow the
    map's zoom and pan. Positions in µm relative to origin_nm."""
    AXIS_PX = 64  # left axis width of the map and of the column profile, so that their x line up

    def __init__(self, layout: pg.GraphicsLayout, row: int, col: int, origin_nm, title: str, name: str):
        """name: what the map shows, for naming its graphs (e.g. "dx")."""
        self.name = name
        self.map = mask_plot(origin_nm, title, data_unit_nm=1000)
        self.rows = pg.PlotItem(axisItems={"left": style.MaskAxis("left", origin_nm[1], 1000)})
        self.columns = pg.PlotItem(axisItems={"bottom": style.MaskAxis("bottom", origin_nm[0], 1000)})
        layout.addItem(self.map, row=row, col=col)
        layout.addItem(self.rows, row=row, col=col + 1)
        layout.addItem(self.columns, row=row + 1, col=col)
        style.style_plot(self.rows, "Mean per row", "error (nm)")
        style.style_plot(self.columns, "Mean per column", "x (µm)", "error (nm)")
        for plot in (self.map, self.columns):
            plot.getAxis("left").setWidth(self.AXIS_PX)
        self.rows.addLine(x=0, pen=pg.mkPen(style.MUTED, style=QtCore.Qt.PenStyle.DashLine))
        style.zero_line(self.columns)
        self.rows.setMouseEnabled(x=True, y=False)
        self.columns.setMouseEnabled(x=False, y=True)
        self.map.vb.sigRangeChanged.connect(self._follow)
        grid = layout.layout
        grid.setColumnStretchFactor(col, 4)
        grid.setColumnStretchFactor(col + 1, 1)
        grid.setRowStretchFactor(row, 3)
        grid.setRowStretchFactor(row + 1, 1)

    def add_profile(self, name: str, color: str, row_um, row_mean, column_um, column_mean):
        """One line in each profile: row_mean (nm) at the rows' y (µm), column_mean at the columns' x (µm)."""
        pen = pg.mkPen(color, width=2)
        self.rows.plot(row_mean, row_um, pen=pen, name=name)
        self.columns.plot(column_um, column_mean, pen=pen, name=name)

    def _follow(self, *_):
        (x0, x1), (y0, y1) = self.map.vb.viewRange()
        self.rows.setYRange(y0, y1, padding=0)
        self.columns.setXRange(x0, x1, padding=0)


class RegistrationView(QtWidgets.QWidget):
    def __init__(self, result: ReleaseResult, use_opengl: bool = True):
        super().__init__()
        r, s = result, result.settings
        self.result = r
        self.error_nm = r.moving.residuals
        self.origin = np.round(r.design_nm.mean(axis=0), -3)  # whole µm
        self.position_um = (r.design_nm - self.origin) / 1000
        distance, _ = cKDTree(r.design_nm).query(r.design_nm, k=2)
        self.pitch_nm = float(np.median(distance[:, 1]))
        self.dot_um = 0.4 * self.pitch_nm / 1000  # a disk of ~0.4 x the pitch when zoomed in
        self.row_y, self.row_mean, _ = row_means(r.design_nm[:, 1], self.error_nm, s.row_gap_nm)
        self.column_x, self.column_mean, _ = row_means(r.design_nm[:, 0], self.error_nm, s.row_gap_nm)
        row_um, column_um = (self.row_y - self.origin[1]) / 1000, (self.column_x - self.origin[0]) / 1000
        low, high = self.position_um.min(axis=0), self.position_um.max(axis=0)
        components = (("Δx", style.DX), ("Δy", style.DY))

        # Colour maps: dx and dy, each with the profiles of its own component; one colour bar.
        self.maps_figure = figure(use_opengl)
        self.maps = []  # (MarginalMap, image, component)
        for component, (name, color) in enumerate(components):
            panel = MarginalMap(self.maps_figure.ci, 0, 2 * component, self.origin, f"{name}: measured − design (nm)",
                                ("dx", "dy")[component])
            panel.add_profile(name, color, row_um, self.row_mean[:, component], column_um,
                              self.column_mean[:, component])
            image = pg.ImageItem(axisOrder="row-major")
            image.setZValue(-100)  # under the grid lines
            panel.map.addItem(image)
            self.maps.append((panel, image, component))
        limit = float(np.percentile(np.abs(self.error_nm), 99)) or 1.0
        self.color_bar = pg.ColorBarItem(values=(-limit, limit), colorMap=JET, width=18)
        axis = self.color_bar.getAxis("right")
        axis.setTextPen(pg.mkPen(style.TEXT))
        axis.setStyle(tickFont=QtGui.QFont(style.FONT, style.TICK_PT))
        axis.enableAutoSIPrefix(False)
        axis.setLabel("error (nm)", color=style.TEXT, **{"font-size": f"{style.LABEL_PT}pt"})
        self.color_bar.sigLevelsChanged.connect(lambda *_: self.draw_maps())
        self.maps_figure.addItem(self.color_bar, row=0, col=4)
        self._link_maps()
        self.maps[0][0].map.vb.setRange(xRange=(low[0], high[0]), yRange=(low[1], high[1]))
        self.redraw = QtCore.QTimer(self, singleShot=True, interval=50)
        self.redraw.timeout.connect(self.draw_maps)
        for panel, _, _ in self.maps:
            panel.map.vb.sigRangeChanged.connect(lambda *_: self.redraw.start())
            panel.map.vb.sigResized.connect(lambda *_: self.redraw.start())

        # Arrows: one map of the error vectors, both components in its profiles.
        self.arrows_figure = figure(use_opengl)
        self.arrow_panel = MarginalMap(self.arrows_figure.ci, 0, 0, self.origin, "Error vectors: measured − design",
                                       "error vectors")
        style.legend(self.arrow_panel.rows)
        for component, (name, color) in enumerate(components):
            self.arrow_panel.add_profile(name, color, row_um, self.row_mean[:, component], column_um,
                                         self.column_mean[:, component])
        self.arrows = pg.PlotCurveItem(connect="pairs", pen=pg.mkPen(style.TEXT, width=1), antialias=False)
        self.arrow_panel.map.addItem(self.arrows)
        self.arrow_panel.map.vb.setRange(xRange=(low[0], high[0]), yRange=(low[1], high[1]))
        self.arrow_scale = QtWidgets.QDoubleSpinBox(decimals=0, minimum=1, maximum=1e7)
        self.arrow_scale.setToolTip("Arrows are drawn this many times longer than the error")
        self.arrow_scale.valueChanged.connect(self.draw_arrows)
        self.arrow_key = QtWidgets.QLabel()

        # Every contact, or the mean over the contacts in each cell of a square grid.
        self.values = QtWidgets.QComboBox()
        self.values.addItems(["Every contact", "Grid average"])
        self.values.setToolTip("Grid average: one value per square cell, the mean of the contacts in it")
        area = np.prod(np.maximum(high - low, 1e-3))  # µm²
        self.cell = QtWidgets.QDoubleSpinBox(decimals=3, minimum=0.01, maximum=1e4, suffix=" µm", singleStep=0.1)
        self.cell.setValue(max(nice_number(np.sqrt(area / 2000)), nice_number(self.pitch_nm / 1000) * 2))  # ~2000 cells
        self.cell.setToolTip("Grid cell size (square)")
        self.grid_lines = QtWidgets.QCheckBox("Grid lines")
        self.grid_lines.setToolTip("Grid lines on the colour maps")
        self.grid_lines.toggled.connect(self.show_grid_lines)

        self.display = QtWidgets.QComboBox()
        self.display.addItems(["Colour maps", "Arrows"])
        self.figures = QtWidgets.QStackedWidget()
        self.figures.addWidget(self.maps_figure)
        self.figures.addWidget(self.arrows_figure)
        self.display.currentIndexChanged.connect(self.show_display)

        window = f"moving-window affine: window {s.moving_window_um:g} µm, step {s.moving_step_um:g} µm"
        summary = QtWidgets.QLabel(f"<b>{len(self.error_nm)} contacts</b>, {len(self.row_y)} rows, "
                                   f"{len(self.column_x)} columns. Error = measured (SEM) − design after the {window}: "
                                   f"{summary_text(self.error_nm)}.")
        summary.setWordWrap(True)
        copy_profiles = QtWidgets.QPushButton("Copy profiles")
        copy_profiles.setToolTip("Copy the mean per row and per column (µm; dx, dy nm) to the clipboard, "
                                 "tab-separated (pastes into Excel)")
        copy_profiles.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.profiles_text("\t")))
        export_profiles = QtWidgets.QPushButton("Export profiles CSV...")
        export_profiles.clicked.connect(self.export_profiles)
        for panel, _, _ in self.maps:
            for plot, items, name in self.graphs_of(panel):
                style.graph_actions(plot, self.maps_figure, items, name)
        for plot, items, name in self.graphs_of(self.arrow_panel):
            style.graph_actions(plot, self.arrows_figure, items, name)

        controls = QtWidgets.QHBoxLayout()
        self.arrow_controls = QtWidgets.QWidget()
        arrow_row = QtWidgets.QHBoxLayout(self.arrow_controls)
        arrow_row.setContentsMargins(0, 0, 0, 0)
        for widget in (QtWidgets.QLabel("Arrow scale ×"), self.arrow_scale, self.arrow_key):
            arrow_row.addWidget(widget)
        for widget in (QtWidgets.QLabel("Show:"), self.display, QtWidgets.QLabel("   Values:"), self.values,
                       QtWidgets.QLabel("cell"), self.cell, QtWidgets.QLabel("  "), self.grid_lines, self.arrow_controls):
            controls.addWidget(widget)
        controls.addStretch(1)
        for button in [copy_profiles, export_profiles] + style.image_menus(self.shown_graphs):
            controls.addWidget(button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(summary)
        layout.addLayout(controls)
        layout.addWidget(self.figures, stretch=1)

        self.values.currentIndexChanged.connect(self.set_values)
        self.cell.valueChanged.connect(self.set_values)
        self.set_values()
        self.show_display(0)

    def grid_mode(self) -> bool:
        return self.values.currentIndex() == 1

    def set_values(self, *_):
        """Every contact or the grid average: the grid computed again, the arrow scale reset, both displays drawn."""
        self.cell.setEnabled(self.grid_mode())
        corner, mean, _ = grid_means(self.result.design_nm, self.error_nm, self.cell.value() * 1000)
        self.grid_corner_um, self.grid_mean = (corner - self.origin) / 1000, mean
        vectors = self.arrow_vectors()[1]
        largest = float(np.percentile(np.linalg.norm(vectors, axis=1), 99)) if len(vectors) else 1.0
        spacing_nm = self.cell.value() * 1000 if self.grid_mode() else self.pitch_nm
        self.arrow_scale.blockSignals(True)
        self.arrow_scale.setValue(nice_number(0.5 * spacing_nm / (largest or 1.0)))  # largest arrows ~ half a spacing
        self.arrow_scale.blockSignals(False)
        self.draw_arrows()
        self.draw_maps()

    def arrow_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Arrow starts (µm, relative to the origin) and errors (nm): every contact, or every cell's mean at its
        centre."""
        if not self.grid_mode():
            return self.position_um, self.error_nm
        rows, cols = np.nonzero(np.isfinite(self.grid_mean[:, :, 0]))
        centres = self.grid_corner_um + (np.column_stack([cols, rows]) + 0.5) * self.cell.value()
        return centres, self.grid_mean[rows, cols]

    def show_display(self, index: int):
        """0: colour maps, 1: arrows."""
        self.figures.setCurrentIndex(index)
        self.arrow_controls.setVisible(index == 1)
        self.grid_lines.setVisible(index == 0)

    def show_grid_lines(self, on: bool):
        for panel, _, _ in self.maps:
            panel.map.showGrid(x=on, y=on, alpha=0.35)

    def draw_maps(self):
        for panel, image, component in self.maps:
            if self.grid_mode():  # one square per cell, the same at every zoom
                cell = self.cell.value()
                rows, cols = self.grid_mean.shape[:2]
                image.setImage(colorize(self.grid_mean[:, :, component], self.color_bar.levels(), background=255),
                               autoLevels=False)
                image.setRect(QtCore.QRectF(*self.grid_corner_um, cols * cell, rows * cell))
            else:
                draw_map(panel.map, image, self.position_um, self.error_nm[:, component], self.color_bar.levels(),
                         self.dot_um, self.devicePixelRatioF(), background=255)

    def draw_arrows(self):
        scale = self.arrow_scale.value()
        start, error = self.arrow_vectors()
        self.arrows.setData(*arrow_segments(start, error * scale / 1000))
        self.arrow_key.setText(f"(1 nm error = {scale / 1000:g} µm on the map)")

    def graphs_of(self, panel: MarginalMap) -> list[tuple[pg.PlotItem, list, str]]:
        """(plot, items to render, name) of a panel's graphs; a colour map comes with the colour bar."""
        bar = [] if panel is self.arrow_panel else [self.color_bar]
        return [(panel.map, [panel.map] + bar, f"{panel.name} map"),
                (panel.rows, [panel.rows], f"{panel.name} mean per row"),
                (panel.columns, [panel.columns], f"{panel.name} mean per column")]

    def shown_graphs(self):
        """For the image menus: the figure shown and its graphs."""
        if self.figures.currentIndex() == 1:
            return self.arrows_figure, [(name, items) for _, items, name in self.graphs_of(self.arrow_panel)]
        return self.maps_figure, [(name, items) for panel, _, _ in self.maps
                                  for _, items, name in self.graphs_of(panel)] + [("colour bar", [self.color_bar])]

    def _link_maps(self):
        """Both colour maps show the same area."""
        (a, _, _), (b, _, _) = self.maps
        busy = [False]

        def follow(source, target):
            def update(*_):
                if not busy[0]:
                    busy[0] = True
                    target.vb.setRange(rect=source.vb.viewRect(), padding=0)
                    busy[0] = False
            source.vb.sigRangeChanged.connect(update)
        follow(a.map, b.map)
        follow(b.map, a.map)

    def profiles(self) -> list[list[str]]:
        """Both profiles as one table: header, then per line a row (y, mask µm; mean dx, dy, nm) and a column
        (x, mask µm; mean dx, dy, nm), blank where one profile has fewer lines than the other."""
        header = ["row y (µm)", "row mean dx (nm)", "row mean dy (nm)",
                  "column x (µm)", "column mean dx (nm)", "column mean dy (nm)"]

        def cells(position, mean, k):
            return [f"{position[k] / 1000:.4f}", *(f"{v:.4f}" for v in mean[k])] if k < len(position) else [""] * 3
        length = max(len(self.row_y), len(self.column_x))
        return [header] + [cells(self.row_y, self.row_mean, k) + cells(self.column_x, self.column_mean, k)
                           for k in range(length)]

    def profiles_text(self, delimiter: str) -> str:
        out = io.StringIO()
        csv.writer(out, delimiter=delimiter, lineterminator="\n").writerows(self.profiles())
        return out.getvalue()

    def export_profiles(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export profiles", "profiles.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel reads the µ
                f.write(self.profiles_text(","))
