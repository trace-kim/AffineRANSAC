"""The release's tabs (docs/SPEC.md D77), pyqtgraph, presentation style (style.py).

StitchingView: tile outlines, overlaps used, design and measured (SEM) centres, at the nominal or the stitched
placement; on request (off at start, D76) the overlaps not matched (filled in the colour of their stitching, D107),
the tiles not stitched (thick red outline) and the tile names; the SEM images on request
(Load SEM images: the app reads them in the background and calls set_images), with an opacity slider; the design
pattern on request (Load design: the .oas contacts filled, one colour per stripe, set_design), so that the files of
neighbouring stripes can be told apart where they overlap. Of a failed run (release.analysis.AnalysisFailed) also
the tiles not measured, with the count per reason; the cursor line names the tiles under the cursor and why each
was not measured.

RegistrationView: the error after the moving-window affine, measured (SEM) − design, in one figure: an arrow map
(arrows coloured by their length dr, jet, own colour bar) and colour maps of dx and dy (jet, shared colour scale,
rasterized to the screen like the research view; draw_map), side by side and showing the same area; values of
every contact, or their mean per cell of a square grid (grid_means); grid lines (on at start). Every map has its
profiles beside it (MarginalMap): the mean per row on its right, rotated so that y runs down the side like the
map's y, and the mean per column below it, of dr beside the arrows and of its own component beside dx and dy;
both follow the map's zoom and pan. Stripes (StripeSelector): the maps and profiles show the contacts of the
stripes ticked. Copy profiles / Export profiles CSV give both profiles as a table; Copy / Save image the whole
figure or one graph (also in each graph's right-click menu).

ContactSizeTab (hidden unless turned on, D95): ROI width and Run measurement (the app measures in the background),
then ContactSizeView: maps of the area, xCD and yCD of every contact (any of them side by side, showing the same area),
the SEM value or SEM − design, each with its own jet colour bar from the 1st to the 99th percentile.

Everything is drawn relative to an origin in whole µm (OpenGL works in float32); the axes show mask µm.
"""

import csv
import io
from collections import Counter

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from scipy.spatial import cKDTree

from affine_ransac.analysis import ContactSizes
from affine_ransac.pipeline import contact_stripes, stripe_ranges
from affine_ransac.registration import error_summary, grid_means, row_means
from affine_ransac.release import style
from affine_ransac.release.analysis import ReleaseResult
from affine_ransac.stitching import placement_corrections
from affine_ransac.view_contact_size import QUANTITIES, size_summary, value_levels
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


def design_item(mask: np.ndarray, center_nm, fov_x_nm: float, color: str) -> pg.ImageItem:
    """A tile's design pattern (read_design_images mask, 1 = contact) in a colour, half transparent, nothing
    between the contacts; above the SEM images."""
    item = tile_image_item(mask, center_nm, fov_x_nm, mask.shape)
    rgb = QtGui.QColor(color)
    item.setLookupTable(np.array([[0, 0, 0, 0], [rgb.red(), rgb.green(), rgb.blue(), 170]], np.uint8))
    item.setLevels((0, 1))
    item.setZValue(-5)
    return item


def name_label(text: str) -> pg.TextItem:
    """A tile's name, centred where it is placed: dark text on a white label, readable over SEM images (grey text on
    grey images was not, D108)."""
    return pg.TextItem(text, color=style.TEXT, anchor=(0.5, 0.5), fill=pg.mkBrush(255, 255, 255, 220),
                       border=pg.mkPen(style.MUTED))


def translucent(color: str, alpha: int = 70) -> QtGui.QBrush:
    """A fill of this colour that lets what is below show through (alpha of 255)."""
    fill = QtGui.QColor(color)
    fill.setAlpha(alpha)
    return QtGui.QBrush(fill)


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
    def __init__(self, result: ReleaseResult, use_opengl: bool = True, on_load_images=None, on_load_design=None,
                 not_measured: dict[int, str] | None = None):
        """on_load_images(): called by Load SEM images (None, or a result without images: button disabled);
        on_load_design(): the same for Load design (a result without .oas files: disabled); not_measured: of a
        failed run, tile index -> why it was not measured (a layer Tiles not measured; None: no such layer)."""
        super().__init__()
        self.result = result
        self.not_measured = not_measured
        r = result
        self.origin = np.round(np.concatenate(r.design_points).mean(axis=0), -3)  # whole µm
        self.shifts = {"nominal": (np.zeros_like(r.centers), np.zeros_like(r.centers)),
                       "stitched": tuple(np.nan_to_num(c) for c in
                                         placement_corrections(r.stitch.corrections, r.design_stitch.corrections)["mean"])}
        self.image_items = []
        self.design_items = []  # (tile index, item)

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
        # Failures: overlaps not matched filled in the colour of their stitching (measured or design), tiles not
        # stitched outlined thick in red, tiles not measured (a failed run) hatched in red.
        self.layers["Overlaps not matched"] = [
            self._rectangles([self._local(f.box) for f in stitch.rejected if f.failed], pg.mkPen(color, width=1),
                             translucent(color), f"{name} overlaps not matched")
            for name, stitch, color in (("Measured", r.stitch, style.SEM), ("Design", r.design_stitch, style.DESIGN))]
        unplaced = sorted(set(r.stitch.unplaced.tolist()) | set(r.design_stitch.unplaced.tolist()))
        self.layers["Tiles not stitched"] = self._rectangles(
            [rectangles[k] for k in unplaced], pg.mkPen(style.FAILURE, width=3), pg.mkBrush(None), "Tiles not stitched")
        if not_measured is not None:
            hatch = QtGui.QBrush(QtGui.QColor(style.FAILURE), QtCore.Qt.BrushStyle.BDiagPattern)
            self.layers["Tiles not measured"] = self._rectangles(
                [rectangles[k] for k in sorted(not_measured)], pg.mkPen(style.FAILURE, width=1), hatch,
                "Tiles not measured")
        self.names = []
        for name, center in zip(r.tile_ids, r.centers):
            text = name_label(name)
            text.setPos(*(center - self.origin))
            self.plot.addItem(text)
            self.names.append(text)
        self.layers["Tile names"] = self.names

        self.boxes = {}
        controls = QtWidgets.QHBoxLayout()  # the placement and the layers
        controls.addWidget(QtWidgets.QLabel("Placement:"))
        self.placement = QtWidgets.QComboBox()
        self.placement.addItems(PLACEMENTS.values())
        self.placement.currentIndexChanged.connect(lambda _: self.update_positions())
        controls.addWidget(self.placement)
        controls.addSpacing(16)
        hidden = {"Overlaps not matched", "Tiles not stitched", "Tiles not measured", "Tile names"}  # off at start (D76)
        for name in self.layers:
            box = QtWidgets.QCheckBox(name)
            box.setChecked(name not in hidden)
            box.toggled.connect(self.update_visibility)
            controls.addWidget(box)
            self.boxes[name] = box
        controls.addStretch(1)
        loading = QtWidgets.QHBoxLayout()  # what is read on request: SEM images, design
        self.images_box = QtWidgets.QCheckBox("SEM images")
        self.images_box.setEnabled(False)
        self.images_box.toggled.connect(self.update_visibility)
        loading.addWidget(self.images_box)
        self.opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, minimum=0, maximum=100, value=100)
        self.opacity.setFixedWidth(120)
        self.opacity.setEnabled(False)
        self.opacity.setToolTip("Opacity of the SEM images")
        self.opacity.valueChanged.connect(self.set_opacity)
        self.opacity_label = QtWidgets.QLabel("100 %")
        loading.addWidget(self.opacity)
        loading.addWidget(self.opacity_label)
        self.load_button = QtWidgets.QPushButton("Load SEM images")
        can_load = on_load_images is not None and bool(r.image_paths)
        self.load_button.setEnabled(can_load)
        self.load_button.setToolTip("Read the SEM images and show them under the centres" if can_load else
                                    "No SEM images: this folder holds contour CSVs" if not r.image_paths else
                                    "Not available")
        if can_load:
            self.load_button.clicked.connect(on_load_images)
        loading.addWidget(self.load_button)
        loading.addSpacing(16)
        self.design_box = QtWidgets.QCheckBox("Design pattern")
        self.design_box.setEnabled(False)
        self.design_box.setToolTip("The .oas contacts, one colour per stripe")
        self.design_box.toggled.connect(self.update_visibility)
        loading.addWidget(self.design_box)
        self.design_button = QtWidgets.QPushButton("Load design")
        can_load_design = on_load_design is not None and any(r.oas_paths)
        self.design_button.setEnabled(can_load_design)
        self.design_button.setToolTip("Read the .oas files and show their contacts, one colour per stripe"
                                      if can_load_design else "No .oas files")
        if can_load_design:
            self.design_button.clicked.connect(on_load_design)
        loading.addWidget(self.design_button)
        loading.addStretch(1)

        self.status = QtWidgets.QLabel(self.status_text())
        self.status.setWordWrap(True)
        self.cursor_label = QtWidgets.QLabel(" ")
        self.cursor_label.setStyleSheet(f"color: {style.MUTED}")
        self.plot.scene().sigMouseMoved.connect(self._show_cursor)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(style.figure_bar("Stitching", style.image_buttons(lambda: self.graphics, "stitching")))
        layout.addWidget(self.status)
        layout.addLayout(controls)
        layout.addLayout(loading)
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

    def _rectangles(self, boxes, pen, brush, name):
        """Boxes (x0, x1, y0, y1, relative to the origin) as one item that QPainter draws, also on an OpenGL viewport:
        there pyqtgraph draws plot lines itself, in their colour and width only, so dashed lines came out solid
        (D107). In the legend with their count."""
        b = np.asarray(boxes, dtype=float).reshape(-1, 4)
        item = pg.BarGraphItem(x0=b[:, 0], y0=b[:, 2], width=b[:, 1] - b[:, 0], height=b[:, 3] - b[:, 2], pen=pen,
                               brush=brush, name=f"{name} ({len(b)})")
        self.plot.addItem(item)
        return item

    def status_text(self) -> str:
        """What the stitching means for the results, worst first (D109), then the counts of both stitchings."""
        r, parts = self.result, []
        n = len(r.tile_ids)
        for name, stitch in (("Measured", r.stitch), ("Design", r.design_stitch)):
            failed = sum(f.failed for f in stitch.rejected)
            parts.append(f"{name}: {n - len(stitch.unplaced)} of {n} tiles stitched, {len(stitch.pairs)} overlaps used, "
                         f"{failed} not matched")
        text = "<br>".join(stitching_verdict(r)) + "<br>" + "; ".join(parts) + f". Method: {r.settings.stitch_method}."
        ticks = [f"<i>{name}</i>" for name, shown in
                 (("Overlaps not matched", any(f.failed for f in r.stitch.rejected + r.design_stitch.rejected)),
                  ("Tiles not stitched", len(r.stitch.unplaced) or len(r.design_stitch.unplaced))) if shown]
        if ticks:
            text += f" Tick {' and '.join(ticks)} to see where."
        if self.not_measured is not None:
            reasons = Counter(self.not_measured.values()).most_common()
            text += f"<br><span style='color:{style.FAILURE}'><b>{len(self.not_measured)} of {n} tiles not " \
                    f"measured:</b></span> " + "; ".join(f"{reason} ({count})" for reason, count in reasons) + \
                    ". Tick <i>Tiles not measured</i> to see where; the line below the plot names the tiles under " \
                    "the cursor."
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
        for k, item in self.design_items:  # a realigned tile's design is not its own file: the files stay put
            item.setPos(*(design_shift[k] if not r.settings.realign else (0.0, 0.0)))

    def update_visibility(self, *_):
        for name, items in self.layers.items():
            for item in items if isinstance(items, list) else [items]:
                item.setVisible(self.boxes[name].isChecked())
        for item in self.image_items:
            item.setVisible(self.images_box.isChecked())
        for _, item in self.design_items:
            item.setVisible(self.design_box.isChecked())

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

    def set_design(self, masks: list[np.ndarray | None]):
        """Show the design pattern: per tile its mask (release.analysis.read_design_images; None: no .oas) at its
        file's position (result.oas_centers), in the colour of the tile's stripe."""
        for _, item in self.design_items:
            self.plot.removeItem(item)
        self.design_items = []
        r = self.result
        for k, mask in enumerate(masks):
            if mask is not None:
                color = style.STRIPES[int(r.stripe[k]) % len(style.STRIPES)]
                item = design_item(mask, r.oas_centers[k] - self.origin, r.sizes[k][0], color)
                self.plot.addItem(item)
                self.design_items.append((k, item))
        self.design_box.setEnabled(True)
        self.design_box.setChecked(True)
        self.design_button.setEnabled(False)
        self.update_positions()
        self.update_visibility()

    def set_opacity(self, percent: int):
        """Opacity of the SEM images, 0 (transparent) to 100 %."""
        for item in self.image_items:
            item.setOpacity(percent / 100)
        self.opacity_label.setText(f"{percent} %")

    def _show_cursor(self, scene_pos):
        pos = self.plot.vb.mapSceneToView(scene_pos)
        point = np.array([pos.x(), pos.y()]) + self.origin
        tiles = self.tiles_at(point)
        self.cursor_label.setText(f"x = {point[0] / 1000:.4f} µm, y = {point[1] / 1000:.4f} µm (mask)"
                                  + (f"   Under the cursor: {tiles}" if tiles else ""))

    def tiles_at(self, point_nm) -> str:
        """The tiles whose (nominal) box holds the point, mask nm; of a failed run each with why it was not measured."""
        r, why = self.result, self.not_measured or {}
        inside = np.flatnonzero(np.all(np.abs(r.centers - point_nm) <= r.sizes / 2, axis=1))
        return ", ".join(r.tile_ids[k] + (f" (not measured: {why[k]})" if k in why else "") for k in inside)


def stitching_verdict(r: ReleaseResult) -> list[str]:
    """One coloured line per finding about the stitching, worst first (D109): an error (overlaps paired one contact row
    off: errors near them can be wrong by a pitch), a warning (tiles not stitched: not measured), or, if neither, OK,
    with a note when overlaps were not matched (left out; every tile still placed through its others)."""
    def line(color, headline, text):
        return f"<span style='color:{color}'><b>{headline}</b></span> {text}"
    names, n, lines = r.tile_ids, len(r.tile_ids), []
    if r.slipped:
        shown = ", ".join(f"{names[i]} / {names[j]}" for i, j in r.slipped[:3]) + (", ..." if len(r.slipped) > 3 else "")
        lines.append(line(style.FAILURE, f"Error: {len(r.slipped)} overlap(s) paired one contact row off?",
                          f"({shown}) Their measured shift is more than max_match_nm away from the one the images taken "
                          f"one after the other predict, so the errors of the tiles around them can be wrong by up to a "
                          f"pitch. The stitching method <i>measurement order first</i> avoids this."))
    unplaced = set(r.stitch.unplaced.tolist()) | set(r.design_stitch.unplaced.tolist())
    if unplaced:
        lines.append(line(style.FAILURE, f"Warning: {len(unplaced)} of {n} tiles not stitched.",
                          "They have no position, so their contacts are missing from the Registration tab (white "
                          "there); its numbers cover the other tiles only."))
    if not lines:
        failed = {name: sum(f.failed for f in stitch.rejected)
                  for name, stitch in (("measured", r.stitch), ("design", r.design_stitch))}
        if any(failed.values()):
            counts = " and ".join(f"{count} {name}" for name, count in failed.items() if count)
            lines.append(line(style.CAUTION, "Stitching OK, with overlaps not matched.",
                              f"All {n} tiles are stitched, so all are measured. {counts} overlap(s) were not matched "
                              f"and are left out: their tiles are placed through their other overlaps."))
        else:
            lines.append(line(style.OK, "Stitching OK.", f"All {n} tiles stitched."))
    if r.slipped is None:
        lines.append("This run was saved before MaskReg kept the check for overlaps paired one row off: see its log.")
    return lines


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

    def add_profile(self, name: str, color: str, row_um=(), row_mean=(), column_um=(), column_mean=()):
        """One line in each profile: row_mean (nm) at the rows' y (µm), column_mean at the columns' x (µm).
        Returns the two lines (row, column), to change their data later."""
        pen = pg.mkPen(color, width=2)
        return self.rows.plot(row_mean, row_um, pen=pen, name=name), self.columns.plot(column_um, column_mean, pen=pen,
                                                                                       name=name)

    def _follow(self, *_):
        (x0, x1), (y0, y1) = self.map.vb.viewRange()
        self.rows.setYRange(y0, y1, padding=0)
        self.columns.setXRange(x0, x1, padding=0)


ARROW_COLOURS = 32  # arrows are coloured by their length in this many steps of the colour scale


def jet_bar(values: tuple[float, float], label: str) -> pg.ColorBarItem:
    """A jet colour bar over values (low, high) with this axis label, in the figures' style."""
    bar = pg.ColorBarItem(values=values, colorMap=JET, width=18)
    axis = bar.getAxis("right")
    axis.setTextPen(pg.mkPen(style.TEXT))
    axis.setStyle(tickFont=QtGui.QFont(style.FONT, style.TICK_PT))
    axis.enableAutoSIPrefix(False)
    axis.setLabel(label, color=style.TEXT, **{"font-size": f"{style.LABEL_PT}pt"})
    return bar


def link_views(plots: list[pg.PlotItem]):
    """The plots show the same area: a zoom or pan in one is copied to the others, as the same centre and the same
    scale (µm per pixel), so plots of different widths still agree. A hidden plot neither leads nor follows."""
    busy = [False]

    def follow(source):
        def update(*_):
            if busy[0] or not source.isVisible() or source.vb.width() <= 0:
                return
            busy[0] = True
            (x0, x1), y_range = source.vb.viewRange()
            per_pixel, centre = (x1 - x0) / source.vb.width(), (x0 + x1) / 2
            for target in plots:
                if target is not source and target.isVisible():
                    half = per_pixel * target.vb.width() / 2
                    target.vb.setRange(xRange=(centre - half, centre + half), yRange=y_range, padding=0)
            busy[0] = False
        source.vb.sigRangeChanged.connect(update)
    for plot in plots:
        follow(plot)


class StripeSelector(QtWidgets.QPushButton):
    """A button whose drop-down lists the stripes, each with a tick box (all ticked at start), and an All button.
    changed() is called when the ticks change; the last ticked stripe cannot be unticked."""

    def __init__(self, labels: list[str], changed):
        super().__init__()
        self.changed = changed
        self.list = QtWidgets.QListWidget()
        for label in labels:
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            self.list.addItem(item)
        self.list.itemChanged.connect(self._ticked)
        everything = QtWidgets.QPushButton("All")
        everything.clicked.connect(lambda: self.set_selected(range(self.list.count())))
        panel = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(panel)
        column.addWidget(self.list)
        column.addWidget(everything)
        menu = QtWidgets.QMenu(self)
        action = QtWidgets.QWidgetAction(menu)
        action.setDefaultWidget(panel)
        menu.addAction(action)
        self.setMenu(menu)
        self.setToolTip("The stripes whose contacts the maps and profiles show (a contact in the overlap of two "
                        "stripes belongs to both)")
        self._label()

    def selected(self) -> list[int]:
        return [k for k in range(self.list.count())
                if self.list.item(k).checkState() == QtCore.Qt.CheckState.Checked]

    def set_selected(self, stripes):
        self.list.blockSignals(True)
        for k in range(self.list.count()):
            self.list.item(k).setCheckState(QtCore.Qt.CheckState.Checked if k in stripes else
                                            QtCore.Qt.CheckState.Unchecked)
        self.list.blockSignals(False)
        self._label()
        self.changed()

    def _ticked(self, item):
        if not self.selected():  # keep one stripe
            self.list.blockSignals(True)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            self.list.blockSignals(False)
            return
        self._label()
        self.changed()

    def _label(self):
        n, total = len(self.selected()), self.list.count()
        self.setText(f"Stripes: all {total}" if n == total else f"Stripes: {n} of {total}")


class RegistrationView(QtWidgets.QWidget):
    def __init__(self, result: ReleaseResult, use_opengl: bool = True):
        super().__init__()
        r = result
        self.result = r
        self.origin = np.round(r.design_nm.mean(axis=0), -3)  # whole µm
        distance, _ = cKDTree(r.design_nm).query(r.design_nm, k=2)
        self.pitch_nm = float(np.median(distance[:, 1]))
        self.dot_um = 0.4 * self.pitch_nm / 1000  # a disk of ~0.4 x the pitch when zoomed in
        self.in_stripe = contact_stripes(r.design_nm[:, 0], r.centers, r.sizes, r.stripe)

        # One figure: the arrow map (arrows coloured by their length dr, profiles of dr, its own colour bar), then
        # the dx and dy colour maps (profiles of their own component, one shared colour bar). Same area in all.
        self.figure = figure(use_opengl)
        grid = self.figure.ci
        self.arrow_panel = MarginalMap(grid, 0, 0, self.origin, "Δr: measured − design (nm)", "arrow")
        self.arrow_profiles = [self.arrow_panel.add_profile("dr", style.DR)]
        self.arrows = [pg.PlotCurveItem(connect="pairs", pen=pg.mkPen(QtGui.QColor(*rgb[:3]), width=1), antialias=False)
                       for rgb in JET.getLookupTable(nPts=ARROW_COLOURS)]
        for item in self.arrows:
            self.arrow_panel.map.addItem(item)
        self.dr_bar = jet_bar((0.0, 1.0), "dr = |error| (nm)")
        self.dr_bar.sigLevelsChanged.connect(lambda *_: self.draw_arrows())
        grid.addItem(self.dr_bar, row=0, col=2)
        grid.layout.setColumnSpacing(2, 24)  # room between its axis label and the dx map's
        self.maps = []  # (MarginalMap, image, component)
        self.map_profiles = []
        for component, (name, color) in enumerate((("Δx", style.DX), ("Δy", style.DY))):
            panel = MarginalMap(grid, 0, 3 + 2 * component, self.origin, f"{name}: measured − design (nm)",
                                ("dx", "dy")[component])
            self.map_profiles.append(panel.add_profile(name, color))
            image = pg.ImageItem(axisOrder="row-major")
            image.setZValue(-100)  # under the grid lines
            panel.map.addItem(image)
            self.maps.append((panel, image, component))
        self.color_bar = jet_bar((-1.0, 1.0), "error (nm)")
        self.color_bar.sigLevelsChanged.connect(lambda *_: self.draw_maps())
        grid.addItem(self.color_bar, row=0, col=7)
        self.panels = [self.arrow_panel] + [panel for panel, _, _ in self.maps]
        link_views([panel.map for panel in self.panels])
        self.redraw = QtCore.QTimer(self, singleShot=True, interval=50)
        self.redraw.timeout.connect(self.draw_maps)
        for panel, _, _ in self.maps:
            panel.map.vb.sigRangeChanged.connect(lambda *_: self.redraw.start())
            panel.map.vb.sigResized.connect(lambda *_: self.redraw.start())

        self.arrow_scale = QtWidgets.QDoubleSpinBox(decimals=0, minimum=1, maximum=1e7)
        self.arrow_scale.setToolTip("Arrows are drawn this many times longer than the error")
        self.arrow_scale.valueChanged.connect(self.draw_arrows)
        self.arrow_key = QtWidgets.QLabel()

        # Every contact, or the mean over the contacts in each cell of a square grid.
        self.values = QtWidgets.QComboBox()
        self.values.addItems(["Every contact", "Grid average"])
        self.values.setToolTip("Grid average: one value per square cell, the mean of the contacts in it")
        area = np.prod(np.maximum(np.ptp(r.design_nm, axis=0) / 1000, 1e-3))  # µm²
        self.cell = QtWidgets.QDoubleSpinBox(decimals=3, minimum=0.01, maximum=1e4, suffix=" µm", singleStep=0.1)
        self.cell.setValue(max(nice_number(np.sqrt(area / 2000)), nice_number(self.pitch_nm / 1000) * 2))  # ~2000 cells
        self.cell.setToolTip("Grid cell size (square)")
        self.grid_lines = QtWidgets.QCheckBox("Grid lines")
        self.grid_lines.setToolTip("Grid lines on the maps")
        self.grid_lines.setChecked(True)
        self.grid_lines.toggled.connect(self.show_grid_lines)
        low, high = stripe_ranges(r.centers, r.sizes, r.stripe)
        self.stripes = StripeSelector([f"Stripe {k + 1}  (x {a / 1000:.1f} to {b / 1000:.1f} µm)"
                                       for k, (a, b) in enumerate(zip(low, high))], self.select_stripes)

        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        copy_profiles = QtWidgets.QPushButton("Copy profiles")
        copy_profiles.setToolTip("Copy the mean per row and per column (µm; dx, dy, dr nm) to the clipboard, "
                                 "tab-separated (pastes into Excel)")
        copy_profiles.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.profiles_text("\t")))
        export_profiles = QtWidgets.QPushButton("Export profiles CSV...")
        export_profiles.clicked.connect(self.export_profiles)
        for panel in self.panels:
            for plot, items, name in self.graphs_of(panel):
                style.graph_actions(plot, self.figure, items, name)

        controls = QtWidgets.QHBoxLayout()
        for widget in (self.stripes, QtWidgets.QLabel("   Values:"), self.values, QtWidgets.QLabel("cell"), self.cell,
                       QtWidgets.QLabel("  "), self.grid_lines, QtWidgets.QLabel("   Arrow scale ×"), self.arrow_scale,
                       self.arrow_key):
            controls.addWidget(widget)
        controls.addStretch(1)
        for button in [copy_profiles, export_profiles] + style.image_menus(self.shown_graphs):
            controls.addWidget(button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addLayout(controls)
        layout.addWidget(self.figure, stretch=1)

        self.values.currentIndexChanged.connect(self.set_values)
        self.cell.valueChanged.connect(self.set_values)
        self.show_grid_lines(True)
        self.select_stripes()

    def select_stripes(self):
        """Show the contacts of the ticked stripes: their maps, profiles, colour scales and summary, the view on
        them. The errors stay those of the moving-window affine fitted on every contact."""
        r, s = self.result, self.result.settings
        selected = self.stripes.selected()
        shown = self.in_stripe[:, selected].any(axis=1)
        self.design_nm, self.error_nm = r.design_nm[shown], r.moving.residuals[shown]
        self.position_um = (self.design_nm - self.origin) / 1000
        length = np.linalg.norm(self.error_nm, axis=1)
        values = np.column_stack([self.error_nm, length])  # dx, dy, dr
        self.row_y, self.row_mean, _ = row_means(self.design_nm[:, 1], values, s.row_gap_nm)
        self.column_x, self.column_mean, _ = row_means(self.design_nm[:, 0], values, s.row_gap_nm)
        row_um, column_um = (self.row_y - self.origin[1]) / 1000, (self.column_x - self.origin[0]) / 1000
        for k, (row_line, column_line) in enumerate(self.map_profiles + self.arrow_profiles):  # dx, dy, dr
            row_line.setData(self.row_mean[:, k], row_um)
            column_line.setData(column_um, self.column_mean[:, k])
        limit = float(np.percentile(np.abs(self.error_nm), 99)) or 1.0
        self.color_bar.setLevels((-limit, limit))
        self.dr_bar.setLevels((0.0, float(np.percentile(length, 99)) or 1.0))

        window = f"moving-window affine: window {s.moving_window_um:g} µm, step {s.moving_step_um:g} µm"
        total = self.in_stripe.shape[1]
        stripes = f"all {total} stripes" if len(selected) == total else \
            f"stripe(s) {', '.join(str(k + 1) for k in selected)} of {total}"
        self.summary.setText(f"<b>{len(self.error_nm)} contacts</b> of {stripes}, {len(self.row_y)} rows, "
                             f"{len(self.column_x)} columns. Error = measured (SEM) − design after the {window}, "
                             f"fitted on every contact: {summary_text(self.error_nm)}.")
        low, high = self.position_um.min(axis=0), self.position_um.max(axis=0)
        self.arrow_panel.map.vb.setRange(xRange=(low[0], high[0]), yRange=(low[1], high[1]))
        self.set_values()

    def grid_mode(self) -> bool:
        return self.values.currentIndex() == 1

    def set_values(self, *_):
        """Every contact or the grid average: the grid computed again, the arrow scale reset, the maps drawn."""
        self.cell.setEnabled(self.grid_mode())
        corner, mean, _ = grid_means(self.design_nm, self.error_nm, self.cell.value() * 1000)
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

    def show_grid_lines(self, on: bool):
        for panel in self.panels:
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
        """The arrows, each coloured by its length (its error's dr) on the dr colour bar."""
        scale = self.arrow_scale.value()
        start, error = self.arrow_vectors()
        low, high = self.dr_bar.levels()
        fraction = (np.linalg.norm(error, axis=1) - low) / (high - low) if high > low else np.zeros(len(error))
        step = np.clip((fraction * ARROW_COLOURS).astype(int), 0, ARROW_COLOURS - 1)
        for k, item in enumerate(self.arrows):
            item.setData(*arrow_segments(start[step == k], error[step == k] * scale / 1000))
        self.arrow_key.setText(f"(1 nm error = {scale / 1000:g} µm on the map)")

    def graphs_of(self, panel: MarginalMap) -> list[tuple[pg.PlotItem, list, str]]:
        """(plot, items to render, name) of a panel's graphs; a map comes with its colour bar."""
        bar, profile = (self.dr_bar, "dr") if panel is self.arrow_panel else (self.color_bar, panel.name)
        return [(panel.map, [panel.map, bar], f"{panel.name} map"),
                (panel.rows, [panel.rows], f"{profile} mean per row"),
                (panel.columns, [panel.columns], f"{profile} mean per column")]

    def shown_graphs(self):
        """For the image menus: the figure and its graphs."""
        return self.figure, [(name, items) for panel in self.panels for _, items, name in self.graphs_of(panel)] + \
            [("dr colour bar", [self.dr_bar]), ("dx, dy colour bar", [self.color_bar])]

    def profiles(self) -> list[list[str]]:
        """Both profiles of the contacts shown, as one table: header, then per line a row (y, mask µm; mean dx, dy,
        dr, nm) and a column (x, mask µm; mean dx, dy, dr, nm), blank where one profile has fewer lines."""
        header = ["row y (µm)", "row mean dx (nm)", "row mean dy (nm)", "row mean dr (nm)",
                  "column x (µm)", "column mean dx (nm)", "column mean dy (nm)", "column mean dr (nm)"]

        def cells(position, mean, k):
            return [f"{position[k] / 1000:.4f}", *(f"{v:.4f}" for v in mean[k])] if k < len(position) else [""] * 4
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


NOT_IMAGES = ("N/A: contour CSVs hold contact centres only. Area, xCD and yCD are measured on the SEM contours and "
              "the .oas outlines of an image folder.")
NEEDS_TILES = ("Measuring needs the tiles of the run in memory: restart the analysis (Analysis > Restart analysis, "
               "F5). While this tab is on, a run keeps them.")


class ContactSizeTab(QtWidgets.QWidget):
    def __init__(self, result: ReleaseResult, use_opengl: bool = True, on_measure=None):
        """The Contact size tab: ROI width, Run measurement, then the maps (ContactSizeView). on_measure(roi_width_nm)
        is called by Run measurement: the app measures in the background, then calls show_sizes. Sizes measured
        before (result.contact_sizes) are shown at once."""
        super().__init__()
        self.result, self.use_opengl, self.view = result, use_opengl, None
        self.roi = QtWidgets.QDoubleSpinBox(decimals=1, maximum=1000.0, suffix=" nm")
        self.roi.setToolTip("xCD / yCD = mean of the chords, one per SEM pixel, in a band this wide through the "
                            "contact centre (0: the single chord through the centre)")
        self.run_button = QtWidgets.QPushButton("Run measurement")
        if on_measure is not None:
            self.run_button.clicked.connect(lambda: on_measure(self.roi.value()))
        self.note = QtWidgets.QLabel()
        bar = QtWidgets.QHBoxLayout()
        for widget in (QtWidgets.QLabel("ROI width:"), self.roi, self.run_button):
            bar.addWidget(widget)
        bar.addWidget(self.note, stretch=1)
        self.layout_ = QtWidgets.QVBoxLayout(self)
        self.layout_.addLayout(bar)
        self.layout_.addStretch(1)  # replaced by the maps
        if result.contact_sizes is not None:
            self.roi.setValue(result.contact_sizes.roi_width_nm)
            self.show_sizes(result.contact_sizes)
        self.update_state(busy=False)

    def update_state(self, busy: bool):
        """Run measurement only with the run's tiles in memory and nothing else running; the note says why not."""
        r = self.result
        self.run_button.setEnabled(r.tiles is not None and not busy)
        if r.kind != "images":
            self.note.setText(NOT_IMAGES)
        elif r.tiles is None:
            self.note.setText(NEEDS_TILES)
        else:
            self.note.setText("" if self.view else "Area, xCD and yCD of every contact: press Run measurement")

    def show_sizes(self, sizes: ContactSizes):
        """Replace the maps (or the empty space) by those of sizes."""
        old = self.layout_.takeAt(1)
        if old.widget() is not None:
            old.widget().deleteLater()
        self.view = ContactSizeView(self.result.design_nm, sizes, self.use_opengl)
        self.layout_.addWidget(self.view, stretch=1)


class ContactSizeView(QtWidgets.QWidget):
    def __init__(self, design_nm: np.ndarray, sizes: ContactSizes, use_opengl: bool = True):
        """design_nm: (N, 2) contact positions, mask nm; sizes: their area (nm²), xCD and yCD (nm) on the SEM and
        on the design, in the same rows."""
        super().__init__()
        self.sizes = sizes
        self.origin = np.round(design_nm.mean(axis=0), -3)  # whole µm
        self.position_um = (design_nm - self.origin) / 1000
        distance, _ = cKDTree(design_nm).query(design_nm, k=2)
        self.dot_um = 0.4 * float(np.median(distance[:, 1])) / 1000  # a disk of ~0.4 x the pitch when zoomed in

        # One figure: per quantity a map and its colour bar; the maps not ticked are hidden (they leave no gap).
        self.figure = figure(use_opengl)
        self.redraw = QtCore.QTimer(self, singleShot=True, interval=50)
        self.redraw.timeout.connect(self.draw_maps)
        self.maps = []  # (plot, image, colour bar) per quantity
        for k, (name, unit) in enumerate(QUANTITIES):
            plot = mask_plot(self.origin, data_unit_nm=1000)
            image = pg.ImageItem(axisOrder="row-major")
            plot.addItem(image)
            bar = jet_bar((0.0, 1.0), f"{name} ({unit})")
            bar.sigLevelsChanged.connect(lambda *_: self.redraw.start())
            self.figure.addItem(plot, row=0, col=2 * k)
            self.figure.addItem(bar, row=0, col=2 * k + 1)
            plot.vb.sigRangeChanged.connect(lambda *_: self.redraw.start())
            plot.vb.sigResized.connect(lambda *_: self.redraw.start())
            style.graph_actions(plot, self.figure, [plot, bar], f"{name} map")
            self.maps.append((plot, image, bar))
        link_views([plot for plot, _, _ in self.maps])

        self.show_boxes = []
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Show:"))
        for k, (name, _) in enumerate(QUANTITIES):
            box = QtWidgets.QCheckBox(name)
            box.setChecked(k == 0)  # the area at start
            box.toggled.connect(self.show_maps)
            controls.addWidget(box)
            self.show_boxes.append(box)
        controls.addWidget(QtWidgets.QLabel("   Value:"))
        self.sem_button = QtWidgets.QRadioButton("SEM")
        self.difference_button = QtWidgets.QRadioButton("SEM − design")
        self.sem_button.setChecked(True)
        self.difference_button.toggled.connect(self.set_values)
        controls.addWidget(self.sem_button)
        controls.addWidget(self.difference_button)
        controls.addStretch(1)
        for button in style.image_menus(self.shown_graphs):
            controls.addWidget(button)

        band = "one chord" if sizes.roi_width_nm == 0 else f"a {sizes.roi_width_nm:g} nm band of chords"
        self.summary = QtWidgets.QLabel("\n".join(
            [f"{len(design_nm)} contacts; xCD and yCD from {band} through the contact centre."]
            + [size_summary(name, unit, sizes.sem[:, k], sizes.design[:, k]) for k, (name, unit) in enumerate(QUANTITIES)]))
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.summary)
        layout.addLayout(controls)
        layout.addWidget(self.figure, stretch=1)
        self.show_maps()  # first: a hidden map would pass its range on while the figure is laid out
        low, high = self.position_um.min(axis=0), self.position_um.max(axis=0)
        self.maps[0][0].vb.setRange(xRange=(low[0], high[0]), yRange=(low[1], high[1]))  # every contact
        self.set_values()

    def values(self, k: int) -> np.ndarray:
        """(N,) what map k shows: the SEM value, or SEM − design."""
        if self.difference_button.isChecked():
            return self.sizes.sem[:, k] - self.sizes.design[:, k]
        return self.sizes.sem[:, k]

    def set_values(self):
        """Titles and colour scales of the chosen value (SEM or SEM − design), then the maps drawn again."""
        difference = self.difference_button.isChecked()
        for k, ((name, unit), (plot, _, bar)) in enumerate(zip(QUANTITIES, self.maps)):
            title = f"{name}, SEM − design ({unit})" if difference else f"{name}, SEM ({unit})"
            plot.setTitle(title, color=style.TEXT, size=f"{style.TITLE_PT}pt", bold=True)
            bar.blockSignals(True)  # one redraw below, not one per bar
            bar.setLevels(value_levels(self.values(k)))
            bar.blockSignals(False)
        self.draw_maps()

    def show_maps(self):
        """Show the maps whose box is ticked, with their colour bars, at the centre and scale of a map shown before:
        its y range, x as wide as each map now is. (Its area would not do: a map made narrower would zoom out to
        fit it, more at every change.)"""
        visible = [plot for plot, _, _ in self.maps if plot.isVisible()]
        (x0, x1), (y0, y1) = visible[0].vb.viewRange() if visible else self.maps[0][0].vb.viewRange()
        for box, (plot, _, bar) in zip(self.show_boxes, self.maps):
            plot.setVisible(box.isChecked())
            bar.setVisible(box.isChecked())
        self.figure.ci.layout.activate()  # the maps' new widths, now
        for box, (plot, _, _) in zip(self.show_boxes, self.maps):
            if box.isChecked() and plot.vb.height() > 0:
                half = (y1 - y0) * plot.vb.width() / plot.vb.height() / 2  # equal x and y scales
                plot.vb.setRange(xRange=((x0 + x1) / 2 - half, (x0 + x1) / 2 + half), yRange=(y0, y1), padding=0)
        self.redraw.start()

    def draw_maps(self):
        for k, (box, (plot, image, bar)) in enumerate(zip(self.show_boxes, self.maps)):
            if box.isChecked():  # a hidden map is drawn when it is shown
                draw_map(plot, image, self.position_um, self.values(k), bar.levels(), self.dot_um,
                         self.devicePixelRatioF(), background=255)

    def shown_graphs(self):
        """For the image menus: the figure and the maps shown, each with its colour bar."""
        return self.figure, [(f"{name} map", [plot, bar]) for box, (name, _), (plot, _, bar)
                             in zip(self.show_boxes, QUANTITIES, self.maps) if box.isChecked()]
