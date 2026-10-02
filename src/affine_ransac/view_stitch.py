"""Interactive stitched view of many tiles (pyqtgraph, OpenGL viewport).

Shows, in mask coordinates: every tile's SEM image, the design (.oas) contours and centres,
the SEM contours and centres from Otsu and from edge refinement, the nominal overlap boxes, and
the contacts flagged as outliers by the pairwise overlap fits. Stitching failures are marked in
magenta: overlaps whose contacts did not match (SEM or design), and tiles that could not be
stitched (these stay at their nominal position in every placement).

- Placement (stitching.placement_corrections): "Nominal" puts each tile at its metadata centre;
  "Stitched" adds each tile's corrections, the SEM ones to the SEM items and the design ones
  (stitch_design) to the design items; "Stitched, first tile fixed" shifts both so the first tile
  stitched in both stays at its nominal position. Switch back and forth to compare.
- Error map (if design errors are given, one set per placement): a second plot beside the SEM
  view, sharing its zoom and pan, with one line per contact from its design position along its
  raw error (registration.design_errors; sign as in registration_error), magnified by the arrow
  scale. It shows the errors of the selected placement. Tiles that were not measured (no error
  computed: not stitched, or design tone flagged) are framed in magenta.
- Layers and tiles can be hidden with the check boxes on the right (hide a tile to see the
  tile beneath it in an overlap).
- Mouse: drag to pan, wheel to zoom, right-click for more options.

Everything is drawn relative to a local origin (the middle of the tiles, in whole µm): the
OpenGL viewport works in float32, which would round mask coordinates (~10⁷ nm) to several nm.
The axes show x − x0, y − y0 in µm; the cursor label shows absolute mask µm.

Data comes from affine_ransac.pipeline (process_tile, stitch_tiles, stitch_design) and
affine_ransac.registration (design_errors); see notebooks/stitch_viewer.ipynb.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from affine_ransac.geometry.frames import pixel_to_tile_nm
from affine_ransac.pipeline import StitchResult, TileResult
from affine_ransac.registration import DesignErrors, error_summary
from affine_ransac.stitching import first_stitched_tile, placement_corrections
from affine_ransac.view_design import polygons_to_path

IMAGES = "SEM images"
DESIGN = "Design contours (.oas)"
DESIGN_CENTRES = "Design centres"
OTSU = "Otsu contours"
OTSU_CENTRES = "Otsu centres"
REFINED = "Refined contours"
REFINED_CENTRES = "Refined centres"
OUTLIERS = "Overlap outliers"
BOXES = "Overlaps used (nominal)"
FAILED = "SEM overlaps not matched"
DESIGN_FAILED = "Design overlaps not matched"
SKIPPED = "Overlaps skipped (few contacts)"
UNSTITCHED = "Tiles not stitched"
ERRORS = "Error arrows (error map)"
TILE_FRAMES = "Tile outlines (error map)"
NOT_MEASURED = "Tiles not measured, no error computed (error map)"

FAILURE_COLOR = "#ff00ff"
COLORS = {
    IMAGES: "#c0c0c0", DESIGN: "#3c8cff", DESIGN_CENTRES: "#3c8cff", OTSU: "#33cc33",
    OTSU_CENTRES: "#33cc33", REFINED: "#ff4040", REFINED_CENTRES: "#ff4040",
    OUTLIERS: "#ffa500", BOXES: "#ffff00", FAILED: FAILURE_COLOR, DESIGN_FAILED: FAILURE_COLOR,
    SKIPPED: "#909090", UNSTITCHED: FAILURE_COLOR, ERRORS: "#ffffff", TILE_FRAMES: "#606060",
    NOT_MEASURED: FAILURE_COLOR,
}
# Off at start: many items (slow to draw); switch on in the panel. Failure markers stay on.
HIDDEN_AT_START = {DESIGN, OTSU, OTSU_CENTRES, REFINED, OUTLIERS, SKIPPED}
SEM_LAYERS = {IMAGES, OTSU, OTSU_CENTRES, REFINED, REFINED_CENTRES, OUTLIERS}  # moved by SEM corrections
DESIGN_LAYERS = {DESIGN, DESIGN_CENTRES}                                      # moved by design corrections
PLACEMENTS = {"nominal": "Nominal (metadata centres)", "mean": "Stitched (corrections average 0)",
              "first": "Stitched, first tile fixed"}


def image_item(tile: TileResult) -> pg.ImageItem:
    """The tile's image in tile-local nm (y up): pixel centres at pixel_to_tile_nm positions."""
    item = pg.ImageItem(tile.image, axisOrder="row-major", levels=(0, 255))
    # Item (0, 0) is the outer corner of pixel (0, 0), i.e. pixel (-0.5, -0.5); 1 item unit = 1 px.
    corner = pixel_to_tile_nm([[-0.5, -0.5]], tile.image.shape, tile.pixel_size_nm)[0]
    s = tile.pixel_size_nm
    item.setTransform(QtGui.QTransform(s, 0, 0, -s, corner[0], corner[1]))  # y flips: pixel down, nm up
    return item


def rectangle(left, right, bottom, top) -> np.ndarray:
    return np.array([[left, bottom], [right, bottom], [right, top], [left, top]])


def path_item(polygons: list[np.ndarray], color, style=QtCore.Qt.PenStyle.SolidLine, width=1):
    item = QtWidgets.QGraphicsPathItem(polygons_to_path(polygons))
    item.setPen(pg.mkPen(color, style=style, width=width))
    return item


def centres_item(points: np.ndarray, color, symbol="+", size=8):
    return pg.ScatterPlotItem(points[:, 0], points[:, 1], symbol=symbol, size=size, pen=pg.mkPen(color), brush=None)


def make_plot(use_opengl: bool) -> pg.PlotWidget:
    """A plot of nm data relative to the viewer's origin, axes labelled x − x0, y − y0 in µm."""
    plot = pg.PlotWidget()
    if use_opengl:
        plot.useOpenGL(True)
    plot.setAspectLocked(True)
    for side, name in (("bottom", "x − x0 (µm)"), ("left", "y − y0 (µm)")):
        plot.setLabel(side, name)
        plot.getAxis(side).setScale(1e-3)  # data is nm
    return plot


def link_views(a: pg.PlotWidget, b: pg.PlotWidget):
    """Keep the same visible area in both plots. (pyqtgraph's setXLink aligns linked views by
    their screen position, which shifts the content of plots placed side by side.)"""
    busy = [False]

    def follow(source, target):
        def update(*_):
            if not busy[0]:
                busy[0] = True
                target.getPlotItem().vb.setRange(rect=source.getPlotItem().vb.viewRect(), padding=0)
                busy[0] = False
        source.getPlotItem().vb.sigRangeChanged.connect(update)

    follow(a, b)
    follow(b, a)


def arrow_segments(start: np.ndarray, vector: np.ndarray, scale: float):
    """x, y arrays of line segments start -> start + scale * vector, for connect="pairs"."""
    end = start + scale * vector
    return np.column_stack([start[:, 0], end[:, 0]]).ravel(), np.column_stack([start[:, 1], end[:, 1]]).ravel()


def outlier_indices(stitch: StitchResult, n_tiles: int) -> list[np.ndarray]:
    """Per tile, the indices of its contacts that some pair fit flagged as an outlier."""
    found = [[] for _ in range(n_tiles)]
    for p in stitch.pairs:
        outliers = ~p.fit.inliers
        found[p.i].extend(p.ia[outliers])
        found[p.j].extend(p.ib[outliers])
    return [np.unique(np.asarray(f, dtype=int)) for f in found]


class StitchViewer(QtWidgets.QWidget):
    def __init__(
        self,
        tiles: list[TileResult],
        tile_ids: list[str],
        design_polygons: list[list[np.ndarray]],
        stitch: StitchResult,
        design_stitch: StitchResult,
        refined: bool = True,
        use_opengl: bool = True,
        errors: dict[str, DesignErrors] | None = None,
        arrow_scale: float = 10.0,
    ):
        """tiles: from process_tile; design_polygons: per tile, read_polygons() of its .oas
        (tile-local nm, [] if none); stitch: SEM stitching (stitch_tiles); design_stitch: design
        stitching (stitch_design); refined: whether the SEM was stitched with the refined (True) or
        the Otsu centres, for the outlier markers; errors: per placement ("nominal", "mean",
        "first"), registration.design_errors with that placement's corrections, shown as an
        error map beside the SEM view (None: no map); arrow_scale: error lines are drawn this
        many times longer than the error."""
        super().__init__()
        self.setWindowTitle(f"Stitch viewer - {len(tiles)} tiles")
        self.tiles = tiles
        self.sem_unplaced = set(stitch.unplaced.tolist())
        self.design_unplaced = set(design_stitch.unplaced.tolist())
        self.reference = first_stitched_tile(stitch.corrections, design_stitch.corrections)
        # Per placement, the (SEM, design) shift added to each tile. Unstitched tiles (NaN) never move.
        self.shifts = {mode: (np.nan_to_num(sem), np.nan_to_num(design))
                       for mode, (sem, design) in placement_corrections(stitch.corrections,
                                                                        design_stitch.corrections).items()}
        centers = np.array([t.center_nm for t in tiles])
        self.origin = np.round(centers.mean(axis=0), -3)  # whole µm, see module docstring
        self.items = []  # (layer, tile index or None, graphics item)
        self.mode = "nominal"

        self.plot = make_plot(use_opengl)
        x0, y0 = self.origin / 1000
        self.plot.setTitle(f"SEM and design - origin x0 = {x0:.0f} µm, y0 = {y0:.0f} µm")

        outliers = outlier_indices(stitch, len(tiles))
        for k, tile in enumerate(tiles):
            self._add_tile(k, tile, design_polygons[k], outliers[k], refined)
        dash, dot, dash_dot = QtCore.Qt.PenStyle.DashLine, QtCore.Qt.PenStyle.DotLine, QtCore.Qt.PenStyle.DashDotLine
        for layer, boxes, style in ((BOXES, [p.box for p in stitch.pairs], dash),
                                    (FAILED, [r.box for r in stitch.rejected if r.failed], dash),
                                    (DESIGN_FAILED, [r.box for r in design_stitch.rejected if r.failed], dash_dot),
                                    (SKIPPED, [r.box for r in stitch.rejected if not r.failed], dot)):
            if boxes:
                width = 1 if layer in (BOXES, SKIPPED) else 2
                self._add(layer, None, path_item([rectangle(*b) - self.origin for b in boxes], COLORS[layer], style, width))

        self.error_plot, self.errors, self.arrow_scale = None, errors, arrow_scale
        if errors is not None:
            self._add_error_map(use_opengl)

        self.status_label = QtWidgets.QLabel(self._status_text(stitch, design_stitch))
        self.cursor_label = QtWidgets.QLabel("x = -, y = - (mask µm)")
        plots = QtWidgets.QSplitter()
        for plot in (self.plot, self.error_plot):
            if plot is not None:
                plots.addWidget(plot)
                plot.scene().sigMouseMoved.connect(lambda pos, plot=plot: self._show_cursor(plot, pos))

        layout = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.status_label)
        left.addWidget(self.cursor_label)
        if self.error_plot is not None:
            left.addWidget(self.error_label)
        left.addWidget(plots, stretch=1)
        layout.addLayout(left, stretch=1)
        layout.addWidget(self._side_panel(tile_ids))

        self.set_placement("nominal")
        self.plot.autoRange()
        self.resize(1900 if errors is not None else 1400, 950)

    def _status_text(self, stitch: StitchResult, design_stitch: StitchResult) -> str:
        parts = []
        for name, result, unplaced in (("SEM", stitch, self.sem_unplaced), ("design", design_stitch, self.design_unplaced)):
            failed = sum(r.failed for r in result.rejected)
            if unplaced or failed:
                parts.append(f"{name}: {len(unplaced)} tile(s) not stitched, {failed} overlap(s) not matched")
        if parts:
            return f"<b style='color:{FAILURE_COLOR}'>Stitching incomplete (magenta) - {'; '.join(parts)}</b>"
        return f"All {len(self.tiles)} tiles stitched, SEM and design"

    def _add(self, layer, k, item, plot=None):
        item.setZValue(0 if layer == IMAGES else 1)
        (plot or self.plot).addItem(item)
        self.items.append((layer, k, item))

    def _add_tile(self, k, tile: TileResult, design_polygons, outliers, refined):
        def local(xy_px):
            return pixel_to_tile_nm(xy_px, tile.image.shape, tile.pixel_size_nm)

        self._add(IMAGES, k, image_item(tile))
        if design_polygons:
            self._add(DESIGN, k, path_item(design_polygons, COLORS[DESIGN]))
        if len(tile.design.centers):
            self._add(DESIGN_CENTRES, k, centres_item(tile.design.centers, COLORS[DESIGN_CENTRES], "x"))
        for layer, centres_layer, found in ((OTSU, OTSU_CENTRES, tile.otsu), (REFINED, REFINED_CENTRES, tile.refined)):
            if found.contours:
                self._add(layer, k, path_item([local(c) for c in found.contours], COLORS[layer]))
                self._add(centres_layer, k, centres_item(local(found.centers), COLORS[centres_layer]))
        if len(outliers):
            found = tile.refined if refined else tile.otsu
            self._add(OUTLIERS, k, centres_item(local(found.centers[outliers]), COLORS[OUTLIERS], "o", 16))
        if k in self.sem_unplaced or k in self.design_unplaced:  # a thick frame at the nominal position
            half_w, half_h = tile.fov_nm / 2
            self._add(UNSTITCHED, k, path_item([rectangle(-half_w, half_w, -half_h, half_h)], COLORS[UNSTITCHED], width=3))

    def _add_error_map(self, use_opengl: bool):
        """The error plot: per tile, a frame, its error lines and a not-measured frame; the lines
        and frames show the selected placement's errors (set_placement)."""
        self.error_plot = make_plot(use_opengl)
        self.error_plot.setTitle("Raw error map (SEM - design)")  # short: a long title widens the plot
        self.error_label = QtWidgets.QLabel()
        link_views(self.plot, self.error_plot)
        self.error_items = {}  # tile -> (lines, dots)
        for k, tile in enumerate(self.tiles):
            half_w, half_h = tile.fov_nm / 2
            frame = [rectangle(-half_w, half_w, -half_h, half_h)]
            self._add(TILE_FRAMES, k, path_item(frame, COLORS[TILE_FRAMES]), self.error_plot)
            self._add(NOT_MEASURED, k, path_item(frame, COLORS[NOT_MEASURED], width=3), self.error_plot)
            lines = pg.PlotCurveItem(connect="pairs", pen=pg.mkPen(COLORS[ERRORS]))
            dots = pg.ScatterPlotItem(symbol="o", size=3, pen=pg.mkPen(COLORS[ERRORS]), brush=None)
            self._add(ERRORS, k, lines, self.error_plot)
            self._add(ERRORS, k, dots, self.error_plot)
            self.error_items[k] = (lines, dots)

    def _show_errors(self):
        """Error lines of the selected placement: from the (stitched) design position along the error."""
        errors = self.errors[self.mode]
        for k, (lines, dots) in self.error_items.items():
            mine = errors.tile == k
            start = errors.design_nm[mine] - self.tiles[k].center_nm  # items sit at the tile centre
            lines.setData(*arrow_segments(start, errors.error_nm[mine], self.arrow_scale))
            dots.setData(start[:, 0], start[:, 1])
        summary = error_summary(errors.error_nm)
        text = (f"Raw error (SEM − design), {PLACEMENTS[self.mode]}: dot = design position, "
                f"line = error ×{self.arrow_scale:g}; {summary['count']} contacts")
        if summary["count"]:
            text += (f", mean ({summary['mean_x_nm']:+.2f}, {summary['mean_y_nm']:+.2f}) nm, "
                     f"3σ ({summary['3sigma_x_nm']:.2f}, {summary['3sigma_y_nm']:.2f}) nm")
        if errors.skipped:
            text += f"; {len(errors.skipped)} tile(s) not measured (magenta)"
        self.error_label.setText(text)

    def set_arrow_scale(self, scale: float):
        self.arrow_scale = scale
        self._show_errors()

    def _side_panel(self, tile_ids):
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(300)
        column = QtWidgets.QVBoxLayout(panel)

        column.addWidget(QtWidgets.QLabel("Placement"))
        self.placement_buttons = {}
        for mode, text in PLACEMENTS.items():
            if mode == "first":
                text += f" ({tile_ids[self.reference] if self.reference is not None else 'no tile stitched'})"
            button = QtWidgets.QRadioButton(text)
            button.setEnabled(mode in self.shifts)
            column.addWidget(button)
            self.placement_buttons[mode] = button
        self.placement_buttons["nominal"].setChecked(True)  # before connecting: the panel is not complete yet
        for mode, button in self.placement_buttons.items():
            button.toggled.connect(lambda on, mode=mode: on and self.set_placement(mode))

        if self.error_plot is not None:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("Error lines ×"))
            self.scale_box = QtWidgets.QDoubleSpinBox()
            self.scale_box.setRange(1, 1e6)
            self.scale_box.setDecimals(0)
            self.scale_box.setValue(self.arrow_scale)
            self.scale_box.valueChanged.connect(self.set_arrow_scale)
            row.addWidget(self.scale_box)
            column.addLayout(row)

        column.addWidget(QtWidgets.QLabel("Layers"))
        self.layer_boxes = {}
        for layer in COLORS:
            box = QtWidgets.QCheckBox(layer)
            box.setChecked(layer not in HIDDEN_AT_START)
            box.setStyleSheet(f"color: {COLORS[layer]}")
            box.toggled.connect(self.update_visibility)
            column.addWidget(box)
            self.layer_boxes[layer] = box

        column.addWidget(QtWidgets.QLabel("Tiles"))
        self.tile_list = QtWidgets.QListWidget()
        for k, tile_id in enumerate(tile_ids):
            entry = QtWidgets.QListWidgetItem(tile_id)
            failures = [name for name, unplaced in (("SEM", self.sem_unplaced), ("design", self.design_unplaced))
                        if k in unplaced]
            if failures:
                entry.setText(f"{tile_id}  ({' and '.join(failures)} not stitched)")
                entry.setForeground(pg.mkColor(FAILURE_COLOR))
            entry.setCheckState(QtCore.Qt.CheckState.Checked)
            self.tile_list.addItem(entry)
        self.tile_list.itemChanged.connect(self.update_visibility)
        column.addWidget(self.tile_list, stretch=1)
        return panel

    def tile_shown(self, k: int) -> bool:
        return self.tile_list.item(k).checkState() == QtCore.Qt.CheckState.Checked

    def update_visibility(self, *_):
        not_measured = self.errors[self.mode].skipped if self.errors is not None else {}
        for layer, k, item in self.items:
            shown = self.layer_boxes[layer].isChecked() and (k is None or self.tile_shown(k))
            if layer == NOT_MEASURED:
                shown = shown and k in not_measured
            item.setVisible(shown)

    def set_placement(self, mode: str):
        """Move every tile's SEM and design items by the placement's corrections (mode "nominal",
        "mean" or "first") and show that placement's errors. Error-map items do not move: their
        data already holds the stitched design positions."""
        self.mode = mode
        sem_shift, design_shift = self.shifts[mode]
        for layer, k, item in self.items:
            if k is None:
                continue
            position = self.tiles[k].center_nm - self.origin
            if layer in SEM_LAYERS:
                position = position + sem_shift[k]
            elif layer in DESIGN_LAYERS:
                position = position + design_shift[k]
            item.setPos(*position)
        if self.errors is not None:
            self._show_errors()
        self.update_visibility()

    def _show_cursor(self, plot, scene_pos):
        pos = plot.getPlotItem().vb.mapSceneToView(scene_pos)
        x, y = (np.array([pos.x(), pos.y()]) + self.origin) / 1000
        self.cursor_label.setText(f"x = {x:.4f} µm, y = {y:.4f} µm (mask)")
