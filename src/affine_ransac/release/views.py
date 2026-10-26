"""The release's tabs (docs/SPEC.md D77), pyqtgraph, presentation style (style.py).

StitchingView: tile outlines, overlaps used, design and measured (SEM) centres, at the nominal or the stitched
placement, under a status line that grades the stitching (stitching_verdict, D109); on request (off at start, D76)
the overlaps not matched (filled in the colour of their stitching, D107), the tiles not stitched (thick red outline)
and the tile names (D108); the SEM images on request (Load SEM images: the app reads them in the background and calls
set_images), with an opacity slider; the design pattern on request (Load design: the .oas contacts filled, one colour
per stripe, set_design), so that the files of neighbouring stripes can be told apart where they overlap; the contours
and centres from the detection and after edge refinement on request (Load contours, set_contours, D113). A click
selects a tile (which image is this?) or a box of an overlap not matched (why?), drawn on top and explained beside
the plot (draw_selection, D112). Of a failed run (release.analysis.AnalysisFailed) also the tiles not measured, with
the count per reason; the cursor line names the tiles under the cursor and why each was not measured.

RegistrationView: the error after the moving-window affine, measured (SEM) − design, in one figure: an arrow map
(arrows coloured by their length dr, jet, own colour bar) and colour maps of dx and dy (jet, shared colour scale,
rasterized to the screen like the research view; draw_map), side by side and showing the same area; values of every
contact (Auto: disks of 0.8 x the pitch, or a symbol size in pixels, D121), or their mean per cell of a square grid
(grid_means); grid lines (on at start); the profiles' line width (D121). Every map has its profiles beside it
(MarginalMap): the mean per row on its right, rotated so that y runs down the side like the map's y, and the mean
per column below it (hidden unless ticked, D118), of dr beside the arrows and of its own component beside dx and dy;
both follow the map's zoom and pan; in grid average, per row and column of cells (band_means, D125). At start and on
View All the maps show the contacts' full height, centred on them, at the maps' own width (fit, D117). Each mean per
row: Auto, or a range typed in the boxes just below it (D119); under them its range (max − min) over all rows, of its
largest ramp and between two lines dragged by hand, each drawn on the graph when ticked (RangeLines, D127).
Stripes (StripeSelector): the maps and profiles show the contacts of the stripes ticked. Copy profiles / Export
profiles CSV give both profiles as a table; Copy / Save image the whole figure, the graphs of dx and dy together
(D116) or one graph (also in each graph's right-click menu and its Copy button, D120).

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
from affine_ransac.overlap import points_in_box
from affine_ransac.pipeline import (PAIRING_GATE, PairResult, RejectedPair, StitchResult, contact_spacing,
                                    contact_stripes, pair_kind, stripe_ranges)
from affine_ransac.registration import band_means, error_summary, grid_means, largest_ramp, noise_sigma, row_means
from affine_ransac.release import style
from affine_ransac.release.analysis import ReleaseResult
from affine_ransac.stitching import placement_corrections
from affine_ransac.view_contact_size import QUANTITIES, size_summary, value_levels
from affine_ransac.view_points import box_outlines, tile_rectangles
from affine_ransac.view_registration import JET, colorize, draw_map

PLACEMENTS = {"stitched": "Stitched", "nominal": "Nominal"}
LAYER_TIPS = {  # the Stitching tab's check boxes
    "Overlaps not matched": "Overlaps where both images (orange) or both .oas files (blue) have at least min_matched "
                            "contacts, but fewer of them pair up: left out of the stitching. Click one to see why",
    "Tiles not stitched": "Tiles that no used overlap links to the others: they have no position and are not measured",
}
HINT = "Click a tile to see which image it is. With Overlaps not matched shown, click a box to see why it was not matched."
CONTOURS = ("Detected contours", "Refined contours", "Detected centres", "Refined centres")  # layers on request
MAX_CONTOURS = 5000  # contacts in view whose contours are drawn; more are too small to read and slow to draw


def mask_plot(origin_nm: np.ndarray, title: str = "", data_unit_nm: float = 1.0,
              view_box: pg.ViewBox | None = None) -> pg.PlotItem:
    """A plot of data relative to origin_nm (data_unit_nm: see MaskAxis), axes in mask µm, equal x and y scales."""
    plot = pg.PlotItem(viewBox=view_box, axisItems={"bottom": style.MaskAxis("bottom", origin_nm[0], data_unit_nm),
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


def rectangles_item(boxes, pen, brush, name: str | None = None) -> pg.BarGraphItem:
    """Boxes (x0, x1, y0, y1) as one item that QPainter draws, also on an OpenGL viewport: there pyqtgraph draws plot
    lines itself, in their colour and width only, so dashed lines came out solid (D107)."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return pg.BarGraphItem(x0=b[:, 0], y0=b[:, 2], width=b[:, 1] - b[:, 0], height=b[:, 3] - b[:, 2], pen=pen,
                           brush=brush, name=name)


def overlaps_of(stitch: StitchResult, tile: int) -> dict:
    """The tile's overlaps in a stitching: neighbour -> its PairResult (used) or RejectedPair (not matched, skipped)."""
    return {o.j if o.i == tile else o.i: o for o in stitch.pairs + stitch.rejected if tile in (o.i, o.j)}


def direction(step) -> str:
    """Where a neighbour lies, from the offset (x, y) of its centre: above, below, left, right, or a corner."""
    dx, dy = step
    if abs(dy) > 2 * abs(dx):
        return "above" if dy > 0 else "below"
    side = "right" if dx > 0 else "left"
    return side if abs(dx) > 2 * abs(dy) else f"{'upper' if dy > 0 else 'lower'} {side}"


STITCHINGS = (  # name, the result's stitching and points, the gate that pairs contacts, plot symbol, text marks
    ("Measured", lambda r: r.stitch, lambda r: r.sem_points, "max_match_nm", "o", "●○"),
    ("Design", lambda r: r.design_stitch, lambda r: r.design_points, "design_max_match_nm", "s", "■□"))


def not_matched_in(r: ReleaseResult, i: int, j: int) -> list[str]:
    """The stitchings ("Measured", "Design") in which the overlap of tiles i and j was not matched."""
    return [name for name, stitch, *_ in STITCHINGS
            if isinstance(e := overlaps_of(stitch(r), i).get(j), RejectedPair) and e.failed]


def tile_details(r: ReleaseResult, tile: int, not_measured: dict | None = None) -> str:
    """A tile in words (HTML, D112): name, stripe, centre and the shift each stitching gave it; then its overlaps
    not matched (in which stitching, with a link overlap:i:j to show why), and the neighbours of the others."""
    def shift(corrections):
        c = corrections[tile]
        return f"<span style='color:{style.FAILURE}'><b>not stitched</b></span> (no position, not measured)" \
            if np.isnan(c).any() else f"({c[0]:+.1f}, {c[1]:+.1f}) nm"

    def where(k):
        return f"{r.tile_ids[k]} ({direction(r.centers[k] - r.centers[tile])})"
    x, y = r.centers[tile] / 1000
    lines = [f"<span style='font-size:13pt'><b>{r.tile_ids[tile]}</b></span>",
             f"stripe {int(r.stripe[tile]) + 1}, centre ({x:.3f}, {y:.3f}) µm",
             f"Shift from the stitching: measured {shift(r.stitch.corrections)}, design "
             f"{shift(r.design_stitch.corrections)}"]
    if not_measured and tile in not_measured:
        lines.append(f"<span style='color:{style.FAILURE}'><b>Not measured</b></span>: {not_measured[tile]}")
    overlaps = [overlaps_of(r.stitch, tile), overlaps_of(r.design_stitch, tile)]
    neighbours = sorted(set(overlaps[0]) | set(overlaps[1]), key=lambda k: r.tile_ids[k])
    failed = [(k, not_matched_in(r, tile, k)) for k in neighbours if not_matched_in(r, tile, k)]
    used = [k for k in neighbours if k not in dict(failed)
            and any(isinstance(found.get(k), PairResult) for found in overlaps)]
    skipped = [k for k in neighbours if k not in dict(failed) and k not in used]
    lines.append(f"<br><b>Overlaps with {len(neighbours)} images</b>")
    if failed:
        lines.append(f"<span style='color:{style.CAUTION}'><b>Not matched</b></span>, so not used to stitch; this image "
                     f"is placed through its other overlaps:")
        lines += [f"&nbsp;• {where(k)}, {' and '.join(names).lower()}: <a href='overlap:{tile}:{k}'>show why</a>"
                  for k, names in failed]
    else:
        lines.append(f"<span style='color:{style.OK}'>Nothing wrong:</span> every overlap with enough contacts "
                     f"was matched.")
    if used:
        lines.append("Used: " + ", ".join(where(k) for k in used))
    if skipped:
        lines.append("Skipped, too few contacts (fewer than min_matched): " + ", ".join(where(k) for k in skipped))
    return "<br>".join(lines)


def overlap_details(r: ReleaseResult, i: int, j: int) -> str:
    """One overlap in words (HTML, D112): what each stitching made of it (not matched: pairs found and needed, each
    image's contacts in it), how to read the contacts drawn, links back to both tiles (tile:k)."""
    a, b = r.tile_ids[i], r.tile_ids[j]
    corner = pair_kind(r.centers, i, j) == "corner"
    lines = [f"<span style='font-size:13pt'><b>Overlap {a} – {b}</b></span>",
             f"{b} lies {direction(r.centers[j] - r.centers[i])} of {a}"
             + (": a corner overlap, a small area with few contacts." if corner else ".")]
    for name, stitch, points, gate, _, marks in STITCHINGS:
        e = overlaps_of(stitch(r), i).get(j)
        what = "images" if name == "Measured" else ".oas files"
        if isinstance(e, PairResult):
            lines.append(f"<br><b>{name}</b>: used, {len(e.ia)} pairs.")
        elif e is not None and e.failed:
            count_a, count_b = (len(points_in_box(points(r)[k], e.box)) for k in (i, j))
            if name == "Measured" and r.settings.stitch_method == "image by image":  # stitch_image_by_image (D139)
                why = (f"Where the placement image by image put the two images, {e.matched} of the {e.in_box} "
                       f"contacts both should see pair up; at least half of them and {r.settings.min_matched} "
                       f"(min_matched) are needed. A pair is a contact in both {what}, the two within "
                       f"{PAIRING_GATE * contact_spacing(r.sem_points):.3g} nm (a quarter of the contact spacing).")
            else:
                why = (f"{e.matched} pair(s) found, {r.settings.min_matched} needed (min_matched): a pair is a "
                       f"contact in both {what}, the two within {getattr(r.settings, gate):g} nm ({gate}).")
            lines.append(f"<br><b>{name}</b>: <span style='color:{style.CAUTION}'><b>not matched</b></span>. {why} "
                         f"{a} has {count_a} contact(s) in the overlap ({marks[0]}), {b} has {count_b} ({marks[1]}).")
        elif e is not None:
            lines.append(f"<br><b>{name}</b>: skipped, only {e.in_box} contact(s) in the overlap.")
    if not_matched_in(r, i, j):
        lines.append(f"<br><b>On the plot</b>, each one's contacts in the overlap, where they were compared: filled for "
                     f"{a} (solid outline), hollow for {b} (dashed outline). Filled inside hollow: the contact is in "
                     f"both. Filled or hollow alone: in one only; next to the other's edge this is normal, as each "
                     f"leaves out the contacts its own edge cuts.")
    lines.append(f"<br>Back to <a href='tile:{i}'>{a}</a> or <a href='tile:{j}'>{b}</a>")
    return "<br>".join(lines)


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
                 not_measured: dict[int, str] | None = None, on_load_contours=None):
        """on_load_images(): called by Load SEM images (None, or a result without images: button disabled);
        on_load_design(): the same for Load design (a result without .oas files: disabled); on_load_contours(): the
        same for Load contours (no images: disabled); not_measured: of a failed run, tile index -> why it was not
        measured (a layer Tiles not measured; None: no such layer)."""
        super().__init__()
        self.result = result
        self.not_measured = not_measured
        self.contours = None  # TileContours, once loaded (set_contours)
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
        self.rectangles = rectangles
        self.layers = {}
        self.layers["Tile outlines"] = self._boxes(rectangles, pg.mkPen(style.TILES, width=1), "Tile outlines")
        self.layers["Overlaps used"] = self._boxes([self._local(p.box) for p in r.stitch.pairs],
                                                   pg.mkPen(style.OVERLAPS, width=1.5), "Overlaps used")
        self.layers["Design centres"] = self.plot.plot(pen=None, symbol="o", symbolSize=4, symbolPen=None,
                                                       symbolBrush=style.DESIGN, name="Design centres")
        used = "detected" if r.settings.refinement == "none" else "refined"  # the centres the run used
        self.layers["Measured centres"] = self.plot.plot(pen=None, symbol="o", symbolSize=3, symbolPen=None,
                                                         symbolBrush=style.SEM, name=f"Measured (SEM) centres, {used}")
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
            box.setToolTip(LAYER_TIPS.get(name, ""))
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
        loading.addSpacing(16)
        for name in CONTOURS:  # enabled once loaded (set_contours)
            box = QtWidgets.QCheckBox(name)
            box.setEnabled(False)
            box.toggled.connect(self.update_visibility)
            loading.addWidget(box)
            self.boxes[name] = box
        self.contours_button = QtWidgets.QPushButton("Load contours")
        can_load_contours = on_load_contours is not None and bool(r.image_paths)
        self.contours_button.setEnabled(can_load_contours)
        self.contours_button.setToolTip(
            "Detect and refine the contacts of every SEM image again, as the analysis did, and show both contours "
            "and centres" if can_load_contours else "No SEM images: this folder holds contour CSVs")
        if can_load_contours:
            self.contours_button.clicked.connect(on_load_contours)
        loading.addWidget(self.contours_button)
        self.contours_note = QtWidgets.QLabel("")
        self.contours_note.setStyleSheet(f"color: {style.MUTED}")
        loading.addWidget(self.contours_note)
        loading.addStretch(1)
        self.redraw_contours = QtCore.QTimer(self, singleShot=True, interval=50)  # after a zoom or pan
        self.redraw_contours.timeout.connect(self.update_contours)
        self.plot.vb.sigRangeChanged.connect(lambda *_: self.redraw_contours.start())

        self.status = QtWidgets.QLabel(self.status_text())
        self.status.setWordWrap(True)
        self.cursor_label = QtWidgets.QLabel(HINT)
        self.cursor_label.setStyleSheet(f"color: {style.MUTED}")
        self.plot.scene().sigMouseMoved.connect(self._show_cursor)
        # What a click selected (D112): None, ("tile", k) or ("overlap", i, j); drawn on top, explained beside the plot.
        self.selection, self.selection_items, self.selection_legend = None, [], []
        self.details = QtWidgets.QTextBrowser()
        self.details.setFixedWidth(400)
        self.details.setOpenLinks(False)
        self.details.anchorClicked.connect(self._link)
        self.details.hide()
        self.plot.scene().sigMouseClicked.connect(self._clicked)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(style.figure_bar("Stitching", style.image_buttons(lambda: self.graphics, "stitching")))
        layout.addWidget(self.status)
        layout.addLayout(controls)
        layout.addLayout(loading)
        figure_row = QtWidgets.QHBoxLayout()
        figure_row.addWidget(self.graphics, stretch=1)
        figure_row.addWidget(self.details)
        layout.addLayout(figure_row, stretch=1)
        layout.addWidget(self.cursor_label)
        # The legend lists the layers shown only: with every layer listed it grew long, and its eye icons hid an item
        # without unticking its box.
        self.legend_entries = [(sample.item, label.text) for sample, label in self.plot.legend.items]
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
        """rectangles_item of the boxes, in the plot and in the legend with their count."""
        item = rectangles_item(boxes, pen, brush, f"{name} ({len(boxes)})")
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
            text += f" Tick {' and '.join(ticks)} to see where" + \
                    ("; click a box to see why." if "<i>Overlaps not matched</i>" in ticks else ".")
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
        if self.contours is not None:
            c = self.contours
            for name, centres in (("Detected centres", c.detected_centres), ("Refined centres", c.refined_centres)):
                xy = centres + self._contour_offsets()
                self.layers[name].setData(xy[:, 0], xy[:, 1])
            self.update_contours()
        self.draw_selection()  # images loaded since come to the front

    def update_visibility(self, *_):
        for name, items in self.layers.items():
            for item in items if isinstance(items, list) else [items]:
                item.setVisible(self.boxes[name].isChecked())
        for item in self.image_items:
            item.setVisible(self.images_box.isChecked())
        for _, item in self.design_items:
            item.setVisible(self.design_box.isChecked())
        self.plot.legend.clear()
        for item, name in self.legend_entries + self.selection_legend:
            if item.isVisible():
                self.plot.legend.addItem(item, name)
        for sample, _ in self.plot.legend.items:  # a click on an entry hides it: untick its layer to match
            sample.sigClicked.connect(self._legend_clicked)

    def _legend_clicked(self, item):
        """A click on a legend entry hid its item: untick its layer too, once the click is handled (unticking rebuilds
        the legend, the clicked entry with it)."""
        for name, items in self.layers.items():
            if any(item is other for other in (items if isinstance(items, list) else [items])):
                QtCore.QTimer.singleShot(0, lambda name=name: self.boxes[name].setChecked(False))

    def _clicked(self, event):
        """A left click on the plot (not on its legend) selects what is there (select_at)."""
        position = event.scenePos()
        if event.button() != QtCore.Qt.MouseButton.LeftButton or event.double() or \
                not self.plot.vb.sceneBoundingRect().contains(position) or \
                self.plot.legend.sceneBoundingRect().contains(position):
            return
        point = self.plot.vb.mapSceneToView(position)
        self.select_at(np.array([point.x(), point.y()]) + self.origin)

    def select_at(self, point_nm):
        """Select what a click at this point (mask nm) is on (D112): with Overlaps not matched shown, a not-matched
        overlap whose box holds it (the smallest, e.g. a corner inside a strip); else the tile whose box holds it (in
        an overlap, the one whose centre is nearer); else nothing."""
        r = self.result

        def holds(box):
            return box[0] <= point_nm[0] <= box[1] and box[2] <= point_nm[1] <= box[3]
        if self.boxes["Overlaps not matched"].isChecked():
            boxes = [f for f in r.stitch.rejected + r.design_stitch.rejected if f.failed and holds(f.box)]
            if boxes:
                f = min(boxes, key=lambda f: (f.box[1] - f.box[0]) * (f.box[3] - f.box[2]))
                return self.select(("overlap", f.i, f.j))
        inside = np.flatnonzero(np.all(np.abs(r.centers - point_nm) <= r.sizes / 2, axis=1))
        nearest = inside[np.argmin(np.linalg.norm(r.centers[inside] - point_nm, axis=1))] if len(inside) else None
        self.select(None if nearest is None else ("tile", int(nearest)))

    def _link(self, url: QtCore.QUrl):
        """A link in the details beside the plot: tile:k or overlap:i:j; an overlap is then shown whole."""
        kind, *numbers = url.toString().split(":")
        numbers = [int(n) for n in numbers]
        self.select((kind, *numbers))
        if kind == "overlap":
            i, j = numbers
            x0, x1, y0, y1 = self._local((overlaps_of(self.result.stitch, i).get(j)
                                          or overlaps_of(self.result.design_stitch, i)[j]).box)
            margin = 0.6 * max(x1 - x0, y1 - y0) + 100  # nm: a corner overlap is only a few hundred nm wide
            self.plot.vb.setRange(xRange=(x0 - margin, x1 + margin), yRange=(y0 - margin, y1 + margin), padding=0)

    def select(self, selection):
        """Select None, ("tile", k) or ("overlap", i, j) and draw it (draw_selection)."""
        self.selection = selection
        self.draw_selection()

    def draw_selection(self):
        """Draw the selection on top and explain it beside the plot (D112).
        A tile: only its outline (black on white, seen on any image) and its name; its SEM image and design pattern
        come to the front, so that it is its own image that shows in its overlaps (tile_details).
        An overlap not matched: only its two tiles, the first outlined solid and the second dashed, both named; the
        overlap's box in the colour of each stitching that did not match it; and each tile's contacts in that box
        where they were compared (nominal placement, as the box), filled for the first and hollow for the second
        (circles measured, squares design), named in the legend (overlap_details)."""
        for item in self.selection_items:
            self.plot.removeItem(item)
        self.selection_items, self.selection_legend = [], []
        for item in self.image_items:
            item.setZValue(-10)
        for _, item in self.design_items:
            item.setZValue(-5)
        if self.selection is None:
            self.details.hide()
        elif self.selection[0] == "tile":
            self._draw_tile(self.selection[1])
        else:
            self._draw_overlap(*self.selection[1:])
        for item in self.selection_items:
            item.setZValue(50)
            self.plot.addItem(item)
        if hasattr(self, "legend_entries"):  # not while the view is being built
            self.update_visibility()

    def _outline(self, tile: int, dashed: bool = False):
        """A tile's outline, black on a white halo, and its name."""
        line = QtCore.Qt.PenStyle.DashLine if dashed else QtCore.Qt.PenStyle.SolidLine
        box = [self.rectangles[tile]]
        label = name_label(self.result.tile_ids[tile])
        label.setPos(*(self.result.centers[tile] - self.origin))
        self.selection_items += [rectangles_item(box, pg.mkPen("w", width=6), pg.mkBrush(None)),
                                 rectangles_item(box, pg.mkPen("k", width=3, style=line), pg.mkBrush(None)), label]

    def _draw_tile(self, tile: int):
        self._outline(tile)
        if self.image_items:
            self.image_items[tile].setZValue(-9)
        for k, item in self.design_items:
            if k == tile:
                item.setZValue(-4)
        self.details.setHtml(tile_details(self.result, tile, self.not_measured))
        self.details.show()

    def _draw_overlap(self, i: int, j: int):
        r = self.result
        self._outline(i)
        self._outline(j, dashed=True)
        for name, stitch, points, _, symbol, _ in STITCHINGS:
            entry = overlaps_of(stitch(r), i).get(j)
            if not (isinstance(entry, RejectedPair) and entry.failed):
                continue
            color = style.SEM if name == "Measured" else style.DESIGN
            self.selection_items.append(rectangles_item([self._local(entry.box)], pg.mkPen(color, width=2.5),
                                                        pg.mkBrush(None)))
            for k, filled in ((i, True), (j, False)):  # where they were compared: nominal, as the box
                xy = points(r)[k][points_in_box(points(r)[k], entry.box)] - self.origin
                if filled:
                    marks = [pg.PlotDataItem(xy[:, 0], xy[:, 1], pen=None, symbol=symbol, symbolSize=9,
                                             symbolBrush="k", symbolPen=pg.mkPen("w", width=1.5))]
                else:  # a black ring on a white one
                    marks = [pg.PlotDataItem(xy[:, 0], xy[:, 1], pen=None, symbol=symbol, symbolSize=15,
                                             symbolBrush=pg.mkBrush(None), symbolPen=pg.mkPen(ring, width=width))
                             for ring, width in (("w", 4), ("k", 2))]
                self.selection_items += marks
                self.selection_legend.append((marks[-1], f"{r.tile_ids[k]}: {name.lower()} contacts in the overlap"))
        self.details.setHtml(overlap_details(r, i, j))
        self.details.show()

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

    def set_contours(self, contours):
        """Show the contours and centres from the detection and after edge refinement (release.analysis
        read_contours, TileContours; D113) as four layers, ticked once loaded (with refinement none the two of the
        detection only, D124); they follow the placement."""
        r, self.contours = self.result, contours
        for name, color in zip(CONTOURS, (style.DETECTED, style.REFINED) * 2):
            if name in self.layers:
                self.plot.removeItem(self.layers[name])
                self.legend_entries = [(item, label) for item, label in self.legend_entries
                                       if item is not self.layers[name]]
            # the detection method and the edge rule (D122) the contours come from
            label = f"{name} ({r.settings.detection if name.startswith('Detected') else r.settings.refinement})"
            if name.endswith("contours"):
                item = pg.PlotDataItem(pen=pg.mkPen(color, width=1.5), connect="finite")
            else:
                item = pg.PlotDataItem(pen=None, symbol="+", symbolSize=9, symbolPen=pg.mkPen(color, width=1.5))
            self.plot.addItem(item)
            self.legend_entries.append((item, label))
            self.layers[name] = item
            # with refinement none the refined contours and centres are the detected ones: not offered (D124)
            offered = name.startswith("Detected") or r.settings.refinement != "none"
            self.boxes[name].setEnabled(offered)
            self.boxes[name].setChecked(offered)
            self.boxes[name].setToolTip("" if offered else "Refinement none: the detected contours and centres are "
                                                           "the measured ones")
        self.contours_button.setEnabled(False)
        self.update_positions()
        self.update_visibility()

    def _contour_offsets(self) -> np.ndarray:
        """(n, 2) what turns each loaded contact's tile-local nm into plot data: its tile's centre and shift at the
        placement shown, minus the origin."""
        tile = self.contours.tile
        return self.result.centers[tile] + self.shifts[self.mode()][0][tile] - self.origin

    def update_contours(self):
        """Draw the contours of the contacts in view (with a margin), at the placement shown; none if more than
        MAX_CONTOURS are in view (too small to read, and slow to draw), and the note beside the button says to zoom
        in."""
        c = self.contours
        if c is None:
            return
        offsets = self._contour_offsets()
        centres = c.refined_centres + offsets
        (x0, x1), (y0, y1) = self.plot.vb.viewRange()
        margin = 0.05 * max(x1 - x0, y1 - y0)
        inside = np.flatnonzero((centres[:, 0] > x0 - margin) & (centres[:, 0] < x1 + margin)
                                & (centres[:, 1] > y0 - margin) & (centres[:, 1] < y1 + margin))
        too_many = len(inside) > MAX_CONTOURS
        shown = self.boxes["Detected contours"].isChecked() or self.boxes["Refined contours"].isChecked()
        self.contours_note.setText("Zoom in to see the contours" if too_many and shown else "")
        if too_many:
            inside = inside[:0]
        lengths = c.start[inside + 1] - c.start[inside]
        rows = np.repeat(c.start[inside] - np.cumsum(lengths) + lengths, lengths) + np.arange(lengths.sum())
        offset = np.repeat(offsets[inside], lengths, axis=0)
        for name, points in (("Detected contours", c.detected), ("Refined contours", c.refined)):
            xy = points[rows] + offset
            self.layers[name].setData(xy[:, 0], xy[:, 1], connect="finite")

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
    if r.slipped and r.settings.stitch_method == "image by image":
        lines.append(line(style.FAILURE, f"Error: {len(r.slipped)} overlap(s) do not line up.",
                          f"({shown}) Where the stitching placed their images, fewer than half of their contacts pair "
                          f"up, another shift of one image pairs clearly more of them, or the two images lie more than "
                          f"overlap_tolerance_nm apart, so the errors of the tiles around them can be wrong."))
    elif r.slipped:
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
        failed = {name: [f for f in stitch.rejected if f.failed]
                  for name, stitch in (("measured", r.stitch), ("design", r.design_stitch))}
        if any(failed.values()):
            counts = " and ".join(f"{len(found)} {name}" for name, found in failed.items() if found)
            corners = sum(pair_kind(r.centers, f.i, f.j) == "corner" for found in failed.values() for f in found)
            total = sum(len(found) for found in failed.values())
            where = "" if not corners else " (all at corners," if corners == total else f" ({corners} at corners,"
            lines.append(line(style.CAUTION, "Stitching OK, with overlaps not matched.",
                              f"All {n} tiles are stitched, so all are measured. {counts} overlap(s) were not matched"
                              + (f"{where} where two images share only a few contacts)" if corners else "")
                              + " and are left out: their tiles are placed through their other overlaps."))
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


class FitViewBox(pg.ViewBox):
    """A view box whose View All (right-click menu) calls view_all() instead of fitting everything in it (D114): a
    colour map's image is drawn for its view, so View All zoomed out a little more every time."""

    def __init__(self, view_all):
        super().__init__()
        self.view_all = view_all

    def autoRange(self, *args, **kwargs):
        self.view_all()


class MarginalMap:
    """A map with the mean per row on its right (rotated: the row's y runs down the side, aligned with the map's
    y) and the mean per column below it (aligned with the map's x), in a GraphicsLayout at (row, col): the map
    takes (row, col), the row profile (row, col + 1), the column profile (row + 2, col); row + 1 is left for the
    controls below the row profile (RowRange). The profiles follow the map's zoom and pan. Positions in µm relative
    to origin_nm."""
    AXIS_PX = 64  # left axis width of the map and of the column profile, so that their x line up

    def __init__(self, layout: pg.GraphicsLayout, row: int, col: int, origin_nm, title: str, name: str, view_all):
        """name: what the map shows, for naming its graphs (e.g. "dx"); view_all(): the map's View All. A
        profile's View All fits its error axis only: the other one follows the map."""
        self.name, self.col = name, col
        self.map = mask_plot(origin_nm, title, data_unit_nm=1000, view_box=FitViewBox(view_all))
        self.rows = pg.PlotItem(viewBox=FitViewBox(lambda: self.rows.vb.enableAutoRange(x=True)),
                                axisItems={"left": style.MaskAxis("left", origin_nm[1], 1000)})
        self.columns = pg.PlotItem(viewBox=FitViewBox(lambda: self.columns.vb.enableAutoRange(y=True)),
                                   axisItems={"bottom": style.MaskAxis("bottom", origin_nm[0], 1000)})
        layout.addItem(self.map, row=row, col=col)
        layout.addItem(self.rows, row=row, col=col + 1)
        layout.addItem(self.columns, row=row + 2, col=col)
        style.style_plot(self.rows, "Mean per row", "error (nm)")
        style.style_plot(self.columns, "Mean per column", "x (µm)", "error (nm)")
        for plot in (self.map, self.rows, self.columns):
            plot.hideButtons()  # its "A" auto-ranged every axis, as View All did
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
        grid.setRowStretchFactor(row + 2, 1)

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


class RowRange(QtWidgets.QWidget):
    """The error range of one mean per row, in boxes just below it (D119): Auto, fitted to its values (the boxes
    show its range), or from the first box to the second (Enter applies and unticks Auto). A zoom with the mouse
    there shows its range in the boxes; View All there ticks Auto again. Over the figure (a child of it), so not in
    its images."""

    def __init__(self, plot: pg.PlotItem, figure: pg.GraphicsView):
        super().__init__(figure)
        self.plot, self.figure = plot, figure
        self.boxes = [QtWidgets.QDoubleSpinBox(decimals=2, minimum=-1e4, maximum=1e4, singleStep=0.5,
                                               keyboardTracking=False) for _ in range(2)]
        self.auto = QtWidgets.QCheckBox("Auto")
        self.auto.setChecked(True)
        self.setToolTip("Error range of this mean per row (nm). Auto: fitted to its values")
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        for widget in (self.boxes[0], QtWidgets.QLabel("to"), self.boxes[1], self.auto):
            row.addWidget(widget)
        self.auto.toggled.connect(self.apply)
        for box in self.boxes:
            box.valueChanged.connect(self.typed)
        plot.vb.sigXRangeChanged.connect(lambda *_: self.auto.isChecked() and self.show_range())
        plot.vb.sigRangeChangedManually.connect(self.zoomed)
        plot.vb.view_all = self.view_all  # its View All (FitViewBox)
        plot.geometryChanged.connect(self.place)

    def apply(self, *_):
        low, high = (box.value() for box in self.boxes)
        if self.auto.isChecked():
            self.plot.vb.enableAutoRange(x=True)
        elif low < high:
            self.plot.setXRange(low, high, padding=0)
        self.show_range()

    def typed(self, *_):
        self.set_auto(False)
        self.apply()

    def zoomed(self, *_):
        self.set_auto(False)
        self.show_range()

    def view_all(self):
        self.set_auto(True)
        self.apply()

    def set_auto(self, on: bool):
        self.auto.blockSignals(True)
        self.auto.setChecked(on)
        self.auto.blockSignals(False)

    def show_range(self):
        for box, value in zip(self.boxes, self.plot.vb.viewRange()[0]):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)

    def place(self):
        """Just below the plot, centred on its graph."""
        size = self.sizeHint()
        left = on_figure(self.figure, self.plot.vb).center().x() - size.width() // 2
        self.setGeometry(QtCore.QRect(QtCore.QPoint(left, on_figure(self.figure, self.plot).bottom() + 2), size))


RANGE_PENS = {  # the min and max lines of each kind of range of a mean per row (D127)
    "all": pg.mkPen(style.ALL_ROWS, width=1.5, style=QtCore.Qt.PenStyle.DotLine),
    "ramp": pg.mkPen(style.RAMP, width=1.5, style=QtCore.Qt.PenStyle.DashLine),
    "manual": pg.mkPen(style.MANUAL, width=1.5),
}


class RangeLines(QtWidgets.QWidget):
    """The range (max − min) of one mean per row, three ways, on one line just below its range boxes (D127): "all"
    over all its rows, "ramp" of its largest ramp (the profile split where it turns by more than the wiggle height,
    largest_ramp), "manual" between two lines dragged by hand. Each has a tick box (off at start) that draws its min
    and max on the graph as two vertical lines in its colour: for all and ramp the rows they come from circled and
    the ramp's rows shaded; the manual lines with their values at the top. The line of ranges is over the figure (a
    child of it), so not in its images; what it draws on the graph is."""
    NOISE_TIMES = 8  # Auto wiggle height: this many times the noise of the row means (noise_sigma)

    def __init__(self, plot: pg.PlotItem, figure: pg.GraphicsView, above: QtWidgets.QWidget):
        """above: the range boxes this line goes under."""
        super().__init__(figure)
        self.plot, self.figure, self.above = plot, figure, above
        self.row_y, self.mean = np.zeros(1), np.zeros(1)  # the profile: its rows' y (the plot's µm) and means
        self.height_nm = 0.0  # wiggle height, 0: Auto
        self.placed = False  # the manual lines: put on the first profile's min and max, then left to the user
        self.lines, self.items, self.boxes = {}, {}, {}
        for kind, pen in RANGE_PENS.items():
            manual = kind == "manual"
            label = {"label": "{value:+.2f}", "labelOpts": {
                "position": 0.97, "anchors": [(0, 0), (1, 0)], "color": style.TEXT,
                "fill": pg.mkBrush(255, 255, 255, 210)}} if manual else {}
            self.lines[kind] = [pg.InfiniteLine(angle=90, pen=pen, movable=manual,
                                                hoverPen=pg.mkPen(pen.color(), width=3), **label) for _ in range(2)]
            self.items[kind] = list(self.lines[kind])
            if not manual:  # the rows of the min and the max
                marks = pg.ScatterPlotItem(symbol="o", size=10, pen=pg.mkPen(pen.color(), width=1.5), brush=None)
                self.items[kind].append(marks)
            self.boxes[kind] = QtWidgets.QCheckBox()
            self.boxes[kind].setIcon(swatch(pen))
            self.boxes[kind].toggled.connect(lambda on, kind=kind: self.show_kind(kind, on))
        self.shade = pg.LinearRegionItem(orientation="horizontal", movable=False, brush=translucent(style.RAMP, 45),
                                         pen=pg.mkPen(None))
        self.items["ramp"].append(self.shade)
        for kind, items in self.items.items():
            for item in items:
                item.setZValue({"all": -5, "ramp": -5, "manual": 10}[kind])  # by hand: over the profile, to drag
                item.setVisible(False)
                plot.addItem(item, ignoreBounds=True)
        self.shade.setZValue(-20)
        self.marks = {kind: self.items[kind][2] for kind in ("all", "ramp")}
        for line in self.lines["manual"]:
            line.sigPositionChanged.connect(self.show_manual)

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        for box in self.boxes.values():
            row.addWidget(box)
        row.addWidget(QtWidgets.QLabel("nm"))
        plot.geometryChanged.connect(self.place)

    def set_profile(self, row_y_um: np.ndarray, mean: np.ndarray):
        """The profile shown: its rows' y (the plot's µm, from the lowest up) and their means (nm)."""
        self.row_y, self.mean = row_y_um, mean
        if not self.placed:
            for line, value in zip(self.lines["manual"], (mean.min(), mean.max())):
                line.setValue(value)
            self.placed = True
        self.update_values()

    def set_height(self, height_nm: float):
        """Wiggles up to this height do not end a ramp; 0: Auto."""
        self.height_nm = height_nm
        self.update_values()

    def wiggle_height(self) -> float:
        return self.height_nm or self.NOISE_TIMES * noise_sigma(self.mean)

    def update_values(self):
        first, last = largest_ramp(self.mean, self.wiggle_height())
        self.shade.setRegion((self.row_y[first], self.row_y[last]))
        wiggle = f"{self.wiggle_height():.2g} nm" + (" (Auto)" if not self.height_nm else "")
        for kind, rows, what in (("all", slice(None), "the mean of all rows"),
                                 ("ramp", slice(first, last + 1), "the largest ramp, shaded (it ends where the "
                                  f"profile turns by more than {wiggle})")):
            y, values = self.row_y[rows], self.mean[rows]
            low, high = values.argmin(), values.argmax()
            for line, k in zip(self.lines[kind], (low, high)):
                line.setValue(values[k])
            self.marks[kind].setData([values[low], values[high]], [y[low], y[high]])
            self.show_range(kind, values[low], values[high], f"Lowest and highest of {what}, their rows circled")
        self.show_manual()

    def show_manual(self, *_):
        low, high = sorted(line.value() for line in self.lines["manual"])
        self.show_range("manual", low, high, "Two lines you drag on the graph")

    def show_range(self, kind: str, low: float, high: float, what: str):
        box = self.boxes[kind]
        box.setText(f"{kind} {high - low:.2f}")
        box.setToolTip(f"{what}: min {low:+.2f} nm, max {high:+.2f} nm, range {high - low:.2f} nm. Tick to draw them")
        self.place()

    def show_kind(self, kind: str, on: bool):
        for item in self.items[kind]:
            item.setVisible(on)

    def place(self):
        """Under the range boxes, centred on the graph, inside the figure."""
        size = self.sizeHint()
        left = on_figure(self.figure, self.plot.vb).center().x() - size.width() // 2
        left = max(0, min(left, self.figure.width() - size.width()))
        top = on_figure(self.figure, self.plot).bottom() + 4 + self.above.sizeHint().height()
        self.setGeometry(QtCore.QRect(QtCore.QPoint(left, top), size))


def swatch(pen: QtGui.QPen) -> QtGui.QIcon:
    """A short stretch of line in this pen, for a tick box that draws such lines (drawn at twice its size)."""
    pixmap = QtGui.QPixmap(40, 16)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    line = QtGui.QPen(pen)
    line.setWidthF(3)
    painter.setPen(line)
    painter.drawLine(2, 8, 38, 8)
    painter.end()
    return QtGui.QIcon(pixmap)


class CopyButton(QtWidgets.QToolButton):
    """A small Copy button over a graph's top-left corner, above its y axis (D120): copies the graph, with the items
    given (a map's colour bar), to the clipboard. Over the figure (a child of it), so not in its images; shown while
    the graph is."""

    def __init__(self, plot: pg.PlotItem, items: list, figure: pg.GraphicsView, name: str):
        super().__init__(figure)
        self.plot, self.figure = plot, figure
        self.setIcon(copy_icon())  # an icon: "Copy" covered the start of a mean per row's title
        self.setToolTip(f"Copy the {name} to the clipboard (paste it into a slide)")
        self.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setImage(style.render_graph(figure, items)))
        plot.geometryChanged.connect(self.place)
        plot.visibleChanged.connect(lambda: self.setVisible(plot.isVisible()))

    def place(self):
        self.setGeometry(QtCore.QRect(on_figure(self.figure, self.plot).topLeft() + QtCore.QPoint(2, 2),
                                      self.sizeHint()))


def copy_icon() -> QtGui.QIcon:
    """Two sheets, one over the other: the usual sign for copy (drawn at twice its size, for sharp high-DPI)."""
    pixmap = QtGui.QPixmap(32, 32)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtGui.QPen(QtGui.QColor(style.MUTED), 2.5))
    painter.setBrush(QtGui.QColor("white"))
    painter.drawRoundedRect(QtCore.QRectF(4, 3, 17, 20), 3, 3)
    painter.drawRoundedRect(QtCore.QRectF(11, 9, 17, 20), 3, 3)
    painter.end()
    return QtGui.QIcon(pixmap)


def on_figure(figure: pg.GraphicsView, item) -> QtCore.QRect:
    """A graphics item's rectangle on the figure widget, where widgets over it are placed."""
    return figure.mapFromScene(item.sceneBoundingRect()).boundingRect().translated(figure.viewport().pos())


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
    """The plots show the same area: a zoom or pan with the mouse in one is copied to the others, as the same centre
    and the same scale (µm per pixel), so plots of different widths still agree. A hidden plot neither leads nor
    follows. Only the user's zooms and pans: copied while the window is resized, a range was taken from a plot laid
    out already and given to one not yet, which then showed less than it should (D114)."""
    def follow(source):
        def update(*_):
            if not source.isVisible() or source.vb.width() <= 0:
                return
            (x0, x1), y_range = source.vb.viewRange()
            per_pixel, centre = (x1 - x0) / source.vb.width(), (x0 + x1) / 2
            for target in plots:
                if target is not source and target.isVisible():
                    half = per_pixel * target.vb.width() / 2
                    target.vb.setRange(xRange=(centre - half, centre + half), yRange=y_range, padding=0)
        source.vb.sigRangeChangedManually.connect(update)
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
        self.arrow_panel = MarginalMap(grid, 0, 0, self.origin, "Δr (nm)", "arrow", self.fit)
        self.arrow_profiles = [self.arrow_panel.add_profile("dr", style.DR)]
        self.arrows = [pg.PlotCurveItem(connect="pairs", pen=pg.mkPen(QtGui.QColor(*rgb[:3]), width=1), antialias=False)
                       for rgb in JET.getLookupTable(nPts=ARROW_COLOURS)]
        for item in self.arrows:
            self.arrow_panel.map.addItem(item)
        self.dr_bar = jet_bar((0.0, 1.0), "|measured − design| (nm)")
        self.dr_bar.sigLevelsChanged.connect(lambda *_: self.draw_arrows())
        grid.addItem(self.dr_bar, row=0, col=2)
        grid.layout.setColumnSpacing(2, 24)  # room between its axis label and the dx map's
        self.maps = []  # (MarginalMap, image, component)
        self.map_profiles = []
        for component, (name, color) in enumerate((("Δx", style.DX), ("Δy", style.DY))):
            panel = MarginalMap(grid, 0, 3 + 2 * component, self.origin, f"{name} (nm)", ("dx", "dy")[component],
                                self.fit)
            self.map_profiles.append(panel.add_profile(name, color))
            image = pg.ImageItem(axisOrder="row-major")
            image.setZValue(-100)  # under the grid lines
            panel.map.addItem(image)
            self.maps.append((panel, image, component))
        self.color_bar = jet_bar((-1.0, 1.0), "measured − design (nm)")
        self.color_bar.sigLevelsChanged.connect(lambda *_: self.draw_maps())
        grid.addItem(self.color_bar, row=0, col=7)
        self.panels = [self.arrow_panel] + [panel for panel, _, _ in self.maps]
        link_views([panel.map for panel in self.panels])
        self.redraw = QtCore.QTimer(self, singleShot=True, interval=50)
        self.redraw.timeout.connect(self.draw_maps)
        for panel, _, _ in self.maps:
            panel.map.vb.sigRangeChanged.connect(lambda *_: self.redraw.start())
            panel.map.vb.sigResized.connect(lambda *_: self.redraw.start())
        # The maps show the contacts' full height (fit) until the user zooms or pans one: then a change of values
        # keeps the area shown.
        self.fitted = True
        for panel in self.panels:
            panel.map.vb.sigRangeChangedManually.connect(lambda *_: setattr(self, "fitted", False))

        self.arrow_scale = QtWidgets.QDoubleSpinBox(decimals=0, minimum=1, maximum=1e7)
        self.arrow_scale.setToolTip("Arrows are drawn this many times longer than the error")
        self.arrow_scale.valueChanged.connect(self.draw_arrows)
        self.arrow_key = QtWidgets.QLabel()

        # Every contact, or the mean over the contacts in each cell of a square grid.
        self.values = QtWidgets.QComboBox()
        self.values.addItems(["Every contact", "Grid average"])
        self.values.setToolTip("Grid average: one value per square cell, the mean of the contacts in it; the means "
                               "per row and per column then average the contacts of each row and column of cells")
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
        self.columns_box = QtWidgets.QCheckBox("Mean per column")
        self.columns_box.setToolTip("Show the mean per column below each map")
        self.columns_box.toggled.connect(self.show_columns)
        self.symbol_size = QtWidgets.QSpinBox(minimum=0, maximum=50, suffix=" px", specialValueText="Auto")
        self.symbol_size.setToolTip("Size of the contacts on the dx and dy maps, pixels across at every zoom. Auto: "
                                    "a disk of 0.8 x the pitch, which grows as you zoom in")
        self.symbol_size.valueChanged.connect(lambda *_: self.draw_maps())
        self.line_width = QtWidgets.QSpinBox(minimum=1, maximum=10, value=2, suffix=" px")
        self.line_width.setToolTip("Width of the lines of the means per row and per column")
        self.line_width.valueChanged.connect(self.set_line_width)
        # Each mean per row has its error range in boxes just below it (D119), and under them its range three ways
        # (D127), in a row of the figure left for them.
        self.row_ranges = [RowRange(panel.rows, self.figure) for panel in self.panels]
        self.range_lines = [RangeLines(panel.rows, self.figure, boxes)
                            for panel, boxes in zip(self.panels, self.row_ranges)]
        room = QtWidgets.QGraphicsWidget()
        room.setMinimumHeight(self.row_ranges[0].sizeHint().height() + self.range_lines[0].sizeHint().height() + 6)
        room.setMaximumHeight(room.minimumHeight())
        grid.addItem(room, row=1, col=0)
        self.wiggle = QtWidgets.QDoubleSpinBox(decimals=2, minimum=0, maximum=1e3, singleStep=0.05, suffix=" nm",
                                               specialValueText="Auto", keyboardTracking=False)
        self.wiggle.setToolTip("Wiggles up to this height do not end a ramp of a mean per row (its range is under it, "
                               f"\"ramp\"). Auto: {RangeLines.NOISE_TIMES} x the noise of each mean per row")
        self.wiggle.valueChanged.connect(lambda height: [lines.set_height(height) for lines in self.range_lines])

        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        copy_profiles = QtWidgets.QPushButton("Copy profiles")
        copy_profiles.setToolTip("Copy the mean per row and per column (µm; dx, dy, dr nm) to the clipboard, "
                                 "tab-separated (pastes into Excel)")
        copy_profiles.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.profiles_text("\t")))
        export_profiles = QtWidgets.QPushButton("Export profiles CSV...")
        export_profiles.clicked.connect(self.export_profiles)
        self.copy_buttons = []  # one over every graph (D120)
        for panel in self.panels:
            for plot, items, name in self.graphs_of(panel):
                style.graph_actions(plot, self.figure, items, name)
                self.copy_buttons.append(CopyButton(plot, items, self.figure, name))

        controls = QtWidgets.QHBoxLayout()
        for widget in (self.stripes, QtWidgets.QLabel("   Values:"), self.values, QtWidgets.QLabel("cell"), self.cell,
                       QtWidgets.QLabel("  "), self.grid_lines, QtWidgets.QLabel("   Arrow scale ×"), self.arrow_scale,
                       self.arrow_key):
            controls.addWidget(widget)
        controls.addStretch(1)
        for button in [copy_profiles, export_profiles] + style.image_menus(self.shown_graphs):
            controls.addWidget(button)
        row_controls = QtWidgets.QHBoxLayout()
        for widget in (self.columns_box, QtWidgets.QLabel("    Symbol size"), self.symbol_size,
                       QtWidgets.QLabel("   Line width"), self.line_width, QtWidgets.QLabel("   Ramp wiggles up to"),
                       self.wiggle):
            row_controls.addWidget(widget)
        row_controls.addStretch(1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addLayout(controls)
        layout.addLayout(row_controls)
        layout.addWidget(self.figure, stretch=1)

        self.values.currentIndexChanged.connect(self.set_values)
        self.cell.valueChanged.connect(self.set_values)
        self.show_grid_lines(True)
        self.show_columns(False)
        self.select_stripes()

    def select_stripes(self):
        """Show the contacts of the ticked stripes: their maps, profiles, colour scales and summary, the view on
        them. The errors stay those of the run's moving-window affine (every contact, or per stripe), not fitted
        again."""
        r, s = self.result, self.result.settings
        selected = self.stripes.selected()
        shown = self.in_stripe[:, selected].any(axis=1)
        self.design_nm, self.error_nm = r.design_nm[shown], r.moving.residuals[shown]
        self.position_um = (self.design_nm - self.origin) / 1000
        length = np.linalg.norm(self.error_nm, axis=1)
        self.values_nm = np.column_stack([self.error_nm, length])  # dx, dy, dr of every contact
        self.contact_rows = row_means(self.design_nm[:, 1], self.values_nm, s.row_gap_nm)[:2]  # (y, mean)
        self.contact_columns = row_means(self.design_nm[:, 0], self.values_nm, s.row_gap_nm)[:2]  # (x, mean)
        limit = float(np.percentile(np.abs(self.error_nm), 99)) or 1.0
        self.color_bar.setLevels((-limit, limit))
        self.dr_bar.setLevels((0.0, float(np.percentile(length, 99)) or 1.0))

        window = f"moving-window affine: window {s.moving_window_um:g} µm, step {s.moving_step_um:g} µm"
        fitted = "fitted per stripe, every image by its own stripe's windows" if s.moving_window_per_stripe else \
            "fitted on every contact"
        total = self.in_stripe.shape[1]
        stripes = f"all {total} stripes" if len(selected) == total else \
            f"stripe(s) {', '.join(str(k + 1) for k in selected)} of {total}"
        self.summary.setText(f"<b>{len(self.error_nm)} contacts</b> of {stripes}, {len(self.contact_rows[0])} rows, "
                             f"{len(self.contact_columns[0])} columns. Error = measured (SEM) − design after the "
                             f"{window}, {fitted}: {summary_text(self.error_nm)}.")
        self.fitted = True  # show them all
        self.set_values()

    def grid_mode(self) -> bool:
        return self.values.currentIndex() == 1

    def set_values(self, *_):
        """Every contact or the grid average: the grid and the profiles computed again, the arrow scale reset, the
        maps fitted again unless zoomed, then drawn."""
        self.cell.setEnabled(self.grid_mode())
        self.symbol_size.setEnabled(not self.grid_mode())  # grid average: cells, no symbols
        corner, mean, _ = grid_means(self.design_nm, self.error_nm, self.cell.value() * 1000)
        self.grid_corner_um, self.grid_mean = (corner - self.origin) / 1000, mean
        self.draw_profiles()
        vectors = self.arrow_vectors()[1]
        largest = float(np.percentile(np.linalg.norm(vectors, axis=1), 99)) if len(vectors) else 1.0
        spacing_nm = self.cell.value() * 1000 if self.grid_mode() else self.pitch_nm
        self.arrow_scale.blockSignals(True)
        self.arrow_scale.setValue(nice_number(0.5 * spacing_nm / (largest or 1.0)))  # largest arrows ~ half a spacing
        self.arrow_scale.blockSignals(False)
        if self.fitted:
            self.fit()
        self.draw_arrows()
        self.draw_maps()

    def draw_profiles(self):
        """The means per row and per column: of each row and column of contacts, or in grid average of the contacts
        of each row and column of cells, at its centre (D125)."""
        if self.grid_mode():
            cell_nm = self.cell.value() * 1000
            self.row_y, self.row_mean, _ = band_means(self.design_nm[:, 1], self.values_nm, cell_nm)
            self.column_x, self.column_mean, _ = band_means(self.design_nm[:, 0], self.values_nm, cell_nm)
        else:
            (self.row_y, self.row_mean), (self.column_x, self.column_mean) = self.contact_rows, self.contact_columns
        row_um, column_um = (self.row_y - self.origin[1]) / 1000, (self.column_x - self.origin[0]) / 1000
        for k, (row_line, column_line) in enumerate(self.map_profiles + self.arrow_profiles):  # dx, dy, dr
            row_line.setData(self.row_mean[:, k], row_um)
            column_line.setData(column_um, self.column_mean[:, k])
        for lines, k in zip(self.range_lines, (2, 0, 1)):  # panels: dr, dx, dy
            lines.set_profile(row_um, self.row_mean[:, k])

    def extent(self) -> tuple[np.ndarray, np.ndarray]:
        """(low, high) x and y, µm, of what the maps show when fitted: the contacts shown with half a pitch around
        them (their disks; arrows half a pitch long), or every cell of the grid."""
        if self.grid_mode():
            rows, cols = self.grid_mean.shape[:2]
            return self.grid_corner_um, self.grid_corner_um + np.array([cols, rows]) * self.cell.value()
        half = self.pitch_nm / 2000
        return self.position_um.min(axis=0) - half, self.position_um.max(axis=0) + half

    def fit(self):
        """Every map shows the full height of the extent, centred on it in x, as wide as the map is: only y is
        fitted (D117). The x range asked for is one pitch wide: with equal x and y scales the view box widens it to
        the map, now and after any resize, so the fit holds and y never moves with the map's width."""
        low, high = self.extent()
        centre, half = (low[0] + high[0]) / 2, self.pitch_nm / 2000
        for panel in self.panels:
            panel.map.vb.setRange(xRange=(centre - half, centre + half), yRange=(low[1], high[1]), padding=0)
        self.fitted = True

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

    def show_columns(self, on: bool):
        """The means per column below the maps, hidden unless ticked (D118). Hidden, they are taken out of the
        layout, which kept their room otherwise, and the maps get their height."""
        grid = self.figure.ci.layout
        for panel in self.panels:
            if on:
                grid.addItem(panel.columns, 2, panel.col)
            else:
                grid.removeItem(panel.columns)
            panel.columns.setVisible(on)

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
                         self.dot_um, self.devicePixelRatioF(), background=255, size_px=self.symbol_size.value())

    def set_line_width(self, width: int):
        """The lines of the means per row and per column, this many pixels wide (D121)."""
        for lines, color in zip(self.map_profiles + self.arrow_profiles, (style.DX, style.DY, style.DR)):
            for line in lines:
                line.setPen(pg.mkPen(color, width=width))

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
        """For the image menus: the figure; the graphs of dx and dy, as laid out, without those of dr (D116); each
        graph shown (a mean per column only while ticked, D118)."""
        shown = [(plot, items, name) for panel in self.panels for plot, items, name in self.graphs_of(panel)
                 if plot.isVisible()]
        dx_dy = [plot for plot, _, name in shown if name.startswith(("dx", "dy"))]
        return self.figure, [("dx and dy: maps, profiles, colour bar", [style.Area(dx_dy + [self.color_bar])])] + \
            [(name, items) for _, items, name in shown] + \
            [("dr colour bar", [self.dr_bar]), ("dx, dy colour bar", [self.color_bar])]

    def profiles(self) -> list[list[str]]:
        """Both profiles as shown, as one table: header, then per line a row (y, mask µm; mean dx, dy, dr, nm) and a
        column (x, mask µm; mean dx, dy, dr, nm), blank where one profile has fewer lines. In grid average the rows
        and columns of cells, named so."""
        row, column = ("grid row", "grid column") if self.grid_mode() else ("row", "column")
        header = [f"{row} y (µm)", f"{row} mean dx (nm)", f"{row} mean dy (nm)", f"{row} mean dr (nm)",
                  f"{column} x (µm)", f"{column} mean dx (nm)", f"{column} mean dy (nm)", f"{column} mean dr (nm)"]

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
