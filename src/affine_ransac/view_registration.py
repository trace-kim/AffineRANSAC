"""Final registration error after the RANSAC affine (docs/SPEC.md S8), pyqtgraph, OpenGL viewport.

The registration error of each contact is its residual after the global affine G:
G(SEM) − design (RansacResult.residuals), for every contact. Positions are relative to the
reference point (design centroid), in µm; errors in nm.

- Error maps of dx and dy: every contact a dot coloured by its error (jet colour map, one shared
  colour scale; drag the colour bar to change it). The two maps share zoom and pan. They are
  rasterized to the screen (rasterize): one image pixel per device pixel, redrawn after every zoom,
  pan or resize. Zoomed out, each contact covers about one pixel and contacts on the same pixel
  are averaged; zoomed in, each contact is a disk of ~0.4 × the contact pitch.
- Row profile: the mean dx and dy of the contacts of each row (registration.row_means: rows
  grouped by design y) against the row's y.
- Reference row profile (optional, raw_error_nm): the same for the stitched SEM − design with no
  affine removed, in its own plot below (the raw error is often far larger than the residual),
  sharing the row axis.
- Extra lines (add_rows): the row means of other sets of contacts, e.g. corrected ones, in both row
  plots (first set yellow dx, green dy; second magenta dx, cyan dy), each with its summary on a
  line of its own.
- Show bar above the plots (instead of legends, which overflow the short row plots): per set of
  lines its name and a dx and a dy check box with the line colour, three sets per row; a box shows
  or hides that component of the set in both row plots, which then rescale in y.
- External reference measurements (add_external, e.g. another tool's result, io.external): the mean
  dx and dy per row as lines with symbols in the row profile, and on the error maps a ring around
  every site in the colour of its error (maps' colour scale, white edges), sized to enclose the contact
  disk there: the contact inside shows our error, the ring the reference, so the same colour means
  agreement (hover a ring for its values). Its dx / dy box also shows or hides its rings on that map.
- Copy rows / Export rows CSV (buttons beside the summary): the row-profile lines that are shown
  (Show bar), as a table (row_table): per set and plot its row y (µm), dx and dy (nm) columns; sets
  have their own rows, so shorter columns end in blanks. Copy puts it on the clipboard tab-separated
  (pastes into Excel), Export writes a CSV (UTF-8 with BOM, so Excel reads the µ).
"""

import csv
import io

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from scipy.spatial import cKDTree

from affine_ransac.registration import error_summary, row_means

# Jet (dark blue -> blue -> cyan -> yellow -> red -> dark red), defined here so that production code
# does not need matplotlib.
JET = pg.ColorMap(pos=[0.0, 0.125, 0.375, 0.625, 0.875, 1.0],
                  color=[(0, 0, 128), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0), (128, 0, 0)])
MAX_DOT_PX = 25  # disk radius cap when zoomed far in
RING_PX = 3  # width of an external reference ring (its white outline is 1 px wider on each side)
SETS_PER_ROW = 3  # Show bar
MAIN_COLORS = ("#ff6040", "#40a0ff")  # dx, dy of the view's own contacts
EXTRA_COLORS = [("#e0c040", "#40d040"), ("#ff60ff", "#40e0e0"), ("#c080ff", "#a0ff60"),  # (dx, dy) of the 1st,
                ("#ff2020", "#3050ff")]                                                  # 2nd... add_rows set
EXTERNAL_COLORS = [("#ffffff", "#b0b0b0"), ("#ffa0c0", "#ffd080")]  # (dx, dy) lines of the 1st, 2nd... add_external set


def rasterize(points: np.ndarray, values: np.ndarray, rect, shape, radius_px: float) -> np.ndarray:
    """Mean of values (N,) drawn as disks of radius_px around points (N, 2) on a (rows, cols) grid
    covering rect = (left, bottom, width, height); row 0 is the bottom. NaN where nothing is drawn."""
    left, bottom, width, height = rect
    rows, cols = shape
    ix = np.floor((points[:, 0] - left) / width * cols).astype(int)
    iy = np.floor((points[:, 1] - bottom) / height * rows).astype(int)
    total, count = np.zeros(shape), np.zeros(shape)
    r = float(radius_px)
    for dy in range(-int(r), int(r) + 1):
        for dx in range(-int(r), int(r) + 1):
            if dx * dx + dy * dy > r * r:
                continue
            x, y = ix + dx, iy + dy
            inside = (x >= 0) & (x < cols) & (y >= 0) & (y < rows)
            np.add.at(total, (y[inside], x[inside]), values[inside])
            np.add.at(count, (y[inside], x[inside]), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def summary_text(error_nm: np.ndarray, what: str = "contacts") -> str:
    """Count, mean, 3σ and max of the errors (N, 2), nm, for a summary line."""
    s = error_summary(error_nm)
    return (f"{s['count']} {what}, mean ({s.get('mean_x_nm', 0):+.3f}, {s.get('mean_y_nm', 0):+.3f}) nm, "
            f"3σ ({s.get('3sigma_x_nm', 0):.3f}, {s.get('3sigma_y_nm', 0):.3f}) nm, max {s.get('max_nm', 0):.3f} nm")


def table_text(table: list[list[str]], delimiter: str) -> str:
    """The table's rows as delimited text (csv quoting, so a comma in a set name is safe)."""
    out = io.StringIO()
    csv.writer(out, delimiter=delimiter, lineterminator="\n").writerows(table)
    return out.getvalue()


def colorize(grid: np.ndarray, levels, transparent: bool = False, background: int = 0) -> np.ndarray:
    """RGB uint8 image of grid in the jet colour map between levels (low, high); NaN (no contact)
    becomes `background` (grey level; 0 = black, the plot background). Opaque by default: ImageItem drew
    NaN and transparent pixels as grey blocks. transparent: RGBA instead, NaN fully transparent (to see
    what lies beneath)."""
    low, high = levels
    lut = JET.getLookupTable(nPts=256, alpha=transparent)
    scaled = (np.nan_to_num(grid) - low) / (high - low) if high > low else np.zeros(grid.shape)
    rgb = lut[np.clip((scaled * 255).astype(int), 0, 255)]
    rgb[np.isnan(grid)] = 0 if transparent else background
    return rgb


def draw_map(plot, image, position_um: np.ndarray, values: np.ndarray, levels, dot_um: float,
             device_ratio: float, transparent: bool = False, background: int = 0, size_px: float = 0) -> float | None:
    """Draw the contacts in plot's view into image (rasterize, colorize), one image pixel per DEVICE
    pixel (with Windows display scaling, e.g. 150 %, a logical pixel is 1.5 device pixels; an image made
    per logical pixel would be enlarged and the contact lattice would alias). position_um (N, 2), values
    (N,); each contact a disk of radius dot_um (at least one pixel, at most MAX_DOT_PX), or size_px
    (logical) pixels across at every zoom if given. Returns the disk radius in logical px, None if the view
    has no size yet. transparent, background: see colorize."""
    view = plot.vb.viewRect()
    pixel_um = np.array(plot.vb.viewPixelSize()) / device_ratio  # (x, y) µm per device pixel
    if pixel_um[0] <= 0 or pixel_um[1] <= 0:
        return None
    cols = int(min(4000, max(1, np.ceil(view.width() / pixel_um[0]))))
    rows = int(min(4000, max(1, np.ceil(view.height() / pixel_um[1]))))
    if size_px > 0:
        radius_px = max(0.0, (size_px * device_ratio - 1) / 2)
    else:
        radius_px = int(np.clip(round(dot_um / pixel_um[0]), 0, MAX_DOT_PX))
    margin = (radius_px + 1) * max(pixel_um)
    x, y = position_um[:, 0], position_um[:, 1]
    near = ((x > view.left() - margin) & (x < view.right() + margin)
            & (y > view.top() - margin) & (y < view.bottom() + margin))  # QRectF: top = smaller y
    rect = (view.left(), view.top(), view.width(), view.height())
    grid = rasterize(position_um[near], values[near], rect, (rows, cols), radius_px)
    image.setImage(colorize(grid, levels, transparent, background), autoLevels=False)
    image.setRect(QtCore.QRectF(*rect))
    return radius_px / device_ratio


class RegistrationView(QtWidgets.QWidget):
    def __init__(
        self,
        design_nm: np.ndarray,
        error_nm: np.ndarray,
        reference_nm: np.ndarray,
        row_gap_nm: float = 10.0,
        use_opengl: bool = True,
        correction: str = "the RANSAC affine",
        raw_error_nm: np.ndarray | None = None,
        name: str = "contacts",
    ):
        """design_nm: (N, 2) contact design positions (mask nm); error_nm: (N, 2) registration error
        G(SEM) − design; reference_nm: (2,) RANSAC reference point; row_gap_nm: see group_rows;
        correction: what G is, for the summary line; raw_error_nm: optional (N, 2) stitched
        SEM − design of the same contacts without any affine, shown as a reference row profile;
        name: these contacts' name in the Show bar."""
        super().__init__()
        self.name = name
        self.line_items, self.line_boxes = {}, {}  # (set name, "dx" / "dy") -> its plot lines / its check box
        self.external_rings = []  # (ring item, its outline item, their dx or dy values) per map of an add_external set
        self.dot_radius_px = 0.0  # contact disk radius on the maps, logical px (rasterize_maps)
        self.show_bar = QtWidgets.QGridLayout()  # SETS_PER_ROW sets per row: more sets add rows, not width
        self.show_bar.addWidget(QtWidgets.QLabel("Show:"), 0, 0)
        self.show_bar.setColumnStretch(1 + 3 * SETS_PER_ROW, 1)
        self.show_sets = []  # names of the sets in the Show bar, in order
        self.setWindowTitle(f"Registration error - {len(error_nm)} contacts")
        self.design_nm, self.reference_nm, self.row_gap_nm = design_nm, reference_nm, row_gap_nm
        self.position_um = (design_nm - reference_nm) / 1000
        if len(design_nm) > 1:
            distance, _ = cKDTree(design_nm).query(design_nm, k=2)
            self.dot_nm = 0.4 * np.median(distance[:, 1])  # ~0.4 x the contact pitch
        else:
            self.dot_nm = 10.0

        self.graphics = pg.GraphicsLayoutWidget()
        if use_opengl:
            self.graphics.useOpenGL(True)
        self.maps = []  # (plot, image, component)
        for col, (component, name) in enumerate(((0, "dx"), (1, "dy"))):
            plot = self.graphics.addPlot(row=0, col=col, title=f"{name} (nm)")
            plot.setAspectLocked(True)
            plot.setLabel("bottom", "x − x_ref (µm)")
            plot.setLabel("left", "y − y_ref (µm)")
            image = pg.ImageItem(axisOrder="row-major")
            plot.addItem(image)
            self.maps.append((plot, image, component))
        limit = float(np.percentile(np.abs(error_nm), 99)) if len(error_nm) else 1.0
        limit = limit if limit > 0 else 1.0
        self.color_bar = pg.ColorBarItem(values=(-limit, limit), colorMap=JET, label="nm")
        # The maps colour their pixels themselves (colorize), so the bar is not attached to the images;
        # moving its handles redraws the maps (and recolours the external markers) with the new limits.
        self.color_bar.sigLevelsChanged.connect(lambda *_: self.rasterize_maps())
        self.color_bar.sigLevelsChanged.connect(lambda *_: self.colour_external())
        self.graphics.addItem(self.color_bar, row=0, col=2)  # own column: both maps keep the same size

        self.row_plot = self.graphics.addPlot(row=1, col=0, colspan=3, title="Mean error per row (averaged over x)")
        self.row_plot.setLabel("bottom", "row y − y_ref (µm)")
        self.row_plot.setLabel("left", "mean error (nm)")
        self.row_curves = [self.row_plot.plot(pen=pg.mkPen(color, width=2)) for color in MAIN_COLORS]
        for curve, axis, color in zip(self.row_curves, ("dx", "dy"), MAIN_COLORS):
            self.add_line(self.name, axis, curve, color)
        self.row_plot.addLine(y=0, pen=pg.mkPen("#808080", style=QtCore.Qt.PenStyle.DashLine))
        self.graphics.ci.layout.setRowStretchFactor(0, 2)
        self.graphics.ci.layout.setRowStretchFactor(1, 1)
        self.raw_plot = None
        if raw_error_nm is not None:
            self._add_raw_plot(raw_error_nm)

        # Re-rasterize shortly after the view stops changing (zooming fires many range changes).
        self.redraw_timer = QtCore.QTimer(self)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.setInterval(50)
        self.redraw_timer.timeout.connect(self.rasterize_maps)
        for plot, _, _ in self.maps:
            plot.vb.sigRangeChanged.connect(lambda *_: self.redraw_timer.start())
            # Also after a resize: a raster made for another size would be resampled, and the
            # regular contact lattice would alias into large blocks.
            plot.vb.sigResized.connect(lambda *_: self.redraw_timer.start())
        self._link_maps()

        self.label = QtWidgets.QLabel()
        self.label.setWordWrap(True)
        self.summary, self.extra_summaries = "", []  # label lines: this view's errors, then add_rows sets
        copy_button = QtWidgets.QPushButton("Copy rows")
        copy_button.setToolTip("Copy the shown row-profile lines (y µm, dx, dy nm) to the clipboard, tab-separated")
        copy_button.clicked.connect(self.copy_rows)
        export_button = QtWidgets.QPushButton("Export rows CSV...")
        export_button.setToolTip("Save the shown row-profile lines (y µm, dx, dy nm) as a CSV file")
        export_button.clicked.connect(self.export_rows)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.label, stretch=1)
        top.addWidget(copy_button, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        top.addWidget(export_button, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(self.show_bar)
        layout.addWidget(self.graphics, stretch=1)
        self.resize(1500, 1000)
        # Start on all contacts (autoRange cannot be used: the images are still empty here).
        low, high = self.position_um.min(axis=0), self.position_um.max(axis=0)
        self.maps[0][0].vb.setRange(xRange=(low[0], high[0]), yRange=(low[1], high[1]))
        self.set_errors(error_nm, correction)

    def _add_raw_plot(self, raw_error_nm: np.ndarray):
        """Mean raw error (stitched, no affine) per row, below the row profile, same row axis."""
        self.raw_plot = self.graphics.addPlot(row=2, col=0, colspan=3,
                                              title="Reference: mean error per row, stitched without affine (averaged over x)")
        self.raw_plot.setLabel("bottom", "row y − y_ref (µm)")
        self.raw_plot.setLabel("left", "mean error (nm)")
        self.raw_plot.setXLink(self.row_plot)
        self.raw_plot.addLine(y=0, pen=pg.mkPen("#808080", style=QtCore.Qt.PenStyle.DashLine))
        row_y, self.raw_row_mean, _ = row_means(self.design_nm[:, 1], raw_error_nm, self.row_gap_nm)
        row_um = (row_y - self.reference_nm[1]) / 1000
        for component, (axis, color) in enumerate(zip(("dx", "dy"), MAIN_COLORS)):
            line = self.raw_plot.plot(row_um, self.raw_row_mean[:, component],
                                      pen=pg.mkPen(color, width=2, style=QtCore.Qt.PenStyle.DotLine))
            self.add_line(self.name, axis, line, color)
        self.graphics.ci.layout.setRowStretchFactor(2, 1)

    def set_errors(self, error_nm: np.ndarray, correction: str):
        """Show new registration errors for the same contacts (the colour scale is kept, so maps
        stay comparable)."""
        self.error_nm = error_nm
        self.row_y, self.row_mean, self.row_count = row_means(self.design_nm[:, 1], error_nm, self.row_gap_nm)
        row_um = (self.row_y - self.reference_nm[1]) / 1000
        for curve, component in zip(self.row_curves, (0, 1)):
            curve.setData(row_um, self.row_mean[:, component])
        self.summary = (f"Registration error after {correction}, G(SEM) − design: {summary_text(error_nm)}; "
                        f"{len(self.row_y)} rows")
        self.label.setText("\n".join([self.summary] + self.extra_summaries))
        self.rasterize_maps()

    def add_rows(self, design_nm: np.ndarray, error_nm: np.ndarray, name: str, raw_error_nm: np.ndarray | None = None,
                 colors: tuple[str, str] | None = None):
        """Mean dx and dy per row of another set of contacts (e.g. corrected; it may hold other
        contacts than this view) as extra lines under `name` in the Show bar: error_nm in the row
        profile and raw_error_nm (if given, and the reference plot exists) in the reference plot.
        Its summary is added below the summary line; the maps do not change. colors: (dx, dy) line
        colours, by default the next pair of EXTRA_COLORS."""
        colors = colors or EXTRA_COLORS[len(self.extra_summaries) % len(EXTRA_COLORS)]
        row_y, mean, _ = row_means(design_nm[:, 1], error_nm, self.row_gap_nm)
        row_um = (row_y - self.reference_nm[1]) / 1000
        for component, (axis, color) in enumerate(zip(("dx", "dy"), colors)):
            line = self.row_plot.plot(row_um, mean[:, component], pen=pg.mkPen(color, width=2))
            self.add_line(name, axis, line, color)
        if raw_error_nm is not None and self.raw_plot is not None:
            _, raw_mean, _ = row_means(design_nm[:, 1], raw_error_nm, self.row_gap_nm)
            for component, (axis, color) in enumerate(zip(("dx", "dy"), colors)):
                line = self.raw_plot.plot(row_um, raw_mean[:, component],
                                          pen=pg.mkPen(color, width=2, style=QtCore.Qt.PenStyle.DotLine))
                self.add_line(name, axis, line, color)
        self.extra_summaries.append(f"{name}: {summary_text(error_nm)}")
        self.label.setText("\n".join([self.summary] + self.extra_summaries))

    def add_external(self, name: str, xy_nm: np.ndarray, error_nm: np.ndarray):
        """An external reference measurement under `name` in the Show bar: xy_nm (N, 2) site
        positions, mask nm; error_nm (N, 2) its dx, dy there, nm (io.external.read_external). Its
        mean per row (rows grouped by the sites' y) goes into the row profile, its sites onto the
        error maps (see the module docstring); its summary is added below the summary line."""
        sets_so_far = len(self.external_rings) // 2  # a ring item on the dx and on the dy map per set
        colors = EXTERNAL_COLORS[sets_so_far % len(EXTERNAL_COLORS)]
        x_um, y_um = ((xy_nm - self.reference_nm) / 1000).T
        row_y, mean, _ = row_means(xy_nm[:, 1], error_nm, self.row_gap_nm)
        row_um = (row_y - self.reference_nm[1]) / 1000
        tips = [f"{name}\nx {x:.3f} µm, y {y:.3f} µm\ndx {dx:+.3f} nm, dy {dy:+.3f} nm"
                for (x, y), (dx, dy) in zip(xy_nm / 1000, error_nm)]
        for component, (axis, color, symbol) in enumerate(zip(("dx", "dy"), colors, ("o", "t"))):
            line = self.row_plot.plot(row_um, mean[:, component], pen=pg.mkPen(color, width=1.5), symbol=symbol,
                                      symbolSize=7, symbolPen=None, symbolBrush=color)
            self.add_line(name, axis, line, color)
            # A ring in the error's colour around the contact at the site (the contact stays visible
            # inside it), on a white ring that makes it easy to find.
            outline = pg.ScatterPlotItem(x_um, y_um, brush=None, pen=pg.mkPen("#ffffff", width=RING_PX + 2))
            ring = pg.ScatterPlotItem(x_um, y_um, data=tips, brush=None, hoverable=True, tip=lambda x, y, data: data)
            for z, item in ((10, outline), (11, ring)):  # above the rasterized map
                item.setZValue(z)
                self.maps[component][0].addItem(item)
                self.add_line(name, axis, item, color)
            self.external_rings.append((ring, outline, error_nm[:, component]))
        self.colour_external()
        self.size_external()
        self.extra_summaries.append(f"{name} (external reference): {summary_text(error_nm, 'sites')}")
        self.label.setText("\n".join([self.summary] + self.extra_summaries))

    def colour_external(self):
        """Colour every external ring with its value in the maps' colour scale (colorize)."""
        for ring, _, values in self.external_rings:
            ring.setPen([pg.mkPen(rgb.tolist(), width=RING_PX) for rgb in colorize(values, self.color_bar.levels())])

    def size_external(self):
        """Make the external rings enclose the contact disks as drawn now (dot_radius_px)."""
        for ring, outline, _ in self.external_rings:
            for item in (ring, outline):
                item.setSize(2 * self.dot_radius_px + RING_PX + 5)

    def add_line(self, name: str, axis: str, line, color: str):
        """Put line (a curve of row_plot or raw_plot, or the markers of an external set on a map)
        under the check box of set `name`, component axis ("dx" / "dy"), in the Show bar; the set's
        name and box are added with its first line."""
        key = (name, axis)
        if key not in self.line_boxes:
            if name not in self.show_sets:  # the set's cells: its name, then its dx and dy boxes
                self.show_sets.append(name)
                label = QtWidgets.QLabel(f"   {name}:")
                label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.show_bar.addWidget(label, *self.show_cell(name, 0))
            swatch = QtGui.QPixmap(18, 6)
            swatch.fill(QtGui.QColor(color))
            box = QtWidgets.QCheckBox(axis)
            box.setIcon(QtGui.QIcon(swatch))
            box.setChecked(True)
            box.toggled.connect(lambda shown, key=key: self.show_lines(key, shown))
            self.show_bar.addWidget(box, *self.show_cell(name, 1 + ("dx", "dy").index(axis)))
            self.line_boxes[key], self.line_items[key] = box, []
        self.line_items[key].append(line)
        line.setVisible(self.line_boxes[key].isChecked())

    def row_table(self) -> list[list[str]]:
        """The row-profile lines whose Show box is checked, as a header row and data rows: per set
        and plot (after the affine, then without affine) its row y − y_ref (µm), dx and dy (nm).
        Blank where a set has fewer rows than the longest, or a component is hidden or NaN."""
        plots = [(self.row_plot, "after affine"), (self.raw_plot, "no affine")]
        groups = {}  # (plot name, set name) -> {"y": row y µm, "dx": nm, "dy": nm}
        for plot, plot_name in plots:
            if plot is None:
                continue
            items = plot.listDataItems()
            for (name, axis), lines in self.line_items.items():
                if not self.line_boxes[name, axis].isChecked():
                    continue
                for line in lines:
                    if line in items:  # not an external set's rings on a map
                        x, y = line.getData()
                        group = groups.setdefault((plot_name, name), {"y": x})
                        group[axis] = y
        columns = []  # (header, values)
        for (plot_name, name), group in groups.items():
            columns.append((f"{name}, {plot_name}: row y - y_ref (µm)", group["y"]))
            columns += [(f"{name}, {plot_name}: {axis} (nm)", group[axis]) for axis in ("dx", "dy") if axis in group]
        length = max((len(values) for _, values in columns), default=0)
        rows = [[f"{values[k]:.6f}" if k < len(values) and np.isfinite(values[k]) else "" for _, values in columns]
                for k in range(length)]
        return [[header for header, _ in columns]] + rows

    def copy_rows(self):
        """Put row_table on the clipboard, tab-separated."""
        QtWidgets.QApplication.clipboard().setText(table_text(self.row_table(), "\t"))

    def export_rows(self):
        """Ask for a file name and write row_table there as CSV."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export row profile", "row_profile.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                f.write(table_text(self.row_table(), ","))

    def show_cell(self, name: str, offset: int) -> tuple[int, int]:
        """(row, column) in the Show bar of set `name`'s name (offset 0), dx box (1) or dy box (2)."""
        k = self.show_sets.index(name)
        return k // SETS_PER_ROW, 1 + 3 * (k % SETS_PER_ROW) + offset

    def show_lines(self, key, shown: bool):
        """Show or hide the lines of key = (set name, "dx" / "dy"); the row plots rescale in y."""
        for line in self.line_items[key]:
            line.setVisible(shown)
        for plot in (self.row_plot, self.raw_plot):
            if plot is not None:
                plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)

    def _link_maps(self):
        """Keep both maps on the same area (as link_views does for two plot widgets)."""
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

    def rasterize_maps(self):
        """Draw the contacts in view into each map's image (draw_map)."""
        for plot, image, component in self.maps:
            radius = draw_map(plot, image, self.position_um, self.error_nm[:, component], self.color_bar.levels(),
                              self.dot_nm / 1000, self.devicePixelRatioF())
            if radius is not None:
                self.dot_radius_px = radius  # both maps show the same area: the same radius
        self.size_external()
