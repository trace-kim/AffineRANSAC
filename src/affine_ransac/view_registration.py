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
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.spatial import cKDTree

from affine_ransac.registration import error_summary, row_means

# Jet (dark blue -> blue -> cyan -> yellow -> red -> dark red), defined here so that production code
# does not need matplotlib.
JET = pg.ColorMap(pos=[0.0, 0.125, 0.375, 0.625, 0.875, 1.0],
                  color=[(0, 0, 128), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0), (128, 0, 0)])
MAX_DOT_PX = 25  # disk radius cap when zoomed far in


def rasterize(points: np.ndarray, values: np.ndarray, rect, shape, radius_px: int) -> np.ndarray:
    """Mean of values (N,) drawn as disks of radius_px around points (N, 2) on a (rows, cols) grid
    covering rect = (left, bottom, width, height); row 0 is the bottom. NaN where nothing is drawn."""
    left, bottom, width, height = rect
    rows, cols = shape
    ix = np.floor((points[:, 0] - left) / width * cols).astype(int)
    iy = np.floor((points[:, 1] - bottom) / height * rows).astype(int)
    total, count = np.zeros(shape), np.zeros(shape)
    r = int(radius_px)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            x, y = ix + dx, iy + dy
            inside = (x >= 0) & (x < cols) & (y >= 0) & (y < rows)
            np.add.at(total, (y[inside], x[inside]), values[inside])
            np.add.at(count, (y[inside], x[inside]), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def colorize(grid: np.ndarray, levels) -> np.ndarray:
    """RGB uint8 image of grid in the jet colour map between levels (low, high); NaN (no contact)
    becomes black, the plot background. Opaque on purpose: ImageItem drew NaN and transparent
    pixels as grey blocks."""
    low, high = levels
    lut = JET.getLookupTable(nPts=256, alpha=False)
    scaled = (np.nan_to_num(grid) - low) / (high - low) if high > low else np.zeros(grid.shape)
    rgb = lut[np.clip((scaled * 255).astype(int), 0, 255)]
    rgb[np.isnan(grid)] = 0
    return rgb


class RegistrationView(QtWidgets.QWidget):
    def __init__(
        self,
        design_nm: np.ndarray,
        error_nm: np.ndarray,
        reference_nm: np.ndarray,
        row_gap_nm: float = 10.0,
        use_opengl: bool = True,
        correction: str = "the RANSAC affine",
    ):
        """design_nm: (N, 2) contact design positions (mask nm); error_nm: (N, 2) registration error
        G(SEM) − design; reference_nm: (2,) RANSAC reference point; row_gap_nm: see group_rows;
        correction: what G is, for the summary line."""
        super().__init__()
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
        # moving its handles redraws the maps with the new limits.
        self.color_bar.sigLevelsChanged.connect(lambda *_: self.rasterize_maps())
        self.graphics.addItem(self.color_bar, row=0, col=2)  # own column: both maps keep the same size

        self.row_plot = self.graphics.addPlot(row=1, col=0, colspan=3, title="Mean error per row (averaged over x)")
        self.row_plot.setLabel("bottom", "row y − y_ref (µm)")
        self.row_plot.setLabel("left", "mean error (nm)")
        self.row_plot.addLegend(offset=(5, 5))
        self.row_curves = [self.row_plot.plot(pen=pg.mkPen("#ff6040", width=2), name="dx"),
                           self.row_plot.plot(pen=pg.mkPen("#40a0ff", width=2), name="dy")]
        self.row_plot.addLine(y=0, pen=pg.mkPen("#808080", style=QtCore.Qt.PenStyle.DashLine))
        self.graphics.ci.layout.setRowStretchFactor(0, 2)
        self.graphics.ci.layout.setRowStretchFactor(1, 1)

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
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.graphics, stretch=1)
        self.resize(1500, 1000)
        # Start on all contacts (autoRange cannot be used: the images are still empty here).
        low, high = self.position_um.min(axis=0), self.position_um.max(axis=0)
        self.maps[0][0].vb.setRange(xRange=(low[0], high[0]), yRange=(low[1], high[1]))
        self.set_errors(error_nm, correction)

    def set_errors(self, error_nm: np.ndarray, correction: str):
        """Show new registration errors for the same contacts (the colour scale is kept, so maps
        stay comparable)."""
        self.error_nm = error_nm
        self.row_y, self.row_mean, self.row_count = row_means(self.design_nm[:, 1], error_nm, self.row_gap_nm)
        row_um = (self.row_y - self.reference_nm[1]) / 1000
        for curve, component in zip(self.row_curves, (0, 1)):
            curve.setData(row_um, self.row_mean[:, component])
        summary = error_summary(error_nm)
        self.label.setText(
            f"Registration error after {correction}, G(SEM) − design: {summary['count']} contacts, "
            f"mean ({summary.get('mean_x_nm', 0):+.3f}, {summary.get('mean_y_nm', 0):+.3f}) nm, "
            f"3σ ({summary.get('3sigma_x_nm', 0):.3f}, {summary.get('3sigma_y_nm', 0):.3f}) nm, "
            f"max {summary.get('max_nm', 0):.3f} nm; {len(self.row_y)} rows")
        self.rasterize_maps()

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
        """Draw the contacts in view into each map's image, one image pixel per DEVICE pixel (with
        Windows display scaling, e.g. 150 %, a logical pixel is 1.5 device pixels; an image made per
        logical pixel would be enlarged and the contact lattice would alias)."""
        scale = self.devicePixelRatioF()
        for plot, image, component in self.maps:
            view = plot.vb.viewRect()
            pixel_um = np.array(plot.vb.viewPixelSize()) / scale  # (x, y) µm per device pixel
            if pixel_um[0] <= 0 or pixel_um[1] <= 0:
                continue
            cols = int(min(4000, max(1, np.ceil(view.width() / pixel_um[0]))))
            rows = int(min(4000, max(1, np.ceil(view.height() / pixel_um[1]))))
            radius_px = int(np.clip(round(self.dot_nm / 1000 / pixel_um[0]), 0, MAX_DOT_PX))
            margin = (radius_px + 1) * max(pixel_um)
            x, y = self.position_um[:, 0], self.position_um[:, 1]
            near = ((x > view.left() - margin) & (x < view.right() + margin)
                    & (y > view.top() - margin) & (y < view.bottom() + margin))  # QRectF: top = smaller y
            rect = (view.left(), view.top(), view.width(), view.height())
            grid = rasterize(self.position_um[near], self.error_nm[near, component], rect, (rows, cols), radius_px)
            image.setImage(colorize(grid, self.color_bar.levels()), autoLevels=False)
            image.setRect(QtCore.QRectF(*rect))
