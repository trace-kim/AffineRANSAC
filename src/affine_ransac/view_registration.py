"""Final registration error after the RANSAC affine (docs/SPEC.md S8), pyqtgraph, OpenGL viewport.

The registration error of each contact is its residual after the global affine G:
G(SEM) − design (RansacResult.residuals), for EVERY contact; outliers are kept and marked.
Positions are relative to the reference point (design centroid), in µm; errors in nm.

- Heatmaps of dx and dy: mean error per square bin (registration.binned_mean_2d), one shared,
  symmetric colour scale (drag the colour bar to change it); outlier contacts ringed in magenta.
  All contacts by default; "Exclude outliers from heatmaps" bins the inliers only (a single
  defect otherwise dominates its bin and the colour scale).
- Trend along y: dx and dy of every contact against y (dots; outliers red), and the mean per y bin
  averaged over x (registration.profile) for all contacts (solid line, bars ±1σ of the bin) and
  for the inliers only (dashed line).
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from affine_ransac.registration import binned_mean_2d, error_summary, profile

OUTLIER, CONTACT, ALL_MEAN, INLIER_MEAN = "#ff4040", "#909090", "#ffffff", "#33cc33"


class RegistrationView(QtWidgets.QWidget):
    def __init__(
        self,
        design_nm: np.ndarray,
        error_nm: np.ndarray,
        inliers: np.ndarray,
        reference_nm: np.ndarray,
        heatmap_bin_nm: float = 1000.0,
        profile_bin_nm: float = 500.0,
        use_opengl: bool = True,
    ):
        """design_nm: (N, 2) contact design positions (mask nm); error_nm: (N, 2) registration error
        G(SEM) − design; inliers: (N,) RANSAC inliers; reference_nm: (2,) RANSAC reference point."""
        super().__init__()
        self.setWindowTitle(f"Registration error - {len(error_nm)} contacts")
        position = design_nm - reference_nm
        outlier = ~inliers

        self.graphics = pg.GraphicsLayoutWidget()
        if use_opengl:
            self.graphics.useOpenGL(True)
        cmap = pg.colormap.get("CET-D1")  # diverging: blue < 0 < red

        self.position, self.error_nm, self.inliers, self.heatmap_bin_nm = position, error_nm, inliers, heatmap_bin_nm
        self.heatmaps = []
        for col, (component, name) in enumerate(((0, "dx"), (1, "dy"))):
            plot = self.graphics.addPlot(row=0, col=col, title=f"{name} (nm), mean per {heatmap_bin_nm / 1000:g} µm bin")
            plot.setAspectLocked(True)
            plot.setLabel("bottom", "x − x_ref (µm)")
            plot.setLabel("left", "y − y_ref (µm)")
            image = pg.ImageItem(axisOrder="row-major")
            plot.addItem(image)
            plot.addItem(pg.ScatterPlotItem(position[outlier, 0] / 1000, position[outlier, 1] / 1000, size=8,
                                            pen=pg.mkPen("#ff00ff"), brush=None))
            self.heatmaps.append([plot, image, None])
        self.color_bar = pg.ColorBarItem(values=(-1, 1), colorMap=cmap, label="nm")
        self.color_bar.setImageItem([image for _, image, _ in self.heatmaps], insert_in=self.heatmaps[1][0])
        self.show_heatmaps(exclude_outliers=False)

        self.profiles = {}
        for col, (component, name) in enumerate(((0, "dx"), (1, "dy"))):
            plot = self.graphics.addPlot(row=1, col=col, title=f"{name} vs y, averaged over x")
            plot.setLabel("bottom", "y − y_ref (µm)")
            plot.setLabel("left", f"{name} (nm)")
            plot.addLegend(offset=(5, 5))
            y_um, value = position[:, 1] / 1000, error_nm[:, component]
            plot.addItem(pg.ScatterPlotItem(y_um[inliers], value[inliers], size=3, pen=None,
                                            brush=pg.mkBrush(CONTACT), name="contact"))
            plot.addItem(pg.ScatterPlotItem(y_um[outlier], value[outlier], size=4, pen=None,
                                            brush=pg.mkBrush(OUTLIER), name="contact, outlier"))
            centers, mean, std, count = profile(position[:, 1], value, profile_bin_nm)
            plot.addItem(pg.ErrorBarItem(x=centers / 1000, y=mean, height=2 * std, pen=pg.mkPen(ALL_MEAN)))
            plot.plot(centers / 1000, mean, pen=pg.mkPen(ALL_MEAN, width=2), symbol="o", symbolSize=5,
                      symbolBrush=ALL_MEAN, name="mean, all contacts (±1σ)")
            if inliers.any():
                c_in, m_in, *_ = profile(position[inliers, 1], value[inliers], profile_bin_nm)
                plot.plot(c_in / 1000, m_in, pen=pg.mkPen(INLIER_MEAN, width=2, style=QtCore.Qt.PenStyle.DashLine),
                          name="mean, inliers")
            self.profiles[name] = (centers, mean, std, count)
            # Start zoomed on the inliers (outliers can be far off); zoom out to see them all.
            inlier_values = value[inliers] if inliers.any() else value
            low, high = inlier_values.min(), inlier_values.max()
            pad = 0.2 * (high - low) + 0.05
            plot.setYRange(low - pad, high + pad, padding=0)

        summary = QtWidgets.QLabel(self._summary_text(error_nm, inliers))
        summary.setWordWrap(True)  # a long single line would widen the window
        self.exclude_box = QtWidgets.QCheckBox("Exclude outliers from heatmaps")
        self.exclude_box.toggled.connect(lambda on: self.show_heatmaps(exclude_outliers=on))
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(summary)
        layout.addWidget(self.exclude_box)
        layout.addWidget(self.graphics, stretch=1)
        self.resize(1500, 1000)

    def show_heatmaps(self, exclude_outliers: bool):
        """Bin the error of all contacts (or the inliers only) and rescale the shared colour bar."""
        use = self.inliers if exclude_outliers else np.ones(len(self.inliers), bool)
        limit = 0.0
        for component, entry in enumerate(self.heatmaps):
            _, image, _ = entry
            grid, x_edges, y_edges = binned_mean_2d(self.position[use], self.error_nm[use, component], self.heatmap_bin_nm)
            image.setImage(grid, autoLevels=False)
            image.setRect(QtCore.QRectF(x_edges[0] / 1000, y_edges[0] / 1000, (x_edges[-1] - x_edges[0]) / 1000,
                                        (y_edges[-1] - y_edges[0]) / 1000))
            entry[2] = grid
            limit = max(limit, np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 0.0)
        limit = limit if limit > 0 else 1.0
        self.color_bar.setLevels((-limit, limit))

    @staticmethod
    def _summary_text(error_nm, inliers) -> str:
        parts = []
        for name, mask in (("all contacts", np.ones(len(error_nm), bool)), ("inliers", inliers)):
            s = error_summary(error_nm[mask])
            if s["count"]:
                parts.append(f"{name}: {s['count']}, mean ({s['mean_x_nm']:+.3f}, {s['mean_y_nm']:+.3f}) nm, "
                             f"3σ ({s['3sigma_x_nm']:.3f}, {s['3sigma_y_nm']:.3f}) nm, max {s['max_nm']:.3f} nm")
        return ("Registration error after the RANSAC affine, G(SEM) − design. " + "; ".join(parts)
                + f"; outliers (magenta rings / red dots): {(~inliers).sum()}")
