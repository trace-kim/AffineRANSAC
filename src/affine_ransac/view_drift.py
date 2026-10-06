"""Drift per stripe (fitting.drift.StripeDrift), for diagnosis (docs/SPEC.md S8, D57), pyqtgraph.

Two plots against the row's y (relative to the reference point, µm), dx above and dy below. Per stripe:
its mean SEM − design per row (thin) and its fitted drift curve (bold), both minus the common straight
line (that line is left for the global affine), so each bold curve is exactly what is subtracted from
that stripe. A check box per stripe shows or hides it in both plots; the y axes rescale. The label
gives the common line and, per stripe, the range of what is subtracted.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from affine_ransac.fitting.drift import StripeDrift

STRIPE_COLORS = ["#ff6040", "#40a0ff", "#e0c040", "#40d040", "#ff60ff", "#40e0e0", "#c080ff", "#a0ff60"]


class StripeDriftView(QtWidgets.QWidget):
    def __init__(self, drift: StripeDrift, reference_nm: np.ndarray, use_opengl: bool = True):
        """drift: the stripes' drift curves (stripe_drift); reference_nm: (2,) reference point, mask nm."""
        super().__init__()
        self.setWindowTitle(f"Drift per stripe - {len(drift.stripes)} stripes")
        self.graphics = pg.GraphicsLayoutWidget()
        if use_opengl:
            self.graphics.useOpenGL(True)
        self.plots = []
        for row, axis in enumerate(("dx", "dy")):
            plot = self.graphics.addPlot(row=row, col=0, title=f"{axis} per stripe: mean per row (thin) and drift curve "
                                                               f"(bold, subtracted), minus the common straight line")
            plot.setLabel("bottom", "row y − y_ref (µm)")
            plot.setLabel("left", f"{axis} (nm)")
            plot.addLine(y=0, pen=pg.mkPen("#808080", style=QtCore.Qt.PenStyle.DashLine))
            if self.plots:
                plot.setXLink(self.plots[0])
            self.plots.append(plot)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Show:"))
        self.lines, self.boxes, ranges = {}, {}, []
        for k, (stripe, curve) in enumerate(sorted(drift.stripes.items())):
            color = STRIPE_COLORS[k % len(STRIPE_COLORS)]
            faint = pg.mkColor(color)
            faint.setAlpha(110)
            row_um = (curve.row_y_nm - reference_nm[1]) / 1000
            line = drift.line(curve.row_y_nm)
            subtracted = curve.curve_nm - line
            self.lines[stripe] = []
            for component, plot in enumerate(self.plots):
                self.lines[stripe].append(plot.plot(row_um, curve.row_mean_nm[:, component] - line[:, component],
                                                    pen=pg.mkPen(faint, width=1)))
                self.lines[stripe].append(plot.plot(row_um, subtracted[:, component], pen=pg.mkPen(color, width=3)))
            swatch = QtGui.QPixmap(18, 6)
            swatch.fill(QtGui.QColor(color))
            box = QtWidgets.QCheckBox(f"stripe {stripe + 1}")
            box.setIcon(QtGui.QIcon(swatch))
            box.setChecked(True)
            box.toggled.connect(lambda shown, stripe=stripe: self.show_stripe(stripe, shown))
            bar.addWidget(box)
            self.boxes[stripe] = box
            ranges.append(f"stripe {stripe + 1}: dx {subtracted[:, 0].min():+.2f}..{subtracted[:, 0].max():+.2f}, "
                          f"dy {subtracted[:, 1].min():+.2f}..{subtracted[:, 1].max():+.2f}")
        bar.addStretch(1)

        common = drift.common
        terms = ", ".join(f"{axis} = {common.line_intercept_nm[c]:+.3f} nm {common.line_slope[c] * 1e6:+.2f} ppm × (y − y_c)"
                          for c, axis in enumerate(("dx", "dy")))
        self.label = QtWidgets.QLabel(
            f"Drift curves: local straight-line fits {common.window_nm / 1000:g} µm tall, per stripe. Common straight "
            f"line (left for the global affine, y_c = {common.line_centre_y_nm / 1000:.3f} µm): {terms}.\n"
            f"Subtracted (nm): " + "; ".join(ranges))
        self.label.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addLayout(bar)
        layout.addWidget(self.graphics, stretch=1)
        self.resize(1500, 950)

    def show_stripe(self, stripe: int, shown: bool):
        """Show or hide one stripe's lines in both plots; the y axes rescale."""
        for line in self.lines[stripe]:
            line.setVisible(shown)
        for plot in self.plots:
            plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
