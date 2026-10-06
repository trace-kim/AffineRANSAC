"""Stitching residuals (docs/SPEC.md S5, D58), pyqtgraph: how well the tiles agree after each stitching.

For every stitching (a name, e.g. "raw input, translation"): the tie residual T_i(a) − T_j(b) of every
tie contact of its used overlaps (pipeline.pair_residuals), summarised per overlap as the RMS length
and plotted against the overlap's y (relative to the reference point, µm). Overlaps within a stripe
(one image above the other) and between stripes (side by side) are in two plots; corner overlaps
are left out of the plots. A check box per stitching shows or hides it; the label gives per stitching
the RMS over all tie contacts of each kind and the largest residual (pipeline.residual_summary). A
correction made before stitching that is right makes the tiles agree better than the raw input.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets

from affine_ransac.pipeline import pair_kind, pair_residuals, residual_summary

COLORS = ["#ff6040", "#40a0ff", "#e0c040", "#40d040", "#ff60ff", "#40e0e0", "#c080ff", "#a0ff60"]


class StitchResidualView(QtWidgets.QWidget):
    def __init__(self, stitchings: dict, centers: np.ndarray, reference_nm: np.ndarray, use_opengl: bool = True):
        """stitchings: name -> (points, stitch, corrections): the points stitched (per tile, mask nm),
        their StitchResult (used pairs and tie contacts) and the corrections found (shifts (n, 2) or
        affines (n, 3, 3)); centers: (n_tiles, 2) tile centres, to tell the overlaps apart;
        reference_nm: (2,) reference point, mask nm."""
        super().__init__()
        self.setWindowTitle(f"Stitching residuals - {len(stitchings)} stitchings")
        self.graphics = pg.GraphicsLayoutWidget()
        if use_opengl:
            self.graphics.useOpenGL(True)
        self.plots = {}
        for row, (kind, title) in enumerate((("vertical", "Within stripes (one image above the other)"),
                                             ("horizontal", "Between stripes (side by side)"))):
            plot = self.graphics.addPlot(row=row, col=0, title=f"{title}: RMS tie residual per overlap after stitching")
            plot.setLabel("bottom", "overlap y − y_ref (µm)")
            plot.setLabel("left", "RMS |T_i(a) − T_j(b)| (nm)")
            plot.showGrid(y=True, alpha=0.3)
            if self.plots:
                plot.setXLink(next(iter(self.plots.values())))
            self.plots[kind] = plot

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Show:"))
        self.items, self.boxes, self.summary, lines = {}, {}, {}, []
        for k, (name, (points, stitch, corrections)) in enumerate(stitchings.items()):
            color = COLORS[k % len(COLORS)]
            residuals = pair_residuals(points, stitch, corrections)
            rms = np.array([np.sqrt((r ** 2).sum(axis=1).mean()) for r in residuals])
            y_um = np.array([((p.box[2] + p.box[3]) / 2 - reference_nm[1]) / 1000 for p in stitch.pairs])
            kinds = np.array([pair_kind(centers, p.i, p.j) for p in stitch.pairs])
            self.items[name] = []
            for kind, plot in self.plots.items():
                chosen = kinds == kind
                self.items[name].append(plot.plot(y_um[chosen], rms[chosen], pen=None, symbol="o", symbolSize=4,
                                                  symbolPen=None, symbolBrush=color))
            swatch = QtGui.QPixmap(18, 6)
            swatch.fill(QtGui.QColor(color))
            box = QtWidgets.QCheckBox(name)
            box.setIcon(QtGui.QIcon(swatch))
            box.setChecked(True)
            box.toggled.connect(lambda shown, name=name: self.show_stitching(name, shown))
            bar.addWidget(box)
            self.boxes[name] = box
            s = residual_summary(points, stitch, corrections, centers)
            self.summary[name] = s
            lines.append(f"{name}: within stripes {s['within stripes RMS (nm)']:.3f} nm, between stripes "
                         f"{s['between stripes RMS (nm)']:.3f} nm, all {s['all overlaps RMS (nm)']:.3f} nm, "
                         f"largest {s['largest (nm)']:.3f} nm")
        bar.addStretch(1)

        self.label = QtWidgets.QLabel("RMS tie residual after stitching:\n" + "\n".join(lines))
        self.label.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addLayout(bar)
        layout.addWidget(self.graphics, stretch=1)
        self.resize(1500, 950)

    def show_stitching(self, name: str, shown: bool):
        """Show or hide one stitching in both plots; the y axes rescale."""
        for item in self.items[name]:
            item.setVisible(shown)
        for plot in self.plots.values():
            plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
