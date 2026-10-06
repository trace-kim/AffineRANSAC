"""Row-to-row pitch along y (docs/SPEC.md S8, D38), pyqtgraph.

Rows are grouped by design y (registration.group_rows); a row's position is the mean y of its
contacts, and the pitch is the spacing between neighbouring rows (registration.row_pitch), plotted
at the rows' mean design y:
- Top: the pitch of the design and of each SEM set (e.g. stitched without affine, after the RANSAC
  affine).
- Bottom: each SEM pitch minus the design pitch (of the same rows).
A set may hold other contacts than the design given (e.g. the affine-stitched SEM): it is then
given with its own design positions and compared with their pitch.
Tile edges (optional) are drawn as faint vertical lines on both plots, so seams can be spotted.
Positions relative to the reference point, in µm; pitches and differences in nm. Clicking a legend
entry hides or shows its curve.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from affine_ransac.registration import row_pitch

COLORS = ["#ff6040", "#40d040", "#40a0ff", "#e0c040"]


class PitchView(QtWidgets.QWidget):
    def __init__(
        self,
        design_nm: np.ndarray,
        sem_sets: dict,
        reference_nm: np.ndarray,
        row_gap_nm: float = 10.0,
        tile_edges_y_nm=(),
        use_opengl: bool = True,
    ):
        """design_nm: (N, 2) contact design positions, mask nm; sem_sets: {name: (N, 2) SEM positions
        of the same contacts, or (design_nm, sem_nm) of other contacts}; reference_nm: (2,) reference
        point; row_gap_nm: see group_rows; tile_edges_y_nm: y of the tile edges (mask nm)."""
        super().__init__()
        self.setWindowTitle(f"Row-to-row pitch - {len(design_nm)} contacts")
        design_y = design_nm[:, 1]
        mid_y, self.design_pitch = row_pitch(design_y, design_y, row_gap_nm)
        self.mid_um = (mid_y - reference_nm[1]) / 1000
        self.pitch, self.difference, set_mid_um = {}, {}, {}
        for name, points in sem_sets.items():
            own_design, sem = points if isinstance(points, tuple) else (design_nm, points)
            own_mid_y, own_design_pitch = row_pitch(own_design[:, 1], own_design[:, 1], row_gap_nm)
            self.pitch[name] = row_pitch(own_design[:, 1], sem[:, 1], row_gap_nm)[1]
            self.difference[name] = self.pitch[name] - own_design_pitch
            set_mid_um[name] = (own_mid_y - reference_nm[1]) / 1000

        self.graphics = pg.GraphicsLayoutWidget()
        if use_opengl:
            self.graphics.useOpenGL(True)
        self.pitch_plot = self.graphics.addPlot(row=0, col=0, title="Row-to-row pitch")
        self.pitch_plot.setLabel("left", "pitch (nm)")
        self.diff_plot = self.graphics.addPlot(row=1, col=0, title="SEM pitch − design pitch")
        self.diff_plot.setLabel("left", "difference (nm)")
        self.diff_plot.setXLink(self.pitch_plot)
        for plot in (self.pitch_plot, self.diff_plot):
            plot.setLabel("bottom", "row y − y_ref (µm)")
            plot.addLegend(offset=(5, 5))
            plot.showGrid(y=True, alpha=0.3)
            for edge in tile_edges_y_nm:
                plot.addItem(pg.InfiniteLine((edge - reference_nm[1]) / 1000, angle=90,
                                             pen=pg.mkPen("#606060", style=QtCore.Qt.PenStyle.DotLine)))
        self.diff_plot.addLine(y=0, pen=pg.mkPen("#808080", style=QtCore.Qt.PenStyle.DashLine))

        self.pitch_plot.plot(self.mid_um, self.design_pitch, pen=pg.mkPen("#c0c0c0", width=2), name="design")
        lines = [f"Design: {self.describe(self.design_pitch)}"]
        for color, (name, pitch) in zip(COLORS, self.pitch.items()):
            self.pitch_plot.plot(set_mid_um[name], pitch, pen=pg.mkPen(color, width=1.5), name=name)
            difference = self.difference[name]
            self.diff_plot.plot(set_mid_um[name], difference, pen=pg.mkPen(color, width=1.5), name=name)
            lines.append(f"{name}: {self.describe(pitch)}; − design: mean {difference.mean():+.3f} nm, "
                         f"3σ {3 * difference.std():.3f} nm, max |·| {np.abs(difference).max():.3f} nm")

        label = QtWidgets.QLabel(f"{len(self.mid_um) + 1} rows. " + "<br>".join(lines))
        label.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(label)
        layout.addWidget(self.graphics, stretch=1)
        self.resize(1500, 950)

    @staticmethod
    def describe(pitch: np.ndarray) -> str:
        if len(pitch) == 0:
            return "no pitch (fewer than 2 rows)"
        return f"pitch mean {pitch.mean():.3f} nm, σ {pitch.std():.3f} nm, range {pitch.min():.3f}..{pitch.max():.3f} nm"
