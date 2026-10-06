"""Stitching inputs without images: design and SEM contact centres, tile boxes and overlaps
(pyqtgraph, OpenGL viewport). Display only. For data that has coordinates but no images, e.g.
the pre-analysed contour CSVs (docs/SPEC.md §4.5).

Everything is drawn at the given (nominal) positions, in mask coordinates relative to an origin
in whole µm (the OpenGL viewport works in float32). Layers (click a legend entry to hide or show):
- design centres (blue) and SEM centres (red), one dot per contact;
- tile boxes (grey): the boxes stitch_tiles used to find the overlaps;
- overlaps used (yellow), skipped (dashed grey: fewer than min_matched contacts matched), not
  matched (dashed magenta: SEM; dash-dot orange: design): enough contacts, but they did not pair;
- tiles not stitched (thick magenta): NaN correction, SEM or design.
Tile names are drawn on request (check box). The cursor position is shown in mask µm.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from affine_ransac.pipeline import StitchResult

DESIGN, SEM = "#3c8cff", "#ff4040"
DASH, DASH_DOT = QtCore.Qt.PenStyle.DashLine, QtCore.Qt.PenStyle.DashDotLine


def box_outlines(boxes) -> tuple[np.ndarray, np.ndarray]:
    """x, y arrays of closed rectangles (x_min, x_max, y_min, y_max), separated by NaN, for
    PlotDataItem(connect="finite")."""
    xs, ys = [], []
    for x0, x1, y0, y1 in boxes:
        xs += [x0, x1, x1, x0, x0, np.nan]
        ys += [y0, y0, y1, y1, y0, np.nan]
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def tile_rectangles(centers: np.ndarray, sizes: np.ndarray) -> list[tuple]:
    return [(c[0] - s[0] / 2, c[0] + s[0] / 2, c[1] - s[1] / 2, c[1] + s[1] / 2) for c, s in zip(centers, sizes)]


class PointsStitchView(QtWidgets.QWidget):
    def __init__(
        self,
        tile_ids: list[str],
        sem_points: list[np.ndarray],
        design_points: list[np.ndarray],
        centers: np.ndarray,
        sizes: np.ndarray,
        stitch: StitchResult,
        design_stitch: StitchResult,
        use_opengl: bool = True,
    ):
        """tile_ids: a name per tile; sem_points, design_points: per tile, (N, 2) mask nm as given
        to stitch_tiles; centers, sizes: (n_tiles, 2) tile boxes given to stitch_tiles; stitch,
        design_stitch: the SEM and design StitchResults."""
        super().__init__()
        self.setWindowTitle(f"Stitching inputs - {len(tile_ids)} tiles")
        all_points = np.concatenate(design_points)
        self.origin = np.floor(all_points.min(axis=0) / 1000) * 1000  # whole µm, nm

        self.plot = pg.PlotWidget()
        if use_opengl:
            self.plot.useOpenGL(True)
        self.plot.setAspectLocked(True)
        for side, name in (("bottom", "x − x0 (µm)"), ("left", "y − y0 (µm)")):
            self.plot.setLabel(side, name)
            self.plot.getAxis(side).setScale(1e-3)  # data is nm
        x0, y0 = self.origin / 1000
        self.plot.setTitle(f"Design and SEM centres, nominal - origin x0 = {x0:.0f} µm, y0 = {y0:.0f} µm")
        self.plot.addLegend(offset=(5, 5))

        self.items = {}
        self._points("Design centres", design_points, DESIGN)
        self._points("SEM centres", sem_points, SEM)
        rectangles = tile_rectangles(centers, sizes)
        self._boxes("Tile boxes", rectangles, pg.mkPen("#808080"))
        self._boxes("Overlaps used (SEM)", [p.box for p in stitch.pairs], pg.mkPen("#ffff00"))
        self._boxes("Overlaps skipped (few contacts)", [r.box for r in stitch.rejected if not r.failed],
                    pg.mkPen("#909090", style=DASH))
        self._boxes("SEM overlaps not matched", [r.box for r in stitch.rejected if r.failed],
                    pg.mkPen("#ff00ff", style=DASH, width=2))
        self._boxes("Design overlaps not matched", [r.box for r in design_stitch.rejected if r.failed],
                    pg.mkPen("#ffa500", style=DASH_DOT, width=2))
        unplaced = sorted(set(stitch.unplaced.tolist()) | set(design_stitch.unplaced.tolist()))
        self._boxes("Tiles not stitched", [rectangles[k] for k in unplaced], pg.mkPen("#ff00ff", width=3))

        self.names = []
        for name, center in zip(tile_ids, centers):
            text = pg.TextItem(name, color="#c0c0c0", anchor=(0.5, 0.5))
            text.setPos(*(center - self.origin))
            text.setVisible(False)
            self.plot.addItem(text)
            self.names.append(text)
        self.names_box = QtWidgets.QCheckBox("Tile names")
        self.names_box.toggled.connect(lambda shown: [t.setVisible(shown) for t in self.names])

        self.status = QtWidgets.QLabel(self._status_text(len(tile_ids), stitch, design_stitch))
        self.status.setWordWrap(True)
        self.cursor_label = QtWidgets.QLabel()
        self.plot.scene().sigMouseMoved.connect(self._show_cursor)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.status, stretch=1)
        top.addWidget(self.names_box)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.plot, stretch=1)
        layout.addWidget(self.cursor_label)
        self.resize(1400, 1000)

    def _points(self, name, points, color):
        xy = np.concatenate(points) - self.origin
        item = self.plot.plot(xy[:, 0], xy[:, 1], pen=None, symbol="o", symbolSize=3, symbolPen=None,
                              symbolBrush=color, name=f"{name} ({len(xy)})")
        self.items[name] = item

    def _boxes(self, name, boxes, pen):
        shifted = [(x0 - self.origin[0], x1 - self.origin[0], y0 - self.origin[1], y1 - self.origin[1])
                   for x0, x1, y0, y1 in boxes]
        x, y = box_outlines(shifted)
        item = self.plot.plot(x, y, pen=pen, connect="finite", name=f"{name} ({len(boxes)})")
        self.items[name] = item

    @staticmethod
    def _status_text(n_tiles, stitch, design_stitch) -> str:
        parts = []
        for name, result in (("SEM", stitch), ("Design", design_stitch)):
            failed = sum(r.failed for r in result.rejected)
            parts.append(f"{name}: {len(result.pairs)} overlaps used, {failed} not matched, "
                         f"{len(result.rejected) - failed} skipped, "
                         f"{n_tiles - len(result.unplaced)} of {n_tiles} tiles stitched")
        return "; ".join(parts)

    def _show_cursor(self, scene_pos):
        pos = self.plot.getPlotItem().vb.mapSceneToView(scene_pos)
        x, y = (np.array([pos.x(), pos.y()]) + self.origin) / 1000
        self.cursor_label.setText(f"x = {x:.4f} µm, y = {y:.4f} µm (mask)")
