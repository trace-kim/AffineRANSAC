"""Translation vs affine stitching side by side (pyqtgraph). Display only.

StitchingComparison is one window, one tab each:
- Stitching (optional): a stitching viewer built by the caller.
- Comparison: the key numbers of every stitching (registration.stitching_summary) in a table, and
  the mean dx / dy per row of contacts (registration.row_means) against the row's y, one line per
  stitching: without affine, after the RANSAC affine, after the moving-window affine.
- One AnalysisWindow per stitching (RANSAC monitor, registration views, tuner, row pitch).

A run is (contacts, ransac, moving): the merged contacts of one stitching (registration.MergedErrors)
and their RansacResult and MovingWindowResult. Rows are plotted relative to the first run's RANSAC
reference point, in µm; errors in nm.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from affine_ransac.registration import row_means
from affine_ransac.view_analysis import AnalysisWindow

COLORS = ["#ffb000", "#00d0ff", "#a0ff60", "#ff60ff"]  # one per run


class ComparisonView(QtWidgets.QWidget):
    def __init__(self, runs: dict, summary: dict, row_gap_nm: float = 10.0, use_opengl: bool = True):
        """runs: name -> (contacts, ransac, moving); summary: name -> stitching_summary(...)."""
        super().__init__()
        self.table = QtWidgets.QTableWidget()
        self._fill_table(summary)

        self.graphics = pg.GraphicsLayoutWidget()
        if use_opengl:
            self.graphics.useOpenGL(True)
        reference = next(iter(runs.values()))[1].reference
        self.plots = {}  # (row title, "dx" / "dy") -> plot
        titles = ("no affine", "after the RANSAC affine", "after the moving-window affine")
        for row, title in enumerate(titles):
            for col, axis in enumerate(("dx", "dy")):
                plot = self.graphics.addPlot(row=row, col=col, title=f"Mean {axis} per row, {title}")
                plot.setLabel("bottom", "row y − y_ref (µm)")
                plot.setLabel("left", f"mean {axis} (nm)")
                plot.addLegend(offset=(5, 5))
                plot.addLine(y=0, pen=pg.mkPen("#808080", style=QtCore.Qt.PenStyle.DashLine))
                if self.plots:
                    plot.setXLink(next(iter(self.plots.values())))
                self.plots[title, axis] = plot

        for color, (name, (contacts, ransac, moving)) in zip(COLORS, runs.items()):
            for title, error in zip(titles, (contacts.error_nm, ransac.residuals, moving.residuals)):
                row_y, mean, _ = row_means(contacts.design_nm[:, 1], error, row_gap_nm)
                row_um = (row_y - reference[1]) / 1000
                for component, axis in enumerate(("dx", "dy")):
                    self.plots[title, axis].plot(row_um, mean[:, component], pen=pg.mkPen(color, width=2), name=name)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self.table)
        splitter.addWidget(self.graphics)
        splitter.setSizes([250, 750])
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(splitter)

    def _fill_table(self, summary: dict):
        """Rows: the summary keys; columns: the runs."""
        names = list(summary)
        keys = list(summary[names[0]]) if names else []
        self.table.setRowCount(len(keys))
        self.table.setColumnCount(len(names))
        self.table.setHorizontalHeaderLabels(names)
        self.table.setVerticalHeaderLabels(keys)
        for col, name in enumerate(names):
            for row, key in enumerate(keys):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{summary[name][key]:.3f}"))
        self.table.resizeColumnsToContents()


class StitchingComparison(QtWidgets.QTabWidget):
    def __init__(
        self,
        runs: dict,
        summary: dict,
        window_um: float,
        step_um: float,
        threshold_nm: float = 0.5,
        seed: int = 0,
        delay_ms: int = 300,
        row_gap_nm: float = 10.0,
        tile_edges_y_nm=(),
        stitch_view: QtWidgets.QWidget | None = None,
        use_opengl: bool = True,
    ):
        """runs: name -> (contacts, ransac, moving), e.g. {"translation": ..., "affine": ...};
        summary: name -> registration.stitching_summary of that run; the other arguments as for
        AnalysisWindow, which is opened once per run (its own tab)."""
        super().__init__()
        self.setWindowTitle(f"AffineRANSAC analysis - stitching {' vs '.join(runs)}")
        if stitch_view is not None:
            self.addTab(stitch_view, "Stitching")
        self.comparison = ComparisonView(runs, summary, row_gap_nm, use_opengl)
        self.addTab(self.comparison, "Comparison")
        self.analyses = {}
        for name, (contacts, ransac, moving) in runs.items():
            window = AnalysisWindow(contacts.sem_nm, contacts.design_nm, ransac, moving, window_um, step_um,
                                    threshold_nm, seed, delay_ms, row_gap_nm, tile_edges_y_nm, use_opengl=use_opengl)
            self.analyses[name] = window
            self.addTab(window, f"Analysis, {name} stitching")
        self.resize(1700, 1050)
