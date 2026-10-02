"""Live monitor of the RANSAC affine fit (pyqtgraph, OpenGL viewport). Display only: it steps
fitting.ransac.ransac_affine_steps() with a timer and redraws after every step.

Plots, for the current iteration's model (search: the random 3-point sample; refit: least squares
on the inliers):
- Map: contacts at their design positions relative to the reference point (design centroid),
  inliers green, outliers red; the 3-point sample as large yellow stars joined by a triangle.
- Residual scatter: G(SEM) − design, dx vs dy (nm), with the threshold circle τ.
- Linear fit: the RAW error (SEM − design, dots) along x and along y, each against x and against
  y, with the model's predicted error as a line (through the reference point; its slopes are the
  model's magnification / rotation / orthogonality terms), and the RESIDUAL after the model
  (G(SEM) − design, crosses) on the same axes: what is left once the line is taken away.
- Inlier count per iteration: this sample (dots) and the best so far (line); the needed number
  of iterations N is in the status line.
- Table: the correction G of this model and of the best one (geometry.affine.report_terms):
  Tx, Ty (nm), Mx, My (ppm = nm per mm), rotation and orthogonality (degrees), each with how far
  that term alone moves the furthest contact (nm at the field edge); below them the affine's own
  coefficients a, b, c, d, tx, ty (x' = a·x + b·y + tx, y' = c·x + d·y + ty, about the reference).

Controls: Run / Pause, Step, Run to end, delay per step; Restart applies τ and the seed.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from affine_ransac.fitting.ransac import RansacStep, ransac_affine_steps, residuals
from affine_ransac.geometry.affine import apply_affine, coefficients, report_terms

INLIER, OUTLIER, SAMPLE, MODEL = "#33cc33", "#ff4040", "#ffff00", "#3c8cff"
RESIDUAL_INLIER, RESIDUAL_OUTLIER = "#00e5ff", "#ffa500"  # residual after the model, drawn as crosses
TERM_LABELS = ([label for label, *_ in report_terms(np.eye(3), np.zeros((1, 2)))]
               + [label for label, _ in coefficients(np.eye(3))])
COLUMNS = ["this model", "edge (nm)", "best so far", "edge (nm)"]


def two_scatters(plot, size=4, names=(None, None)):
    """An inlier and an outlier scatter item (filled dots) in plot."""
    items = [pg.ScatterPlotItem(size=size, pen=None, brush=pg.mkBrush(color), name=name)
             for color, name in zip((INLIER, OUTLIER), names)]
    for item in items:
        plot.addItem(item)
    return items


def two_cross_scatters(plot, size=6, names=(None, None)):
    """An inlier and an outlier scatter item drawn as crosses (residuals) in plot."""
    items = [pg.ScatterPlotItem(size=size, symbol="x", pen=pg.mkPen(color), brush=pg.mkBrush(color), name=name)
             for color, name in zip((RESIDUAL_INLIER, RESIDUAL_OUTLIER), names)]
    for item in items:
        plot.addItem(item)
    return items


class RansacMonitor(QtWidgets.QWidget):
    def __init__(
        self,
        sem: np.ndarray,
        design: np.ndarray,
        threshold_nm: float = 0.5,
        seed: int = 0,
        confidence: float = 0.999,
        max_iters: int = 10_000,
        min_area_fraction: float = 1e-3,
        refine_iters: int = 3,
        interval_ms: int = 300,
        use_opengl: bool = True,
    ):
        """sem, design: (N, 2) matched points in mask nm, e.g. MergedErrors.sem_nm / design_nm."""
        super().__init__()
        self.setWindowTitle(f"RANSAC monitor - {len(sem)} contacts")
        self.sem, self.design = sem, design
        self.options = dict(confidence=confidence, max_iters=max_iters, min_area_fraction=min_area_fraction,
                            refine_iters=refine_iters)
        self.raw_error = sem - design  # SEM − design, the error the affine explains

        self.graphics = pg.GraphicsLayoutWidget()
        if use_opengl:
            self.graphics.useOpenGL(True)
        self._make_plots()
        panel = self._make_panel(threshold_nm, seed, interval_ms)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.graphics, stretch=1)
        layout.addWidget(panel)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)
        self.restart()
        self.resize(1700, 950)

    # --- layout -------------------------------------------------------------------------------
    def _make_plots(self):
        g = self.graphics
        self.map_plot = g.addPlot(row=0, col=0, rowspan=3, title="Contacts (design position)")
        self.map_plot.setAspectLocked(True)
        self.map_plot.setLabel("bottom", "x − x_ref (µm)")
        self.map_plot.setLabel("left", "y − y_ref (µm)")
        self.map_points = two_scatters(self.map_plot)
        self.sample_triangle = pg.PlotCurveItem(pen=pg.mkPen(SAMPLE, width=2))
        self.sample_points = pg.ScatterPlotItem(size=18, symbol="star", pen=pg.mkPen("k"), brush=pg.mkBrush(SAMPLE))
        self.map_plot.addItem(self.sample_triangle)
        self.map_plot.addItem(self.sample_points)

        self.residual_plot = g.addPlot(row=0, col=1, title="Residual G(SEM) − design")
        self.residual_plot.setAspectLocked(True)
        self.residual_plot.setLabel("bottom", "dx (nm)")
        self.residual_plot.setLabel("left", "dy (nm)")
        self.residual_points = two_scatters(self.residual_plot)
        self.threshold_circle = pg.PlotCurveItem(pen=pg.mkPen(SAMPLE, style=QtCore.Qt.PenStyle.DashLine))
        self.residual_plot.addItem(self.threshold_circle)

        self.count_plot = g.addPlot(row=0, col=2, title="Inliers per iteration")
        self.count_plot.setLabel("bottom", "iteration")
        self.sample_counts = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush("#c0c0c0"))
        self.best_counts = pg.PlotCurveItem(pen=pg.mkPen(INLIER, width=2))
        self.count_plot.addItem(self.sample_counts)
        self.count_plot.addItem(self.best_counts)

        # Linear fit: error component (0 = x, 1 = y) against position axis (0 = x, 1 = y):
        # raw error (dots), the model's line, and the residual after the model (crosses).
        self.linear = {}
        for row, component in ((1, 0), (2, 1)):
            for col, axis in ((1, 0), (2, 1)):
                name = "xy"
                plot = g.addPlot(row=row, col=col, title=f"d{name[component]} vs {name[axis]}")
                plot.setLabel("bottom", f"{name[axis]} − {name[axis]}_ref (µm)")
                plot.setLabel("left", f"d{name[component]} (nm)")
                first = not self.linear
                if first:
                    plot.addLegend(offset=(5, 5))
                raw = two_scatters(plot, size=3, names=("raw, inlier", "raw, outlier") if first else (None, None))
                line = pg.PlotCurveItem(pen=pg.mkPen(MODEL, width=2), name="model" if first else None)
                plot.addItem(line)
                residual = two_cross_scatters(plot, names=("residual, inlier", "residual, outlier") if first
                                              else (None, None))
                self.linear[component, axis] = (raw, line, residual)
        # Short titles above: a long plot title sets a minimum width and pushes plots off-screen.
        for col, stretch in enumerate((2, 1, 1)):
            g.ci.layout.setColumnStretchFactor(col, stretch)

    def _make_panel(self, threshold_nm, seed, interval_ms):
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(420)
        column = QtWidgets.QVBoxLayout(panel)

        buttons = QtWidgets.QGridLayout()
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.clicked.connect(self.toggle_run)
        step_button = QtWidgets.QPushButton("Step")
        step_button.clicked.connect(self.step)
        end_button = QtWidgets.QPushButton("Run to end")
        end_button.clicked.connect(self.run_to_end)
        restart_button = QtWidgets.QPushButton("Restart")
        restart_button.clicked.connect(self.restart)
        for i, button in enumerate((self.run_button, step_button, end_button, restart_button)):
            buttons.addWidget(button, i // 2, i % 2)
        column.addLayout(buttons)

        form = QtWidgets.QFormLayout()
        self.delay_box = QtWidgets.QSpinBox()
        self.delay_box.setRange(0, 5000)
        self.delay_box.setSuffix(" ms")
        self.delay_box.setValue(interval_ms)
        self.delay_box.valueChanged.connect(self.timer_interval)
        self.threshold_box = QtWidgets.QDoubleSpinBox()
        self.threshold_box.setDecimals(2)
        self.threshold_box.setRange(0.01, 1000)
        self.threshold_box.setSuffix(" nm")
        self.threshold_box.setValue(threshold_nm)
        self.seed_box = QtWidgets.QSpinBox()
        self.seed_box.setRange(0, 2 ** 31 - 1)
        self.seed_box.setValue(seed)
        form.addRow("Delay per step", self.delay_box)
        form.addRow("Threshold τ (restart)", self.threshold_box)
        form.addRow("Seed (restart)", self.seed_box)
        column.addLayout(form)

        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        column.addWidget(self.status)

        self.table = QtWidgets.QTableWidget(len(TERM_LABELS), len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setVerticalHeaderLabels(TERM_LABELS)
        self.table.setToolTip("Correction G (SEM -> design). edge (nm): how far the term alone moves the "
                              "furthest contact. ppm = nm per mm.")
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        column.addWidget(self.table)
        column.addStretch(1)
        return panel

    # --- control ------------------------------------------------------------------------------
    def timer_interval(self, ms: int):
        self.timer.setInterval(ms)

    def restart(self):
        """Start a new run with the threshold and seed from the panel."""
        self.timer.stop()
        self.run_button.setText("Run")
        self.threshold_nm = self.threshold_box.value()
        self.steps = ransac_affine_steps(self.sem, self.design, self.threshold_nm, seed=self.seed_box.value(),
                                         **self.options)
        self.history = []  # (iteration, sample inliers, best inliers) of the search steps
        self.last_step = None
        self.done = False
        self.timer.setInterval(self.delay_box.value())
        self.status.setText(f"Ready: {len(self.sem)} contacts, τ = {self.threshold_nm:g} nm. Press Run or Step.")

    def toggle_run(self):
        if self.timer.isActive():
            self.timer.stop()
            self.run_button.setText("Run")
        elif not self.done:
            self.timer.start()
            self.run_button.setText("Pause")

    def step(self):
        """Advance one RANSAC iteration and redraw. Returns False when the run is finished."""
        step = next(self.steps, None)
        if step is None:
            self._finish()
            return False
        self.last_step = step
        if step.stage == "search":
            self.history.append((step.iteration, step.inliers.sum(), step.best_inliers.sum()))
        self.show_step(step)
        return True

    def run_to_end(self):
        """Run the remaining iterations without drawing each one, then show the final state."""
        self.timer.stop()
        last = self.last_step
        for last in self.steps:
            if last.stage == "search":
                self.history.append((last.iteration, last.inliers.sum(), last.best_inliers.sum()))
        if last is not None:
            self.last_step = last
            self.show_step(last)
        self._finish()

    def _finish(self):
        self.timer.stop()
        self.run_button.setText("Run")
        self.done = True
        self.status.setText(self.status.text() + "\nFinished: the last refit is the final affine G.")

    # --- drawing ------------------------------------------------------------------------------
    def show_step(self, step: RansacStep):
        """Draw one step: this iteration's model if it has one, else the best so far."""
        model = step.model if step.model is not None else step.best_model
        inliers = step.inliers if step.model is not None else step.best_inliers
        ref = step.reference
        position = self.design - ref  # nm, relative to the reference point
        groups = (inliers, ~inliers)

        for item, mask in zip(self.map_points, groups):
            item.setData(position[mask, 0] / 1000, position[mask, 1] / 1000)
        if step.sample is not None:
            corners = position[np.append(step.sample, step.sample[0])] / 1000
            self.sample_triangle.setData(corners[:, 0], corners[:, 1])
            self.sample_points.setData(corners[:3, 0], corners[:3, 1])
        else:
            self.sample_triangle.setData([], [])
            self.sample_points.setData([], [])

        if model is not None:
            residual = residuals(model, self.sem - ref, position)
            for item, mask in zip(self.residual_points, groups):
                item.setData(residual[mask, 0], residual[mask, 1])
            angle = np.linspace(0, 2 * np.pi, 100)
            self.threshold_circle.setData(self.threshold_nm * np.cos(angle), self.threshold_nm * np.sin(angle))
            self._show_linear(model, position, groups, residual)
        self._show_history()
        self._show_table(step)

        kind = "degenerate sample (rejected)" if step.degenerate else ("refit (least squares on inliers)"
                                                                       if step.stage == "refit" else "3-point sample")
        n = len(self.sem)
        text = (f"{step.stage.capitalize()} iteration {step.iteration}: {kind}\n"
                f"Inliers: this model {inliers.sum()} / {n}, best {step.best_inliers.sum()} / {n} "
                f"({step.best_inliers.mean():.1%})\n"
                f"Iterations needed (adaptive N): {step.needed_iterations}; τ = {self.threshold_nm:g} nm")
        if step.improved and step.stage == "search":
            text += "\nNew best model."
        self.status.setText(text)

    def _show_linear(self, model, position, groups, residual):
        """Raw error and residual vs position, and the model's predicted error along each axis
        through the reference."""
        # The model maps SEM -> design, so its predicted SEM − design at a point p is p − G(p).
        for (component, axis), (scatters, line, residual_scatters) in self.linear.items():
            for item, mask in zip(scatters, groups):
                item.setData(position[mask, axis] / 1000, self.raw_error[mask, component])
            for item, mask in zip(residual_scatters, groups):
                item.setData(position[mask, axis] / 1000, residual[mask, component])
            span = np.array([position[:, axis].min(), position[:, axis].max()])
            along = np.zeros((2, 2))
            along[:, axis] = span
            predicted = along - apply_affine(model, along)
            line.setData(span / 1000, predicted[:, component])

    def _show_history(self):
        if not self.history:
            return
        iteration, sample, best = np.array(self.history).T
        self.sample_counts.setData(iteration, sample)
        self.best_counts.setData(iteration, best)

    def _show_table(self, step: RansacStep):
        points = self.design - step.reference
        for col, model in ((0, step.model), (2, step.best_model)):
            if model is None:
                rows = [(label, None, None) for label in TERM_LABELS]
            else:
                rows = report_terms(model, points) + [(label, value, None) for label, value in coefficients(model)]
            for row, (label, value, edge) in enumerate(rows):
                # Degrees are tiny (1 µrad = 0.0000573°); a, b, c, d differ from 1 / 0 by ~1e-6.
                digits = 6 if "°" in label else 9 if label in ("a", "b", "c", "d") else 3
                texts = ("-", "-") if value is None else (f"{value:+.{digits}f}", "" if edge is None else f"{edge:.3f}")
                for offset, text in enumerate(texts):
                    self.table.setItem(row, col + offset, QtWidgets.QTableWidgetItem(text))
