"""Interactive tuner for the moving-window affine (docs/SPEC.md S8, D36, D37), pyqtgraph.

Sliders set the window height and step (µm); the moving-window affine (moving_window_affine) is
recomputed shortly after a slider stops moving and every plot updates:
- Left: the registration error maps and mean error per row (view_registration.RegistrationView)
  of the moving-window residuals; the row plot also shows the RANSAC row means (dashed). The colour
  scale is kept from the RANSAC residuals, so colours compare with the RANSAC view.
- Right: one affine term of each window (chosen in the list) against the window's centre y. Each
  window's affine is relative to its own reference point (design centroid of the window), so its
  Tx, Ty is the shift there; the RANSAC affine is recentred on the same points (dashed) to compare.
  Correction G, units as in report_terms.
Positions relative to the RANSAC reference point, in µm; errors in nm.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from affine_ransac.fitting.moving_window import moving_window_affine
from affine_ransac.geometry.affine import recentre, report_terms
from affine_ransac.registration import row_means
from affine_ransac.view_registration import RegistrationView

TICK_UM = 0.5  # slider resolution


def term_values(model: np.ndarray) -> dict:
    """{label: value} of the affine terms (report_terms; the edge values are not used)."""
    return {label: value for label, value, _ in report_terms(model, np.zeros((1, 2)))}


class SliderBox(QtWidgets.QWidget):
    """A slider and a spin box showing the same value in µm (TICK_UM steps)."""

    changed = QtCore.Signal(float)

    def __init__(self, name: str, low_um: float, high_um: float, value_um: float):
        super().__init__()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(round(low_um / TICK_UM), round(high_um / TICK_UM))
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(1)
        self.spin.setSingleStep(TICK_UM)
        self.spin.setRange(low_um, high_um)
        self.spin.setSuffix(" µm")
        self.slider.valueChanged.connect(lambda ticks: self.spin.setValue(ticks * TICK_UM))
        self.spin.valueChanged.connect(self._spin_changed)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel(name))
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.spin)
        self.spin.setValue(value_um)

    def _spin_changed(self, value_um: float):
        self.slider.blockSignals(True)
        self.slider.setValue(round(value_um / TICK_UM))
        self.slider.blockSignals(False)
        self.changed.emit(value_um)

    def value(self) -> float:
        return self.spin.value()


class MovingWindowTuner(QtWidgets.QWidget):
    def __init__(
        self,
        sem_nm: np.ndarray,
        design_nm: np.ndarray,
        ransac,
        window_um: float = 40.0,
        step_um: float = 5.0,
        row_gap_nm: float = 10.0,
        use_opengl: bool = True,
        raw_error_nm: np.ndarray | None = None,
    ):
        """sem_nm, design_nm: (N, 2) matched contacts, mask nm; ransac: the RansacResult of the same
        contacts (its residuals, model and reference are shown for comparison); raw_error_nm:
        optional stitched SEM − design without affine, shown as a reference row profile."""
        super().__init__()
        self.setWindowTitle(f"Moving-window affine tuner - {len(design_nm)} contacts")
        self.sem_nm, self.design_nm, self.ransac = sem_nm, design_nm, ransac
        self.reference_um = ransac.reference / 1000
        height_um = np.ptp(design_nm[:, 1]) / 1000
        self.window = SliderBox("Window", 1.0, max(1.0, np.ceil(height_um)), window_um)
        self.step = SliderBox("Step", TICK_UM, max(1.0, np.ceil(height_um)), step_um)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)

        # Left: the registration view, with the RANSAC row means added as dashed lines.
        self.view = RegistrationView(design_nm, ransac.residuals, ransac.reference, row_gap_nm, use_opengl,
                                     correction="the RANSAC affine", raw_error_nm=raw_error_nm)
        row_y, ransac_rows, _ = row_means(design_nm[:, 1], ransac.residuals, row_gap_nm)
        row_um = row_y / 1000 - self.reference_um[1]
        for component, (name, color) in enumerate((("dx RANSAC", "#ff6040"), ("dy RANSAC", "#40a0ff"))):
            self.view.row_plot.plot(row_um, ransac_rows[:, component], name=name,
                                    pen=pg.mkPen(color, width=1, style=QtCore.Qt.PenStyle.DashLine))

        # Right: one term per window against the window's centre y.
        self.term = QtWidgets.QComboBox()
        self.term.addItems(list(term_values(ransac.model)))
        self.term_plot = pg.PlotWidget(title="Affine term per window (correction G)")
        self.term_plot.setLabel("bottom", "window centre y − y_ref (µm)")
        self.term_plot.addLegend(offset=(5, 5))
        self.term_curve = self.term_plot.plot(pen=pg.mkPen("#40d040", width=2), symbol="o", symbolSize=5,
                                              name="moving window")
        self.term_ransac = self.term_plot.plot(pen=pg.mkPen("#c0c0c0", style=QtCore.Qt.PenStyle.DashLine),
                                               name="RANSAC (global, at the same points)")
        self.term.currentTextChanged.connect(lambda *_: self.show_term())
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.addWidget(self.term)
        right_layout.addWidget(self.term_plot, stretch=1)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.view)
        splitter.addWidget(right)
        splitter.setSizes([1100, 500])
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.window)
        layout.addWidget(self.step)
        layout.addWidget(self.status)
        layout.addWidget(splitter, stretch=1)
        self.resize(1700, 1050)

        # Recompute shortly after the sliders stop moving (dragging fires many changes).
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.recompute)
        self.window.changed.connect(lambda *_: self.timer.start())
        self.step.changed.connect(lambda *_: self.timer.start())
        self.result = None
        self.recompute()

    def recompute(self):
        window_um, step_um = self.window.value(), self.step.value()
        try:
            result = moving_window_affine(self.sem_nm, self.design_nm, window_um * 1000, step_um * 1000)
        except ValueError as error:  # step > window, or a window with < 3 contacts: keep the last result
            self.status.setText(f"<span style='color:#ff5050'>Not updated: {error}</span>")
            return
        self.result = result
        self.status.setText(f"{len(result.centres_nm)} windows of {window_um:g} µm, step {step_um:g} µm, "
                            f"{result.counts.min()}..{result.counts.max()} contacts per window")
        self.view.set_errors(result.residuals, f"the moving-window affine ({window_um:g} µm, step {step_um:g} µm)")
        self.show_term()

    def show_term(self):
        if self.result is None:
            return
        label = self.term.currentText()
        y_um = self.result.centres_nm / 1000 - self.reference_um[1]
        self.term_curve.setData(y_um, [term_values(model)[label] for model in self.result.models])
        ransac = [term_values(recentre(self.ransac.model, reference - self.ransac.reference))[label]
                  for reference in self.result.references]
        self.term_ransac.setData(y_um, ransac)
        self.term_plot.setLabel("left", label)
