"""Frequency analysis of one mean per row or per column of the Registration tab, in a window of its own
(docs/SPEC.md D173, D174, D175).

The profile as the tab showed it when the window opened, along a horizontal axis; below it what is left after
removing parts of it; then the amplitude spectrum of both in cycles per µm, with the period in nm on the top axis:
least-squares sines (Lomb–Scargle), or an FFT of the profile resampled evenly (D175). Removed (D174): the slow part on request (Remove slow part: a running median, or a straight line per ramp, D175: the
sawtooth), and the sine of every picked peak, all picked sines fitted together on what the slow part leaves; the part
removed is drawn over the profile. Markers on the spectrum, off at start: the limit of the row spacing (a higher
frequency shows folded below it) and the image step with its multiples (where an error made inside every image would
show). A click on the spectrum picks the peak nearest to it in what is left: its sine, as removed, a stem up to its
amplitude (D176), with its period, its amplitude and the shorter periods it may have folded from written beside it.
A click on a picked peak drops it.
"""

from contextlib import contextmanager

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from affine_ransac import spectrum
from affine_ransac.release import style

MIN_POINTS = 4  # fewer points give no spectrum
CLICK_PX = 8  # a click on the spectrum picks the highest peak this many screen pixels around it
WINDOW_ROWS = 20  # the running median's window at start: this many median row spacings


@contextmanager
def busy():
    """A wait cursor while a spectrum is computed (about a second for 4000 points)."""
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()


def period_text(period_nm: float) -> str:
    return f"{period_nm:.0f}" if period_nm >= 100 else f"{period_nm:.3g}"


class PeriodAxis(pg.AxisItem):
    """A spectrum's top axis: the period (nm) at each tick of its frequency axis (cycles per µm)."""

    def tickStrings(self, values, scale, spacing):
        return [period_text(1000 / v) if v > 0 else "" for v in values]


def marker(frequency_nm: float, pen: QtGui.QPen, label: str = "", side: int = 0) -> pg.InfiniteLine:
    """A vertical line on the spectrum at this frequency (cycles per nm), named half-way up (below the legend), on its
    right (side 0) or its left (side 1)."""
    options = {"label": label, "labelOpts": {"position": 0.55, "anchors": [(side, 0), (side, 0)], "color": style.TEXT,
                                             "fill": pg.mkBrush(255, 255, 255, 210)}} if label else {}
    line = pg.InfiniteLine(frequency_nm * 1000, angle=90, pen=pen, movable=False, **options)
    line.setVisible(False)
    return line


def show_line(plot: pg.PlotItem, line: pg.PlotDataItem, name: str, on: bool):
    """Show a line and its legend entry, or hide both."""
    line.setVisible(on)
    plot.legend.removeItem(line)
    if on:
        plot.legend.addItem(line, name)


class SpectrumWindow(QtWidgets.QWidget):
    def __init__(self, parent, name: str, about: str, axis: str, position_nm, values_nm, color: str,
                 image_step_nm: float | None):
        """name: the profile's graph (e.g. "dy mean per row"); about: what it shows (stripes, values); axis: "x" or
        "y", along which the positions (nm) run; values (nm) the profile's; image_step_nm: the tiles' step along
        that axis, None if there is none."""
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"Frequency analysis: {name}")
        order = np.argsort(position_nm)
        self.position = np.asarray(position_nm, float)[order]
        self.values = np.asarray(values_nm, float)[order]
        self.spacing = spectrum.row_spacing(self.position)
        self.method = QtWidgets.QComboBox()
        self.method.addItems(["Lomb–Scargle", "FFT, resampled evenly"])
        self.method.setToolTip("Lomb–Scargle: a least-squares sine at each frequency, for rows at any spacing. FFT: the "
                               "profile resampled every median row spacing, straight across gaps, Hann window. Picked "
                               "peaks are measured by least squares either way")
        self.frequency, self.amplitude = self.spectrum_of(self.values)  # cycles per nm; nm
        self.picked = []  # the picked peaks' frequencies, cycles per nm

        self.figure = pg.GraphicsLayoutWidget()
        x_label, um = f"{axis} (µm)", self.position / 1000
        self.profile = self.figure.addPlot(row=0, col=0)
        style.style_plot(self.profile, name, x_label, "error (nm)")
        style.legend(self.profile)
        style.zero_line(self.profile)
        self.profile.plot(um, self.values, pen=pg.mkPen(color, width=2), name="profile")
        removed_pen = pg.mkPen(style.REMOVED, width=1.5, style=QtCore.Qt.PenStyle.DashLine)
        self.removed_line = self.profile.plot(um, np.zeros_like(um), pen=removed_pen)
        self.remainder_plot = self.figure.addPlot(row=1, col=0)
        style.style_plot(self.remainder_plot, "", x_label, "error (nm)")  # its title names its one line
        style.zero_line(self.remainder_plot)
        self.remainder_plot.setXLink(self.profile)
        self.remainder_line = self.remainder_plot.plot(um, self.values, pen=pg.mkPen(style.REMAINDER, width=2))

        self.spectrum = self.figure.addPlot(row=2, col=0, axisItems={"top": PeriodAxis("top")})
        style.style_plot(self.spectrum, "", "frequency (cycles per µm)", "amplitude (nm)")
        self.style_top_axis()
        style.legend(self.spectrum)
        self.profile_spectrum = self.spectrum.plot(self.frequency * 1000, self.amplitude, pen=pg.mkPen(color, width=2),
                                                   name="profile")
        self.remainder_spectrum = self.spectrum.plot(self.frequency * 1000, self.amplitude,
                                                     pen=pg.mkPen(style.REMAINDER, width=1.5))
        # The picked sines as removed, a stem each up to its amplitude: not on a spectrum line, which can be hidden.
        self.stems = self.spectrum.plot([], [], pen=removed_pen, connect="pairs")
        self.stems.visibleChanged.connect(self.show_labels)  # their labels go with them when the legend hides them
        self.labels = []
        self.figure.scene().sigMouseClicked.connect(self.clicked)

        self.method.currentIndexChanged.connect(self.change_method)
        self.slow_box = QtWidgets.QCheckBox("Remove slow part:")
        self.slow_box.setToolTip("Remove the profile's slow part (its sawtooth), leaving the faster signal")
        self.slow_box.toggled.connect(self.update_removal)
        self.slow_method = QtWidgets.QComboBox()
        self.slow_method.addItems(["running median over", "straight line per ramp"])
        self.slow_method.setToolTip("Running median: assumes no shape. Straight line per ramp: the profile split where it "
                                    "turns by more than the wiggle height, as the tab's ramp range")
        self.window = QtWidgets.QDoubleSpinBox(decimals=3, minimum=0.001, maximum=1e4, singleStep=0.1, suffix=" µm",
                                               keyboardTracking=False)
        self.window.setValue(round(WINDOW_ROWS * self.spacing / 1000, 3))
        self.window.setToolTip("Window of the running median. Shorter than a few periods of the signal to keep, it "
                               "removes part of it; longer, it leaves more of the slow part at its drops")
        self.wiggle = QtWidgets.QDoubleSpinBox(decimals=2, minimum=0.01, maximum=1e3, singleStep=0.05, suffix=" nm",
                                               keyboardTracking=False)
        # Not the tab's Auto (8 x the row-to-row noise): a fast signal that moves a lot from row to row raises that
        # above the sawtooth, and no ramp is found.
        self.wiggle.setValue(round(float(np.ptp(self.values)) / 2, 2))
        self.wiggle.setToolTip("Wiggles up to this height do not end a ramp. At start half the profile's range")
        for box in (self.slow_method, self.window, self.wiggle):
            (box.currentIndexChanged if box is self.slow_method else box.valueChanged).connect(self.change_slow_part)
        self.limit_box, self.image_box = self.marker_boxes(image_step_nm)
        menus = style.image_menus(lambda: (self.figure, [(name, [self.profile]),
                                                         (f"{name} after removal", [self.remainder_plot]),
                                                         (f"{name} spectrum", [self.spectrum])]))
        controls = QtWidgets.QHBoxLayout()
        for widget in (QtWidgets.QLabel("Spectrum:"), self.method, QtWidgets.QLabel("    "), self.limit_box,
                       self.image_box):
            controls.addWidget(widget)
        controls.addStretch(1)
        for button in menus:
            controls.addWidget(button)
        removal = QtWidgets.QHBoxLayout()
        for widget in (self.slow_box, self.slow_method, self.window, QtWidgets.QLabel("   wiggles up to"),
                       self.wiggle):
            removal.addWidget(widget)
        removal.addStretch(1)
        length_um = np.ptp(self.position) / 1000
        summary = QtWidgets.QLabel(
            f"<b>{name}</b>: {about}. {len(self.position)} points over {length_um:.1f} µm, {self.spacing:.1f} nm apart "
            f"(median): periods shorter than {2 * self.spacing:.0f} nm show folded to longer ones, and peaks closer than "
            f"{1000 * spectrum.resolution(self.position):.3g} cycles per µm are not told apart. Click a peak to measure "
            "it and remove its sine; click it again to put it back.")
        summary.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(summary)
        layout.addLayout(controls)
        layout.addLayout(removal)
        layout.addWidget(self.figure, stretch=1)
        self.resize(1000, 900)
        self.change_slow_part()
        self.update_removal()

    def style_top_axis(self):
        self.spectrum.showAxis("top")
        axis = self.spectrum.getAxis("top")
        axis.setPen(pg.mkPen(style.MUTED))
        axis.setTextPen(pg.mkPen(style.TEXT))
        axis.setStyle(tickFont=QtGui.QFont(style.FONT, style.TICK_PT), tickTextOffset=6)
        axis.setLabel("period (nm)", color=style.TEXT, **{"font-size": f"{style.LABEL_PT}pt", "font-family": style.FONT})

    def marker_boxes(self, image_step_nm: float | None) -> tuple[QtWidgets.QCheckBox, QtWidgets.QCheckBox]:
        """Tick boxes, off at start, each drawing its markers on the spectrum: the limit of the row spacing, and every
        multiple of one over the image step up to that limit."""
        top = spectrum.limit(self.position)
        limit_pen = pg.mkPen(style.LIMIT, width=1.5, style=QtCore.Qt.PenStyle.DashLine)
        image_pen = pg.mkPen(style.IMAGE_STEP, width=1, style=QtCore.Qt.PenStyle.DotLine)
        self.limit_lines = [marker(top, limit_pen, "limit", side=1)]  # at the right end: named on its left
        steps = np.arange(1, int(top * image_step_nm) + 1) / image_step_nm if image_step_nm else []
        self.image_lines = [marker(f, image_pen, "image step" if k == 0 else "") for k, f in enumerate(steps)]
        for line in self.limit_lines + self.image_lines:
            self.spectrum.addItem(line, ignoreBounds=True)
        limit_box = QtWidgets.QCheckBox("Limit of the row spacing")
        limit_box.setIcon(style.swatch(limit_pen))
        limit_box.setToolTip(f"Points {self.spacing:.1f} nm apart show a period shorter than {2 * self.spacing:.0f} nm "
                             "folded to a longer one: frequencies above this line show below it")
        image_box = QtWidgets.QCheckBox("Image step")
        image_box.setIcon(style.swatch(image_pen))
        if image_step_nm:
            image_box.setToolTip(f"Every multiple of one over the image step ({image_step_nm / 1000:.3g} µm): an error "
                                 "made inside every image would show at these frequencies")
        else:
            image_box.setEnabled(False)
            image_box.setToolTip("The tiles have no step along this axis")
        for box, lines in ((limit_box, self.limit_lines), (image_box, self.image_lines)):
            box.toggled.connect(lambda on, lines=lines: [line.setVisible(on) for line in lines])
        return limit_box, image_box

    def spectrum_of(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(frequencies, cycles per nm; amplitudes, nm) of values at the rows, by the method chosen (D175)."""
        with busy():
            if self.method.currentIndex() == 1:
                return spectrum.fft_spectrum(self.position, values)
            frequency = spectrum.frequency_grid(self.position)
            return frequency, spectrum.amplitude_spectrum(self.position, values, frequency)

    def change_method(self, *_):
        self.frequency, self.amplitude = self.spectrum_of(self.values)
        self.profile_spectrum.setData(self.frequency * 1000, self.amplitude)
        self.update_removal()

    def change_slow_part(self, *_):
        """Only the setting of the slow part's method chosen is active; the removal follows while it is ticked."""
        ramps = self.slow_method.currentIndex() == 1
        self.window.setEnabled(not ramps)
        self.wiggle.setEnabled(ramps)
        if self.slow_box.isChecked():
            self.update_removal()

    def slow_part(self) -> np.ndarray:
        if not self.slow_box.isChecked():
            return np.zeros_like(self.values)
        if self.slow_method.currentIndex() == 1:
            return spectrum.line_per_ramp(self.position, self.values, self.wiggle.value())
        return spectrum.running_median(self.position, self.values, self.window.value() * 1000)

    def update_removal(self, *_):
        """What is removed (the slow part if ticked, plus the picked sines fitted together on what it leaves), drawn
        over the profile; what is left and its spectrum; the picked peaks."""
        slow = self.slow_part()
        self.fitted, sines = spectrum.fit_sines(self.position, self.values - slow, self.picked)
        self.remainder = self.values - slow - sines
        removing = self.slow_box.isChecked() or bool(self.picked)
        self.remainder_amplitude = self.spectrum_of(self.remainder)[1] if removing else self.amplitude
        self.removed_line.setData(self.position / 1000, slow + sines)
        show_line(self.profile, self.removed_line, "removed", removing)
        self.remainder_line.setData(self.position / 1000, self.remainder)
        self.remainder_plot.setTitle("After removal" if removing else "After removal: nothing removed",
                                     color=style.TEXT, size=f"{style.TITLE_PT}pt", bold=True)
        self.remainder_spectrum.setData(self.frequency * 1000, self.remainder_amplitude)
        show_line(self.spectrum, self.remainder_spectrum, "after removal", removing)
        self.draw_picked()

    def clicked(self, event):
        """A left click on the spectrum picks or drops a peak, unless the legend took it (to hide or show a line)."""
        position = event.scenePos()
        if event.button() != QtCore.Qt.MouseButton.LeftButton or event.isAccepted() or \
                not self.spectrum.vb.sceneBoundingRect().contains(position):
            return
        within = max(spectrum.resolution(self.position), CLICK_PX * self.spectrum.vb.viewPixelSize()[0] / 1000)
        self.pick(self.spectrum.vb.mapSceneToView(position).x() / 1000, within)

    def pick(self, near: float, within: float):
        """Drop the picked peak nearest to this frequency (cycles per nm) if one is within `within` of it, else pick
        the top of the peak nearest to it in what is left."""
        close = [f for f in self.picked if abs(f - near) <= within]
        if close:
            self.picked.remove(min(close, key=lambda f: abs(f - near)))
        else:
            self.picked.append(spectrum.peak_near(self.position, self.remainder, self.frequency,
                                                  self.remainder_amplitude, near, within))
        self.update_removal()

    def draw_picked(self):
        """Every picked peak's sine as removed: a stem from zero up to its amplitude (fitted with the others), named
        "removed sines" in the legend, with its period, its amplitude and the shorter periods it may have folded from
        beside its top."""
        for label in self.labels:
            self.spectrum.removeItem(label)
        self.labels = []
        frequency = np.array(self.picked)
        self.stems.setData(np.repeat(frequency * 1000, 2),
                           np.column_stack([np.zeros(len(frequency)), self.fitted]).ravel())
        show_line(self.spectrum, self.stems, "removed sines", bool(self.picked))
        middle = np.mean(self.spectrum.vb.viewRange()[0]) / 1000
        for f, a in zip(frequency, self.fitted):
            folded = ", ".join(period_text(1 / g) for g in spectrum.folded_from(f, self.spacing))
            # Hanging from the stem's top, on the side with more room: a tall stem's label stays in the graph.
            label = pg.TextItem(f"period {period_text(1 / f)} nm, amplitude {a:.2f} nm\nor folded from {folded} nm",
                                color=style.TEXT, anchor=(float(f > middle), 0), fill=pg.mkBrush(255, 255, 255, 210))
            label.setPos(f * 1000, a)
            label.setVisible(self.stems.isVisible())
            self.spectrum.addItem(label, ignoreBounds=True)
            self.labels.append(label)

    def show_labels(self):
        """The picked peaks' labels are shown while their stems are."""
        for label in self.labels:
            label.setVisible(self.stems.isVisible())
