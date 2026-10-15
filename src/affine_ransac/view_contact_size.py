"""Contact size maps: area, xCD and yCD of every contact (analysis.ContactSizes), pyqtgraph.

- Show: Area / xCD / yCD check boxes choose the maps (one, two or all three side by side; Area at start).
- Value: the SEM value (measured on the SEM contour), or SEM − design (the .oas outline).
- Every contact a dot coloured by its value (jet), drawn like the registration error maps
  (view_registration.draw_map); each map has its own colour bar, from the 1st to the 99th percentile
  (also for SEM − design: a size bias between SEM and design is normal, a scale centred on 0 would hide
  the variation; drag the bar to change it); the maps share zoom and pan. Positions relative to the reference point, µm; area in nm², CDs in nm.
- The summary gives per quantity the mean and 3σ of the SEM value, the design value and SEM − design.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.spatial import cKDTree

from affine_ransac.view_registration import JET, draw_map

QUANTITIES = (("Area", "nm²"), ("xCD", "nm"), ("yCD", "nm"))  # the columns of ContactSizes


def value_levels(values: np.ndarray) -> tuple[float, float]:
    """Colour scale of a map: the 1st to the 99th percentile of the values (NaN left out)."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [1, 99])
    return (float(low), float(high)) if high > low else (float(low) - 1, float(low) + 1)


def size_summary(name: str, unit: str, sem: np.ndarray, design: np.ndarray) -> str:
    """One summary line of a quantity: mean and 3σ of the SEM value, the design value and SEM − design."""
    def stats(values):
        values = values[np.isfinite(values)]
        return f"mean {values.mean():.2f}, 3σ {3 * values.std():.2f}" if len(values) else "no value"
    missing = int((~np.isfinite(sem - design)).sum())
    return (f"{name} ({unit}): SEM {stats(sem)}; design {stats(design)}; SEM − design {stats(sem - design)}"
            + (f"; {missing} contact(s) without a value" if missing else ""))


class ContactSizeView(QtWidgets.QWidget):
    def __init__(self, design_nm: np.ndarray, sem_sizes: np.ndarray, design_sizes: np.ndarray,
                 reference_nm: np.ndarray, use_opengl: bool = True):
        """design_nm: (N, 2) contact positions, mask nm; sem_sizes, design_sizes: (N, 3) area (nm²), xCD
        and yCD (nm) of the same contacts on the SEM and the design; reference_nm: (2,) the point the
        positions are shown relative to (the RANSAC reference, as on the other maps)."""
        super().__init__()
        self.sem_sizes, self.design_sizes = sem_sizes, design_sizes
        self.position_um = (design_nm - reference_nm) / 1000
        if len(design_nm) > 1:
            distance, _ = cKDTree(design_nm).query(design_nm, k=2)
            self.dot_um = 0.4 * np.median(distance[:, 1]) / 1000  # ~0.4 x the contact pitch
        else:
            self.dot_um = 0.01

        self.show_boxes = []
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Show:"))
        for k, (name, _) in enumerate(QUANTITIES):
            box = QtWidgets.QCheckBox(name)
            box.setChecked(k == 0)
            box.toggled.connect(self.show_maps)
            bar.addWidget(box)
            self.show_boxes.append(box)
        bar.addSpacing(30)
        bar.addWidget(QtWidgets.QLabel("Value:"))
        self.sem_button = QtWidgets.QRadioButton("SEM")
        self.difference_button = QtWidgets.QRadioButton("SEM − design")
        self.sem_button.setChecked(True)
        self.difference_button.toggled.connect(self.set_values)
        bar.addWidget(self.sem_button)
        bar.addWidget(self.difference_button)
        bar.addStretch(1)

        self.maps = []  # (widget, plot, image, colour bar) per quantity
        row = QtWidgets.QHBoxLayout()
        for name, unit in QUANTITIES:
            widget = pg.GraphicsLayoutWidget()
            if use_opengl:
                widget.useOpenGL(True)
            plot = widget.addPlot(row=0, col=0)
            plot.setAspectLocked(True)
            plot.setLabel("bottom", "x − x_ref (µm)")
            plot.setLabel("left", "y − y_ref (µm)")
            image = pg.ImageItem(axisOrder="row-major")
            plot.addItem(image)
            color_bar = pg.ColorBarItem(values=(0, 1), colorMap=JET, label=unit)
            color_bar.sigLevelsChanged.connect(lambda *_: self.draw_maps())  # the maps colour themselves
            widget.addItem(color_bar, row=0, col=1)
            row.addWidget(widget)
            self.maps.append((widget, plot, image, color_bar))
        self._link_maps()

        # Redraw shortly after the view stops changing (zooming fires many range changes) or is resized.
        self.redraw_timer = QtCore.QTimer(self)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.setInterval(50)
        self.redraw_timer.timeout.connect(self.draw_maps)
        for _, plot, _, _ in self.maps:
            plot.vb.sigRangeChanged.connect(lambda *_: self.redraw_timer.start())
            plot.vb.sigResized.connect(lambda *_: self.redraw_timer.start())

        self.label = QtWidgets.QLabel("\n".join(
            size_summary(name, unit, sem_sizes[:, k], design_sizes[:, k]) for k, (name, unit) in enumerate(QUANTITIES)))
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addLayout(bar)
        layout.addLayout(row, stretch=1)
        if len(design_nm):  # start on all contacts
            low, high = self.position_um.min(axis=0), self.position_um.max(axis=0)
            self.maps[0][1].vb.setRange(xRange=(low[0], high[0]), yRange=(low[1], high[1]))
        self.show_maps()
        self.set_values()

    def values(self, k: int) -> np.ndarray:
        """(N,) what map k shows: the SEM value, or SEM − design."""
        if self.difference_button.isChecked():
            return self.sem_sizes[:, k] - self.design_sizes[:, k]
        return self.sem_sizes[:, k]

    def set_values(self):
        """Titles and colour scales for the chosen value (SEM or SEM − design), then redraw."""
        difference = self.difference_button.isChecked()
        for k, ((name, unit), (_, plot, _, color_bar)) in enumerate(zip(QUANTITIES, self.maps)):
            plot.setTitle(f"{name}, SEM − design ({unit})" if difference else f"{name}, SEM ({unit})")
            color_bar.blockSignals(True)  # one redraw below, not one per bar
            color_bar.setLevels(value_levels(self.values(k)))
            color_bar.blockSignals(False)
        self.draw_maps()

    def show_maps(self):
        """Show the maps whose box is checked, on the area of the first one shown (a hidden map has no size,
        so it did not keep the area while hidden)."""
        shown = [m for box, m in zip(self.show_boxes, self.maps) if box.isChecked()]
        for box, (widget, _, _, _) in zip(self.show_boxes, self.maps):
            widget.setVisible(box.isChecked())
        for _, plot, _, _ in shown[1:]:
            plot.vb.setRange(rect=shown[0][1].vb.viewRect(), padding=0)
        self.redraw_timer.start()

    def draw_maps(self):
        for k, (box, (_, plot, image, color_bar)) in enumerate(zip(self.show_boxes, self.maps)):
            if box.isChecked():  # a hidden map is drawn when it is shown (show_maps)
                draw_map(plot, image, self.position_um, self.values(k), color_bar.levels(), self.dot_um,
                         self.devicePixelRatioF())

    def _link_maps(self):
        """Keep all maps on the same area (as RegistrationView._link_maps for its two)."""
        busy = [False]

        def follow(source):
            def update(*_):
                if not busy[0]:
                    busy[0] = True
                    for _, target, _, _ in self.maps:
                        if target is not source:
                            target.vb.setRange(rect=source.vb.viewRect(), padding=0)
                    busy[0] = False
            source.vb.sigRangeChanged.connect(update)

        for _, plot, _, _ in self.maps:
            follow(plot)
