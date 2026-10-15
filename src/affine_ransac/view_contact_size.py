"""Contact size maps: area, xCD and yCD of every contact (analysis.ContactSizes), pyqtgraph.

- Show: Area / xCD / yCD check boxes choose the maps (one, two or all three side by side; Area at start).
- Value: the SEM value (measured on the SEM contour), or SEM − design (the .oas outline).
- Every contact a dot coloured by its value (jet), drawn like the registration error maps
  (view_registration.draw_map); each map has its own colour bar, from the 1st to the 99th percentile
  (also for SEM − design: a size bias between SEM and design is normal, a scale centred on 0 would hide
  the variation; drag the bar to change it); the maps share zoom and pan. Positions relative to the reference point, µm; area in nm², CDs in nm.
- The summary gives per quantity the mean and 3σ of the SEM value, the design value and SEM − design.
- Behind the map (image runs in memory, off at start): the SEM images and the design (.oas) outlines, each
  tile placed like the contacts (the corrections of the run's placement); the map is then transparent
  between the contacts.

ContactSizeTab is the analysis window's tab: the measurement runs only when asked (Run measurement, with
an ROI width), in a background thread (analysis.measure_contact_sizes needs the images of an image run in
memory); the maps (ContactSizeView) appear when it is done. Sizes saved with a result are shown at once.
"""

import threading
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from scipy.spatial import cKDTree

from affine_ransac.analysis import AnalysisResult, measure_contact_sizes
from affine_ransac.stitching import placement_corrections
from affine_ransac.view_registration import JET, draw_map
from affine_ransac.view_stitch import COLORS, DESIGN, image_item, path_item

QUANTITIES = (("Area", "nm²"), ("xCD", "nm"), ("yCD", "nm"))  # the columns of ContactSizes
TO_UM = QtGui.QTransform.fromScale(1e-3, 1e-3)  # item nm -> map µm


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
                 reference_nm: np.ndarray, use_opengl: bool = True, sem_tiles: list | None = None,
                 design_tiles: list | None = None):
        """design_nm: (N, 2) contact positions, mask nm; sem_sizes, design_sizes: (N, 3) area (nm²), xCD
        and yCD (nm) of the same contacts on the SEM and the design; reference_nm: (2,) the point the
        positions are shown relative to (the RANSAC reference, as on the other maps).
        sem_tiles: optional [(TileResult, centre_nm)], design_tiles: optional [(polygons, centre_nm)]: what
        can be shown behind the map, each tile's image or .oas polygons (tile-local nm) placed at centre_nm
        (2,), mask nm (tile_backgrounds); None: that box is disabled."""
        super().__init__()
        self.sem_sizes, self.design_sizes = sem_sizes, design_sizes
        self.reference_nm = np.asarray(reference_nm, dtype=float)
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
        bar.addSpacing(30)
        bar.addWidget(QtWidgets.QLabel("Behind the map:"))
        self.backgrounds = {}  # box -> (tiles, item maker, z); items made the first time the box is checked
        self.background_items = {}  # box -> per map, its items
        for name, tiles, make, z in (("SEM images", sem_tiles, self._image_item, -10),
                                     ("Design", design_tiles, self._design_item, 10)):
            box = QtWidgets.QCheckBox(name)
            box.setEnabled(tiles is not None)
            if tiles is None:
                box.setToolTip("Needs the SEM images of an image run in memory: analyse the image folder again")
            box.toggled.connect(self.show_backgrounds)
            bar.addWidget(box)
            self.backgrounds[box] = (tiles, make, z)
        self.sem_box, self.design_box = self.backgrounds
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
        transparent = self.sem_box.isChecked()  # black between the contacts would hide the images
        for k, (box, (_, plot, image, color_bar)) in enumerate(zip(self.show_boxes, self.maps)):
            if box.isChecked():  # a hidden map is drawn when it is shown (show_maps)
                draw_map(plot, image, self.position_um, self.values(k), color_bar.levels(), self.dot_um,
                         self.devicePixelRatioF(), transparent)

    def show_backgrounds(self):
        """Show the SEM images / design outlines whose box is checked (made on first use), then redraw."""
        for box, (tiles, make, z) in self.backgrounds.items():
            if box.isChecked() and box not in self.background_items:
                self.background_items[box] = []
                for _, plot, _, _ in self.maps:
                    items = [make(thing, centre_nm) for thing, centre_nm in tiles]
                    for item in items:
                        item.setZValue(z)  # images below the map (z 0), outlines above it
                        plot.addItem(item)
                    self.background_items[box].append(items)
            for items in self.background_items.get(box, []):
                for item in items:
                    item.setVisible(box.isChecked())
        self.draw_maps()

    def _image_item(self, tile, centre_nm):
        item = image_item(tile)  # tile-local nm
        item.setTransform(item.transform() * TO_UM)
        item.setPos(*((centre_nm - self.reference_nm) / 1000))
        return item

    def _design_item(self, polygons, centre_nm):
        item = path_item(polygons, COLORS[DESIGN])  # tile-local nm, cosmetic pen
        item.setTransform(TO_UM)
        item.setPos(*((centre_nm - self.reference_nm) / 1000))
        return item

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


def tile_backgrounds(result: AnalysisResult) -> tuple[list, list] | tuple[None, None]:
    """What ContactSizeView can show behind the maps of an image run in memory: [(TileResult, centre_nm)]
    and [(.oas polygons, centre_nm)], each tile at its nominal centre plus the SEM / design correction of
    the placement the contacts were merged with (settings.ransac_placement), like the contacts on the map.
    Tiles not stitched are left out. (None, None) without the images."""
    if result.images is None:
        return None, None
    sem_shift, design_shift = placement_corrections(result.stitchings["raw"].stitch.corrections,
                                                    result.design_stitch.corrections)[result.settings.ransac_placement]
    sem_tiles, design_tiles = [], []
    for k, (tile, polygons) in enumerate(zip(result.images.tiles, result.images.design_polygons)):
        if np.isfinite(sem_shift[k]).all():
            sem_tiles.append((tile, tile.center_nm + sem_shift[k]))
        if np.isfinite(design_shift[k]).all() and len(polygons):
            design_tiles.append((polygons, tile.center_nm + design_shift[k]))
    return sem_tiles, design_tiles


class _Measurement(QtCore.QObject):
    """measure_contact_sizes in a daemon thread; progress, the ContactSizes or the error arrive as signals
    in the GUI thread (as app.Job)."""
    progress = QtCore.Signal(int, int)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def start(self, result: AnalysisResult, roi_width_nm: float):
        threading.Thread(target=self._run, args=(result, roi_width_nm), daemon=True).start()

    def _run(self, result, roi_width_nm):
        try:
            sizes = measure_contact_sizes(result, roi_width_nm, progress=self.progress.emit)
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
        else:
            self.done.emit(sizes)


class ContactSizeTab(QtWidgets.QWidget):
    def __init__(self, result: AnalysisResult, use_opengl: bool = True):
        """The Contact size tab of an image run's result: ROI width, Run measurement, then the maps.
        result.contact_sizes is set when a measurement is done (so Save results keeps it)."""
        super().__init__()
        self.result, self.use_opengl, self.view = result, use_opengl, None
        self.roi = QtWidgets.QDoubleSpinBox(decimals=1, maximum=1000.0, suffix=" nm")
        self.roi.setToolTip("xCD / yCD = mean of the chords, one per SEM pixel, in a band this wide through the "
                            "contact centre (0: the single chord through the centre)")
        self.run_button = QtWidgets.QPushButton("Run measurement")
        self.run_button.clicked.connect(self.run)
        self.status = QtWidgets.QLabel()
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("ROI width:"))
        bar.addWidget(self.roi)
        bar.addWidget(self.run_button)
        bar.addWidget(self.status, stretch=1)
        self.layout_ = QtWidgets.QVBoxLayout(self)
        self.layout_.addLayout(bar)
        self.layout_.addStretch(1)  # replaced by the maps

        self.measurement = _Measurement()
        self.measurement.progress.connect(lambda done, total: self.status.setText(f"Measuring: {done} of {total} tiles"))
        self.measurement.done.connect(self._done)
        self.measurement.failed.connect(self._failed)
        if result.images is None:
            self.run_button.setEnabled(False)
            self.status.setText("Measuring needs the SEM images in memory: analyse the image folder again "
                                "(sizes measured before a result was saved are shown)")
        else:
            self.status.setText("Area, xCD and yCD of every contact: press Run measurement")
        if result.contact_sizes is not None:
            self.roi.setValue(result.contact_sizes.roi_width_nm)
            self._show(result.contact_sizes)

    def run(self):
        """Start the measurement with the ROI width in the box."""
        self.run_button.setEnabled(False)
        self.status.setText("Measuring...")
        self.started = time.perf_counter()
        self.measurement.start(self.result, self.roi.value())

    def _done(self, sizes):
        self.result.contact_sizes = sizes
        self._show(sizes)
        self.status.setText(f"Measured {len(sizes.sem)} contacts in {time.perf_counter() - self.started:.0f} s")
        self.run_button.setEnabled(True)

    def _failed(self, text: str):
        self.status.setText(f"Measurement failed: {text}")
        self.run_button.setEnabled(True)

    def _show(self, sizes):
        """Replace the maps (or the empty space) by a ContactSizeView of sizes."""
        main = self.result.sets["uncorrected"]
        old = self.layout_.takeAt(1)
        if old.widget() is not None:
            old.widget().deleteLater()
        self.view = ContactSizeView(main.design_nm, sizes.sem, sizes.design, main.ransac.reference, self.use_opengl,
                                    *tile_backgrounds(self.result))
        band = "one chord" if sizes.roi_width_nm == 0 else f"a {sizes.roi_width_nm:g} nm band of chords"
        self.view.label.setText(f"xCD, yCD from {band} through the centre.\n" + self.view.label.text())
        self.layout_.addWidget(self.view, stretch=1)
