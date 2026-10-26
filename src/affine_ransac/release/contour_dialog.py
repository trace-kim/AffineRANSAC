"""Contour diagnosis (docs/SPEC.md D129): try the contour settings (analysis.CONTOUR_SETTINGS, D128) on one SEM image
before a run uses them.

ContourDialog shows a loaded image, as the contacts are found in it (with the background flattened if asked, D130),
with the contacts the settings find in it, exactly as a run finds them (pipeline.find_contacts): the detected contours and centres (the detection method's), the refined contours and centres
(after edge refinement: what the run measures) and, on request, the threshold mask (the pixels the threshold takes as
contact). One line says in words how many contacts were found and why the other regions were left out. Every change of
a setting finds them again, in a background thread, for the newest settings only. Apply to parameters: values() gives
the settings with those shown, which the app keeps for the next run; Cancel leaves the parameters as they were.

Coordinates are the pixel frame: (x, y) px, y down, pixel centres at whole numbers (docs/SPEC.md §3).
"""

import html
import threading
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from affine_ransac.analysis import CHOICES, Settings, check_settings, contour_options
from affine_ransac.app import editor, editor_value
from affine_ransac.features.contact import DetectedContacts, contact_mask
from affine_ransac.io.sem_image import load_sem_image
from affine_ransac.pipeline import find_contacts, prepared_image
from affine_ransac.release import style
from affine_ransac.release.analysis import drawable_outlines
from affine_ransac.release.views import figure

GROUPS = (("Pre-processing", ("background_px",)),
          ("Detection (thresholding)", ("detection", "dark_contacts", "detection_blur_px", "min_area_px",
                                        "max_area_px", "min_solidity", "min_contrast")),
          ("Edge refinement", ("refinement", "search_px", "edge_step_px", "edge_blur_px", "normal_span")))
REFINEMENT_OPTIONS = ("search_px", "edge_step_px", "edge_blur_px", "normal_span")  # unused with refinement none
STEPS = {"background_px": 5, "detection_blur_px": 0.1, "min_area_px": 5, "max_area_px": 100, "min_solidity": 0.01, "min_contrast": 0.5,
         "search_px": 0.5, "edge_step_px": 0.05, "edge_blur_px": 0.1}  # spin box steps (the others: 1)
LAYERS = ("Threshold mask", "Detected contours", "Detected centres", "Refined contours", "Refined centres")
IMAGE_FILES = "SEM images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;All files (*)"
NO_IMAGE = "Load an SEM image (<i>Load image...</i>) to see the contacts these settings find in it."


@dataclass
class Diagnosis:
    """What the window shows of one image with one set of settings."""
    image: np.ndarray           # the image the contacts are found in (pipeline.prepared_image)
    detected: DetectedContacts  # after the contrast check
    refined: DetectedContacts   # what the run measures (refinement none: the detected ones)
    low_contrast: int           # detected contacts the contrast check dropped
    mask: np.ndarray            # (rows, cols) uint8, 1 = taken as contact by the threshold


def diagnose(image: np.ndarray, s: Settings) -> Diagnosis:
    """The image the contacts are found in, the contacts the settings find, as a run finds them, and the threshold
    mask."""
    prepared = prepared_image(image, s.background_px, s.dark_contacts)
    # find_contacts would flatten the image first: given it flattened (~0.1-0.3 s), it need not do it again
    detected, refined, low_contrast = find_contacts(prepared, **contour_options(replace(s, background_px=0.0)))
    mask, _ = contact_mask(prepared, s.dark_contacts, s.detection_blur_px, s.detection)
    return Diagnosis(prepared, detected, refined, low_contrast, mask)


def summary(d: Diagnosis, s: Settings) -> str:
    """How many contacts were found and why the other regions were left out, in words (HTML)."""
    reasons = {"border": "touching the image border", "small": f"smaller than {s.min_area_px} px",
               "large": f"larger than {s.max_area_px} px", "not solid": f"less solid than {s.min_solidity:g}"}
    left_out = [f"{count} {reasons[why]}" for why, count in d.detected.left_out.items() if count]
    if d.low_contrast:
        left_out.append(f"{d.low_contrast} below the contrast of {s.min_contrast:g} noise σ")
    flattened = " of the flattened image" if s.background_px > 0 else ""
    text = (f"<b>{len(d.refined.centers)} contacts found</b> ({s.detection}, threshold at grey level "
            f"{d.detected.threshold:.0f}{flattened}). ")
    return text + (f"Regions left out: {', '.join(left_out)}." if left_out else "")


def applies(name: str, s: Settings) -> bool:
    """Whether a contour setting changes anything, given the others: min_solidity for band only, the refinement's
    options unless refinement is none."""
    if name == "min_solidity":
        return s.detection == "band"
    return name not in REFINEMENT_OPTIONS or s.refinement != "none"


def changed_signal(widget: QtWidgets.QWidget):
    """The signal of an editor (app.editor) that says its value changed."""
    if isinstance(widget, QtWidgets.QComboBox):
        return widget.currentTextChanged
    if isinstance(widget, QtWidgets.QCheckBox):
        return widget.toggled
    return widget.valueChanged


def place(item: pg.ImageItem, shape):
    """Put an image item in the pixel frame: pixel (row, column) centred at (x, y) = (column, row)."""
    item.setRect(QtCore.QRectF(-0.5, -0.5, shape[1], shape[0]))


class ContourDialog(QtWidgets.QDialog):
    searched = QtCore.Signal(int, object)  # (search number, Diagnosis or the exception), from the background thread

    def __init__(self, settings: Settings, folder: str = "", use_opengl: bool = False, parent=None):
        """settings: the parameters to start from (values() keeps the others); folder: where Load image starts."""
        super().__init__(parent)
        self.setWindowTitle("Contour diagnosis")
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMaximizeButtonHint)
        self.settings, self.folder = settings, folder
        self.image = None     # the SEM image loaded (uint8)
        self.shown = None     # the image shown: the one the contacts are found in
        self.request = 0      # number of the newest search asked for
        self.running = False  # a search runs in the background thread
        self.searched.connect(self._searched)
        self.timer = QtCore.QTimer(self, singleShot=True, interval=150)  # a search once the typing pauses
        self.timer.timeout.connect(self.search)

        # The figure: the image, the threshold mask over it, the contours and centres on top.
        self.graphics = figure(use_opengl)
        self.plot = self.graphics.addPlot()
        style.style_plot(self.plot, x_label="x (px)", y_label="y (px)", grid=False)  # a grid would cross the image
        self.plot.invertY(True)  # image rows run down
        self.plot.setAspectLocked(True)
        style.legend(self.plot)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_item.setZValue(-10)
        self.plot.addItem(self.image_item)
        mask = pg.ImageItem(axisOrder="row-major")
        color = QtGui.QColor(style.THRESHOLD)
        mask.setLookupTable(np.array([[0, 0, 0, 0], [color.red(), color.green(), color.blue(), 110]], np.uint8))
        mask.setZValue(-5)
        self.layers = {"Threshold mask": mask}
        for name in LAYERS[1:]:
            color = style.DETECTED if name.startswith("Detected") else style.REFINED
            if name.endswith("contours"):
                self.layers[name] = pg.PlotDataItem(pen=pg.mkPen(color, width=1.5), connect="finite")
            else:
                self.layers[name] = pg.PlotDataItem(pen=None, symbol="+", symbolSize=9,
                                                    symbolPen=pg.mkPen(color, width=1.5))
        for item in self.layers.values():
            self.plot.addItem(item)

        self.boxes = {}
        show = QtWidgets.QHBoxLayout()
        show.addWidget(QtWidgets.QLabel("Show:"))
        for name in LAYERS:
            box = QtWidgets.QCheckBox(name)
            box.setChecked(name != "Threshold mask")  # the mask covers the image: off at start (D76)
            box.toggled.connect(self.update_visibility)
            show.addWidget(box)
            self.boxes[name] = box
        self.boxes["Threshold mask"].setToolTip("The pixels the threshold takes as contact (for band: everything "
                                                "but the bright band), before the regions are checked")
        show.addStretch(1)

        # The settings, in two groups; what each does in its tooltip.
        help_texts = {f.name: f.metadata["help"] for f in fields(Settings)}
        self.editors = {}
        panel = QtWidgets.QVBoxLayout()
        note = QtWidgets.QLabel("Hover a setting to see what it does.")
        note.setStyleSheet(f"color: {style.MUTED}")
        panel.addWidget(note)
        for title, names in GROUPS:
            group = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(group)
            for name in names:
                widget = editor(getattr(settings, name), CHOICES.get(name))
                if isinstance(widget, QtWidgets.QAbstractSpinBox):
                    widget.setMinimum(0)
                    widget.setSingleStep(STEPS.get(name, 1))
                label = QtWidgets.QLabel(name.upper())
                for item in (label, widget):
                    item.setToolTip(help_texts[name])
                form.addRow(label, widget)
                changed_signal(widget).connect(self.changed)
                self.editors[name] = widget
            panel.addWidget(group)
        panel.addStretch(1)

        self.load_button = QtWidgets.QPushButton("Load image...")
        self.load_button.setToolTip("Open one SEM image to try the settings on")
        self.load_button.clicked.connect(self.choose_image)
        self.name_label = QtWidgets.QLabel("")
        self.summary = QtWidgets.QLabel(NO_IMAGE, wordWrap=True)
        self.busy_label = QtWidgets.QLabel("")
        self.busy_label.setStyleSheet(f"color: {style.MUTED}")
        self.cursor_label = QtWidgets.QLabel("")
        self.cursor_label.setStyleSheet(f"color: {style.MUTED}")
        self.plot.scene().sigMouseMoved.connect(self._show_cursor)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.apply_button.setText("Apply to parameters")
        self.apply_button.setToolTip("Keep these settings for the next run (Analysis > Parameters lists them)")
        for button in buttons.buttons():  # Enter in a setting must not close the window
            button.setAutoDefault(False)
            button.setDefault(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.load_button)
        top.addWidget(self.name_label)
        top.addStretch(1)
        for button in style.image_buttons(lambda: self.graphics, "contour diagnosis"):
            top.addWidget(button)
        status = QtWidgets.QHBoxLayout()
        status.addWidget(self.summary, stretch=1)
        status.addWidget(self.busy_label)
        middle = QtWidgets.QHBoxLayout()
        middle.addWidget(self.graphics, stretch=1)
        side = QtWidgets.QWidget()
        side.setLayout(panel)
        side.setFixedWidth(330)
        middle.addWidget(side)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.cursor_label, stretch=1)
        bottom.addWidget(buttons)
        layout = QtWidgets.QVBoxLayout(self)
        for row in (top, status, show):
            layout.addLayout(row)
        layout.addLayout(middle, stretch=1)
        layout.addLayout(bottom)
        self.changed()
        self.resize(1500, 950)

    def values(self) -> Settings:
        """The settings given, with the contour settings shown."""
        return replace(self.settings, **{name: editor_value(widget) for name, widget in self.editors.items()})

    def changed(self, *_):
        """A setting changed: grey out the settings that change nothing now, then search again once the typing
        pauses."""
        s = self.values()
        for name, widget in self.editors.items():
            widget.setEnabled(applies(name, s))
        for name in ("Refined contours", "Refined centres"):  # with refinement none they are the detected ones (D124)
            self.boxes[name].setEnabled(s.refinement != "none")
        self.update_visibility()
        self.timer.start()

    def choose_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "SEM image", self.folder, IMAGE_FILES)
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        """Show the image, whole, and find its contacts."""
        try:
            image = load_sem_image(path)
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.warning(self, "Contour diagnosis", f"The image could not be read:\n{error}")
            return
        self.image, self.shown, self.folder = image, image, str(Path(path).parent)
        self.image_item.setImage(image, levels=(0, 255))
        place(self.image_item, image.shape)
        self.name_label.setText(f"{Path(path).name}   ({image.shape[1]} × {image.shape[0]} px)")
        self.plot.setRange(xRange=(-0.5, image.shape[1] - 0.5), yRange=(-0.5, image.shape[0] - 0.5), padding=0.02)
        self.search()

    def search(self):
        """Find the contacts with the settings shown, in the background. While a search runs, the newest settings
        wait for it (_searched starts them)."""
        if self.image is None:
            return
        self.request += 1  # the result of any earlier search is out of date now
        try:
            check_settings(self.values(), "images")
        except ValueError as error:
            self.summary.setText(f"<span style='color:{style.FAILURE}'>{html.escape(str(error))}</span>")
            self.apply_button.setEnabled(False)
            return
        self.apply_button.setEnabled(True)
        if not self.running:
            self._start()

    def _start(self):
        self.running = True
        self.busy_label.setText("Finding the contacts...")
        number, image, s = self.request, self.image, self.values()

        def work():
            try:
                found = diagnose(image, s)
            except Exception as error:
                found = error
            try:
                self.searched.emit(number, found)
            except RuntimeError:  # the window was closed meanwhile
                pass
        threading.Thread(target=work, daemon=True).start()

    def _searched(self, number: int, found):
        self.running = False
        if number != self.request:  # the settings changed meanwhile
            self.search()
            return
        self.busy_label.setText("")
        if isinstance(found, Exception):
            self.summary.setText(f"<span style='color:{style.FAILURE}'>The contacts could not be found: "
                                 f"{html.escape(str(found))}</span>")
            return
        self.show_diagnosis(found)

    def show_diagnosis(self, d: Diagnosis):
        self.shown = d.image
        self.image_item.setImage(d.image, levels=(0, 255))
        self.layers["Threshold mask"].setImage(d.mask, levels=(0, 1))
        place(self.layers["Threshold mask"], d.mask.shape)
        for name, contacts in (("Detected", d.detected), ("Refined", d.refined)):
            xy = drawable_outlines(contacts.contours)
            self.layers[f"{name} contours"].setData(xy[:, 0], xy[:, 1], connect="finite")
            self.layers[f"{name} centres"].setData(contacts.centers[:, 0], contacts.centers[:, 1])
        self.summary.setText(summary(d, self.values()))
        self.update_visibility()

    def update_visibility(self, *_):
        """Show the layers ticked (and offered); the legend lists the contours and centres shown, each with the method
        or rule it comes from."""
        s = self.values()
        for name, item in self.layers.items():
            item.setVisible(self.boxes[name].isChecked() and self.boxes[name].isEnabled())
        self.plot.legend.clear()
        for name in LAYERS[1:]:
            if self.layers[name].isVisible():
                self.plot.legend.addItem(self.layers[name],
                                         f"{name} ({s.detection if name.startswith('Detected') else s.refinement})")

    def _show_cursor(self, scene_pos):
        """The cursor's position and the grey level of the pixel under it in the image shown (to compare with the
        threshold)."""
        if self.shown is None:
            return
        pos = self.plot.vb.mapSceneToView(scene_pos)
        column, row = round(pos.x()), round(pos.y())
        rows, cols = self.shown.shape
        grey = f", grey level {self.shown[row, column]}" if 0 <= row < rows and 0 <= column < cols else ""
        self.cursor_label.setText(f"x = {pos.x():.1f} px, y = {pos.y():.1f} px{grey}")
