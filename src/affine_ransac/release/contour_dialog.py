"""Contour diagnosis (docs/SPEC.md D129): try the contour settings (analysis.CONTOUR_SETTINGS, D128) on one SEM image
before a run uses them.

ContourDialog shows a loaded image, as the contacts are found in it (with the background flattened, D130, denoised,
D132, and upscaled, D131, if asked), with the contacts the settings find in it, exactly as a run finds them
(pipeline.find_contacts): the detected contours and centres (the detection method's), the refined contours and centres
(after edge refinement: what the run measures) and, on request, the threshold mask (the pixels the threshold takes as
contact). One line says how many contacts were found and why the other regions were left out. Get contours starts a
background search. Auto retrieve also searches after settings changes and image loads. Apply to parameters: values() gives
the settings with those shown, which the app keeps for the next run; Cancel leaves the parameters as they were.
Original image (D160) shows the image as loaded instead, under the same layers.
With PATTERN lines (D153) it finds horizontal lines instead (pipeline.find_lines): the contour layers draw their
detected and refined edges, and the line says how many lines were found, their edge roughness and the noise per column.
With PATTERN contours (D155) it finds the contours of any pattern (pipeline.find_contours): the contour layers draw
them, closed or open, and the line says how many were found and how many of them are open.
With EQUALIZE (D167) the preview shows the equalized image used to choose the threshold. Image previews update
independently of contour retrieval (D168).

Coordinates are the pixel frame: (x, y) px, y down, pixel centres at whole numbers (docs/SPEC.md §3).
"""

import html
import threading
import time
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from affine_ransac.analysis import (CHOICES, PATTERN_SETTINGS, Settings, any_shape_options, check_settings,
                                    contour_options, line_options)
from affine_ransac.app import editor, editor_value
from affine_ransac.features.contact import DetectedContacts, contact_mask_and_image, threshold_image, upscaled_image
from affine_ransac.features.contours import Contours
from affine_ransac.features.lines import LineEdges, column_noise_px, roughness_px
from affine_ransac.io.sem_image import load_sem_image
from affine_ransac.pipeline import find_contacts, find_contours, find_lines, prepared_image
from affine_ransac.step_log import StepLog, timed
from affine_ransac.release import style
from affine_ransac.release.analysis import drawable_outlines
from affine_ransac.release.views import figure

GROUPS = (("Pattern", ("pattern",)),
          ("Pre-processing", ("background_px", "denoise", "median_px", "nlm_h", "tv_weight", "upscale")),
          ("Detection (thresholding)", ("detection", "dark_contacts", "detection_blur_px", "equalize", "clahe_clip",
                                        "clahe_tiles", "min_area_px", "max_area_px", "min_solidity", "min_contrast")),
          ("Edge refinement", ("refinement", "search_px", "edge_step_px", "edge_blur_px", "normal_span")))
REFINEMENT_OPTIONS = ("search_px", "edge_step_px", "edge_blur_px", "normal_span")  # unused with refinement none
DENOISE_OPTIONS = {"median_px": "median", "nlm_h": "non-local means", "tv_weight": "total variation"}  # each method's
EQUALIZE_OPTIONS = ("clahe_clip", "clahe_tiles")  # used by adaptive equalization only
STEPS = {"background_px": 5, "median_px": 2, "tv_weight": 0.01, "detection_blur_px": 0.1, "clahe_clip": 0.5,
         "min_area_px": 5, "max_area_px": 100, "min_solidity": 0.01, "min_contrast": 0.5, "search_px": 0.5,
         "edge_step_px": 0.05, "edge_blur_px": 0.1}  # spin box steps (the others: 1)
LAYERS = ("Threshold mask", "Detected contours", "Detected centres", "Refined contours", "Refined centres")
IMAGE_FILES = "SEM images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;All files (*)"
NO_IMAGE = "Load an SEM image to try the contour settings."
READY = "Click <i>Get contours</i> to retrieve contours for these settings."


@dataclass
class Diagnosis:
    """What the window shows of one image with one set of settings."""
    image: np.ndarray           # the prepared image or, with equalization, the threshold input
    detected: DetectedContacts | LineEdges | Contours  # after the contrast check (lines, contours: none)
    refined: DetectedContacts | LineEdges | Contours   # what the run measures (refinement none: the detected ones)
    low_contrast: int           # detected contacts the contrast check dropped
    mask: np.ndarray            # uint8, the pixels of `image`: 1 = taken as contact by the threshold
    log: StepLog = field(default_factory=StepLog)  # the time of each step of the search (log_entry, D157)


def preparation_key(s: Settings) -> tuple:
    return (s.background_px, s.dark_contacts if s.background_px > 0 else None, s.denoise,
            next((getattr(s, name) for name, method in DENOISE_OPTIONS.items() if s.denoise == method), None))


def threshold_image_options(s: Settings) -> tuple:
    return s.detection_blur_px * s.upscale, s.equalize, s.clahe_clip, s.clahe_tiles


def preview_key(s: Settings) -> tuple:
    blur, method, clip, tiles = threshold_image_options(s)
    equalization = () if method == "none" else (method, blur)
    if method == "adaptive":
        equalization += (clip, tiles)
    return preparation_key(s), s.upscale, equalization


class PreparedImageCache:
    """One loaded image's native-resolution preprocessing, reused by sequential diagnosis searches.

    Loaded images are not edited in place. A new image gets a new cache, so an older search still finishing in
    the background cannot replace its preparation. Only the active preprocessing options belong in the key.
    """

    def __init__(self):
        self.image = self.key = self.flat = None

    def prepare(self, image: np.ndarray, s: Settings, log: StepLog) -> np.ndarray:
        key = preparation_key(s)
        if self.image is image and self.key == key:
            with timed(log, "reuse flattened and denoised image"):
                return self.flat
        flat = prepared_image(image, s.background_px, s.dark_contacts, s.denoise, s.median_px, s.nlm_h,
                              s.tv_weight, log=log)
        self.image, self.key, self.flat = image, key, flat
        return flat


def preview_image(image: np.ndarray, s: Settings, preparation: PreparedImageCache) -> np.ndarray:
    prepared = upscaled_image(preparation.prepare(image, s, StepLog()), s.upscale)
    if s.equalize != "none":
        return threshold_image(prepared, *threshold_image_options(s))[0]
    return prepared


def diagnose(image: np.ndarray, s: Settings, preparation: PreparedImageCache | None = None) -> Diagnosis:
    """The preview image, the contacts the settings find, as a run finds them, and the threshold
    mask. preparation: the window's single-image cache; omitted, preprocessing runs afresh."""
    log = StepLog()
    flat = (preparation if preparation is not None else PreparedImageCache()).prepare(image, s, log)
    # find_contacts would flatten and denoise the image first: given it so (up to a few s), it need not do it again
    given = replace(s, background_px=0.0, denoise="none")
    if s.pattern == "lines":
        with timed(log, "find the lines and their edges"):
            (detected, refined), low_contrast = find_lines(flat, **line_options(given)), 0
    elif s.pattern == "contours":
        (detected, refined), low_contrast = find_contours(flat, **any_shape_options(given), log=log), 0
    else:
        seconds = {}
        detected, refined, low_contrast = find_contacts(flat, **contour_options(given), seconds=seconds)
        log.add("detect the contacts (upscale, threshold mask, regions, contrast check)", seconds["detect"])
        log.add("refine their edges", seconds["refine"])
    with timed(log, "upscale and threshold mask again, for the display"):
        blur, equalize, clip, tiles = threshold_image_options(s)
        shown = upscaled_image(flat, s.upscale)
        mask, _, pixels = contact_mask_and_image(shown, s.dark_contacts, blur, s.detection, equalize, clip, tiles)
        if equalize != "none":
            shown = pixels
    return Diagnosis(shown, detected, refined, low_contrast, mask, log)


NOT_TIMED = "not timed (between the steps; waiting for the window)"


def log_entry(d: Diagnosis, s: Settings, name: str, shape: tuple[int, int], total: float) -> str:
    """One search as text for the window's log (D157): when, the image and the settings that decide the time; the
    time of every step (d.log), with its share of `total` (seconds from the start of the search until it was drawn)
    and the longest marked; what is left of the total as NOT_TIMED; then the counts the times depend on."""
    seconds = dict(d.log.seconds)
    seconds[NOT_TIMED] = max(total - sum(seconds.values()), 0.0)
    longest = max(seconds, key=seconds.get)
    lines = [f"{time.strftime('%H:%M:%S')}  {name}, {shape[1]} × {shape[0]} px.  PATTERN {s.pattern}, DETECTION "
             f"{s.detection}, REFINEMENT {s.refinement}, UPSCALE {s.upscale}, DENOISE {s.denoise}, BACKGROUND_PX "
             f"{s.background_px:g}, MIN_AREA_PX {s.min_area_px}",
             f"  {total:.2f} s in all:"]
    lines += [f"  {t:8.2f} s {100 * t / max(total, 1e-9):4.0f} %  {step}" + ("   <== the longest" * (step == longest))
              for step, t in seconds.items()]
    rounds = d.log.counts.get("total variation rounds")
    if rounds is not None:  # its time: the rounds' count times the time of one round over the whole image
        each = d.log.seconds["total variation: its rounds"] / rounds
        lines.append(f"  Total variation: {rounds} rounds (at most 200: it stops once the image hardly changes), "
                     f"{1e3 * each:.1f} ms each.")
    return "\n".join(lines + ["  " + line for line in count_lines(d, s)]) + "\n"


def count_lines(d: Diagnosis, s: Settings) -> list[str]:
    """What a search worked on, in words: the numbers its time depends on."""
    pixels = f"{d.image.size:,} pixels searched"
    if s.pattern == "lines":
        return [f"{pixels}. {len(d.refined.upper):,} lines."]
    if s.pattern == "contacts":
        return [f"{pixels}. {len(d.refined.centers):,} contacts. Regions left out: {d.detected.left_out}."]
    found, c, left = d.refined, d.log.counts, d.detected.left_out
    regions, counts = np.unique(found.region, return_counts=True)
    most = int(np.argmax(counts)) if len(counts) else None
    points = np.array([len(p) for p in found.points])
    lines = [f"{pixels}. Mask: {c['regions']:,} regions; {c['kept']:,} kept, {left['small']:,} smaller than "
             f"MIN_AREA_PX, {left['large']:,} larger than MAX_AREA_PX.",
             f"Bounding boxes of the kept regions, all together: {c['box_pixels'] / d.image.size:.1f} times the image.",
             f"Boundaries around holes inside the kept regions: {c['holes']:,} (holes are not left out by size).",
             f"{len(points):,} contours ({int(found.closed.sum()):,} closed, {int((~found.closed).sum()):,} open), "
             f"{int(points.sum()):,} edge points."]
    if most is not None:
        lines.append(f"Most contours in one region: {int(counts[most]):,}, with "
                     f"{int(points[found.region == regions[most]].sum()):,} edge points.")
    return lines


def summary(d: Diagnosis, s: Settings) -> str:
    """How many contacts were found and why the other regions were left out, in words (HTML)."""
    if s.pattern == "lines":
        return lines_summary(d, s)
    reasons = {"border": "touching the image border", "small": f"smaller than {s.min_area_px} px",
               "large": f"larger than {s.max_area_px} px", "not solid": f"less solid than {s.min_solidity:g}"}
    left_out = [f"{count} {reasons[why]}" for why, count in d.detected.left_out.items() if count]
    if d.low_contrast:
        left_out.append(f"{d.low_contrast} below the contrast of {s.min_contrast:g} noise σ")
    if s.pattern == "contours":
        closed = d.refined.closed
        found = (f"{len(closed)} contours found</b>: {int(closed.sum())} closed, {int((~closed).sum())} open, "
                 f"ending at the image border")
    else:
        found = f"{len(d.refined.centers)} contacts found</b>"
    text = f"<b>{found} ({threshold_text(d, s)}). "
    return text + (f"Regions left out: {', '.join(left_out)}." if left_out else "")


def threshold_text(d: Diagnosis, s: Settings) -> str:
    """The detection method and its threshold, in words: a grey level of the image searched, or with adaptive
    equalization of the equalized image (D167)."""
    level = f"threshold at grey level {d.detected.threshold:.0f}"
    if s.equalize == "adaptive":
        return f"{s.detection} after adaptive equalization, {level} of the equalized image"
    if s.equalize == "global":
        return f"{s.detection} on the equalized histogram, {level} before equalization"
    return f"{s.detection}, {level}{of_the_image(s)}"


def lines_summary(d: Diagnosis, s: Settings) -> str:
    """How many lines and edges were found, their roughness and noise per column, and how many regions were left out,
    in words (HTML)."""
    lines = d.refined
    upper, lower = (int(np.isfinite(edges).any(axis=1).sum()) for edges in (lines.upper, lines.lower))
    text = f"<b>{len(lines.upper)} lines found</b> ({upper} upper and {lower} lower edges; {threshold_text(d, s)}). "
    if upper + lower:
        shared = " (reads low: neighbouring columns share noise in a denoised or upscaled image)" \
            if s.denoise != "none" or s.upscale > 1 else ""
        text += (f"Edge roughness {roughness_px(lines):.2f} px rms, noise per column "
                 f"{column_noise_px(lines):.2f} px{shared}. ")
    return text + (f"Regions left out: {lines.left_out} not crossing the image." if lines.left_out else "")


def of_the_image(s: Settings) -> str:
    """What was done to the image before the search, in words ("" if nothing)."""
    steps = [step for step, done in (("flattened", s.background_px > 0),
                                     (f"denoised ({s.denoise})", s.denoise != "none"),
                                     (f"upscaled {s.upscale}×", s.upscale > 1)) if done]
    if len(steps) > 1:
        steps = [", ".join(steps[:-1]) + " and " + steps[-1]]
    return f" of the image {steps[0]}" if steps else ""


def layer_name(name: str, s: Settings) -> str:
    """A layer's name as shown: lines have edges, not contours."""
    return name.replace("contours", "edges") if s.pattern == "lines" else name


def applies(name: str, s: Settings) -> bool:
    """Whether a contour setting changes anything, given the others: min_solidity for band only, each denoising
    method's option for that method only, the CLAHE options for adaptive equalization only, the refinement's options
    unless refinement is none; for lines and for contours, only the settings that find them (PATTERN_SETTINGS)."""
    if name != "pattern" and name not in PATTERN_SETTINGS[s.pattern]:
        return False
    if name == "min_solidity":
        return s.detection == "band"
    if name in DENOISE_OPTIONS:
        return s.denoise == DENOISE_OPTIONS[name]
    if name in EQUALIZE_OPTIONS:
        return s.equalize == "adaptive"
    return name not in REFINEMENT_OPTIONS or s.refinement != "none"


def drawable_edges(lines: LineEdges) -> np.ndarray:
    """Every edge of the lines as one (rows, 2) float32 line (x = column, y) to draw with connect="finite": NaN where
    an edge is missing and after each edge."""
    edges = np.column_stack([lines.edges, np.full(len(lines.edges), np.nan)])
    x = np.tile(np.append(np.arange(edges.shape[1] - 1, dtype=float), np.nan), len(edges))
    return np.column_stack([x, edges.ravel()]).astype(np.float32)


def drawable_contours(contours: Contours) -> np.ndarray:
    """The contours as one (rows, 2) float32 line to draw with connect="finite": a closed one with its first point
    again, an open one as it is, each followed by a NaN row."""
    nan = np.full((1, 2), np.nan)
    parts = [np.vstack([points, points[:1], nan] if closed else [points, nan])
             for points, closed in zip(contours.points, contours.closed)]
    return np.concatenate(parts).astype(np.float32) if parts else np.empty((0, 2), np.float32)


def changed_signal(widget: QtWidgets.QWidget):
    """The signal of an editor (app.editor) that says its value changed."""
    if isinstance(widget, QtWidgets.QComboBox):
        return widget.currentTextChanged
    if isinstance(widget, QtWidgets.QCheckBox):
        return widget.toggled
    return widget.valueChanged


def place(item: pg.ImageItem, shape):
    """Put an image item over an image of this shape in the pixel frame: pixel (row, column) centred at (x, y) =
    (column, row). An item with more pixels (the image upscaled) covers the same area."""
    item.setRect(QtCore.QRectF(-0.5, -0.5, shape[1], shape[0]))


class ContourDialog(QtWidgets.QDialog):
    searched = QtCore.Signal(int, bool, object)  # revision, retrieve contours, result or exception

    def __init__(self, settings: Settings, folder: str = "", use_opengl: bool = False, parent=None):
        """settings: the parameters to start from (values() keeps the others); folder: where Load image starts."""
        super().__init__(parent)
        self.setWindowTitle("Contour diagnosis")
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMaximizeButtonHint)
        self.settings, self.folder = settings, folder
        self.image = None     # the SEM image loaded (uint8)
        self.preparation = PreparedImageCache()
        self.prepared = self.previewed = self.shown = None
        self.has_result = False
        self.revision = 0  # changes to settings or to the loaded image invalidate earlier work
        self.pending_search = None  # True for auto, False for a click, None if no extraction is queued
        self.running = False
        self.name = ""        # the file name of the image loaded
        self.search_started = 0.0   # when the search running (or the last one) began (time.perf_counter)
        self.searched_with = settings  # its settings
        self.searched.connect(self._searched)
        self.timer = QtCore.QTimer(self, singleShot=True, interval=150)  # wait for a pause in editing
        self.timer.timeout.connect(self._refresh)

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
        self.original_box = QtWidgets.QCheckBox("Original image")
        self.original_box.setToolTip("Show the image as loaded, before preprocessing and equalization")
        self.original_box.toggled.connect(self.show_image)
        show.addWidget(self.original_box)
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
        self.search_button = QtWidgets.QPushButton("Get contours", autoDefault=False)
        self.search_button.clicked.connect(self.search)
        self.auto_box = QtWidgets.QCheckBox("Auto retrieve")
        self.auto_box.setToolTip("Retrieve contours whenever you load an image or change the settings")
        self.auto_box.toggled.connect(self._automatic_changed)
        self.name_label = QtWidgets.QLabel("")
        self.summary = QtWidgets.QLabel(NO_IMAGE, wordWrap=True)
        self.busy_label = QtWidgets.QLabel("")
        self.busy_label.setStyleSheet(f"color: {style.MUTED}")
        self.cursor_label = QtWidgets.QLabel("")
        self.cursor_label.setStyleSheet(f"color: {style.MUTED}")
        self.plot.scene().sigMouseMoved.connect(self._show_cursor)
        # The log (D157): every search adds the time of each of its steps, the newest last.
        self.log_box = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log_box.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.log_box.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_box.setMaximumBlockCount(5000)
        self.log_box.setPlaceholderText("Log: the time of every step of each search, the newest last")
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
        top.addWidget(self.search_button)
        top.addWidget(self.auto_box)
        for button in style.image_buttons(lambda: self.graphics, "contour diagnosis"):
            top.addWidget(button)
        status = QtWidgets.QHBoxLayout()
        status.addWidget(self.summary, stretch=1)
        status.addWidget(self.busy_label)
        left = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)  # the figure over the log; drag to resize
        left.addWidget(self.graphics)
        left.addWidget(self.log_box)
        left.setStretchFactor(0, 1)
        left.setSizes([700, 190])
        middle = QtWidgets.QHBoxLayout()
        middle.addWidget(left, stretch=1)
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
        """Invalidate the old contours and refresh the preview once editing pauses."""
        self.revision += 1
        self.pending_search = None
        self.timer.stop()
        if self.has_result:
            self.has_result = False
            self.layers["Threshold mask"].clear()
            for name in LAYERS[1:]:
                self.layers[name].setData([], [])
        s = self.values()
        for name, widget in self.editors.items():
            widget.setEnabled(applies(name, s))
        for name, box in self.boxes.items():
            box.setText(layer_name(name, s))
            no_centres = s.pattern != "contacts" and name.endswith("centres")  # only contacts have a centre
            detected = s.refinement == "none" and name.startswith("Refined")  # with none, the detected ones (D124)
            box.setEnabled(not (no_centres or detected))
        self.update_visibility()
        self.summary.setText(READY if self.image is not None else NO_IMAGE)
        try:
            check_settings(s, "images")
        except ValueError as error:
            self.summary.setText(f"<span style='color:{style.FAILURE}'>{html.escape(str(error))}</span>")
            valid = False
        else:
            valid = True
        self.apply_button.setEnabled(valid)
        self.search_button.setEnabled(valid and self.image is not None)
        if self.search_button.isEnabled():
            self.timer.start()

    def choose_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "SEM image", self.folder, IMAGE_FILES)
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        """Show the new image and discard the results of the previous one."""
        try:
            image = load_sem_image(path)
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.warning(self, "Contour diagnosis", f"The image could not be read:\n{error}")
            return
        self.image, self.prepared, self.folder, self.name = image, None, str(Path(path).parent), Path(path).name
        self.previewed = None
        self.preparation = PreparedImageCache()
        self.show_image()
        self.name_label.setText(f"{Path(path).name}   ({image.shape[1]} × {image.shape[0]} px)")
        self.plot.setRange(xRange=(-0.5, image.shape[1] - 0.5), yRange=(-0.5, image.shape[0] - 0.5), padding=0.02)
        self.changed()

    def _automatic_changed(self, enabled):
        if enabled and self.search_button.isEnabled() and not self.has_result:
            self.timer.start()
        if not enabled and self.pending_search is True:
            self.pending_search = None

    def _refresh(self):
        if self.auto_box.isChecked() and not self.has_result and self.pending_search is None:
            self.pending_search = True
        self._start_if_ready()

    def search(self):
        """Retrieve contours for the current settings, after any work already running."""
        self.timer.stop()
        self.pending_search = False
        self._start_if_ready()

    def _start_if_ready(self):
        if self.running or self.timer.isActive() or not self.search_button.isEnabled():
            return
        s = self.values()
        retrieve = self.pending_search is not None
        self.pending_search = None
        if retrieve or preview_key(s) != self.previewed:
            self._start(s, retrieve)

    def _start(self, s: Settings, retrieve: bool):
        self.running = True
        self.busy_label.setText(f"Finding the {s.pattern}..." if retrieve else "Updating the image...")
        if retrieve and not self.has_result:
            self.summary.clear()
        revision, image = self.revision, self.image
        preparation = self.preparation  # a newly loaded image gets its own, while this search finishes
        self.search_started, self.searched_with = time.perf_counter(), s

        def work():
            try:
                found = diagnose(image, s, preparation) if retrieve else preview_image(image, s, preparation)
            except Exception as error:
                found = error
            try:
                self.searched.emit(revision, retrieve, found)
            except RuntimeError:  # the window was closed meanwhile
                pass
        threading.Thread(target=work, daemon=True).start()

    def _searched(self, revision: int, retrieve: bool, found):
        self.running = False
        self.busy_label.setText("")
        if revision == self.revision:
            self._show_result(retrieve, found)
        self._start_if_ready()

    def _show_result(self, retrieve: bool, found):
        if not retrieve or not isinstance(found, Exception):
            self.previewed = preview_key(self.searched_with)
        if isinstance(found, Exception):
            task = f"Find the {self.searched_with.pattern}" if retrieve else "Update the image"
            self.summary.setText(f"<span style='color:{style.FAILURE}'>{task} failed: {html.escape(str(found))}</span>")
            return
        if not retrieve:
            self.prepared = found
            self.show_image()
            return
        back = time.perf_counter()
        self.has_result = True
        if self.pending_search is True:
            self.pending_search = None  # an in-flight retrieval just supplied the current result
        self.show_diagnosis(found)
        shown = time.perf_counter()
        self.graphics.viewport().repaint()  # drawn now, not when the window next gets to it, so that it is timed
        found.log.add("give the image, mask and contours to the plot", shown - back)
        found.log.add("draw them on screen (again at every zoom and pan)", time.perf_counter() - shown)
        self.log_box.appendPlainText(log_entry(found, self.searched_with, self.name, self.image.shape,
                                               time.perf_counter() - self.search_started))

    def show_image(self, *_):
        """Show the latest preview or the original image, both in the original pixel frame."""
        if self.image is None:
            return
        self.shown = self.image if self.original_box.isChecked() or self.prepared is None else self.prepared
        self.image_item.setImage(self.shown, levels=(0, 255))
        place(self.image_item, self.image.shape)

    def show_diagnosis(self, d: Diagnosis):
        self.prepared = d.image
        self.show_image()
        self.layers["Threshold mask"].setImage(d.mask, levels=(0, 1))
        place(self.layers["Threshold mask"], self.image.shape)
        for name, found in (("Detected", d.detected), ("Refined", d.refined)):
            if isinstance(found, DetectedContacts):
                xy, centres = drawable_outlines(found.contours), found.centers
            else:
                xy = drawable_edges(found) if isinstance(found, LineEdges) else drawable_contours(found)
                centres = np.empty((0, 2))
            self.layers[f"{name} contours"].setData(xy[:, 0], xy[:, 1], connect="finite")
            self.layers[f"{name} centres"].setData(centres[:, 0], centres[:, 1])
        self.summary.setText(summary(d, self.values()))
        self.update_visibility()

    def update_visibility(self, *_):
        """Show the layers ticked (and offered); the legend lists the contours and centres shown, each with the method
        or rule it comes from."""
        s = self.values()
        for name, item in self.layers.items():
            item.setVisible(self.has_result and self.boxes[name].isChecked() and self.boxes[name].isEnabled())
        self.plot.legend.clear()
        for name in LAYERS[1:]:
            if self.layers[name].isVisible():
                self.plot.legend.addItem(self.layers[name], f"{layer_name(name, s)} "
                                         f"({s.detection if name.startswith('Detected') else s.refinement})")

    def _show_cursor(self, scene_pos):
        """The cursor's position and the grey level of the pixel under it in the image shown (to compare with the
        threshold; upscaled: of its own, smaller pixel)."""
        if self.shown is None:
            return
        pos = self.plot.vb.mapSceneToView(scene_pos)
        factor = self.shown.shape[0] // self.image.shape[0]
        column, row = int(np.floor((pos.x() + 0.5) * factor)), int(np.floor((pos.y() + 0.5) * factor))
        rows, cols = self.shown.shape
        grey = f", grey level {self.shown[row, column]}" if 0 <= row < rows and 0 <= column < cols else ""
        self.cursor_label.setText(f"x = {pos.x():.1f} px, y = {pos.y():.1f} px{grey}")
