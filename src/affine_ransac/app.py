"""Standalone analyzer: analyse a data folder, or open saved results, in the analysis window
(docs/SPEC.md D64). Start it with:  python -m affine_ransac.app

File menu:
- Open folder: a folder of contour CSVs, or of SEM images (+ ContourCAD/*.oas + metadata CSV; the
  metadata reader io/metadata.py exists only on the data machine). A dialog shows the settings of that
  kind of folder (the last used ones are remembered); the analysis (analysis.py) runs in the background
  with its log below, then the analysis window (view_analysis.analysis_window) fills the main window.
- Open results / Save results: one .npz per analysis (results_file).
- Load external measurement: a txt reference result (X, Y µm, dX, dY nm; io.external), shown in every
  registration view; the sign of dX, dY is chosen per file.
View menu: the log, and OpenGL drawing (switch it off if the plots stay black; used for the next window).
"""

import json
import threading
import time
import traceback
import warnings
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from affine_ransac.analysis import CHOICES, Settings, analyse_contour_folder, analyse_images, settings_for
from affine_ransac.io.external import read_external
from affine_ransac.results_file import load_result, save_result
from affine_ransac.view_analysis import analysis_window

SIGNS = ("as in the file (measured − design)", "flipped (the file has design − measured)")


def folder_kind(folder: Path) -> str:
    """"images" if the folder has SEM images or a ContourCAD folder, else "contour CSV"."""
    return "images" if (folder / "ContourCAD").is_dir() or any(folder.glob("*.jpg")) else "contour CSV"


def image_records(folder):
    """The tiles of an image folder, from the metadata reader (task T001, only on the data machine)."""
    try:
        from affine_ransac.io.metadata import read_tile_index
    except ImportError as error:
        raise RuntimeError("Image folders need affine_ransac.io.metadata (task T001), which exists only on the "
                           "machine with the data") from error
    return read_tile_index(folder)


class Job(QtCore.QObject):
    """Runs work(log) in a background thread: log lines arrive as `message`, the return value as `done`,
    an error with its traceback as `failed`, all in the GUI thread. The library's warnings (e.g. a
    stitching failure) are logged too. A daemon thread: quitting during a run just abandons it."""
    message = QtCore.Signal(str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def start(self, work):
        threading.Thread(target=self._run, args=(work,), daemon=True).start()

    def _run(self, work):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.showwarning = lambda message, *_: self.message.emit(f"Warning: {message}")
                result = work(self.message.emit)
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.done.emit(result)


def editor(value, choices=None) -> QtWidgets.QWidget:
    """An input widget for one setting value."""
    if choices:
        widget = QtWidgets.QComboBox()
        widget.addItems(list(choices))
        widget.setCurrentText(value)
    elif isinstance(value, bool):
        widget = QtWidgets.QCheckBox()
        widget.setChecked(value)
    elif isinstance(value, int):
        widget = QtWidgets.QSpinBox()
        widget.setRange(-10 ** 9, 10 ** 9)
        widget.setValue(value)
    else:
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(-1e9, 1e9)
        widget.setDecimals(3)
        widget.setValue(value)
    return widget


def editor_value(widget: QtWidgets.QWidget):
    if isinstance(widget, QtWidgets.QComboBox):
        return widget.currentText()
    if isinstance(widget, QtWidgets.QCheckBox):
        return widget.isChecked()
    return widget.value()


class SettingsDialog(QtWidgets.QDialog):
    """The settings a folder of this kind uses: name (as in the notebooks), value, what it does."""

    def __init__(self, settings: Settings, kind: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Settings for a {kind} folder")
        self.settings, self.editors = settings, {}
        grid = QtWidgets.QGridLayout()
        for row, f in enumerate(f for f in fields(Settings) if f.name in settings_for(kind)):
            widget = editor(getattr(settings, f.name), CHOICES.get(f.name))
            if f.name == "ransac_threshold_nm":
                widget.setEnabled(False)  # fixed: the global affine is RANSAC with τ = 0.5 nm (user rule)
            help_label = QtWidgets.QLabel(f.metadata["help"], wordWrap=True)
            help_label.setStyleSheet("color: gray")
            for col, item in enumerate((QtWidgets.QLabel(f.name.upper()), widget, help_label)):
                grid.addWidget(item, row, col)
            self.editors[f.name] = widget
        grid.setColumnStretch(2, 1)  # the help texts take the free width
        inner = QtWidgets.QWidget()
        inner.setLayout(grid)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(scroll)
        layout.addWidget(buttons)
        self.resize(1000, 700)

    def values(self) -> Settings:
        """The settings with the values in the dialog (the other kind's settings unchanged)."""
        return replace(self.settings, **{name: editor_value(widget) for name, widget in self.editors.items()})


class AnalyzerWindow(QtWidgets.QMainWindow):
    """The main window: File and View menus, the analysis window in the middle, the log below."""

    def __init__(self, preferences: QtCore.QSettings | None = None):
        """preferences: where the last settings and folder are remembered (default: the user's
        AffineRANSAC/Analyzer settings)."""
        super().__init__()
        self.preferences = preferences or QtCore.QSettings("AffineRANSAC", "Analyzer")
        self.result = None    # the AnalysisResult shown
        self.analysis = None  # its AnalysisWindow (not self.window: that is QWidget.window())
        self.busy = False
        self.setWindowTitle("AffineRANSAC analyzer")
        self.setCentralWidget(QtWidgets.QLabel("Open a data folder or saved results (File menu).",
                                               alignment=QtCore.Qt.AlignmentFlag.AlignCenter))

        self.log_view = QtWidgets.QPlainTextEdit(readOnly=True)
        dock = QtWidgets.QDockWidget("Log")
        dock.setObjectName("log")
        dock.setWidget(self.log_view)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.progress = QtWidgets.QProgressBar(maximumWidth=200, visible=False)
        self.progress.setRange(0, 0)  # busy, no percentage
        self.statusBar().addPermanentWidget(self.progress)

        file_menu = self.menuBar().addMenu("&File")
        self.actions = {}
        for name, text, shortcut, slot in (("open_folder", "Open &folder...", "Ctrl+O", self.choose_folder),
                                           ("open_results", "&Open results...", "Ctrl+Shift+O", self.choose_results),
                                           ("save_results", "&Save results...", "Ctrl+S", self.choose_save),
                                           ("external", "Load &external measurement...", "Ctrl+E", self.choose_external)):
            action = file_menu.addAction(text)
            action.setShortcut(QtGui.QKeySequence(shortcut))
            action.triggered.connect(slot)
            self.actions[name] = action
        file_menu.addSeparator()
        file_menu.addAction("&Quit", self.close)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(dock.toggleViewAction())
        self.opengl_action = view_menu.addAction("Draw with OpenGL (off if the plots stay black)")
        self.opengl_action.setCheckable(True)
        self.opengl_action.setChecked(self.preferences.value("use_opengl", "true") == "true")
        self.opengl_action.toggled.connect(lambda on: self.preferences.setValue("use_opengl", "true" if on else "false"))

        self.job = Job()
        self.job.message.connect(self.log)
        self.job.done.connect(self._job_done)
        self.job.failed.connect(self._job_failed)
        self.update_actions()
        self.resize(1700, 1150)

    # --- menu actions: the dialogs, then the work ----------------------------------------------
    def choose_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Data folder", self.last_folder())
        if not folder:
            return
        self.preferences.setValue("last_folder", folder)
        kind = folder_kind(Path(folder))
        dialog = SettingsDialog(self.saved_settings(), kind, self)
        if dialog.exec():
            settings = dialog.values()
            self.preferences.setValue("settings", json.dumps(asdict(settings)))
            self.analyse(folder, kind, settings)

    def choose_results(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open results", self.last_folder(), "Results (*.npz)")
        if path:
            self.open_results(path)

    def choose_save(self):
        suggestion = str(Path(self.last_folder()) / f"{Path(self.result.folder).name}_results.npz")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save results", suggestion, "Results (*.npz)")
        if path:
            self.save_results(path)

    def choose_external(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "External measurement: X, Y (µm), dX, dY (nm)",
                                                        self.last_folder(), "Text files (*.txt);;All files (*)")
        if not path:
            return
        sign, ok = QtWidgets.QInputDialog.getItem(self, "Sign of dX, dY", f"{Path(path).name}: dX, dY are",
                                                  list(SIGNS), 0, False)
        if ok:
            self.add_external(path, flip_sign=sign == SIGNS[1])

    # --- the work ------------------------------------------------------------------------------
    def analyse(self, folder: str, kind: str, settings: Settings):
        """Analyse the folder in the background, then show the result."""
        self.log(f"=== Analysing {folder} ({kind} folder)")
        if kind == "images":
            def work(log):
                return analyse_images(folder, image_records(folder), settings, log)
        else:
            def work(log):
                return analyse_contour_folder(folder, settings, log)
        self.run(f"Analysing {Path(folder).name}...", work, self.show_result)

    def open_results(self, path: str):
        self.log(f"=== Opening {path}")

        def work(log):
            result = load_result(path)
            log(f"Results of {result.folder} ({result.kind} folder). Log of that run:")
            for line in result.log:
                log("    " + line)
            return result
        self.run("Opening results...", work, self.show_result)

    def save_results(self, path: str):
        result = self.result

        def work(log):
            save_result(result, path)
            log(f"Saved {path}" + (" (without the images: reopened, the Stitching tab shows the points)"
                                   if result.images is not None else ""))
        self.run("Saving results...", work, lambda _: None)

    def add_external(self, path: str, flip_sign: bool = False):
        """Show an external measurement in every registration view (io.external.read_external)."""
        try:
            xy_nm, error_nm = read_external(path, flip_sign)
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "External measurement", str(error))
            return
        name = Path(path).stem + (" (sign flipped)" if flip_sign else "")
        self.analysis.add_external(name, xy_nm, error_nm)
        design = self.result.sets["uncorrected"].design_nm  # a check of the units and coordinates:
        inside = np.all((xy_nm > design.min(axis=0) - 1000) & (xy_nm < design.max(axis=0) + 1000), axis=1)  # ±1 µm
        self.log(f"External measurement '{name}': {len(xy_nm)} sites, {inside.sum()} inside the analysed field"
                 + ("" if inside.any() else " - check its units (X, Y in µm) and coordinates"))

    def show_result(self, result):
        """Build the analysis window of a result and show it in the main window."""
        self.statusBar().showMessage("Drawing...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            window = analysis_window(result, use_opengl=self.opengl_action.isChecked())
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.result, self.analysis = result, window
        self.setCentralWidget(window)  # deletes the previous window
        self.setWindowTitle(f"AffineRANSAC analyzer - {Path(result.folder).name} ({result.kind} folder)")
        self.statusBar().showMessage(f"{len(result.tile_ids)} tiles, {len(result.sets['uncorrected'].design_nm)} "
                                     f"contacts", 10000)
        self.update_actions()

    # --- background jobs and state ----------------------------------------------------------------
    def run(self, title: str, work, on_done):
        """work(log) in the background; on_done(its return value) afterwards, in the GUI thread."""
        self.busy, self.on_done = True, on_done
        self.statusBar().showMessage(title)
        self.progress.setVisible(True)
        self.update_actions()
        self.job.start(work)

    def _job_done(self, value):
        self._finish()
        self.on_done(value)

    def _job_failed(self, text: str):
        self._finish()
        self.log(text)
        QtWidgets.QMessageBox.critical(self, "AffineRANSAC analyzer", text.strip().splitlines()[-1])

    def _finish(self):
        self.busy = False
        self.progress.setVisible(False)
        self.statusBar().clearMessage()
        self.update_actions()

    def update_actions(self):
        for name, action in self.actions.items():
            needs_result = name in ("save_results", "external")
            action.setEnabled(not self.busy and (self.result is not None or not needs_result))

    def log(self, text: str):
        self.log_view.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")

    def last_folder(self) -> str:
        return self.preferences.value("last_folder", "")

    def saved_settings(self) -> Settings:
        """The settings used last (defaults for any not saved, e.g. added since)."""
        try:
            saved = json.loads(self.preferences.value("settings", "") or "{}")
        except ValueError:
            saved = {}
        known = {f.name for f in fields(Settings)}
        return Settings(**{name: value for name, value in saved.items() if name in known})

    def closeEvent(self, event):
        if self.busy and QtWidgets.QMessageBox.question(self, "AffineRANSAC analyzer",
                                                        "Still working. Quit anyway?") != \
                QtWidgets.QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        event.accept()


def main():
    app = pg.mkQApp("AffineRANSAC analyzer")
    window = AnalyzerWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
