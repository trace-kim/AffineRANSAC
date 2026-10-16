"""MaskReg: the minimal mask registration analyzer released to users (docs/SPEC.md D77).
Start it with:  python -m affine_ransac.release.app

- File > Browse FTP server: connect to an FTP server (pre-set in maskreg.ini, config.py), walk through its folders
  and download one to analyse it: kept at <download folder>/<host>/<remote path>; only new or changed files are
  fetched, then the folder is opened as above (analysed again if anything new arrived).
- File > Open folder: a folder of contour CSVs or of SEM images. If it holds a saved analysis (cache.py), that is
  shown at once; otherwise the folder is analysed (release/analysis.py) with the current parameters and the
  result saved there.
- Analysis > Restart analysis: analyse the open folder again (current parameters) and save over the old result.
- Analysis > Parameters: the parameters (hidden from the main window); after a change, offers a restart.
- View: the log, and GPU drawing (OpenGL; checked at start whether it works here, else drawn on the CPU).
- Status bar: the step being done, a progress bar with its percentage, and the time so far.
"""

import configparser
import json
import threading
import time
import traceback
import warnings
from dataclasses import asdict, fields, replace
from pathlib import Path

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from affine_ransac.analysis import Settings
from affine_ransac.app import SettingsDialog, folder_kind, image_records
from affine_ransac.release import ftp as remote
from affine_ransac.release import style
from affine_ransac.release.analysis import DEFAULTS, PARAMETERS, analyse_folder, read_display_images
from affine_ransac.release.cache import load_cache, save_cache
from affine_ransac.release.config import (HIDDEN, config_path, default_download_root, read_ftp_config,
                                          read_hidden_settings)
from affine_ransac.release.ftp_dialog import FtpDialog
from affine_ransac.release.views import RegistrationView, StitchingView

APP = "MaskReg"
TITLE = "MaskReg - Mask Registration Analyzer"


def opengl_works() -> bool:
    """Whether an OpenGL context can be made here with a real (not the Windows 1.1 fallback) renderer. Without
    one, OpenGL plots stay black, so the figures are drawn on the CPU instead."""
    try:
        context, surface = QtGui.QOpenGLContext(), QtGui.QOffscreenSurface()
        surface.create()
        if not context.create() or not context.makeCurrent(surface):
            return False
        renderer = str(context.functions().glGetString(0x1F01) or "")  # GL_RENDERER
        context.doneCurrent()
        return bool(renderer) and "GDI Generic" not in renderer
    except Exception:
        return False


class Job(QtCore.QObject):
    """Runs work(log, progress) in a background thread. Log lines (and the library's warnings) arrive as
    `message`, progress(fraction, stage) as `progress`, the return value as `done`, an error with its traceback
    as `failed`, all in the GUI thread."""
    message = QtCore.Signal(str)
    progress = QtCore.Signal(float, str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def start(self, work):
        threading.Thread(target=self._run, args=(work,), daemon=True).start()

    def _run(self, work):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.showwarning = lambda message, *_: self.message.emit(f"Warning: {message}")
                result = work(self.message.emit, self.progress.emit)
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.done.emit(result)


class StatusBar(QtWidgets.QStatusBar):
    """What is being done (step, progress bar with percentage, time so far), or when idle what is shown."""

    def __init__(self):
        super().__init__()
        self.stage = QtWidgets.QLabel()
        self.bar = QtWidgets.QProgressBar(minimumWidth=260, maximumWidth=360, textVisible=True)
        self.bar.setRange(0, 1000)
        self.bar.setFormat("%p%")
        self.clock = QtWidgets.QLabel()
        self.idle = QtWidgets.QLabel()
        self.addWidget(self.idle, 1)
        self.addWidget(self.stage, 1)
        self.addPermanentWidget(self.bar)
        self.addPermanentWidget(self.clock)
        self.started = 0.0
        self.timer = QtCore.QTimer(self, interval=500)
        self.timer.timeout.connect(self._tick)
        self.set_idle("Open a data folder: File > Open folder (Ctrl+O)")

    def begin(self, title: str):
        self.started = time.time()
        self.title = title
        self.idle.hide()
        for widget in (self.stage, self.bar, self.clock):
            widget.show()
        self.update_progress(0.0, "Starting")
        self.timer.start()

    def update_progress(self, fraction: float, stage: str):
        self.bar.setValue(round(1000 * min(max(fraction, 0.0), 1.0)))
        self.stage.setText(f"{self.title}  -  {stage}")
        self._tick()

    def _tick(self):
        seconds = int(time.time() - self.started)
        self.clock.setText(f"  {seconds // 60}:{seconds % 60:02d}  ")

    def set_idle(self, text: str):
        self.timer.stop()
        for widget in (self.stage, self.bar, self.clock):
            widget.hide()
        self.idle.setText(text)
        self.idle.show()


class MaskRegWindow(QtWidgets.QMainWindow):
    def __init__(self, preferences: QtCore.QSettings | None = None, cache_dir: str | None = None):
        """preferences: where the parameters, the last folder and the drawing choice are kept (default: the
        user's MaskReg settings); cache_dir: the user's cache folder (used when a data folder is read-only)."""
        super().__init__()
        self.preferences = preferences or QtCore.QSettings(APP, APP)
        self.cache_dir = cache_dir or QtCore.QStandardPaths.writableLocation(
            QtCore.QStandardPaths.StandardLocation.CacheLocation)
        self.folder = None   # the open folder
        self.result = None   # its ReleaseResult
        self.images = None   # its SEM images, once loaded (kept when the tabs are rebuilt)
        self.busy = False
        self.setWindowTitle(TITLE)
        self.welcome()

        self.log_view = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log_dock = QtWidgets.QDockWidget("Log")
        self.log_dock.setObjectName("log")
        self.log_dock.setWidget(self.log_view)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()
        self.status = StatusBar()
        self.setStatusBar(self.status)

        self.actions = {}
        file_menu = self.menuBar().addMenu("&File")
        self._action(file_menu, "open", "&Open folder...", "Ctrl+O", self.choose_folder)
        self._action(file_menu, "ftp", "Browse &FTP server...", "Ctrl+Shift+O", self.browse_ftp)
        file_menu.addSeparator()
        file_menu.addAction("&Quit", self.close)
        analysis_menu = self.menuBar().addMenu("&Analysis")
        self._action(analysis_menu, "restart", "&Restart analysis", "F5", self.restart)
        analysis_menu.addSeparator()
        self._action(analysis_menu, "parameters", "&Parameters...", "Ctrl+P", self.edit_parameters)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.log_dock.toggleViewAction())
        self.gpu_action = view_menu.addAction("Draw with the &GPU (OpenGL)")
        self.gpu_action.setCheckable(True)
        saved = self.preferences.value("use_opengl", "")
        self.gpu_action.setChecked(saved == "true" if saved else opengl_works())
        self.gpu_action.toggled.connect(self.set_gpu)
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("&About MaskReg", self.about)

        self.job = Job()
        self.job.message.connect(self.log)
        self.job.progress.connect(self.status.update_progress)
        self.job.done.connect(self._job_done)
        self.job.failed.connect(self._job_failed)
        self.update_actions()
        self.resize(1600, 1050)

    def _action(self, menu, name, text, shortcut, slot):
        action = menu.addAction(text)
        action.setShortcut(QtGui.QKeySequence(shortcut))
        action.triggered.connect(slot)
        self.actions[name] = action

    def welcome(self):
        label = QtWidgets.QLabel("<h2>MaskReg</h2><p>Open a data folder with <b>File &gt; Open folder</b> (Ctrl+O).</p>"
                                 "<p>A folder analysed before opens its saved result at once.</p>",
                                 alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"color: {style.MUTED}")
        self.setCentralWidget(label)

    # --- menu actions ---------------------------------------------------------------------------
    def choose_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Data folder", self.preferences.value("last_folder", ""))
        if folder:
            self.preferences.setValue("last_folder", folder)
            self.open_folder(folder)

    def browse_ftp(self):
        path = config_path()
        try:
            config = read_ftp_config(path)
        except configparser.Error as error:
            QtWidgets.QMessageBox.warning(self, TITLE, f"{path} could not be read, so it is ignored:\n{error}")
            config = {}
        root = config.get("download_folder") or default_download_root()
        dialog = FtpDialog(self.preferences, config, root, self)
        if dialog.exec() and dialog.chosen:
            self.download_and_open(dialog.login(), dialog.chosen, root)

    def download_and_open(self, login, remote_path: str, root: str):
        """Download the remote folder (new or changed files only), then open it; analyse it again if anything new
        arrived, since a saved analysis would not hold it."""
        host, port, user, password = login
        folder = remote.local_folder(root, host, remote_path)
        self.log(f"=== Downloading ftp://{host}{remote_path} to {folder}")

        def work(log, progress):
            ftp = remote.connect(host, port, user, password)
            try:
                return remote.download_folder(ftp, remote_path, folder, log, progress)
            finally:
                ftp.close()

        def downloaded(count):
            self.preferences.setValue("last_folder", str(folder))
            if count:
                self.folder = str(folder)
                self.restart()
            else:
                self.open_folder(str(folder))
        self.run(f"Downloading {remote_path}", work, downloaded)

    def open_folder(self, folder: str):
        """Show the folder's saved analysis, or analyse it (and save) if there is none."""
        self.log(f"=== Opening {folder}")
        settings, cache_dir = self.settings(), self.cache_dir

        def work(log, progress):
            progress(0.0, "Looking for a saved analysis")
            cached = load_cache(folder, cache_dir)
            if cached is not None:
                result, saved = cached
                log(f"Saved analysis of {saved} opened (Analysis > Restart analysis runs it again)")
                return result
            log("No saved analysis: analysing the folder")
            return self._analyse_and_save(folder, settings, log, progress)
        self.folder = folder
        self.run(f"Opening {Path(folder).name}", work, self.show_new_result)

    def restart(self):
        folder, settings = self.folder, self.settings()
        self.log(f"=== Restarting the analysis of {folder}")
        self.run(f"Analysing {Path(folder).name}",
                 lambda log, progress: self._analyse_and_save(folder, settings, log, progress), self.show_new_result)

    def _analyse_and_save(self, folder, settings, log, progress):
        """Analyse the folder (in the background thread) and save the result; a failed save is only logged."""
        kind = folder_kind(Path(folder))
        log(f"Analysing a {kind} folder")
        records = image_records(folder) if kind == "images" else None
        result = analyse_folder(folder, kind, settings, records, log, progress)
        progress(1.0, "Saving the result")
        try:
            log(f"Result saved: {save_cache(result, self.cache_dir)}")
        except OSError as error:
            log(f"!! {error}")
        return result

    def edit_parameters(self):
        kind = self.result.kind if self.result is not None else None
        names = PARAMETERS[kind] if kind else list(dict.fromkeys(PARAMETERS["images"] + PARAMETERS["contour CSV"]))
        dialog = SettingsDialog(self.settings(), kind or "any", self, names)
        dialog.setWindowTitle("Parameters")
        if not dialog.exec():
            return
        settings = dialog.values()
        self.preferences.setValue("settings", json.dumps(asdict(settings)))
        if self.result is not None and settings != self.result.settings and QtWidgets.QMessageBox.question(
                self, TITLE, "The parameters differ from those of the analysis shown. Restart the analysis now?") == \
                QtWidgets.QMessageBox.StandardButton.Yes:
            self.restart()

    def load_images(self):
        paths = self.result.image_paths
        self.log(f"=== Loading {len(paths)} SEM images")
        self.run("Loading SEM images", lambda log, progress: read_display_images(paths, progress), self.show_images)

    def set_gpu(self, on: bool):
        self.preferences.setValue("use_opengl", "true" if on else "false")
        if self.result is not None:
            self.show_result(self.result)  # draw again with the new choice

    def about(self):
        QtWidgets.QMessageBox.about(self, "About MaskReg", "<b>MaskReg</b> - mask registration analyzer<br><br>"
                                    "Stitches the SEM tiles of a folder, compares every contact with the design and "
                                    "reports the registration error, measured − design, after a moving-window "
                                    "affine along y.")

    # --- showing a result -------------------------------------------------------------------------
    def show_new_result(self, result):
        self.images = None  # those of another result
        self.show_result(result)

    def show_images(self, images):
        self.images = images
        self.stitching_view.set_images(images)
        self.status.set_idle(f"{len(images)} SEM images loaded (Stitching tab, SEM images)")

    def show_result(self, result):
        self.status.set_idle("Drawing...")
        QtWidgets.QApplication.processEvents()
        use_opengl = self.gpu_action.isChecked()
        tabs = QtWidgets.QTabWidget()
        self.stitching_view = StitchingView(result, use_opengl, self.load_images)
        if self.images is not None:
            self.stitching_view.set_images(self.images)
        self.registration_view = RegistrationView(result, use_opengl)
        tabs.addTab(self.stitching_view, "Stitching")
        tabs.addTab(self.registration_view, "Registration")
        tabs.setCurrentIndex(1)
        self.setCentralWidget(tabs)
        self.result = result
        self.setWindowTitle(f"{TITLE} - {Path(result.folder).name}")
        self.status.set_idle(f"{Path(result.folder).name}: {len(result.tile_ids)} tiles, {len(result.design_nm)} "
                             f"contacts ({result.kind} folder)  |  drawing on the {'GPU' if use_opengl else 'CPU'}")
        self.update_actions()

    # --- background jobs --------------------------------------------------------------------------
    def run(self, title: str, work, on_done):
        """work(log, progress) in the background; on_done(its return value) afterwards, in the GUI thread."""
        self.busy, self.on_done = True, on_done
        self.status.begin(title)
        self.update_actions()
        self.job.start(work)

    def _job_done(self, value):
        self.busy = False
        self.on_done(value)
        self.update_actions()

    def _job_failed(self, text: str):
        self.busy = False
        self.log(text)
        self.status.set_idle("Failed - see View > Log")
        self.update_actions()
        QtWidgets.QMessageBox.critical(self, TITLE, text.strip().splitlines()[-1])

    def update_actions(self):
        self.actions["open"].setEnabled(not self.busy)
        self.actions["ftp"].setEnabled(not self.busy)
        self.actions["restart"].setEnabled(not self.busy and self.folder is not None)
        self.actions["parameters"].setEnabled(not self.busy)
        if self.result is not None:
            self.stitching_view.load_button.setEnabled(
                not self.busy and self.images is None and bool(self.result.image_paths))

    def log(self, text: str):
        self.log_view.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")

    def settings(self) -> Settings:
        """The parameters chosen last (the release's defaults for any not saved); the hidden ones (config.HIDDEN)
        only from maskreg.ini, never from what was saved, so removing them from the file turns them off."""
        try:
            saved = json.loads(self.preferences.value("settings", "") or "{}")
        except ValueError:
            saved = {}
        known = {f.name for f in fields(Settings)} - set(HIDDEN)
        try:
            hidden = read_hidden_settings(config_path())
        except (ValueError, configparser.Error) as error:
            self.log(f"!! maskreg.ini [analysis] ignored: {error}")
            hidden = {}
        for name, value in hidden.items():
            self.log(f"Hidden parameter from {config_path()}: {name} = {value:g}")
        return replace(DEFAULTS, **{name: value for name, value in saved.items() if name in known}, **hidden)

    def closeEvent(self, event):
        if self.busy and QtWidgets.QMessageBox.question(self, TITLE, "Still working. Quit anyway?") != \
                QtWidgets.QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        event.accept()


def main():
    style.setup()
    app = pg.mkQApp(APP)
    window = MaskRegWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
