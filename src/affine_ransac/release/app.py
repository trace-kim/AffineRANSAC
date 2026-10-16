"""MaskReg: the minimal mask registration analyzer released to users (docs/SPEC.md D77).
Start it with:  python -m affine_ransac.release.app

- File > Browse FTP server: connect to an FTP server (pre-set in maskreg.ini, config.py), walk through its folders
  and download one to analyse it: kept at <download folder>/<host>/<remote path>; only new or changed files are
  fetched, then the folder is opened as above (analysed again if anything new arrived).
- File > Folders: the folder of maskreg.ini and the download folder (folders_dialog.py).
- File > Open folder: a folder of contour CSVs or of SEM images. If it holds a saved analysis (cache.py), that is
  shown at once; otherwise the folder is analysed (release/analysis.py) with the current parameters and the
  result saved there. File > Open recent: the last folders whose result was shown, opened the same way.
- Analysis > Restart analysis: analyse the open folder again (current parameters) and save the new result; the old
  one is kept as a backup (the last 5, cache.py).
- Analysis > Parameters: the parameters (hidden from the main window); after a change, offers a restart.
- View: the log, GPU drawing (OpenGL; checked at start whether it works here, else drawn on the CPU), and the
  Contact size tab (hidden unless turned on; remembered): area, xCD and yCD of every contact, measured on request
  on the tiles of the run, which an image run keeps in memory only while the tab is on.
- Status bar: the step being done, a progress bar with its percentage, and the time so far.
"""

import configparser
import json
import os
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
from affine_ransac.release.analysis import (DEFAULTS, PARAMETERS, analyse_folder, measure_contact_sizes,
                                           read_design_images, read_display_images)
from affine_ransac.release.cache import load_cache, save_cache
from affine_ransac.release.config import config_path, download_root, read_ftp_config
from affine_ransac.release.folders_dialog import FoldersDialog
from affine_ransac.release.ftp_dialog import FtpDialog
from affine_ransac.release.views import ContactSizeTab, RegistrationView, StitchingView

APP = "MaskReg"
TITLE = "MaskReg - Mask Registration Analyzer"
RECENT = 10  # folders in File > Open recent


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
        self.design = None   # its design pattern masks, once loaded (the same)
        self.size_tab = None  # the Contact size tab, when shown
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
        self.recent_menu = file_menu.addMenu("Open &recent")
        self.recent_menu.aboutToShow.connect(self.fill_recent_menu)
        self._action(file_menu, "ftp", "Browse &FTP server...", "Ctrl+Shift+O", self.browse_ftp)
        file_menu.addSeparator()
        file_menu.addAction("Fol&ders...", lambda: FoldersDialog(self.preferences, self).exec())
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
        self.size_action = view_menu.addAction("&Contact size tab")
        self.size_action.setCheckable(True)
        self.size_action.setChecked(self.preferences.value("contact_size_tab", "") == "true")  # hidden by default
        self.size_action.toggled.connect(self.set_size_tab)
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
        label = QtWidgets.QLabel("<h2>MaskReg</h2><p>Open a data folder with <b>File &gt; Open folder</b> (Ctrl+O), "
                                 "or one opened before with <b>File &gt; Open recent</b>.</p>"
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

    def recent_folders(self) -> list[str]:
        try:
            return json.loads(self.preferences.value("recent_folders", "") or "[]")
        except ValueError:
            return []

    def remember_folder(self, folder: str):
        """Put the folder at the top of File > Open recent (once; at most RECENT folders)."""
        folder = str(Path(folder))
        same = lambda other: os.path.normcase(other) == os.path.normcase(folder)
        recent = [folder] + [f for f in self.recent_folders() if not same(f)]
        self.preferences.setValue("recent_folders", json.dumps(recent[:RECENT]))

    def fill_recent_menu(self):
        """File > Open recent: the folders shown last, newest first; a folder gone from the disk is greyed."""
        self.recent_menu.clear()
        recent = self.recent_folders()
        for folder in recent:
            found = Path(folder).is_dir()
            action = self.recent_menu.addAction(f"{Path(folder).name}   ({folder}){'' if found else '  - not found'}")
            action.setEnabled(found)
            action.triggered.connect(lambda checked=False, f=folder: self.open_recent(f))
        if not recent:
            self.recent_menu.addAction("No folders opened yet").setEnabled(False)
        self.recent_menu.addSeparator()
        self.recent_menu.addAction("Clear list", lambda: self.preferences.setValue("recent_folders", "[]"))

    def open_recent(self, folder: str):
        self.preferences.setValue("last_folder", folder)
        self.open_folder(folder)

    def browse_ftp(self):
        path = config_path(self.preferences.value("config_folder", ""))
        try:
            config = read_ftp_config(path)
        except configparser.Error as error:
            QtWidgets.QMessageBox.warning(self, TITLE, f"{path} could not be read, so it is ignored:\n{error}")
            config = {}
        root = download_root(self.preferences.value("download_folder", ""), config)
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
        settings, cache_dir, keep_tiles = self.settings(), self.cache_dir, self.size_action.isChecked()
        self._release_tiles()

        def work(log, progress):
            progress(0.0, "Looking for a saved analysis")
            cached = load_cache(folder, cache_dir)
            if cached is not None:
                result, saved = cached
                log(f"Saved analysis of {saved} opened (Analysis > Restart analysis runs it again)")
                return result
            log("No saved analysis: analysing the folder")
            return self._analyse_and_save(folder, settings, keep_tiles, log, progress)
        self.folder = folder
        self.run(f"Opening {Path(folder).name}", work, self.show_new_result)

    def restart(self):
        folder, settings, keep_tiles = self.folder, self.settings(), self.size_action.isChecked()
        self.log(f"=== Restarting the analysis of {folder}")
        self._release_tiles()
        self.run(f"Analysing {Path(folder).name}",
                 lambda log, progress: self._analyse_and_save(folder, settings, keep_tiles, log, progress),
                 self.show_new_result)

    def _release_tiles(self):
        """Let go of the tiles the shown result keeps before another run loads its own: both at once could exhaust
        the memory."""
        if self.result is not None:
            self.result.tiles = None

    def _analyse_and_save(self, folder, settings, keep_tiles, log, progress):
        """Analyse the folder (in the background thread) and save the result; a failed save is only logged.
        keep_tiles: an image folder's result keeps its tiles, to measure the contact sizes on."""
        kind = folder_kind(Path(folder))
        log(f"Analysing a {kind} folder")
        records = image_records(folder) if kind == "images" else None
        result = analyse_folder(folder, kind, settings, records, log, progress, keep_tiles)
        progress(1.0, "Saving the result")
        try:
            log(f"Result saved: {save_cache(result, self.cache_dir, log=log)}")
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

    def load_design(self):
        result = self.result
        self.log(f"=== Loading the design of {len(result.oas_paths)} tiles")
        self.run("Loading the design", lambda log, progress: read_design_images(result, progress), self.show_design)

    def show_design(self, masks):
        self.design = masks
        self.stitching_view.set_design(masks)
        self.status.set_idle(f"Design of {sum(m is not None for m in masks)} tiles loaded (Stitching tab, "
                             f"Design pattern)")

    def measure_sizes(self, roi_width_nm: float):
        """Measure the contact sizes on the run's tiles in the background, then show them and save them with the
        result (a reopened folder shows them)."""
        result, cache_dir = self.result, self.cache_dir
        self.log(f"=== Measuring the contact sizes (ROI width {roi_width_nm:g} nm)")

        def work(log, progress):
            sizes = measure_contact_sizes(result, roi_width_nm,
                                          lambda done, total: progress(done / total, f"Measuring tile {done} of {total}"))
            log(f"Contact sizes of {len(sizes.sem)} contacts measured")
            progress(1.0, "Saving the result")
            try:
                saved = save_cache(replace(result, contact_sizes=sizes), cache_dir, backup=False)  # the same run
                log(f"Result saved with them: {saved}")
            except OSError as error:
                log(f"!! {error}")
            return sizes
        self.run("Measuring contact sizes", work, self.show_sizes)

    def show_sizes(self, sizes):
        self.result.contact_sizes = sizes
        self.size_tab.show_sizes(sizes)
        self.status.set_idle(f"Contact sizes of {len(sizes.sem)} contacts measured (Contact size tab)")

    def set_size_tab(self, on: bool):
        """Show or hide the Contact size tab (remembered). Hidden, the shown result's tiles are let go (memory)."""
        self.preferences.setValue("contact_size_tab", "true" if on else "false")
        if self.result is None:
            return
        if on:
            self._add_size_tab(self.result, self.gpu_action.isChecked())
            self.tabs.setCurrentWidget(self.size_tab)
        else:
            self.tabs.removeTab(self.tabs.indexOf(self.size_tab))
            self.size_tab.deleteLater()
            self.size_tab = None
            self._release_tiles()
        self.update_actions()

    def _add_size_tab(self, result, use_opengl: bool):
        self.size_tab = ContactSizeTab(result, use_opengl, self.measure_sizes)
        self.tabs.addTab(self.size_tab, "Contact size")

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
        self.images = self.design = None  # those of another result
        self.show_result(result)
        self.remember_folder(result.folder)

    def show_images(self, images):
        self.images = images
        self.stitching_view.set_images(images)
        self.status.set_idle(f"{len(images)} SEM images loaded (Stitching tab, SEM images)")

    def show_result(self, result):
        self.status.set_idle("Drawing...")
        QtWidgets.QApplication.processEvents()
        use_opengl = self.gpu_action.isChecked()
        tabs = QtWidgets.QTabWidget()
        self.stitching_view = StitchingView(result, use_opengl, self.load_images, self.load_design)
        if self.images is not None:
            self.stitching_view.set_images(self.images)
        if self.design is not None:
            self.stitching_view.set_design(self.design)
        self.registration_view = RegistrationView(result, use_opengl)
        tabs.addTab(self.stitching_view, "Stitching")
        tabs.addTab(self.registration_view, "Registration")
        self.tabs, self.size_tab = tabs, None
        if self.size_action.isChecked():
            self._add_size_tab(result, use_opengl)
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
        self.recent_menu.menuAction().setEnabled(not self.busy)
        self.actions["ftp"].setEnabled(not self.busy)
        self.actions["restart"].setEnabled(not self.busy and self.folder is not None)
        self.actions["parameters"].setEnabled(not self.busy)
        self.size_action.setEnabled(not self.busy)
        if self.size_tab is not None:
            self.size_tab.update_state(self.busy)
        if self.result is not None:
            self.stitching_view.load_button.setEnabled(
                not self.busy and self.images is None and bool(self.result.image_paths))
            self.stitching_view.design_button.setEnabled(
                not self.busy and self.design is None and any(self.result.oas_paths))

    def log(self, text: str):
        self.log_view.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")

    def settings(self) -> Settings:
        """The parameters chosen last (the release's defaults for any not saved)."""
        try:
            saved = json.loads(self.preferences.value("settings", "") or "{}")
        except ValueError:
            saved = {}
        known = {f.name for f in fields(Settings)}
        return replace(DEFAULTS, **{name: value for name, value in saved.items() if name in known})

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
