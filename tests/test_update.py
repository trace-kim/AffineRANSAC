"""Automatic updates of MaskReg (affine_ransac.release.update and the window's part), against a local pyftpdlib
server with an anonymous, read-only login like the real update server."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import subprocess
import sys
import threading
import time

import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from affine_ransac.release import app as release_app
from affine_ransac.release.update import download_setup, find_update, newest_setup, read_update_source, run_setup
from test_release import wait

pytest.importorskip("pyftpdlib")
from pyftpdlib.authorizers import DummyAuthorizer  # noqa: E402
from pyftpdlib.handlers import FTPHandler  # noqa: E402
from pyftpdlib.servers import ThreadedFTPServer  # noqa: E402

SETUPS = {"MaskReg-setup-1.0.0.exe": b"a" * 10, "MaskReg-setup-1.9.0.exe": b"b" * 20,
          "MaskReg-setup-1.10.0.exe": b"c" * 3_000_000, "notes.txt": b"x"}


@pytest.fixture
def source(tmp_path):
    """A local FTP server whose /MaskReg folder holds SETUPS, anonymous read-only. Yields its update source."""
    folder = tmp_path / "server" / "MaskReg"
    folder.mkdir(parents=True)
    for name, content in SETUPS.items():
        (folder / name).write_bytes(content)
    authorizer = DummyAuthorizer()
    authorizer.add_anonymous(str(tmp_path / "server"))  # read-only

    class Handler(FTPHandler):
        pass
    Handler.authorizer = authorizer
    server = ThreadedFTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield {"host": "127.0.0.1", "port": server.address[1], "folder": "/MaskReg"}
    server.close_all()


def test_the_newest_setup_is_chosen_by_its_version_numbers():
    assert newest_setup(SETUPS) == ("1.10.0", "MaskReg-setup-1.10.0.exe")  # 10 > 9, though "1.10" < "1.9" as text
    assert newest_setup(["notes.txt", "MaskReg-setup-2.0.exe", "MaskReg-setup-2.0.0.exe.part"]) is None


def test_the_update_server_is_read_from_update_ini(tmp_path):
    path = tmp_path / "update.ini"
    assert read_update_source(path) is None  # no file: no updates
    path.write_text("[update]\nhost = 10.0.0.5\nfolder = /MaskReg\n", encoding="utf-8")
    assert read_update_source(path) == {"host": "10.0.0.5", "port": 21, "folder": "/MaskReg"}
    path.write_text("[update]\nport = 2121\n", encoding="utf-8")
    assert read_update_source(path) is None  # no host


def test_only_a_newer_setup_is_found_and_it_is_downloaded_whole(source, tmp_path):
    assert find_update(source, "1.9.5") == ("1.10.0", "MaskReg-setup-1.10.0.exe", 3_000_000)
    assert find_update(source, "1.10.0") is None and find_update(source, "2.0.0") is None

    folder = tmp_path / "download"
    (folder).mkdir()
    (folder / "MaskReg-setup-0.9.0.exe").write_bytes(b"old")  # an earlier download goes
    fractions = []
    setup = download_setup(source, "MaskReg-setup-1.10.0.exe", 3_000_000, folder, fractions.append)
    assert setup.read_bytes() == SETUPS["MaskReg-setup-1.10.0.exe"] and [p.name for p in folder.iterdir()] == [setup.name]
    assert fractions[-1] == 1.0 and len(fractions) > 1
    with pytest.raises(OSError, match="incomplete"):
        download_setup(source, "MaskReg-setup-1.9.0.exe", 21, folder)


@pytest.mark.skipif(sys.platform != "win32", reason="the installer is a Windows program")
def test_the_installer_runs_once_maskreg_has_quit_then_maskreg_starts_again(tmp_path):
    folder = tmp_path / "it's a folder"  # a space and a ' in every path
    folder.mkdir()
    quit_mark, ran, started = folder / "quit.txt", folder / "setup ran.txt", folder / "started.txt"
    setup, restart = folder / "MaskReg-setup-9.0.0.bat", folder / "MaskReg.bat"  # stand-ins for the real programs
    setup.write_text(f'@if exist "{quit_mark}" (echo %* > "{ran}") else (echo too early > "{ran}")\n')
    restart.write_text(f'@echo started > "{started}"\n')
    maskreg = subprocess.Popen([sys.executable, "-c",  # stands in for MaskReg, quitting 2 s later
                                "import sys, time; time.sleep(2); open(sys.argv[1], 'w').close()", str(quit_mark)])

    run_setup(setup, restart, maskreg.pid)
    end = time.time() + 30
    while not started.exists() and time.time() < end:
        time.sleep(0.1)
    assert started.exists()
    assert ran.read_text().strip() == f'/SILENT /SUPPRESSMSGBOXES /NORESTART /LOG="{setup.with_suffix(".log")}"'


def make_window(tmp_path, source):
    pg.mkQApp()
    preferences = QtCore.QSettings(str(tmp_path / "preferences.ini"), QtCore.QSettings.Format.IniFormat)
    preferences.setValue("use_opengl", "false")
    return release_app.MaskRegWindow(preferences, cache_dir=str(tmp_path / "user cache"), update_source=source)


def until(condition, seconds=20):
    end = time.time() + seconds
    while not condition() and time.time() < end:
        pg.mkQApp().processEvents()
        time.sleep(0.01)
    assert condition()


def test_a_newer_version_is_offered_and_installed_if_the_user_agrees(source, tmp_path, monkeypatch):
    asked, started = [], []
    answers = [QtWidgets.QMessageBox.StandardButton.No, QtWidgets.QMessageBox.StandardButton.Yes]
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda parent, title, text: asked.append(text) or answers.pop(0))
    monkeypatch.setattr(release_app, "run_setup", started.append)
    window = make_window(tmp_path, source)

    window.check_for_update()  # at start; No: nothing downloaded, asked again next time
    until(lambda: asked)
    assert "MaskReg 1.10.0 is available" in asked[0] and not started and window.update_action.isEnabled()

    window.check_for_update()  # Yes: downloaded, the installer started, MaskReg closes
    until(lambda: len(asked) == 2)
    wait(window)
    assert len(started) == 1 and started[0].read_bytes() == SETUPS["MaskReg-setup-1.10.0.exe"]
    assert "Downloading MaskReg 1.10.0" in window.log_view.toPlainText()


def test_while_a_job_runs_the_question_waits_until_it_is_done(source, tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda parent, title, text: asked.append(text) or QtWidgets.QMessageBox.StandardButton.No)
    window = make_window(tmp_path, source)
    gate = threading.Event()
    window.run("Analysing", lambda log, progress: gate.wait(20), lambda value: None)
    window.check_for_update()
    until(lambda: window.update_found is not None)
    assert not asked  # not in the middle of the analysis
    gate.set()
    wait(window)
    assert len(asked) == 1


def test_without_a_server_or_with_none_reachable(source, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda parent, title, text: shown.append(text))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda parent, title, text: shown.append(text))
    assert not make_window(tmp_path, None).update_action.isEnabled()  # not the installed MaskReg: no updates

    window = make_window(tmp_path, dict(source, port=1))  # nothing listens there
    window.check_for_update()  # at start: only logged
    until(lambda: "Update check failed" in window.log_view.toPlainText())
    assert not shown
    window.check_for_update(quiet=False)  # asked for: said
    until(lambda: shown)
    assert shown[0].startswith("Could not look for updates")

    monkeypatch.setattr(release_app, "__version__", "1.10.0")
    window = make_window(tmp_path, source)
    window.check_for_update(quiet=False)
    until(lambda: len(shown) == 2)
    assert "is the newest version" in shown[1]
