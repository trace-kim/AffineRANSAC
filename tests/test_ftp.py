"""The FTP browser and download of the release app (affine_ransac.release.ftp, ftp_dialog, config), against a
local pyftpdlib server."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import ftplib
import threading
from pathlib import Path

import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore

from affine_ransac.release import app as release_app
from affine_ransac.release import ftp as remote
from affine_ransac.release.config import CONFIG_ENV, config_path, read_ftp_config, read_hidden_settings
from affine_ransac.release.ftp_dialog import FtpDialog
from test_analysis import write_contour_folder
from test_release import make_window, wait

pytest.importorskip("pyftpdlib")
from pyftpdlib.authorizers import DummyAuthorizer  # noqa: E402
from pyftpdlib.handlers import FTPHandler  # noqa: E402
from pyftpdlib.servers import ThreadedFTPServer  # noqa: E402


@pytest.fixture
def server(tmp_path):
    """A local FTP server (its root: tmp_path/'server') holding two folders (one with a space and a subfolder)
    and a file; user 'mask' / 'reg'. Yields its port."""
    root = tmp_path / "server"
    (root / "Chip 1" / "tiles").mkdir(parents=True)
    (root / "Chip 1" / "tiles" / "CD002.jpg").write_bytes(b"jpeg")
    (root / "Chip 1" / "CD001.csv").write_text("x")
    (root / "chip2").mkdir()
    (root / "notes.txt").write_text("x")
    authorizer = DummyAuthorizer()
    authorizer.add_user("mask", "reg", str(root))

    class Handler(FTPHandler):
        pass
    Handler.authorizer = authorizer
    ftp_server = ThreadedFTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=ftp_server.serve_forever, daemon=True).start()
    yield ftp_server.address[1]
    ftp_server.close_all()


def test_folders_and_files_are_listed_apart_and_sorted(server):
    ftp = remote.connect("127.0.0.1", server, "mask", "reg")
    assert remote.list_folder(ftp, "/") == (["Chip 1", "chip2"], [("notes.txt", 1)])
    assert remote.list_folder(ftp, "/Chip 1") == (["tiles"], [("CD001.csv", 1)])
    ftp.quit()


def test_without_mlsd_the_list_output_is_parsed(server, monkeypatch):
    def no_mlsd(*args, **kwargs):
        raise ftplib.error_perm("500 Command not understood")
    ftp = remote.connect("127.0.0.1", server, "mask", "reg")
    monkeypatch.setattr(ftp, "mlsd", no_mlsd)
    assert remote.list_folder(ftp, "/Chip 1") == (["tiles"], [("CD001.csv", 1)])
    ftp.quit()


def test_a_wrong_password_raises(server):
    with pytest.raises(ftplib.error_perm):
        remote.connect("127.0.0.1", server, "mask", "wrong")


def test_unix_and_dos_list_lines():
    lines = ["drwxr-xr-x    2 mask  staff     4096 Oct 15 12:00 Chip 1",
             "-rw-r--r--    1 mask  staff       10 Oct 15 12:00 notes.txt",
             "drwxr-xr-x    2 mask  staff     4096 Oct 15 12:00 .",
             "lrwxrwxrwx    1 mask  staff        4 Oct 15 12:00 link -> Chip 1",
             "10-15-26  12:00PM       <DIR>          chip2",
             "10-15-26  12:00PM                  10 CD001.csv",
             "total 8"]
    assert remote.parse_list(lines) == (["Chip 1", "chip2"], [("notes.txt", 10), ("CD001.csv", 10)])


def test_parent_and_join():
    assert remote.parent("/a/b") == "/a" and remote.parent("/a/") == "/" and remote.parent("/") == "/"
    assert remote.join("/", "Chip 1") == "/Chip 1" and remote.join("/a", "b") == "/a/b"


def test_a_remote_folder_is_kept_under_the_host_and_its_path():
    assert remote.local_folder("D:/data", "ftp.lab", "/2026/Chip 1") == Path("D:/data", "ftp.lab", "2026", "Chip 1")
    assert remote.local_folder("D:/data", "ftp.lab", "/") == Path("D:/data", "ftp.lab")
    assert remote.safe_name("a:b?c. ") == "a_b_c" and remote.safe_name("..") == "_"


def test_download_copies_the_tree_and_then_only_new_or_changed_files(server, tmp_path):
    ftp = remote.connect("127.0.0.1", server, "mask", "reg")
    local = tmp_path / "copy"
    assert remote.download_folder(ftp, "/Chip 1", local, log=lambda line: None) == 2
    assert (local / "CD001.csv").read_text() == "x" and (local / "tiles" / "CD002.jpg").read_bytes() == b"jpeg"
    assert not list(local.rglob("*.part"))

    assert remote.download_folder(ftp, "/Chip 1", local, log=lambda line: None) == 0  # all there
    (tmp_path / "server" / "Chip 1" / "CD001.csv").write_text("changed")
    (local / "tiles" / "CD002.jpg").unlink()
    assert remote.download_folder(ftp, "/Chip 1", local, log=lambda line: None) == 2
    assert (local / "CD001.csv").read_text() == "changed"
    with pytest.raises(FileNotFoundError):
        remote.download_folder(ftp, "/chip2", tmp_path / "empty", log=lambda line: None)
    ftp.quit()


def test_the_config_file_gives_the_ftp_keys(tmp_path, monkeypatch):
    path = tmp_path / "maskreg.ini"
    assert read_ftp_config(path) == {}  # no file
    path.write_text("[ftp]\nhost = ftp.lab\nuser = mask\npassword = a%b\n", encoding="utf-8")
    assert read_ftp_config(path) == {"host": "ftp.lab", "user": "mask", "password": "a%b"}
    monkeypatch.setenv(CONFIG_ENV, str(path))
    assert config_path() == path


def test_the_config_file_gives_the_hidden_parameters(tmp_path):
    path = tmp_path / "maskreg.ini"
    assert read_hidden_settings(path) == {}  # no file
    path.write_text("[ftp]\nhost = ftp.lab\n[analysis]\nrealign_search_um = 12.5\n", encoding="utf-8")
    assert read_hidden_settings(path) == {"realign_search_um": 12.5}
    path.write_text("[analysis]\nmax_match_nm = 30\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        read_hidden_settings(path)


def make_preferences(tmp_path):
    pg.mkQApp()
    return QtCore.QSettings(str(tmp_path / "preferences.ini"), QtCore.QSettings.Format.IniFormat)


def test_the_dialog_connects_and_walks_through_folders(server, tmp_path):
    preferences = make_preferences(tmp_path)
    dialog = FtpDialog(preferences, {}, str(tmp_path / "downloads"))
    dialog.host.setText("127.0.0.1")
    dialog.port.setValue(server)
    dialog.user.setText("mask")
    dialog.password.setText("reg")
    dialog.connect_button.click()

    shown = lambda: [dialog.entries.item(k).text() for k in range(dialog.entries.count())]
    assert dialog.path == "/" and shown() == ["Chip 1/", "chip2/", "notes.txt"]
    assert not dialog.up_button.isEnabled()
    dialog.open_item(dialog.entries.item(0))
    assert dialog.path == "/Chip 1" and shown() == ["tiles/", "CD001.csv"]
    dialog.open_item(dialog.entries.item(1))  # a file: nothing happens
    assert dialog.path == "/Chip 1"
    dialog.up_button.click()
    assert dialog.path == "/"

    # Host, port and user are remembered for next time; the typed password is not.
    assert preferences.value("ftp_host") == "127.0.0.1" and preferences.value("ftp_user") == "mask"
    assert "reg" not in [str(preferences.value(key)) for key in preferences.allKeys()]
    dialog.connect_button.click()  # disconnect
    assert dialog.ftp is None and dialog.entries.count() == 0 and dialog.host.isEnabled()


def test_a_failed_connection_is_reported_in_the_dialog(server, tmp_path):
    dialog = FtpDialog(make_preferences(tmp_path), {}, str(tmp_path / "downloads"))
    dialog.host.setText("127.0.0.1")
    dialog.port.setValue(server)
    dialog.user.setText("mask")
    dialog.password.setText("wrong")
    dialog.connect_button.click()
    assert dialog.ftp is None and dialog.message.text().startswith("Could not connect")


def test_a_preset_server_connects_at_once_and_a_folder_is_chosen(server, tmp_path):
    config = {"host": "127.0.0.1", "port": str(server), "user": "mask", "password": "reg"}
    dialog = FtpDialog(make_preferences(tmp_path), config, str(tmp_path / "downloads"))
    pg.mkQApp().processEvents()  # the connection starts once the event loop runs
    assert dialog.ftp is not None and dialog.path == "/"
    assert dialog.target() == "/"  # nothing selected: the folder shown
    dialog.entries.setCurrentRow(0)
    assert dialog.target() == "/Chip 1"
    assert str(tmp_path / "downloads" / "127.0.0.1" / "Chip 1") in dialog.target_label.text()
    dialog.download_button.click()
    assert dialog.chosen == "/Chip 1" and dialog.ftp is None  # closed, and disconnected
    assert dialog.login() == ("127.0.0.1", server, "mask", "reg")


def test_the_window_downloads_a_folder_analyses_it_and_reopens_the_saved_result(server, tmp_path, monkeypatch):
    (tmp_path / "server" / "run1").mkdir()
    write_contour_folder(tmp_path / "server" / "run1")
    window = make_window(tmp_path)
    runs = []
    analyse = release_app.analyse_folder
    monkeypatch.setattr(release_app, "analyse_folder", lambda *args, **kwargs: runs.append(1) or analyse(*args, **kwargs))
    login, root = ("127.0.0.1", server, "mask", "reg"), tmp_path / "downloads"
    local = root / "127.0.0.1" / "run1"

    window.download_and_open(login, "/run1", str(root))
    wait(window)
    assert window.result is not None and Path(window.result.folder) == local and len(runs) == 1
    window.download_and_open(login, "/run1", str(root))  # nothing new: the saved result opens
    wait(window)
    assert len(runs) == 1 and Path(window.result.folder) == local
    window.close()
