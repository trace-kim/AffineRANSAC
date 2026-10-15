"""The FTP browser of the release app (affine_ransac.release.ftp, ftp_dialog), against a local pyftpdlib server."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # must be set before Qt starts

import ftplib
import threading

import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore

from affine_ransac.release import ftp as remote
from affine_ransac.release.ftp_dialog import FtpDialog

pytest.importorskip("pyftpdlib")
from pyftpdlib.authorizers import DummyAuthorizer  # noqa: E402
from pyftpdlib.handlers import FTPHandler  # noqa: E402
from pyftpdlib.servers import ThreadedFTPServer  # noqa: E402


@pytest.fixture
def server(tmp_path):
    """A local FTP server whose root holds two folders (one with a space) and a file; user 'mask' / 'reg'."""
    (tmp_path / "Chip 1" / "tiles").mkdir(parents=True)
    (tmp_path / "Chip 1" / "CD001.csv").write_text("x")
    (tmp_path / "chip2").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    authorizer = DummyAuthorizer()
    authorizer.add_user("mask", "reg", str(tmp_path))

    class Handler(FTPHandler):
        pass
    Handler.authorizer = authorizer
    ftp_server = ThreadedFTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=ftp_server.serve_forever, daemon=True).start()
    yield ftp_server.address[1]
    ftp_server.close_all()


def test_folders_and_files_are_listed_apart_and_sorted(server):
    ftp = remote.connect("127.0.0.1", server, "mask", "reg")
    assert remote.list_folder(ftp, "/") == (["Chip 1", "chip2"], ["notes.txt"])
    assert remote.list_folder(ftp, "/Chip 1") == (["tiles"], ["CD001.csv"])
    ftp.quit()


def test_without_mlsd_the_list_output_is_parsed(server, monkeypatch):
    def no_mlsd(*args, **kwargs):
        raise ftplib.error_perm("500 Command not understood")
    ftp = remote.connect("127.0.0.1", server, "mask", "reg")
    monkeypatch.setattr(ftp, "mlsd", no_mlsd)
    assert remote.list_folder(ftp, "/Chip 1") == (["tiles"], ["CD001.csv"])
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
    assert remote.parse_list(lines) == (["Chip 1", "chip2"], ["notes.txt", "CD001.csv"])


def test_parent_and_join():
    assert remote.parent("/a/b") == "/a" and remote.parent("/a/") == "/" and remote.parent("/") == "/"
    assert remote.join("/", "Chip 1") == "/Chip 1" and remote.join("/a", "b") == "/a/b"


def test_the_dialog_connects_and_walks_through_folders(server, tmp_path):
    pg.mkQApp()
    preferences = QtCore.QSettings(str(tmp_path / "preferences.ini"), QtCore.QSettings.Format.IniFormat)
    dialog = FtpDialog(preferences)
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

    # Host, port and user are remembered for next time; the password is not.
    assert preferences.value("ftp_host") == "127.0.0.1" and preferences.value("ftp_user") == "mask"
    assert "reg" not in [str(preferences.value(key)) for key in preferences.allKeys()]
    dialog.connect_button.click()  # disconnect
    assert dialog.ftp is None and dialog.entries.count() == 0 and dialog.host.isEnabled()


def test_a_failed_connection_is_reported_in_the_dialog(server, tmp_path):
    pg.mkQApp()
    preferences = QtCore.QSettings(str(tmp_path / "preferences.ini"), QtCore.QSettings.Format.IniFormat)
    dialog = FtpDialog(preferences)
    dialog.host.setText("127.0.0.1")
    dialog.port.setValue(server)
    dialog.user.setText("mask")
    dialog.password.setText("wrong")
    dialog.connect_button.click()
    assert dialog.ftp is None and dialog.message.text().startswith("Could not connect")
