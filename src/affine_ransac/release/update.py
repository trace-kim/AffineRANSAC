"""Automatic updates of MaskReg (docs/SPEC.md D101). No Qt here: the window asks and runs the steps (app.py).

The installed MaskReg looks, at every start and on Help > Check for updates, in a folder of an FTP server (anonymous
login, read-only) for the newest MaskReg-setup-X.Y.Z.exe. If it is newer than this MaskReg, the user is asked (No: asked
again at the next start); Yes downloads it and quits MaskReg, and once MaskReg is gone the installer runs silently over
it and MaskReg starts again (run_setup).

The server is in update.ini next to MaskReg.exe, which packaging/build.py puts into the installer from
packaging/update.ini (kept out of git):

    [update]
    host = 192.168.0.10
    port = 21
    folder = /MaskReg
"""

import configparser
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from affine_ransac.release import ftp as remote

UPDATE_INI = "update.ini"
SETUP = re.compile(r"MaskReg-setup-(\d+)\.(\d+)\.(\d+)\.exe", re.IGNORECASE)


def read_update_source(path) -> dict | None:
    """The update server of an update.ini: {host, port, folder}; None if there is no file or no [update] host.
    A malformed file raises configparser.Error (or ValueError for the port)."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path, encoding="utf-8")
    if not parser.get("update", "host", fallback=""):
        return None
    section = parser["update"]
    return {"host": section["host"], "port": section.getint("port", 21), "folder": section.get("folder", "/")}


def installed_source() -> dict | None:
    """The update server of this MaskReg; None unless it runs as the installed program (MaskReg.exe) with one."""
    if not getattr(sys, "frozen", False):
        return None
    return read_update_source(Path(sys.executable).parent / UPDATE_INI)


def version_numbers(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def newest_setup(names) -> tuple[str, str] | None:
    """(version, file name) of the newest MaskReg-setup-X.Y.Z.exe among the file names, compared as numbers
    (1.10.0 after 1.9.0); None if there is none."""
    found = [(match.groups(), name) for name in names if (match := SETUP.fullmatch(name))]
    if not found:
        return None
    numbers, name = max(found, key=lambda item: tuple(int(n) for n in item[0]))
    return ".".join(str(int(n)) for n in numbers), name


def find_update(source: dict, current: str) -> tuple[str, str, int] | None:
    """(version, file name, size in bytes) of the newest setup on the server if it is newer than `current`, else
    None. Raises OSError / ftplib.Error if the server cannot be reached."""
    ftp = remote.connect(source["host"], source["port"])  # anonymous
    try:
        _, files = remote.list_folder(ftp, source["folder"])
    finally:
        ftp.close()
    sizes = dict(files)
    newest = newest_setup(sizes)
    if newest is None or version_numbers(newest[0]) <= version_numbers(current):
        return None
    version, name = newest
    return version, name, sizes[name]


def download_setup(source: dict, name: str, size: int, folder: Path, progress=lambda fraction: None) -> Path:
    """Download the setup `name` into `folder` (emptied first: older downloads go); raises OSError if it did not
    arrive whole. Returns its path."""
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    target = folder / name
    ftp = remote.connect(source["host"], source["port"])
    try:
        with open(target, "wb") as out:
            def write(block: bytes):
                out.write(block)
                progress(out.tell() / max(size, 1))
            ftp.retrbinary(f"RETR {remote.join(source['folder'], name)}", write, blocksize=1 << 20)
    finally:
        ftp.close()
    if size >= 0 and target.stat().st_size != size:
        raise OSError(f"{name} arrived incomplete ({target.stat().st_size} of {size} bytes)")
    return target


def run_setup(setup: Path, restart: Path | None = None, pid: int | None = None):
    """Install the setup once MaskReg (process `pid`, this one) has quit, then start MaskReg again (`restart`, this
    MaskReg.exe): the new version, or the old one if the installation failed. A hidden PowerShell does it, so it
    outlives MaskReg; the installer runs silently (a progress window, no questions) and logs to <setup>.log.
    Waiting first matters: the installer gives up on files still in use (exit code 5) and rolls back."""
    restart = restart or Path(sys.executable)
    pid = os.getpid() if pid is None else pid
    log_option = quoted('/LOG="' + str(setup.with_suffix(".log")) + '"')
    script = (f"Wait-Process -Id {pid} -Timeout 120 -ErrorAction SilentlyContinue; "
              f"Start-Process -Wait -FilePath {quoted(setup)} -ArgumentList '/SILENT','/SUPPRESSMSGBOXES','/NORESTART',"
              f"{log_option}; "
              f"Start-Process -FilePath {quoted(restart)}")
    subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                     creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP, close_fds=True)


def quoted(text) -> str:
    """A PowerShell string literal of `text` (a path may hold spaces or a ')."""
    return "'" + str(text).replace("'", "''") + "'"
