"""Browsing an FTP server (docs/SPEC.md D82): connect, and list the folders and files of a remote folder.

Plain FTP with the standard library's ftplib (passive mode, the default). No Qt here: the dialog is ftp_dialog.py.
"""

import ftplib
import posixpath

TIMEOUT_S = 15  # for the connection and every command, so an unreachable server fails instead of hanging


def connect(host: str, port: int = 21, user: str = "", password: str = "") -> ftplib.FTP:
    """A logged-in connection (anonymous when user is empty). Raises OSError / ftplib.Error on failure."""
    ftp = ftplib.FTP(timeout=TIMEOUT_S)
    ftp.connect(host, port)
    try:
        ftp.login(user or "anonymous", password)
    except ftplib.Error:
        ftp.close()
        raise
    return ftp


def list_folder(ftp: ftplib.FTP, path: str) -> tuple[list[str], list[str]]:
    """The names of the folders and of the files in the remote folder `path`, each sorted (case-insensitive).
    Uses MLSD where the server has it, else parses LIST (Unix `ls -l` or DOS/IIS format)."""
    try:
        entries = [(name, facts.get("type", "").lower()) for name, facts in ftp.mlsd(path, facts=["type"])]
        folders = [name for name, kind in entries if kind == "dir"]
        files = [name for name, kind in entries if kind == "file"]
    except ftplib.error_perm:  # 500/502: MLSD not supported
        lines = []
        ftp.retrlines(f"LIST {path}", lines.append)
        folders, files = parse_list(lines)
    return sorted(folders, key=str.lower), sorted(files, key=str.lower)


def parse_list(lines: list[str]) -> tuple[list[str], list[str]]:
    """Folder and file names from LIST lines. Unix: 'drwxr-xr-x 2 user group 4096 Jan 01 12:00 name';
    DOS: '01-01-24  12:00PM  <DIR>  name' or '01-01-24  12:00PM  1234  name'. Links and other lines are skipped."""
    folders, files = [], []
    for line in lines:
        parts = line.split(None, 8)
        if len(parts) == 9 and line[0] in "d-":  # Unix: the name is the 9th field (may hold spaces)
            name = parts[8]
            (folders if line[0] == "d" else files).append(name)
            continue
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[0][:1].isdigit():  # DOS
            name = parts[3]
            (folders if parts[2].upper() == "<DIR>" else files).append(name)
    return ([n for n in folders if n not in (".", "..")], files)


def parent(path: str) -> str:
    """The remote folder above `path` ('/' stays '/')."""
    return posixpath.dirname(path.rstrip("/")) or "/"


def join(path: str, name: str) -> str:
    return posixpath.join(path, name)
