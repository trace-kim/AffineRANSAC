"""Browsing and downloading from an FTP server (docs/SPEC.md D82, D83): connect, list a remote folder, and copy a
remote folder (with its subfolders) into a local one.

Plain FTP with the standard library's ftplib (passive mode, the default). No Qt here: the dialog is ftp_dialog.py.
"""

import ftplib
import posixpath
import re
from pathlib import Path

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


def list_folder(ftp: ftplib.FTP, path: str) -> tuple[list[str], list[tuple[str, int]]]:
    """The folders (names) and the files (name, size in bytes; -1 if unknown) in the remote folder `path`, each
    sorted by name (case-insensitive). Uses MLSD where the server has it, else parses LIST (Unix or DOS/IIS)."""
    try:
        entries = [(name, facts) for name, facts in ftp.mlsd(path, facts=["type", "size"])]
        folders = [name for name, facts in entries if facts.get("type", "").lower() == "dir"]
        files = [(name, int(facts.get("size", -1))) for name, facts in entries
                 if facts.get("type", "").lower() == "file"]
    except ftplib.error_perm:  # 500/502: MLSD not supported
        lines = []
        ftp.retrlines(f"LIST {path}", lines.append)
        folders, files = parse_list(lines)
    return sorted(folders, key=str.lower), sorted(files, key=lambda file: file[0].lower())


def parse_list(lines: list[str]) -> tuple[list[str], list[tuple[str, int]]]:
    """Folder names and (file name, size) from LIST lines. Unix: 'drwxr-xr-x 2 user group 4096 Jan 01 12:00 name';
    DOS: '01-01-24  12:00PM  <DIR>  name' or '01-01-24  12:00PM  1234  name'. Links and other lines are skipped."""
    folders, files = [], []
    for line in lines:
        parts = line.split(None, 8)
        if len(parts) == 9 and line[0] in "d-":  # Unix: the name is the 9th field (may hold spaces)
            if line[0] == "d":
                folders.append(parts[8])
            else:
                files.append((parts[8], int(parts[4])))
            continue
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[0][:1].isdigit():  # DOS
            if parts[2].upper() == "<DIR>":
                folders.append(parts[3])
            else:
                files.append((parts[3], int(parts[2])))
    return [n for n in folders if n not in (".", "..")], files


def walk(ftp: ftplib.FTP, path: str) -> list[tuple[str, int]]:
    """Every file under the remote folder `path`, subfolders included: (path relative to it with '/', size)."""
    folders, files = list_folder(ftp, path)
    found = list(files)
    for folder in folders:
        found += [(f"{folder}/{relative}", size) for relative, size in walk(ftp, join(path, folder))]
    return found


def local_folder(root: str | Path, host: str, remote_path: str) -> Path:
    """Where a remote folder is kept: <root>/<host>/<remote path>, so the same remote folder always lands in the
    same place (and its saved analysis is found again)."""
    return Path(root, safe_name(host), *[safe_name(part) for part in remote_path.split("/") if part])


def safe_name(name: str) -> str:
    """A remote name usable on Windows: characters it forbids become '_', no trailing dots or spaces."""
    return re.sub(r'[<>:"/\\|?*]', "_", name).rstrip(". ") or "_"


def download_folder(ftp: ftplib.FTP, remote_path: str, local: Path, log=print,
                    progress=lambda fraction, stage: None) -> int:
    """Copy every file under the remote folder into `local` (same layout). A file already there with the same size
    is kept. Each file is written as '<name>.part' and renamed when complete, so an interrupted download is
    fetched again next time. Returns the number of files downloaded; an empty remote folder raises."""
    files = walk(ftp, remote_path)
    if not files:
        raise FileNotFoundError(f"No files in {remote_path} on the server")
    targets = {relative: Path(local, *[safe_name(part) for part in relative.split("/")]) for relative, _ in files}
    todo = [(relative, size) for relative, size in files
            if size < 0 or not targets[relative].is_file() or targets[relative].stat().st_size != size]
    log(f"{len(files)} files on the server: {len(files) - len(todo)} already here, {len(todo)} to download")
    total = sum(max(size, 0) for _, size in todo) or 1
    done = 0
    for k, (relative, size) in enumerate(todo, 1):
        progress(done / total, f"Downloading file {k} of {len(todo)}")
        target = targets[relative]
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        with open(part, "wb") as out:
            ftp.retrbinary(f"RETR {join(remote_path, relative)}", out.write, blocksize=1 << 20)
        part.replace(target)
        done += max(size, 0)
    progress(1.0, "Download complete")
    return len(todo)


def parent(path: str) -> str:
    """The remote folder above `path` ('/' stays '/')."""
    return posixpath.dirname(path.rstrip("/")) or "/"


def join(path: str, name: str) -> str:
    return posixpath.join(path, name)
