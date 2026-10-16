"""Pre-set settings of MaskReg (docs/SPEC.md D83), so users need not type the FTP server in.

An administrator writes maskreg.ini (plain text, so the password is readable by whoever can read the file):

    [ftp]
    host = ftp.example.com
    port = 21
    user = maskreg
    password = secret
    download_folder = D:/MaskReg data

Every key is optional. The file is maskreg.ini in the folder the user chose (File > Folders), or else the one named
by the MASKREG_CONFIG environment variable, or else %LOCALAPPDATA%/MaskReg/maskreg.ini (on Windows; the user's
config folder elsewhere). Downloads are kept in the folder the user chose, else the file's download_folder, else
Documents/MaskReg data.
"""

import configparser
import os
from pathlib import Path

from pyqtgraph.Qt import QtCore

CONFIG_NAME = "maskreg.ini"
CONFIG_ENV = "MASKREG_CONFIG"


def config_path(folder: str = "") -> Path:
    """The settings file: maskreg.ini in `folder` (the user's choice; "" = none chosen), else the default."""
    if folder:
        return Path(folder, CONFIG_NAME)
    if os.environ.get(CONFIG_ENV):
        return Path(os.environ[CONFIG_ENV])
    folder = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.GenericConfigLocation)
    return Path(folder, "MaskReg", CONFIG_NAME)


def read_ftp_config(path: Path) -> dict[str, str]:
    """The [ftp] keys of the file ({} if there is no file). A malformed file raises configparser.Error."""
    parser = configparser.ConfigParser(interpolation=None)  # a '%' in a password stays a '%'
    parser.read(path, encoding="utf-8")
    return dict(parser["ftp"]) if parser.has_section("ftp") else {}


def download_root(chosen: str, config: dict[str, str]) -> str:
    """Where downloaded folders are kept: the user's choice ("" = none chosen), else the config file's
    download_folder, else Documents/MaskReg data."""
    return chosen or config.get("download_folder") or default_download_root()


def default_download_root() -> str:
    documents = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.DocumentsLocation)
    return str(Path(documents, "MaskReg data"))
