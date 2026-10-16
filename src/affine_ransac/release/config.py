"""Pre-set settings of MaskReg (docs/SPEC.md D83), so users need not type the FTP server in.

An administrator writes maskreg.ini (plain text, so the password is readable by whoever can read the file):

    [ftp]
    host = ftp.example.com
    port = 21
    user = maskreg
    password = secret
    download_folder = D:/MaskReg data

Every key is optional. The file is the one named by the MASKREG_CONFIG environment variable, or else
%LOCALAPPDATA%/MaskReg/maskreg.ini (on Windows; the user's config folder elsewhere).
"""

import configparser
import os
from pathlib import Path

from pyqtgraph.Qt import QtCore

CONFIG_NAME = "maskreg.ini"
CONFIG_ENV = "MASKREG_CONFIG"


def config_path() -> Path:
    if os.environ.get(CONFIG_ENV):
        return Path(os.environ[CONFIG_ENV])
    folder = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.GenericConfigLocation)
    return Path(folder, "MaskReg", CONFIG_NAME)


def read_ftp_config(path: Path) -> dict[str, str]:
    """The [ftp] keys of the file ({} if there is no file). A malformed file raises configparser.Error."""
    parser = configparser.ConfigParser(interpolation=None)  # a '%' in a password stays a '%'
    parser.read(path, encoding="utf-8")
    return dict(parser["ftp"]) if parser.has_section("ftp") else {}


def default_download_root() -> str:
    """Where downloaded data is kept unless the config file says otherwise: Documents/MaskReg data."""
    documents = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.DocumentsLocation)
    return str(Path(documents, "MaskReg data"))
