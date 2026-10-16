"""File > Folders... of MaskReg (docs/SPEC.md D96): the folder of the settings file (maskreg.ini, config.py) and
the folder where FTP downloads are kept, chosen by the user and kept in the preferences ("config_folder",
"download_folder"). An empty field means the default, shown greyed in it.
"""

import configparser

from pyqtgraph.Qt import QtCore, QtWidgets

from affine_ransac.release.config import CONFIG_NAME, config_path, download_root, read_ftp_config


class FoldersDialog(QtWidgets.QDialog):
    def __init__(self, preferences: QtCore.QSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Folders")
        self.preferences = preferences
        self.config_folder = QtWidgets.QLineEdit(preferences.value("config_folder", ""))
        self.download_folder = QtWidgets.QLineEdit(preferences.value("download_folder", ""))
        self.config_note = QtWidgets.QLabel(wordWrap=True)
        self.download_note = QtWidgets.QLabel("Each server folder is kept in <this folder>/<host>/<its path>.",
                                              wordWrap=True)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QFormLayout(self)
        layout.addRow(f"Settings file ({CONFIG_NAME}) in", self._with_browse(self.config_folder))
        layout.addRow("", self.config_note)
        layout.addRow("FTP downloads kept in", self._with_browse(self.download_folder))
        layout.addRow("", self.download_note)
        layout.addRow(buttons)
        self.config_folder.textChanged.connect(self.update_notes)
        self.update_notes()
        self.resize(760, self.sizeHint().height())

    def _with_browse(self, field: QtWidgets.QLineEdit) -> QtWidgets.QWidget:
        button = QtWidgets.QPushButton("Browse...")
        button.clicked.connect(lambda: self.browse(field))
        row = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(field, 1)
        box.addWidget(button)
        return row

    def browse(self, field: QtWidgets.QLineEdit):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder", field.text() or field.placeholderText())
        if folder:
            field.setText(folder)

    def update_notes(self):
        """The defaults (as placeholders) and whether the settings file is there, for the folder typed now."""
        path = config_path(self.config_folder.text().strip())
        self.config_folder.setPlaceholderText(f"default: {config_path().parent}")
        try:
            config = read_ftp_config(path)
            found = "found" if path.is_file() else "not found there: the FTP fields start empty"
        except configparser.Error as error:
            config, found = {}, f"cannot be read ({error}): ignored"
        self.config_note.setText(f"{path}: {found}")
        self.download_folder.setPlaceholderText(f"default: {download_root('', config)}")

    def accept(self):
        self.preferences.setValue("config_folder", self.config_folder.text().strip())
        self.preferences.setValue("download_folder", self.download_folder.text().strip())
        super().accept()
