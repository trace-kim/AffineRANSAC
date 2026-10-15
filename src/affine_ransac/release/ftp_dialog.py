"""The FTP browser of MaskReg (docs/SPEC.md D82, D83): connect to a server, walk through its folders and choose
one to download and analyse.

The fields start from the pre-set config (config.py), else from what was typed last (host, port and user are
remembered in the preferences; a typed password never is). With a pre-set host it connects at once.
Double-click a folder (or press Enter) to open it; Up goes to the folder above. *Download and analyse* takes the
selected folder, or the folder shown when none is selected; the window then does the download (app.py).
"""

import ftplib

from pyqtgraph.Qt import QtCore, QtWidgets

from affine_ransac.release import ftp as remote


class FtpDialog(QtWidgets.QDialog):
    def __init__(self, preferences: QtCore.QSettings, config: dict[str, str], download_root: str, parent=None):
        """config: the pre-set [ftp] keys (may be empty); download_root: where downloaded folders are kept."""
        super().__init__(parent)
        self.setWindowTitle("FTP server")
        self.preferences = preferences
        self.download_root = download_root
        self.ftp = None      # the open connection
        self.path = "/"      # the remote folder shown
        self.chosen = None   # the remote folder to download, once chosen

        def start(key, default=""):
            return config.get(key) or preferences.value(f"ftp_{key}", default)
        self.host = QtWidgets.QLineEdit(start("host"), placeholderText="server name or address")
        self.port = QtWidgets.QSpinBox(minimum=1, maximum=65535, value=int(start("port", 21)))
        self.user = QtWidgets.QLineEdit(start("user"), placeholderText="empty: anonymous")
        self.password = QtWidgets.QLineEdit(config.get("password", ""), echoMode=QtWidgets.QLineEdit.EchoMode.Password)
        self.connect_button = QtWidgets.QPushButton("Connect", default=True)
        self.connect_button.clicked.connect(self.toggle_connection)
        form = QtWidgets.QHBoxLayout()
        for label, widget in (("Host", self.host), ("Port", self.port), ("User", self.user),
                              ("Password", self.password)):
            form.addWidget(QtWidgets.QLabel(label))
            form.addWidget(widget, 2 if widget is self.host else 1)
        form.addWidget(self.connect_button)

        self.up_button = QtWidgets.QPushButton("Up")
        self.up_button.clicked.connect(lambda: self.show_folder(remote.parent(self.path)))
        self.path_label = QtWidgets.QLabel(textInteractionFlags=QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(self.up_button)
        path_row.addWidget(self.path_label, 1)

        self.entries = QtWidgets.QListWidget()
        self.entries.itemActivated.connect(self.open_item)  # double-click or Enter
        self.entries.currentItemChanged.connect(self.update_state)
        self.message = QtWidgets.QLabel()
        self.target_label = QtWidgets.QLabel(wordWrap=True)
        self.download_button = QtWidgets.QPushButton("Download and analyse")
        self.download_button.clicked.connect(self.choose)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.target_label, 1)
        bottom.addWidget(self.download_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(path_row)
        layout.addWidget(self.entries, 1)
        layout.addWidget(self.message)
        layout.addLayout(bottom)
        self.update_state()
        self.resize(800, 600)
        if config.get("host"):
            QtCore.QTimer.singleShot(0, self.toggle_connection)  # once the dialog is shown

    def login(self) -> tuple[str, int, str, str]:
        """(host, port, user, password) as entered."""
        return self.host.text().strip(), self.port.value(), self.user.text(), self.password.text()

    def toggle_connection(self):
        if self.ftp is not None:
            self.disconnect()
            return
        host, port, user, password = self.login()
        if not host:
            self.message.setText("Enter the server's name or address.")
            return
        for key, value in (("ftp_host", host), ("ftp_port", port), ("ftp_user", user)):
            self.preferences.setValue(key, value)
        self.message.setText(f"Connecting to {host}...")
        with WaitCursor():
            try:
                self.ftp = remote.connect(host, port, user, password)
                start = self.ftp.pwd()
            except (OSError, ftplib.Error, EOFError) as error:
                self.ftp = None
                self.message.setText(f"Could not connect: {error}")
                return
        self.show_folder(start)

    def disconnect(self):
        try:
            self.ftp.quit()
        except (OSError, ftplib.Error, EOFError):
            self.ftp.close()
        self.ftp = None
        self.entries.clear()
        self.path_label.clear()
        self.message.setText("Disconnected.")
        self.update_state()

    def show_folder(self, path: str):
        """List the remote folder `path`: its folders first (with a trailing /), then its files."""
        with WaitCursor():
            try:
                folders, files = remote.list_folder(self.ftp, path)
            except (OSError, ftplib.Error, EOFError) as error:
                self.message.setText(f"Could not open {path}: {error}")
                self.update_state()
                return
        self.path = path
        self.path_label.setText(path)
        self.entries.clear()
        folder_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirIcon)
        file_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon)
        for name in folders:
            item = QtWidgets.QListWidgetItem(folder_icon, name + "/")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            self.entries.addItem(item)
        for name, _ in files:
            self.entries.addItem(QtWidgets.QListWidgetItem(file_icon, name))  # no data: not opened
        self.message.setText(f"{len(folders)} folders, {len(files)} files")
        self.update_state()

    def open_item(self, item: QtWidgets.QListWidgetItem):
        name = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if name is not None:
            self.show_folder(remote.join(self.path, name))

    def target(self) -> str:
        """The remote folder *Download and analyse* takes: the selected folder, else the folder shown."""
        item = self.entries.currentItem()
        name = item.data(QtCore.Qt.ItemDataRole.UserRole) if item is not None else None
        return remote.join(self.path, name) if name is not None else self.path

    def choose(self):
        self.chosen = self.target()
        self.accept()

    def update_state(self):
        connected = self.ftp is not None
        self.connect_button.setText("Disconnect" if connected else "Connect")
        for widget in (self.host, self.port, self.user, self.password):
            widget.setEnabled(not connected)
        self.up_button.setEnabled(connected and self.path != "/")
        self.download_button.setEnabled(connected)
        if connected:
            local = remote.local_folder(self.download_root, self.login()[0], self.target())
            self.target_label.setText(f"{self.target()}  →  {local}")
        else:
            self.target_label.setText(f"Downloads are kept in {self.download_root}")

    def done(self, result):
        if self.ftp is not None:
            self.disconnect()
        super().done(result)


class WaitCursor:
    """The busy cursor while a (short, blocking) FTP command runs."""

    def __enter__(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

    def __exit__(self, *_):
        QtWidgets.QApplication.restoreOverrideCursor()
