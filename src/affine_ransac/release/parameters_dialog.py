"""Analysis > Parameters... of MaskReg (docs/SPEC.md D77, D136): the research app's settings dialog with a row on
top to save the parameters under a name the user chooses and load them again, since datasets need different
parameters.

The sets are kept in the preferences ("parameter_sets": JSON, name -> {parameter: value}), beside the parameters
chosen last ("settings"). A set holds every parameter a user can change (USER_PARAMETERS), those of both kinds of
folder, whichever kind the dialog shows. Loading puts a set in the dialog, and OK uses it as any change would. A
parameter the set does not have (added to MaskReg after the set was saved) takes its default. The settings users
cannot change keep their values.
"""

import json
from dataclasses import replace

from pyqtgraph.Qt import QtCore, QtWidgets

from affine_ransac.analysis import Settings, current_names
from affine_ransac.app import SettingsDialog, set_editor_value
from affine_ransac.release import style
from affine_ransac.release.analysis import DEFAULTS, PARAMETERS

USER_PARAMETERS = list(dict.fromkeys(PARAMETERS["images"] + PARAMETERS["contour CSV"]))  # both kinds of folder
YES = QtWidgets.QMessageBox.StandardButton.Yes


def read_sets(preferences: QtCore.QSettings) -> dict[str, dict]:
    """The saved sets by name ({} if none, or if they cannot be read)."""
    try:
        sets = json.loads(preferences.value("parameter_sets", "") or "{}")
    except ValueError:
        return {}
    return sets if isinstance(sets, dict) else {}


def write_sets(preferences: QtCore.QSettings, sets: dict[str, dict]):
    preferences.setValue("parameter_sets", json.dumps(sets))


def stored(settings: Settings) -> dict:
    """What a set keeps of the settings: every parameter a user can change."""
    return {name: getattr(settings, name) for name in USER_PARAMETERS}


def loaded(settings: Settings, values: dict) -> Settings:
    """The settings with a set's parameters, read in today's names; one the set does not have takes its default."""
    values = current_names(values)
    return replace(settings, **{name: values.get(name, getattr(DEFAULTS, name)) for name in USER_PARAMETERS})


class ParametersDialog(SettingsDialog):
    """The parameters `names` of `settings`, with the saved sets on top: Load, Save as..., Delete."""

    def __init__(self, settings: Settings, names: list[str], preferences: QtCore.QSettings, parent=None):
        super().__init__(settings, "", parent, names)
        self.setWindowTitle("Parameters")
        self.preferences = preferences
        self.saved = QtWidgets.QComboBox(minimumWidth=260)
        self.saved.setPlaceholderText("none saved yet")
        self.load_button = QtWidgets.QPushButton("Load")
        self.save_button = QtWidgets.QPushButton("Save as...")
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.note = QtWidgets.QLabel()
        self.note.setStyleSheet(f"color: {style.MUTED}")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Saved parameters"))
        row.addWidget(self.saved)
        for button, slot in ((self.load_button, self.load), (self.save_button, self.save),
                             (self.delete_button, self.delete)):
            button.setAutoDefault(False)  # Enter in a value still means OK
            button.clicked.connect(slot)
            row.addWidget(button)
        row.addWidget(self.note, 1)
        self.layout().insertLayout(0, row)

        # Open on the set the parameters in use come from, if they are one's.
        in_use = [name for name, values in read_sets(preferences).items() if loaded(settings, values) == settings]
        self.fill(min(in_use, key=str.casefold) if in_use else None)
        if in_use:
            self.note.setText(f"The parameters in use are '{self.saved.currentText()}'")

    def fill(self, chosen: str | None):
        """List the saved sets by name, `chosen` selected (None: the first)."""
        names = sorted(read_sets(self.preferences), key=str.casefold)
        self.saved.clear()
        self.saved.addItems(names)
        if names:  # with placeholder text, adding items selects none
            self.saved.setCurrentIndex(names.index(chosen) if chosen in names else 0)
        for widget in (self.saved, self.load_button, self.delete_button):
            widget.setEnabled(bool(names))

    def load(self):
        """Put the chosen set in the dialog: OK uses it, Cancel does not."""
        name = self.saved.currentText()
        self.settings = loaded(self.settings, read_sets(self.preferences)[name])
        for parameter, widget in self.editors.items():
            set_editor_value(widget, getattr(self.settings, parameter))
        self.note.setText(f"'{name}' loaded: OK uses these parameters")

    def save(self):
        """Save the parameters in the dialog under the name the user types; an existing name only if they agree."""
        name, ok = QtWidgets.QInputDialog.getText(self, "Save parameters", "Name:", text=self.saved.currentText())
        name = name.strip()
        if not ok or not name:
            return
        sets = read_sets(self.preferences)
        if name in sets and QtWidgets.QMessageBox.question(
                self, "Save parameters", f"Replace the saved parameters '{name}'?") != YES:
            return
        sets[name] = stored(self.values())
        write_sets(self.preferences, sets)
        self.fill(name)
        self.note.setText(f"Saved as '{name}'")

    def delete(self):
        name = self.saved.currentText()
        if QtWidgets.QMessageBox.question(self, "Delete parameters", f"Delete the saved parameters '{name}'?") != YES:
            return
        sets = read_sets(self.preferences)
        sets.pop(name, None)
        write_sets(self.preferences, sets)
        self.fill(None)
        self.note.setText(f"'{name}' deleted")
