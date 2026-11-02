"""View > Log of MaskReg (docs/SPEC.md D148): every log message with its time, styled so that what matters is seen
at a glance. The kind of a message comes from how it starts, as the analysis writes them:
- "=== ..."       a step begins (opening a folder, loading images): bold, on a grey band
- "!! ..."        a problem worth knowing (tiles not measured, overlaps one row off): dark red, on a pale red band
- "Traceback ..." a step failed (the whole traceback): the same
- "Warning: ..."  a warning from the libraries: amber
In every line the numbers (counts, nm, %) are bold and the time is grey. Only the look changes: the text is as logged,
and a copy of it is plain text.
"""

import re
import time

from pyqtgraph.Qt import QtGui, QtWidgets

from affine_ransac.release import style

# A number on its own, with its sign and unit: not one inside a name, a path or a file name (tile_12, Chip1-1-3, a.12).
NUMBER = re.compile(r"(?<![\w.:/\\-])[+\-−]?\d+(?:\.\d+)*(?:\s?(?:nm|µm|um|px|%))?(?![\w/\\]|\.\w)")

KINDS = (("===", "heading"), ("!!", "problem"), ("Traceback", "problem"), ("Warning:", "warning"))
TEXT_COLOUR = {"problem": style.PROBLEM, "warning": style.CAUTION}
BAND_COLOUR = {"heading": style.HEADING_BAND, "problem": style.PROBLEM_BAND}


def message_kind(text: str) -> str:
    """'heading', 'problem', 'warning' or 'plain', from how the message starts."""
    for start, kind in KINDS:
        if text.startswith(start):
            return kind
    return "plain"


def char_format(colour: str, bold: bool = False) -> QtGui.QTextCharFormat:
    fmt = QtGui.QTextCharFormat()
    fmt.setForeground(QtGui.QColor(colour))
    if bold:
        fmt.setFontWeight(QtGui.QFont.Weight.Bold)
    return fmt


def block_format(kind: str) -> QtGui.QTextBlockFormat:
    """A line's format: its band (the whole width of the log), if its kind has one."""
    fmt = QtGui.QTextBlockFormat()
    if kind in BAND_COLOUR:
        fmt.setBackground(QtGui.QColor(BAND_COLOUR[kind]))
    return fmt


class LogView(QtWidgets.QPlainTextEdit):
    """The log: read-only, on white (the colours above are for a light background, whatever the system theme)."""

    def __init__(self):
        super().__init__(readOnly=True)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("white"))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(style.TEXT))
        self.setPalette(palette)

    def add(self, text: str):
        """Append a message (one or more lines; the time before the first), styled by its kind. If the end of the log
        was in view, it stays in view."""
        kind = message_kind(text)
        bar = self.verticalScrollBar()
        follow = bar.value() == bar.maximum()
        cursor = QtGui.QTextCursor(self.document())
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        for i, line in enumerate(text.rstrip("\n").split("\n")):
            if self.document().isEmpty():
                cursor.setBlockFormat(block_format(kind))
            else:
                cursor.insertBlock(block_format(kind))
            if i == 0:
                cursor.insertText(f"[{time.strftime('%H:%M:%S')}] ", char_format(style.MUTED))
            self.insert_line(cursor, line, kind)
        if follow:
            bar.setValue(bar.maximum())

    @staticmethod
    def insert_line(cursor: QtGui.QTextCursor, line: str, kind: str):
        """The line in its kind's colour (a heading all bold), its numbers bold."""
        colour, bold = TEXT_COLOUR.get(kind, style.TEXT), kind == "heading"
        start = 0
        for number in NUMBER.finditer(line):
            cursor.insertText(line[start:number.start()], char_format(colour, bold))
            cursor.insertText(number.group(), char_format(colour, True))
            start = number.end()
        cursor.insertText(line[start:], char_format(colour, bold))
