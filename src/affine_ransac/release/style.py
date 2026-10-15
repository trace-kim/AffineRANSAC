"""Presentation style of the release's figures (docs/SPEC.md D77): white background, dark text, light grid,
fonts large enough for a slide, and Copy image / Save image buttons that render a figure at twice the screen
resolution (crisp when pasted into PowerPoint).

Line colours come from a colour-blind-checked categorical palette (blue, orange, aqua; dataviz reference
palette, light mode). Error maps keep the jet scale of the research views (user, D35); empty pixels are white.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

TEXT, MUTED, GRID = "#1b1b1a", "#52514e", "#d8d8d4"
DX, DY = "#2a78d6", "#eb6834"         # the two error components, in every figure
DESIGN, SEM = "#2a78d6", "#eb6834"    # stitching view: design and measured centres
TILES, OVERLAPS, FAILURE = "#9a9a96", "#1baf7a", "#e34948"
FONT = "Segoe UI"
TICK_PT, LABEL_PT, TITLE_PT = 10, 11, 12
EXPORT_SCALE = 2  # Copy / Save image: pixels per screen pixel


class MaskAxis(pg.AxisItem):
    """An axis of data drawn relative to an origin (the OpenGL viewport works in float32) that shows absolute
    mask µm: tick value + origin. data_unit_nm: nm per data unit (1: data in nm, 1000: data in µm)."""

    def __init__(self, orientation: str, origin_nm: float, data_unit_nm: float = 1.0):
        super().__init__(orientation)
        self.origin_um = origin_nm / 1000
        self.setScale(data_unit_nm / 1000)

    def tickStrings(self, values, scale, spacing):
        decimals = max(0, int(np.ceil(-np.log10(spacing * scale)))) if spacing * scale > 0 else 0
        return [f"{v * scale + self.origin_um:.{decimals}f}" for v in values]


def setup():
    """Global pyqtgraph options: white figures, smooth lines. Call before making any figure."""
    pg.setConfigOptions(background="w", foreground=TEXT, antialias=True)


def style_plot(plot: pg.PlotItem, title: str = "", x_label: str = "", y_label: str = "", grid: bool = True):
    """Title, axis labels, fonts and a light grid on a plot."""
    if title:
        plot.setTitle(title, color=TEXT, size=f"{TITLE_PT}pt", bold=True)
    tick_font = QtGui.QFont(FONT, TICK_PT)
    for side, label in (("bottom", x_label), ("left", y_label)):
        axis = plot.getAxis(side)
        axis.setPen(pg.mkPen(MUTED))
        axis.setTextPen(pg.mkPen(TEXT))
        axis.setStyle(tickFont=tick_font, tickTextOffset=6)
        axis.enableAutoSIPrefix(False)  # the labels carry the units; no "(x0.001)"
        if label:
            axis.setLabel(label, color=TEXT, **{"font-size": f"{LABEL_PT}pt", "font-family": FONT})
    if grid:
        plot.showGrid(x=True, y=True, alpha=0.25)


def legend(plot: pg.PlotItem) -> pg.LegendItem:
    """A legend in the plot's top-left corner, white with a thin border."""
    item = plot.addLegend(offset=(10, 10), brush=pg.mkBrush(255, 255, 255, 230), pen=pg.mkPen(GRID),
                          labelTextColor=TEXT, labelTextSize=f"{TICK_PT}pt")
    return item


def zero_line(plot: pg.PlotItem):
    plot.addLine(y=0, pen=pg.mkPen(MUTED, width=1, style=QtCore.Qt.PenStyle.DashLine))


def render_image(view: pg.GraphicsView, scale: int = EXPORT_SCALE) -> QtGui.QImage:
    """The figure as shown, rendered at `scale` x its size on screen through the scene (so an OpenGL
    viewport, which cannot always be grabbed, renders too)."""
    source = view.viewport().rect()
    image = QtGui.QImage(source.size() * scale, QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtGui.QColor("white"))
    painter = QtGui.QPainter(image)
    painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
    view.render(painter, QtCore.QRectF(image.rect()), source)
    painter.end()
    return image


def image_buttons(figure, name: str) -> list[QtWidgets.QPushButton]:
    """Copy image (clipboard, for slides) and Save image... (PNG) buttons for the figure (pg.GraphicsView) that
    figure() returns when a button is pressed."""
    copy = QtWidgets.QPushButton("Copy image")
    copy.setToolTip(f"Copy the {name} figure to the clipboard (paste it into a slide)")
    copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setImage(render_image(figure())))
    save = QtWidgets.QPushButton("Save image...")
    save.setToolTip(f"Save the {name} figure as a PNG file")

    def save_png():
        path, _ = QtWidgets.QFileDialog.getSaveFileName(save, "Save image", f"{name}.png", "PNG image (*.png)")
        if path:
            render_image(figure()).save(path)
    save.clicked.connect(save_png)
    return [copy, save]


def figure_bar(title: str, buttons: list[QtWidgets.QWidget]) -> QtWidgets.QHBoxLayout:
    """A row above a figure: its name in bold, buttons on the right."""
    bar = QtWidgets.QHBoxLayout()
    label = QtWidgets.QLabel(f"<b>{title}</b>")
    bar.addWidget(label)
    bar.addStretch(1)
    for button in buttons:
        bar.addWidget(button)
    return bar
