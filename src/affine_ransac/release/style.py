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
DR = "#1baf7a"                        # the error's length |measured − design|
DESIGN, SEM = "#2a78d6", "#eb6834"    # stitching view: design and measured centres
TILES, OVERLAPS, FAILURE = "#9a9a96", "#1baf7a", "#e34948"
OK, CAUTION = "#14804a", "#b06f00"   # status texts: nothing wrong; worth knowing, results not affected (red: FAILURE)
DETECTED, REFINED = "#c2189b", "#00a8c8"  # contours and centres from the detection and after edge refinement
STRIPES = ["#2a78d6", "#9b4dca", "#1baf7a", "#c7951a"]  # design pattern: one colour per stripe, in turn
ALL_ROWS, RAMP, MANUAL = "#7d7d79", "#9b4dca", TEXT  # min and max lines of a mean per row: all rows, ramp, by hand
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

    def tickSpacing(self, minVal, maxVal, size):
        """pyqtgraph's spacing, with the labelled ticks far enough apart for their labels (1, 2, 5 x 10^k):
        pyqtgraph puts two or more on any axis, and on a narrow map their labels ran into each other (D114)."""
        levels = super().tickSpacing(minVal, maxVal, size)
        if not levels or size <= 0 or maxVal == minVal:
            return levels
        major = levels[0][0]
        while major * size / abs(maxVal - minVal) < self.label_px(minVal, maxVal, major):
            major *= 2.5 if round(major / 10 ** np.floor(np.log10(major))) == 2 else 2  # 1 -> 2 -> 5 -> 10
        return [(major, 0)] + levels[1:]

    def label_px(self, low: float, high: float, spacing: float) -> float:
        """Room a tick label takes along the axis, px, with a gap: the widest label at the axis ends (bottom, top)
        or a line of text (left, right). low, high, spacing as tickSpacing has them (scaled)."""
        metrics = QtGui.QFontMetrics(self.style["tickFont"] or QtGui.QFont())
        if self.orientation in ("left", "right"):
            return metrics.height() + 8
        labels = self.tickStrings([low / self.scale, high / self.scale], self.scale, spacing / self.scale)
        return max(metrics.horizontalAdvance(label) for label in labels) + 12


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


class Area:
    """Graphs taken together as one: the rectangle of the figure that holds them, as laid out (for render_graph;
    e.g. two maps with their profiles below and beside them)."""

    def __init__(self, items: list):
        self.items = items

    def sceneBoundingRect(self) -> QtCore.QRectF:
        rect = QtCore.QRectF()
        for item in self.items:
            rect = rect.united(item.sceneBoundingRect())
        return rect


def render_graph(view: pg.GraphicsView, items: list, scale: int = EXPORT_SCALE) -> QtGui.QImage:
    """Some graphs of a figure as shown (e.g. a map and its colour bar; an Area), side by side in one image, at
    `scale` x their size on screen."""
    # Cut from the whole figure rendered at that scale: rendering a part of the view alone misplaced the axis texts.
    whole = render_image(view, scale)
    sources = [view.mapFromScene(item.sceneBoundingRect()).boundingRect() for item in items]
    sources = [QtCore.QRect(r.x() * scale, r.y() * scale, r.width() * scale, r.height() * scale) for r in sources]
    image = QtGui.QImage(sum(r.width() for r in sources), max(r.height() for r in sources),
                         QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtGui.QColor("white"))
    painter = QtGui.QPainter(image)
    left = 0
    for source in sources:
        painter.drawImage(left, 0, whole, source.x(), source.y(), source.width(), source.height())
        left += source.width()
    painter.end()
    return image


def save_png(parent, image: QtGui.QImage, name: str):
    path, _ = QtWidgets.QFileDialog.getSaveFileName(parent, "Save image", f"{name}.png", "PNG image (*.png)")
    if path:
        image.save(path)


def image_menus(graphs) -> list[QtWidgets.QPushButton]:
    """Copy image and Save image buttons, each with a menu: the whole figure, or one graph of it. graphs() gives
    (figure view, [(graph name, [its graphics items]), ...]) of what is shown when a menu opens."""
    copy = menu_button("Copy image", graphs, lambda button, image, name: QtWidgets.QApplication.clipboard().setImage(image))
    save = menu_button("Save image", graphs, save_png)
    return [copy, save]


def menu_button(text: str, graphs, use) -> QtWidgets.QPushButton:
    """A button whose menu lists "Whole figure" and the graphs of graphs() (see image_menus); choosing one
    renders it and calls use(button, image, name)."""
    button = QtWidgets.QPushButton(text)
    button.setToolTip(f"{text}: the whole figure or one graph, at twice the screen resolution")
    menu = QtWidgets.QMenu(button)
    button.setMenu(menu)

    def fill():
        menu.clear()
        view, parts = graphs()
        menu.addAction("Whole figure", lambda: use(button, render_image(view), "figure"))
        for name, items in parts:
            menu.addAction(name, lambda name=name, items=items: use(button, render_graph(view, items), name))
    menu.aboutToShow.connect(fill)
    return button


def graph_actions(plot: pg.PlotItem, view: pg.GraphicsView, items: list, name: str):
    """Copy this graph / Save this graph... in the plot's right-click menu."""
    menu = plot.vb.menu
    menu.addSeparator()
    menu.addAction("Copy this graph", lambda: QtWidgets.QApplication.clipboard().setImage(render_graph(view, items)))
    menu.addAction("Save this graph...", lambda: save_png(view, render_graph(view, items), name))


def figure_bar(title: str, buttons: list[QtWidgets.QWidget]) -> QtWidgets.QHBoxLayout:
    """A row above a figure: its name in bold, buttons on the right."""
    bar = QtWidgets.QHBoxLayout()
    label = QtWidgets.QLabel(f"<b>{title}</b>")
    bar.addWidget(label)
    bar.addStretch(1)
    for button in buttons:
        bar.addWidget(button)
    return bar
