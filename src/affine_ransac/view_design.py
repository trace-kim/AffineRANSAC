"""Interactive viewer for OASIS design files (pyqtgraph).

Usage:
    python -m affine_ransac.view_design FILE.oas                 # all layers
    python -m affine_ransac.view_design FILE.oas --layer 1/0     # one layer
    python -m affine_ransac.view_design FILE.oas --cell TOP      # choose top cell

Mouse: drag to pan, wheel to zoom, right-click for more options.
The cursor position (in nm) is shown above the plot.
"""

import argparse
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from affine_ransac.io.design import contact_centers, list_layers, load_layout, read_polygons


def parse_layer(text: str) -> tuple[int, int]:
    """Parse 'layer/datatype' (e.g. '1/0') into a tuple of ints."""
    layer, datatype = text.split("/")
    return int(layer), int(datatype)


def polygons_to_path(polygons: list[np.ndarray]):
    """Combine all polygons into one QPainterPath.

    Drawing one path is much faster than one graphics item per polygon.
    A NaN row between polygons breaks the outline, so they stay separate.
    """
    gap = np.array([[np.nan, np.nan]])
    pieces = []
    for p in polygons:
        pieces += [p, p[:1], gap]  # outline, back to first vertex to close it, then a break
    xy = np.concatenate(pieces)
    return pg.arrayToQPath(xy[:, 0], xy[:, 1], connect="finite")


def add_layer(plot, polygons: list[np.ndarray], color, label: str):
    """Draw one layer: filled polygons plus a '+' at each polygon centroid."""
    fill = pg.mkColor(color)
    fill.setAlpha(90)
    outline = QtWidgets.QGraphicsPathItem(polygons_to_path(polygons))
    outline.setPen(pg.mkPen(color))
    outline.setBrush(pg.mkBrush(fill))
    plot.addItem(outline)

    centers, _ = contact_centers(polygons)
    markers = pg.ScatterPlotItem(
        centers[:, 0], centers[:, 1], symbol="+", size=8, pen=pg.mkPen(color), brush=None,
        name=f"{label} ({len(polygons)})",
    )
    plot.addItem(markers)


def build_window(path, layer=None, cell=None):
    """Create the viewer window (without starting the Qt event loop)."""
    layout = load_layout(path)
    layers = [layer] if layer else list_layers(layout)
    print(f"Database unit: {layout.dbu * 1000} nm")
    print(f"Layers: {list_layers(layout)}")

    win = pg.GraphicsLayoutWidget(title=f"Design viewer - {Path(path).name}")
    cursor_label = win.addLabel("x = -, y = -", row=0, col=0, justify="left")
    plot = win.addPlot(row=1, col=0)
    plot.setAspectLocked(True)  # 1 nm in x looks the same as 1 nm in y
    plot.showGrid(x=True, y=True, alpha=0.3)
    # Data is in nm. Declaring the axis unit as metres with a 1e-9 scale lets
    # pyqtgraph pick the right SI prefix (nm, um, mm) as you zoom.
    for side, name in (("bottom", "x"), ("left", "y")):
        plot.setLabel(side, name, units="m")
        plot.getAxis(side).setScale(1e-9)
    plot.addLegend()

    for i, (lay, dt) in enumerate(layers):
        polygons = read_polygons(layout, lay, dt, cell)
        print(f"Layer {lay}/{dt}: {len(polygons)} polygons")
        if polygons:
            add_layer(plot, polygons, pg.intColor(i, hues=8), f"{lay}/{dt}")
    plot.autoRange()

    def show_cursor(scene_pos):
        pos = plot.vb.mapSceneToView(scene_pos)
        cursor_label.setText(f"x = {pos.x():.1f} nm, y = {pos.y():.1f} nm")

    plot.scene().sigMouseMoved.connect(show_cursor)
    win.resize(900, 750)
    return win


def main():
    parser = argparse.ArgumentParser(description="Interactive viewer for OASIS files.")
    parser.add_argument("path", help="Path to the .oas file")
    parser.add_argument("--layer", type=parse_layer, help="Show only this layer, as L/D (e.g. 1/0)")
    parser.add_argument("--cell", help="Top cell name (needed if the file has several top cells)")
    args = parser.parse_args()

    app = pg.mkQApp("Design viewer")
    win = build_window(args.path, args.layer, args.cell)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
