"""Interactive viewer for SEM images (pyqtgraph).

Usage:
    python -m affine_ransac.view_sem IMAGE.jpg
    python -m affine_ransac.view_sem IMAGE.jpg --detect                    # overlay contacts
    python -m affine_ransac.view_sem IMAGE.jpg --detect --databar-rows 64  # ignore data bar

Mouse: drag to pan, wheel to zoom, right-click for more options.
Adjust contrast with the histogram on the right.
The label above the image shows the pixel under the cursor: u (column), v (row,
pointing down, as in the image file) and its grey value.
"""

import argparse
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from affine_ransac.features.contact import DetectedContacts, detect_contacts
from affine_ransac.io.sem_image import crop_databar, load_sem_image
from affine_ransac.view_design import polygons_to_path

# Our arrays are image[row, col]; tell pyqtgraph so it doesn't transpose them.
pg.setConfigOptions(imageAxisOrder="row-major")


def pixel_at(image: np.ndarray, x: float, y: float):
    """Return (u, v, value) for the pixel containing view point (x, y), or None if outside.

    Pixel centres are at integer coordinates, so pixel u covers x in [u - 0.5, u + 0.5).
    """
    u, v = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
    rows, cols = image.shape
    if 0 <= u < cols and 0 <= v < rows:
        return u, v, int(image[v, u])
    return None


def add_contacts(plot, contacts: DetectedContacts):
    """Overlay detected contact contours (green) and centres (red '+')."""
    if not contacts.contours:
        return
    outlines = QtWidgets.QGraphicsPathItem(polygons_to_path(contacts.contours))
    outlines.setPen(pg.mkPen("g"))
    plot.addItem(outlines)
    plot.addItem(pg.ScatterPlotItem(
        contacts.centers[:, 0], contacts.centers[:, 1],
        symbol="+", size=10, pen=pg.mkPen("r"), brush=None,
    ))


def build_window(path, contacts: DetectedContacts | None = None):
    """Create the viewer window (without starting the Qt event loop).

    If contacts are given, their contours and centres are drawn over the image.
    """
    image = load_sem_image(path)
    rows, cols = image.shape
    print(f"{Path(path).name}: {cols} x {rows} px, {image.dtype}, "
          f"values {image.min()}..{image.max()}")

    plot = pg.PlotItem()
    plot.invertY(True)  # row 0 at the top, like the image file
    plot.setAspectLocked(True)
    plot.setLabel("bottom", "u (column, px)")
    plot.setLabel("left", "v (row, px)")

    image_view = pg.ImageView(view=plot)
    # Shift by half a pixel so pixel centres land on integer coordinates (SPEC §3).
    image_view.setImage(image, pos=(-0.5, -0.5), autoRange=True, autoLevels=True)
    if contacts is not None:
        add_contacts(plot, contacts)

    cursor_label = QtWidgets.QLabel("u = -, v = -, value = -")

    def show_cursor(scene_pos):
        pos = plot.vb.mapSceneToView(scene_pos)
        hit = pixel_at(image, pos.x(), pos.y())
        if hit:
            cursor_label.setText(f"u = {hit[0]}, v = {hit[1]}, value = {hit[2]}")

    plot.scene().sigMouseMoved.connect(show_cursor)

    win = QtWidgets.QWidget()
    win.setWindowTitle(f"SEM viewer - {Path(path).name}")
    layout = QtWidgets.QVBoxLayout(win)
    layout.addWidget(cursor_label)
    layout.addWidget(image_view)
    win.image_view = image_view  # kept for tests / later overlays
    win.resize(1100, 850)
    return win


def main():
    parser = argparse.ArgumentParser(description="Interactive viewer for SEM images.")
    parser.add_argument("path", help="Path to the image file (e.g. .jpg)")
    parser.add_argument("--detect", action="store_true", help="Detect contacts and overlay them")
    parser.add_argument("--databar-rows", type=int, default=0,
                        help="Rows of data bar at the bottom to ignore when detecting")
    parser.add_argument("--bright", action="store_true", help="Contacts are bright, not dark")
    args = parser.parse_args()

    contacts = None
    if args.detect:
        image = crop_databar(load_sem_image(args.path), args.databar_rows)
        contacts = detect_contacts(image, dark_contacts=not args.bright)
        print(f"Detected {len(contacts.centers)} contacts (Otsu threshold {contacts.threshold:.0f})")

    app = pg.mkQApp("SEM viewer")
    win = build_window(args.path, contacts)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
