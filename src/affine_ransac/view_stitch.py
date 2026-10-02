"""Interactive stitched view of many tiles (pyqtgraph, OpenGL viewport).

Shows, in mask coordinates: every tile's SEM image, the design (.oas) contours and centres,
the SEM contours and centres from Otsu and from edge refinement, the nominal overlap boxes, and
the contacts flagged as outliers by the pairwise overlap fits.

- Placement: "Nominal" puts each tile at its metadata centre; "Stitched" adds the tile's
  stitching correction. SEM items move, the design stays. Switch back and forth to compare.
- Layers and tiles can be hidden with the check boxes on the right (hide a tile to see the
  tile beneath it in an overlap).
- Mouse: drag to pan, wheel to zoom, right-click for more options.

Everything is drawn relative to a local origin (the middle of the tiles, in whole µm): the
OpenGL viewport works in float32, which would round mask coordinates (~10⁷ nm) to several nm.
The axes show x − x0, y − y0 in µm; the cursor label shows absolute mask µm.

Data comes from affine_ransac.pipeline (process_tile, stitch_tiles); see
notebooks/stitch_viewer.ipynb.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from affine_ransac.geometry.frames import pixel_to_tile_nm
from affine_ransac.pipeline import StitchResult, TileResult
from affine_ransac.view_design import polygons_to_path

IMAGES = "SEM images"
DESIGN = "Design contours (.oas)"
DESIGN_CENTRES = "Design centres"
OTSU = "Otsu contours"
OTSU_CENTRES = "Otsu centres"
REFINED = "Refined contours"
REFINED_CENTRES = "Refined centres"
OUTLIERS = "Overlap outliers"
BOXES = "Overlap boxes (nominal)"

COLORS = {
    IMAGES: "#c0c0c0", DESIGN: "#3c8cff", DESIGN_CENTRES: "#3c8cff", OTSU: "#33cc33",
    OTSU_CENTRES: "#33cc33", REFINED: "#ff4040", REFINED_CENTRES: "#ff4040",
    OUTLIERS: "#ffa500", BOXES: "#ffff00",
}
SEM_LAYERS = {IMAGES, OTSU, OTSU_CENTRES, REFINED, REFINED_CENTRES, OUTLIERS}  # moved by placement


def image_item(tile: TileResult) -> pg.ImageItem:
    """The tile's image in tile-local nm (y up): pixel centres at pixel_to_tile_nm positions."""
    item = pg.ImageItem(tile.image, axisOrder="row-major", levels=(0, 255))
    # Item (0, 0) is the outer corner of pixel (0, 0), i.e. pixel (-0.5, -0.5); 1 item unit = 1 px.
    corner = pixel_to_tile_nm([[-0.5, -0.5]], tile.image.shape, tile.pixel_size_nm)[0]
    s = tile.pixel_size_nm
    item.setTransform(QtGui.QTransform(s, 0, 0, -s, corner[0], corner[1]))  # y flips: pixel down, nm up
    return item


def path_item(polygons: list[np.ndarray], color, style=QtCore.Qt.PenStyle.SolidLine):
    item = QtWidgets.QGraphicsPathItem(polygons_to_path(polygons))
    item.setPen(pg.mkPen(color, style=style))
    return item


def centres_item(points: np.ndarray, color, symbol="+", size=8):
    return pg.ScatterPlotItem(points[:, 0], points[:, 1], symbol=symbol, size=size, pen=pg.mkPen(color), brush=None)


def outlier_indices(stitch: StitchResult, n_tiles: int) -> list[np.ndarray]:
    """Per tile, the indices of its contacts that some pair fit flagged as an outlier."""
    found = [[] for _ in range(n_tiles)]
    for p in stitch.pairs:
        outliers = ~p.fit.inliers
        found[p.i].extend(p.ia[outliers])
        found[p.j].extend(p.ib[outliers])
    return [np.unique(np.asarray(f, dtype=int)) for f in found]


class StitchViewer(QtWidgets.QWidget):
    def __init__(
        self,
        tiles: list[TileResult],
        tile_ids: list[str],
        design_polygons: list[list[np.ndarray]],
        stitch: StitchResult,
        refined: bool = True,
        use_opengl: bool = True,
    ):
        """tiles: from process_tile; design_polygons: per tile, read_polygons() of its .oas
        (tile-local nm, [] if none); stitch: from stitch_tiles; refined: whether stitch_tiles was
        given the refined (True) or the Otsu centres, for the outlier markers."""
        super().__init__()
        self.setWindowTitle(f"Stitch viewer - {len(tiles)} tiles")
        self.tiles, self.corrections = tiles, stitch.corrections
        centers = np.array([t.center_nm for t in tiles])
        self.origin = np.round(centers.mean(axis=0), -3)  # whole µm, see module docstring
        self.items = []  # (layer, tile index or None, graphics item)

        self.plot = pg.PlotWidget()
        if use_opengl:
            self.plot.useOpenGL(True)
        self.plot.setAspectLocked(True)
        x0, y0 = self.origin / 1000
        self.plot.setTitle(f"origin x0 = {x0:.0f} µm, y0 = {y0:.0f} µm")
        for side, name in (("bottom", "x − x0 (µm)"), ("left", "y − y0 (µm)")):
            self.plot.setLabel(side, name)
            self.plot.getAxis(side).setScale(1e-3)  # data is nm

        outliers = outlier_indices(stitch, len(tiles))
        for k, tile in enumerate(tiles):
            self._add_tile(k, tile, design_polygons[k], outliers[k], refined)
        boxes = [np.array([[left, bottom], [right, bottom], [right, top], [left, top]]) - self.origin
                 for left, right, bottom, top in (p.box for p in stitch.pairs)]
        if boxes:
            self._add(BOXES, None, path_item(boxes, COLORS[BOXES], QtCore.Qt.PenStyle.DashLine))

        self.cursor_label = QtWidgets.QLabel("x = -, y = - (mask µm)")
        self.plot.scene().sigMouseMoved.connect(self._show_cursor)

        layout = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.cursor_label)
        left.addWidget(self.plot)
        layout.addLayout(left, stretch=1)
        layout.addWidget(self._side_panel(tile_ids))

        self.set_placement(stitched=False)
        self.update_visibility()
        self.plot.autoRange()
        self.resize(1400, 950)

    def _add(self, layer, k, item):
        item.setZValue(0 if layer == IMAGES else 1)
        self.plot.addItem(item)
        self.items.append((layer, k, item))

    def _add_tile(self, k, tile: TileResult, design_polygons, outliers, refined):
        def local(xy_px):
            return pixel_to_tile_nm(xy_px, tile.image.shape, tile.pixel_size_nm)

        self._add(IMAGES, k, image_item(tile))
        if design_polygons:
            self._add(DESIGN, k, path_item(design_polygons, COLORS[DESIGN]))
        if len(tile.design_centers):
            self._add(DESIGN_CENTRES, k, centres_item(tile.design_centers, COLORS[DESIGN_CENTRES], "x"))
        for layer, centres_layer, found in ((OTSU, OTSU_CENTRES, tile.otsu), (REFINED, REFINED_CENTRES, tile.refined)):
            if found.contours:
                self._add(layer, k, path_item([local(c) for c in found.contours], COLORS[layer]))
                self._add(centres_layer, k, centres_item(local(found.centers), COLORS[centres_layer]))
        if len(outliers):
            found = tile.refined if refined else tile.otsu
            self._add(OUTLIERS, k, centres_item(local(found.centers[outliers]), COLORS[OUTLIERS], "o", 16))

    def _side_panel(self, tile_ids):
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(260)
        column = QtWidgets.QVBoxLayout(panel)

        column.addWidget(QtWidgets.QLabel("Placement"))
        self.nominal_button = QtWidgets.QRadioButton("Nominal (metadata centres)")
        self.stitched_button = QtWidgets.QRadioButton("Stitched (corrected)")
        self.nominal_button.setChecked(True)
        self.stitched_button.toggled.connect(lambda on: self.set_placement(stitched=on))
        column.addWidget(self.nominal_button)
        column.addWidget(self.stitched_button)

        column.addWidget(QtWidgets.QLabel("Layers"))
        self.layer_boxes = {}
        for layer in COLORS:
            box = QtWidgets.QCheckBox(layer)
            box.setChecked(True)
            box.setStyleSheet(f"color: {COLORS[layer]}")
            box.toggled.connect(self.update_visibility)
            column.addWidget(box)
            self.layer_boxes[layer] = box

        column.addWidget(QtWidgets.QLabel("Tiles"))
        self.tile_list = QtWidgets.QListWidget()
        for tile_id in tile_ids:
            entry = QtWidgets.QListWidgetItem(tile_id)
            entry.setCheckState(QtCore.Qt.CheckState.Checked)
            self.tile_list.addItem(entry)
        self.tile_list.itemChanged.connect(self.update_visibility)
        column.addWidget(self.tile_list, stretch=1)
        return panel

    def tile_shown(self, k: int) -> bool:
        return self.tile_list.item(k).checkState() == QtCore.Qt.CheckState.Checked

    def update_visibility(self, *_):
        for layer, k, item in self.items:
            item.setVisible(self.layer_boxes[layer].isChecked() and (k is None or self.tile_shown(k)))

    def set_placement(self, stitched: bool):
        """Move every tile's items to its nominal or its stitched (corrected) position."""
        for layer, k, item in self.items:
            if k is None:
                continue
            position = self.tiles[k].center_nm - self.origin
            if stitched and layer in SEM_LAYERS:
                position = position + self.corrections[k]
            item.setPos(*position)

    def _show_cursor(self, scene_pos):
        pos = self.plot.getPlotItem().vb.mapSceneToView(scene_pos)
        x, y = (np.array([pos.x(), pos.y()]) + self.origin) / 1000
        self.cursor_label.setText(f"x = {x:.4f} µm, y = {y:.4f} µm (mask)")
