"""Read design patterns from an OASIS (.oas) file.

All returned coordinates are in nanometres (float64), in the design frame (y up).
The cell hierarchy is flattened, so every returned polygon is at its final position.
"""

from pathlib import Path

import klayout.db as kdb
import numpy as np


def load_layout(path: str | Path) -> kdb.Layout:
    """Load an OASIS (or GDS) file into a KLayout Layout object."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    layout = kdb.Layout()
    layout.read(str(path))
    return layout


def list_layers(layout: kdb.Layout) -> list[tuple[int, int]]:
    """Return the (layer, datatype) pairs present in the layout, sorted."""
    return sorted((info.layer, info.datatype) for info in layout.layer_infos())


def get_top_cell(layout: kdb.Layout, name: str | None = None) -> kdb.Cell:
    """Return the named cell, or the single top cell if no name is given."""
    if name is not None:
        cell = layout.cell(name)
        if cell is None:
            raise ValueError(f"Cell '{name}' not found in layout")
        return cell

    tops = layout.top_cells()
    if len(tops) != 1:
        names = [c.name for c in tops]
        raise ValueError(f"Layout has {len(tops)} top cells {names}; pass a cell name")
    return tops[0]


def _shape_iterator(layout: kdb.Layout, layer: int, datatype: int, cell_name: str | None):
    """Iterator over all shapes on (layer, datatype), flattening the hierarchy."""
    layer_index = layout.find_layer(layer, datatype)
    if layer_index is None:
        raise ValueError(f"Layer {layer}/{datatype} not found; available: {list_layers(layout)}")
    return get_top_cell(layout, cell_name).begin_shapes_rec(layer_index)


def _polygon_vertices(polygon: kdb.Polygon) -> np.ndarray:
    """Outer hull vertices of a KLayout polygon as an (N, 2) array in database units."""
    return np.array([(p.x, p.y) for p in polygon.each_point_hull()], dtype=np.float64)


def read_contacts(
    layout: kdb.Layout,
    layer: int,
    datatype: int,
    cell_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Centre and size of every contact on (layer, datatype). Use this in the pipeline.

    Returns:
        centers: (N, 2) contact centres in nm (area centroid).
        sizes:   (N, 2) bounding-box width and height in nm.

    Fast path: for a box, the centroid is simply the bounding-box centre, so no
    vertices are extracted. Other polygons (e.g. OPC-shaped contacts) use the
    exact area centroid. Text shapes are skipped.
    """
    nm_per_dbu = layout.dbu * 1000.0  # layout.dbu is in micrometres
    centers, sizes = [], []

    it = _shape_iterator(layout, layer, datatype, cell_name)
    while not it.at_end():
        shape = it.shape()
        if shape.is_box() or shape.is_polygon() or shape.is_simple_polygon() or shape.is_path():
            box = shape.bbox().transformed(it.trans())
            sizes.append((box.width(), box.height()))
            if shape.is_box():
                center = box.center()
                centers.append((center.x, center.y))
            else:
                vertices = _polygon_vertices(shape.polygon.transformed(it.trans()))
                centers.append(tuple(polygon_centroid(vertices)))
        it.next()

    if not centers:
        return np.empty((0, 2)), np.empty((0, 2))
    return (
        np.array(centers, dtype=np.float64) * nm_per_dbu,
        np.array(sizes, dtype=np.float64) * nm_per_dbu,
    )


def read_polygons(
    layout: kdb.Layout,
    layer: int,
    datatype: int,
    cell_name: str | None = None,
) -> list[np.ndarray]:
    """Return every polygon on (layer, datatype) as an (N, 2) array of vertices in nm.

    Used for drawing (viewer). It is slower than read_contacts because it extracts
    every vertex, so the pipeline should use read_contacts instead.
    Boxes and paths are converted to polygons. Text shapes are skipped.
    Holes inside polygons are ignored (only the outer hull is returned).
    """
    nm_per_dbu = layout.dbu * 1000.0  # layout.dbu is in micrometres
    polygons = []

    it = _shape_iterator(layout, layer, datatype, cell_name)
    while not it.at_end():
        shape_polygon = it.shape().polygon  # None for texts
        if shape_polygon is not None:
            placed = shape_polygon.transformed(it.trans())
            polygons.append(_polygon_vertices(placed) * nm_per_dbu)
        it.next()
    return polygons


def polygon_centroid(vertices: np.ndarray) -> np.ndarray:
    """Area centroid (x, y) of a simple polygon, using the shoelace formula."""
    x, y = vertices[:, 0], vertices[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0
    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return np.array([cx, cy])


def contact_centers(polygons: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Feature point and size for each contact polygon.

    Returns:
        centers: (N, 2) polygon centroids in nm.
        sizes:   (N, 2) bounding-box width and height in nm.
    """
    if not polygons:
        return np.empty((0, 2)), np.empty((0, 2))
    centers = np.array([polygon_centroid(p) for p in polygons])
    sizes = np.array([p.max(axis=0) - p.min(axis=0) for p in polygons])
    return centers, sizes
