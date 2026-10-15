"""Read design patterns from an OASIS (.oas) file.

All returned coordinates are in nanometres (float64), in the design frame (y up),
exactly as stored in the file (no shift). Our per-tile .oas files are centred on
(0, 0) = the centre of the SEM image (tile-local frame, docs/SPEC.md §3).
The cell hierarchy is flattened, so every returned polygon is at its final position.
"""

from pathlib import Path

import klayout.db as kdb
import numpy as np

from affine_ransac.geometry.polygon import polygon_area, polygon_centroid  # noqa: F401 (also re-exported)


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


def layers_with_shapes(layout: kdb.Layout) -> list[tuple[int, int]]:
    """The (layer, datatype) pairs that hold at least one polygon, box or path (texts do not count), sorted."""
    found = []
    for info in layout.layer_infos():
        index = layout.find_layer(info.layer, info.datatype)
        if any(not kdb.Region(top.begin_shapes_rec(index)).is_empty() for top in layout.top_cells()):
            found.append((info.layer, info.datatype))
    return sorted(found)


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


def _merged_polygons(layout: kdb.Layout, layer: int, datatype: int, cell_name: str | None) -> list[kdb.Polygon]:
    """All shapes on (layer, datatype) merged: shapes that touch or overlap become one polygon."""
    return list(kdb.Region(_shape_iterator(layout, layer, datatype, cell_name)).merged().each())


def _area_centroid(polygon: kdb.Polygon) -> np.ndarray:
    """Area centroid (x, y) of a KLayout polygon in database units, holes subtracted."""
    if polygon.is_box():
        center = polygon.bbox().center()
        return np.array([center.x, center.y], dtype=np.float64)
    hull = _polygon_vertices(polygon)
    holes = [np.array([(p.x, p.y) for p in polygon.each_point_hole(i)], dtype=np.float64)
             for i in range(polygon.holes())]
    areas = [polygon_area(hull)] + [-polygon_area(h) for h in holes]
    centroids = [polygon_centroid(hull)] + [polygon_centroid(h) for h in holes]
    return np.average(centroids, axis=0, weights=areas)


def read_contacts(
    layout: kdb.Layout,
    layer: int,
    datatype: int,
    cell_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Centre and size of every contact on (layer, datatype). Use this in the pipeline.

    A contact may be stored as several fractures (trapezoids, boxes) that touch or overlap, so
    the shapes are merged first (KLayout boolean OR): one merged pattern = one contact.

    Returns:
        centers: (N, 2) contact centres in nm (area centroid of the merged pattern).
        sizes:   (N, 2) bounding-box width and height of the merged pattern in nm.

    Text shapes are skipped.
    """
    nm_per_dbu = layout.dbu * 1000.0  # layout.dbu is in micrometres
    merged = _merged_polygons(layout, layer, datatype, cell_name)
    if not merged:
        return np.empty((0, 2)), np.empty((0, 2))

    centers = np.array([_area_centroid(polygon) for polygon in merged])
    sizes = np.array([(p.bbox().width(), p.bbox().height()) for p in merged], dtype=np.float64)
    return centers * nm_per_dbu, sizes * nm_per_dbu


def read_contacts_tone_reversed(
    layout: kdb.Layout,
    layer: int,
    datatype: int,
    cell_name: str | None = None,
    frame_nm: tuple[float, float] | None = None,
    drop_border: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Like read_contacts, for a TONE-REVERSED file: the drawn shapes are the area
    AROUND the holes, so the contacts are the empty space between them.

    Empty space = frame minus the drawn shapes (boolean operation in KLayout).
    Each separate empty region is one contact; centre = its area centroid.

    Args:
        frame_nm: (width, height) of the frame, centred at (0, 0), e.g. the SEM FOV.
            If None, the bounding box of the drawn shapes is used.
        drop_border: drop empty regions touching the frame edge (contacts cut off
            by the frame, or gaps between the drawn area and the frame).

    Returns:
        centers: (N, 2) in nm; sizes: (N, 2) bounding-box width and height in nm.
    """
    nm_per_dbu = layout.dbu * 1000.0  # layout.dbu is in micrometres
    regions = _reversed_regions(layout, layer, datatype, cell_name, frame_nm, drop_border)
    if not regions:
        return np.empty((0, 2)), np.empty((0, 2))
    centers = [polygon_centroid(_polygon_vertices(region)) for region in regions]
    sizes = [(region.bbox().width(), region.bbox().height()) for region in regions]
    return np.array(centers) * nm_per_dbu, np.array(sizes, dtype=np.float64) * nm_per_dbu


def _reversed_regions(layout: kdb.Layout, layer: int, datatype: int, cell_name: str | None,
                      frame_nm: tuple[float, float] | None, drop_border: bool) -> list[kdb.Polygon]:
    """The contacts of a tone-reversed file: frame minus the drawn shapes, one polygon per contact
    (arguments as read_contacts_tone_reversed)."""
    nm_per_dbu = layout.dbu * 1000.0
    drawn = kdb.Region(_shape_iterator(layout, layer, datatype, cell_name))  # flattened; texts ignored
    if frame_nm is None:
        frame = drawn.bbox()
    else:
        half_w = round(frame_nm[0] / 2 / nm_per_dbu)
        half_h = round(frame_nm[1] / 2 / nm_per_dbu)
        frame = kdb.Box(-half_w, -half_h, half_w, half_h)

    regions = []
    for region in (kdb.Region(frame) - drawn).each():
        box = region.bbox()
        on_border = (box.left <= frame.left or box.right >= frame.right
                     or box.bottom <= frame.bottom or box.top >= frame.top)
        if not (drop_border and on_border):
            regions.append(region)
    return regions


def contact_outlines(
    layout: kdb.Layout,
    layer: int,
    datatype: int,
    tone_reversed: bool = False,
    frame_nm: tuple[float, float] | None = None,
) -> list[np.ndarray]:
    """Outline of every contact, (M, 2) vertices in nm, in the order of the centres of read_contacts
    (tone_reversed False) or read_contacts_tone_reversed (True, with frame_nm; border contacts
    dropped). For measuring a contact's area and CDs (geometry.polygon.contact_sizes). The outer hull
    only: a contact has no holes."""
    nm_per_dbu = layout.dbu * 1000.0
    if tone_reversed:
        polygons = _reversed_regions(layout, layer, datatype, None, frame_nm, drop_border=True)
    else:
        polygons = _merged_polygons(layout, layer, datatype, None)
    return [_polygon_vertices(p) * nm_per_dbu for p in polygons]


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
    OASIS cannot store holes: a polygon with holes is stored as one outline with
    zero-width cut lines into each hole, so it is returned (and drawn) that way.
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


def list_shapes(
    layout: kdb.Layout,
    layer: int,
    datatype: int,
    cell_name: str | None = None,
) -> list[dict]:
    """Every shape on (layer, datatype) as it is stored, for inspecting a file.

    One dict per shape (texts skipped), hierarchy flattened:
        kind: "box", "polygon", "path"; cell: name of the cell the shape is defined in;
        vertices: (N, 2) outer hull in nm, placed; area_nm2: exact area (holes removed).
    """
    nm_per_dbu = layout.dbu * 1000.0
    shapes = []

    it = _shape_iterator(layout, layer, datatype, cell_name)
    while not it.at_end():
        shape = it.shape()
        polygon = shape.polygon  # None for texts
        if polygon is not None:
            placed = polygon.transformed(it.trans())
            kind = "box" if shape.is_box() else "path" if shape.is_path() else "polygon"
            shapes.append({
                "kind": kind,
                "cell": layout.cell(it.cell_index()).name,
                "vertices": _polygon_vertices(placed) * nm_per_dbu,
                "area_nm2": placed.area() * nm_per_dbu**2,
            })
        it.next()
    return shapes


def merge_shapes(
    layout: kdb.Layout,
    layer: int,
    datatype: int,
    cell_name: str | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Union of all shapes on (layer, datatype): overlapping or touching shapes become one.

    Returns:
        merged: list of (N, 2) outer-hull vertices in nm, one per merged pattern.
        labels: (number of shapes,) index into merged for each shape, in list_shapes order;
            -1 for a shape with no area (it is not part of any merged polygon).
    """
    nm_per_dbu = layout.dbu * 1000.0
    merged = _merged_polygons(layout, layer, datatype, cell_name)

    # Each shape lies inside exactly one merged polygon: look among those whose bounding box
    # contains the shape's bounding box, and check by overlap only if there are several.
    boxes = np.array([(b.left, b.bottom, b.right, b.top) for b in (m.bbox() for m in merged)]).reshape(-1, 4)
    labels = []
    it = _shape_iterator(layout, layer, datatype, cell_name)
    while not it.at_end():
        polygon = it.shape().polygon  # None for texts
        if polygon is not None:
            placed = polygon.transformed(it.trans())
            b = placed.bbox()
            inside = ((boxes[:, 0] <= b.left) & (boxes[:, 1] <= b.bottom)
                      & (boxes[:, 2] >= b.right) & (boxes[:, 3] >= b.top))
            candidates = np.flatnonzero(inside)
            if len(candidates) > 1:
                candidates = [j for j in candidates
                              if not (kdb.Region(merged[j]) & kdb.Region(placed)).is_empty()]
            labels.append(candidates[0] if len(candidates) else -1)
        it.next()

    vertices = [_polygon_vertices(m) * nm_per_dbu for m in merged]
    return vertices, np.array(labels, dtype=int)


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
