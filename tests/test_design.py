import klayout.db as kdb
import numpy as np
import pytest

from affine_ransac.io.design import (
    contact_centers,
    list_layers,
    list_shapes,
    load_layout,
    merge_shapes,
    polygon_centroid,
    read_contacts,
    read_contacts_tone_reversed,
    read_polygons,
)
from sample_design import write_contact_array

# Expected centres of the default 3x2 array: pitch 200 nm in x, 300 nm in y.
EXPECTED_CENTERS = np.array([[x, y] for y in (0, 300) for x in (0, 200, 400)], dtype=float)


def sort_rows(points):
    return points[np.lexsort((points[:, 0], points[:, 1]))]


def test_reads_flattened_contact_array(tmp_path):
    path = tmp_path / "a.oas"
    write_contact_array(path)

    polygons = read_polygons(load_layout(path), 1, 0)
    centers, sizes = contact_centers(polygons)

    assert len(polygons) == 6  # the text label on 1/0 is skipped
    np.testing.assert_allclose(sort_rows(centers), EXPECTED_CENTERS)
    np.testing.assert_allclose(sizes, 100.0)


def test_coordinates_are_nm_regardless_of_dbu(tmp_path):
    path = tmp_path / "fine_dbu.oas"
    write_contact_array(path, dbu_nm=0.25)

    centers, sizes = contact_centers(read_polygons(load_layout(path), 1, 0))

    np.testing.assert_allclose(sort_rows(centers), EXPECTED_CENTERS)
    np.testing.assert_allclose(sizes, 100.0)


def test_read_contacts_matches_polygon_path(tmp_path):
    path = tmp_path / "a.oas"
    write_contact_array(path, dbu_nm=0.25)
    layout = load_layout(path)

    centers, sizes = read_contacts(layout, 1, 0)

    assert len(centers) == 6  # text label skipped
    np.testing.assert_allclose(sort_rows(centers), EXPECTED_CENTERS)
    np.testing.assert_allclose(sizes, 100.0)


def test_read_contacts_non_box_uses_area_centroid(tmp_path):
    # An L-shaped contact (not a box) placed via a rotated + mirrored instance.
    layout = kdb.Layout()
    layout.dbu = 0.001  # 1 nm
    top, child = layout.create_cell("TOP"), layout.create_cell("L")
    l_shape = [(0, 0), (200, 0), (200, 100), (100, 100), (100, 200), (0, 200)]
    child.shapes(layout.layer(1, 0)).insert(kdb.Polygon([kdb.Point(x, y) for x, y in l_shape]))
    # Rotate 90 degrees, mirror, then shift by (1000, 500).
    top.insert(kdb.CellInstArray(child.cell_index(), kdb.Trans(1, True, 1000, 500)))
    path = tmp_path / "l.oas"
    layout.write(str(path))

    centers, sizes = read_contacts(load_layout(path), 1, 0)

    # Unplaced centroid is (500/6, 500/6) nm. Mirror about x then rotate 90 deg maps (x, y) -> (y, x).
    np.testing.assert_allclose(centers, [[1000 + 500 / 6, 500 + 500 / 6]])
    np.testing.assert_allclose(sizes, [[200, 200]])
    # Same answer as the slow polygon path.
    np.testing.assert_allclose(centers, contact_centers(read_polygons(load_layout(path), 1, 0))[0])


def test_layer_selection(tmp_path):
    path = tmp_path / "a.oas"
    write_contact_array(path)
    layout = load_layout(path)

    assert list_layers(layout) == [(1, 0), (2, 0)]
    assert len(read_polygons(layout, 2, 0)) == 1
    with pytest.raises(ValueError, match="not found"):
        read_polygons(layout, 9, 0)


def test_multiple_top_cells_need_a_name(tmp_path):
    layout = kdb.Layout()
    layer = layout.layer(1, 0)
    for name in ("A", "B"):
        layout.create_cell(name).shapes(layer).insert(kdb.Box(0, 0, 10, 10))
    path = tmp_path / "two_tops.oas"
    layout.write(str(path))

    loaded = load_layout(path)
    with pytest.raises(ValueError, match="top cells"):
        read_polygons(loaded, 1, 0)
    assert len(read_polygons(loaded, 1, 0, cell_name="B")) == 1


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_layout(tmp_path / "nope.oas")


def test_polygon_centroid_of_l_shape():
    # L-shape: 2x1 bar along the bottom plus a 1x1 square on its left end.
    l_shape = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
    # Bar (area 2, centroid (1, 0.5)) + square (area 1, centroid (0.5, 1.5))
    np.testing.assert_allclose(polygon_centroid(l_shape), [5 / 6, 5 / 6])


def write_tone_reversed(path, polygons, dbu_nm=1.0):
    """Write the given klayout polygons/boxes (in nm) on layer 1/0 of a TOP cell."""
    layout = kdb.Layout()
    layout.dbu = dbu_nm / 1000.0
    shapes = layout.create_cell("TOP").shapes(layout.layer(1, 0))
    for polygon in polygons:
        shapes.insert(polygon)
    layout.write(str(path))
    return load_layout(path)


def test_tone_reversed_single_polygon_with_holes(tmp_path):
    # Drawn: a 1000 x 1000 nm square centred at (0, 0) with two rectangular holes cut out.
    drawn = kdb.Polygon(kdb.Box(-500, -500, 500, 500))
    drawn.insert_hole(kdb.Box(-300, -100, -100, 100))  # centre (-200, 0), 200 x 200
    drawn.insert_hole(kdb.Box(100, -50, 250, 100))  # centre (175, 25), 150 x 150
    layout = write_tone_reversed(tmp_path / "rev.oas", [drawn])

    centers, sizes = read_contacts_tone_reversed(layout, 1, 0)

    np.testing.assert_allclose(sort_rows(centers), [[-200, 0], [175, 25]])
    np.testing.assert_allclose(sizes[np.lexsort((centers[:, 0], centers[:, 1]))], [[200, 200], [150, 150]])
    # The normal reader sees the drawn shape itself as ONE "contact": the wrong answer here.
    assert len(read_contacts(layout, 1, 0)[0]) == 1


def test_tone_reversed_separate_shapes_and_border(tmp_path):
    # Drawn: four separate bars enclosing a 200 x 200 nm hole centred at (100, 0), inside
    # a 1200 x 1000 nm frame. The empty space between the bars and the frame edge is one
    # region touching the border.
    # Box coordinates are in database units; with dbu = 0.5 nm that is 2 units per nm.
    bars = [
        kdb.Box(-600, -800, 0, 800),     # left:   x -300..0,   y -400..400 nm
        kdb.Box(400, -800, 1000, 800),   # right:  x  200..500, y -400..400 nm
        kdb.Box(0, -800, 400, -200),     # bottom: x    0..200, y -400..-100 nm
        kdb.Box(0, 200, 400, 800),       # top:    x    0..200, y  100..400 nm
    ]
    layout = write_tone_reversed(tmp_path / "bars.oas", bars, dbu_nm=0.5)

    centers, sizes = read_contacts_tone_reversed(layout, 1, 0, frame_nm=(1200, 1000))
    np.testing.assert_allclose(centers, [[100, 0]])  # border region dropped
    np.testing.assert_allclose(sizes, [[200, 200]])

    centers, _ = read_contacts_tone_reversed(layout, 1, 0, frame_nm=(1200, 1000), drop_border=False)
    assert len(centers) == 2  # the hole + the border region


def test_oasis_stores_holes_as_cut_lines(tmp_path):
    # OASIS has no holes: a polygon with a hole comes back as ONE outline that runs
    # into the hole and back (a "keyhole"). read_polygons returns it as stored.
    drawn = kdb.Polygon(kdb.Box(-500, -500, 500, 500))
    drawn.insert_hole(kdb.Box(-100, -100, 100, 100))
    layout = write_tone_reversed(tmp_path / "rev.oas", [drawn])

    polygons = read_polygons(layout, 1, 0)
    assert len(polygons) == 1
    assert [-100, -100] in polygons[0].tolist()  # the hole's corner is on the outline


def write_fractured(path):
    """One 100 x 100 nm contact at (0, 0) as three OVERLAPPING boxes, one 100 x 60 nm contact at
    (300, 0) as two TOUCHING boxes, and a separate 50 x 50 nm contact at (600, 0) placed twice
    on top of itself through a child cell. Layer 1/0, dbu 1 nm, plus a text (skipped)."""
    layout = kdb.Layout()
    layout.dbu = 0.001
    top, child = layout.create_cell("TOP"), layout.create_cell("CHILD")
    shapes = top.shapes(layout.layer(1, 0))
    for box in [(-50, -50, 10, 50), (-20, -50, 50, 50), (-50, -10, 50, 10)]:
        shapes.insert(kdb.Box(*box))
    shapes.insert(kdb.Box(250, -30, 300, 30))
    shapes.insert(kdb.Box(300, -30, 350, 30))
    shapes.insert(kdb.Box(575, -25, 625, 25))
    shapes.insert(kdb.Text("label", kdb.Trans()))
    child.shapes(layout.layer(1, 0)).insert(kdb.Box(-25, -25, 25, 25))
    top.insert(kdb.CellInstArray(child.cell_index(), kdb.Trans(600, 0)))
    layout.write(str(path))
    return load_layout(path)


def test_list_shapes_reports_each_stored_shape(tmp_path):
    shapes = list_shapes(write_fractured(tmp_path / "f.oas"), 1, 0)

    assert len(shapes) == 7  # text skipped
    assert all(s["kind"] == "box" for s in shapes)
    assert sorted(s["cell"] for s in shapes) == ["CHILD"] + ["TOP"] * 6
    assert sorted(s["area_nm2"] for s in shapes)[-1] == 70 * 100  # box -20..50 x -50..50 nm
    child = next(s for s in shapes if s["cell"] == "CHILD")
    np.testing.assert_allclose(child["vertices"].min(axis=0), [575, -25])  # placed at (600, 0)


def test_merge_shapes_joins_overlapping_and_touching(tmp_path):
    layout = write_fractured(tmp_path / "f.oas")

    merged, labels = merge_shapes(layout, 1, 0)

    assert len(merged) == 3
    centers = sort_rows(np.array([polygon_centroid(m) for m in merged]))
    np.testing.assert_allclose(centers, [[0, 0], [300, 0], [600, 0]], atol=1e-9)
    # Labels follow list_shapes order (KLayout's order, not insertion order): each shape's
    # label is the merged pattern it lies in, so 3 + 2 + 2 shapes share 3 labels.
    shapes = list_shapes(layout, 1, 0)
    assert len(labels) == len(shapes) == 7
    pattern_x = np.array([polygon_centroid(m)[0] for m in merged])
    shape_x = np.array([s["vertices"][:, 0].mean() for s in shapes])
    nearest = np.abs(shape_x[:, None] - pattern_x[None, :]).argmin(axis=1)
    np.testing.assert_array_equal(labels, nearest)
    assert sorted(np.bincount(labels)) == [2, 2, 3]


def test_read_contacts_merges_fractures(tmp_path):
    # write_fractured: 3 overlapping boxes, 2 touching boxes, 2 copies of one box (via a cell).
    centers, sizes = read_contacts(write_fractured(tmp_path / "f.oas"), 1, 0)

    order = np.argsort(centers[:, 0])
    np.testing.assert_allclose(centers[order], [[0, 0], [300, 0], [600, 0]], atol=1e-9)
    np.testing.assert_allclose(sizes[order], [[100, 100], [100, 60], [50, 50]])


def test_read_contacts_merges_trapezoid_fractures(tmp_path):
    # An octagon-like contact centred at (40, -20) nm, stored as 3 touching trapezoids/box,
    # placed in a child cell; and a separate square 300 nm away. dbu 0.5 nm.
    layout = kdb.Layout()
    layout.dbu = 0.0005
    top, child = layout.create_cell("TOP"), layout.create_cell("C")
    shapes = child.shapes(layout.layer(1, 0))
    to_dbu = 2  # dbu per nm
    pieces = [
        [(-50, 20), (50, 20), (30, 50), (-30, 50)],      # top trapezoid
        [(-50, -20), (50, -20), (50, 20), (-50, 20)],    # middle box
        [(-30, -50), (30, -50), (50, -20), (-50, -20)],  # bottom trapezoid
    ]
    for piece in pieces:
        shapes.insert(kdb.Polygon([kdb.Point(x * to_dbu, y * to_dbu) for x, y in piece]))
    top.insert(kdb.CellInstArray(child.cell_index(), kdb.Trans(40 * to_dbu, -20 * to_dbu)))
    top.shapes(layout.layer(1, 0)).insert(kdb.Box(640, -40, 680, 0))  # 20 x 20 nm at (330, -10) nm
    path = tmp_path / "trap.oas"
    layout.write(str(path))

    centers, sizes = read_contacts(load_layout(path), 1, 0)

    order = np.argsort(centers[:, 0])
    np.testing.assert_allclose(centers[order], [[40, -20], [330, -10]], atol=1e-9)
    np.testing.assert_allclose(sizes[order], [[100, 100], [20, 20]])


def test_read_contacts_centroid_subtracts_holes(tmp_path):
    # A 100 x 100 nm square with a 40 x 40 nm hole off-centre (hole centre (20, 0)).
    # Area 10000 - 1600 = 8400; centroid x = (0 * 10000 - 20 * 1600) / 8400.
    drawn = kdb.Polygon(kdb.Box(-50, -50, 50, 50))
    drawn.insert_hole(kdb.Box(0, -20, 40, 20))
    layout = write_tone_reversed(tmp_path / "hole.oas", [drawn])

    centers, _ = read_contacts(layout, 1, 0)

    np.testing.assert_allclose(centers, [[-20 * 1600 / 8400, 0]], atol=1e-9)
