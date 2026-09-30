import klayout.db as kdb
import numpy as np
import pytest

from affine_ransac.io.design import (
    contact_centers,
    list_layers,
    load_layout,
    polygon_centroid,
    read_contacts,
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
