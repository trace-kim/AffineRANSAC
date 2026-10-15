import numpy as np

from affine_ransac.geometry.polygon import contact_sizes, polygon_area


def ellipse(center, a, b, n=2000):
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + a * np.cos(angle), center[1] + b * np.sin(angle)])


def box(center, w, h):
    x, y = center
    return np.array([(x - w / 2, y - h / 2), (x + w / 2, y - h / 2), (x + w / 2, y + h / 2), (x - w / 2, y + h / 2)])


def test_single_chord_through_the_centre_of_boxes_and_ellipses():
    outlines = [box((10, 20), 40, 30), ellipse((-50, 5), 25, 15)]
    sizes = contact_sizes(outlines, np.array([(10, 20), (-50, 5)]), roi_width=0, spacing=1)
    np.testing.assert_allclose(sizes[0], [1200, 40, 30])
    np.testing.assert_allclose(sizes[1, 0], np.pi * 25 * 15, rtol=1e-4)
    np.testing.assert_allclose(sizes[1, 1:], [50, 30], rtol=1e-5)


def test_roi_band_averages_chords_one_spacing_apart():
    # Ellipse: the horizontal chord at height y is 2a sqrt(1 - y^2 / b^2), the vertical one at x likewise.
    a, b = 30.0, 20.0
    sizes = contact_sizes([ellipse((0, 0), a, b, n=20000)], np.zeros((1, 2)), roi_width=10, spacing=2)
    offsets = np.array([-5, -3, -1, 1, 3, 5.0])  # 10 // 2 + 1 = 6 chords, centred on the centre
    np.testing.assert_allclose(sizes[0, 1], np.mean(2 * a * np.sqrt(1 - offsets ** 2 / b ** 2)), rtol=1e-5)
    np.testing.assert_allclose(sizes[0, 2], np.mean(2 * b * np.sqrt(1 - offsets ** 2 / a ** 2)), rtol=1e-5)


def test_box_edge_on_a_chord_line_and_lines_missing_the_outline():
    # Band 30 on a box 20 high: chords at -15..15 every 5; the ones at |y| > 10 miss, the edges count.
    sizes = contact_sizes([box((0, 0), 40, 20)], np.zeros((1, 2)), roi_width=30, spacing=5)
    assert sizes[0, 1] == 40
    assert np.isnan(contact_sizes([box((0, 0), 4, 4)], np.array([[100.0, 100.0]]), 0, 1)[0, 1:]).all()


def test_area_and_vertex_order_and_empty_input():
    square = box((3, 4), 10, 10)
    np.testing.assert_allclose(contact_sizes([square[::-1]], np.array([[3.0, 4.0]]), 0, 1), [[100, 10, 10]])
    assert polygon_area(square) == 100
    assert contact_sizes([], np.empty((0, 2)), 0, 1).shape == (0, 3)
