import numpy as np

from affine_ransac.mosaic import build_mosaic


def test_tiles_land_at_their_positions_and_overlaps_average():
    s = 2.0  # nm per pixel
    a = np.full((4, 6), 10, np.uint8)
    b = np.full((4, 6), 30, np.uint8)
    # b is 4 px (8 nm) to the right of a: they overlap in 2 columns.
    mosaic, extent = build_mosaic([a, b], np.array([[0.0, 0.0], [8.0, 0.0]]), s)

    assert mosaic.shape == (4, 10)
    np.testing.assert_allclose(mosaic[:, :4], 10)
    np.testing.assert_allclose(mosaic[:, 4:6], 20)  # overlap: average
    np.testing.assert_allclose(mosaic[:, 6:], 30)
    # a spans 6 px * 2 nm centred at 0 -> x from -6; b ends at 8 + 6 = 14. y from -4 to 4.
    np.testing.assert_allclose(extent, (-6, 14, -4, 4))


def test_sub_pixel_placement():
    s = 1.0
    tile = np.zeros((5, 5), np.uint8)
    tile[2, 2] = 100  # a bright pixel at the tile centre
    ref = np.zeros((5, 5), np.uint8)
    # Two tiles; the second centred 0.5 px right of a pixel grid of the first.
    mosaic, _ = build_mosaic([ref, tile], np.array([[0.0, 0.0], [10.5, 0.0]]), s)
    row = mosaic[2]
    peak = np.nansum(np.arange(len(row)) * np.nan_to_num(row)) / np.nansum(np.nan_to_num(row))
    assert abs(peak - 12.5) < 1e-3  # first tile's centre pixel is mosaic x = 2; +10.5 px
