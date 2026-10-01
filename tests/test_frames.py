import numpy as np

from affine_ransac.geometry.frames import pixel_to_tile_nm


def test_pixel_to_tile_nm_centre_and_axes():
    shape = (101, 201)  # rows, cols -> centre pixel is (x=100, y=50)
    xy_px = [
        [100, 50],  # centre
        [110, 50],  # 10 px right
        [100, 40],  # 10 px UP in the image (smaller pixel y)
        [0, 0],     # top-left pixel
    ]

    xy = pixel_to_tile_nm(xy_px, shape, pixel_size_nm=2.0)

    np.testing.assert_allclose(xy, [
        [0, 0],
        [20, 0],
        [0, 20],  # y points up, so moving up in the image is +y
        [-200, 100],
    ])


def test_even_size_centre_falls_between_pixels():
    xy = pixel_to_tile_nm([[0, 0], [1, 1]], (2, 2), pixel_size_nm=1.0)
    np.testing.assert_allclose(xy, [[-0.5, 0.5], [0.5, -0.5]])
