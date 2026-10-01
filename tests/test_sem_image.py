import cv2
import numpy as np
import pytest

from affine_ransac.io.sem_image import load_sem_image
from sample_sem import write_sem_like


def test_loads_grayscale_uint8(tmp_path):
    path = tmp_path / "tile.jpg"
    centers = write_sem_like(path, rows=300, cols=400, databar_px=40)

    image = load_sem_image(path)

    assert image.shape == (300, 400)
    assert image.dtype == np.uint8
    # Hole centres are dark, and the data bar at the bottom is (nearly) black.
    u, v = centers[0].astype(int)
    assert image[v, u] < 80
    assert np.median(image[-40:]) < 20


def test_colour_jpeg_is_converted_to_gray(tmp_path):
    path = tmp_path / "colour.jpg"
    colour = np.zeros((20, 30, 3), dtype=np.uint8)
    colour[:] = (0, 0, 255)  # pure red in OpenCV's BGR order
    cv2.imencode(".jpg", colour)[1].tofile(str(path))

    image = load_sem_image(path)

    assert image.shape == (20, 30)
    assert abs(int(image.mean()) - 76) <= 3  # standard luma of red: 0.299 * 255


def test_non_ascii_path(tmp_path):
    folder = tmp_path / "한글_폴더"
    folder.mkdir()
    path = folder / "이미지.jpg"
    write_sem_like(path, rows=100, cols=100, databar_px=0)

    assert load_sem_image(path).shape == (100, 100)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sem_image(tmp_path / "nope.jpg")


def test_corrupt_file_raises(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not a jpeg")
    with pytest.raises(ValueError, match="decode"):
        load_sem_image(path)
