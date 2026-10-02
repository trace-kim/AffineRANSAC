import cv2
import numpy as np
import pytest

from affine_ransac.pipeline import process_tile, stitch_tiles
from sample_design import write_contact_array
from sample_sem import render_tile

FOV, SIZE = 720.0, 512
LATTICE = np.array([(x, y) for x in np.arange(-1530, 1530, 90.0) for y in np.arange(-1530, 1530, 90.0)])
# 2 x 2 tiles, 600 nm apart: ~120 nm overlap strips. True position = nominal + stage error.
NOMINAL = np.array([[0.0, 0.0], [0.0, -600.0], [600.0, 0.0], [600.0, -600.0]])
ERRORS = np.array([[1.2, -0.7], [-3.1, 2.4], [0.5, 1.5], [2.0, -2.0]])


def write_tile(path, true_center, seed):
    ok, encoded = cv2.imencode(".jpg", render_tile(LATTICE, true_center, FOV, SIZE, seed=seed),
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
    encoded.tofile(str(path))


def test_process_tile_finds_sem_and_design_contacts(tmp_path):
    write_tile(tmp_path / "tile.jpg", true_center=(0.0, 0.0), seed=1)
    write_contact_array(tmp_path / "tile.oas", nx=3, ny=2, pitch_x_nm=200, pitch_y_nm=300)

    tile = process_tile(tmp_path / "tile.jpg", tmp_path / "tile.oas", (5000.0, -2000.0), (FOV, FOV), layer=(1, 0))

    assert tile.pixel_size_nm == FOV / SIZE
    assert len(tile.refined.centers) == len(tile.otsu.centers) > 20
    assert len(tile.design_centers) == 6  # tile-local nm, as stored in the file
    np.testing.assert_allclose(tile.design_centers.min(axis=0), (0, 0))
    # SEM points in mask nm: near lattice positions shifted by the tile centre. (render_tile's
    # wide rims leave small background gaps that are detected too; only real contacts count.)
    points = (tile.sem_points_nm() - (5000.0, -2000.0))[tile.refined.areas > 200]
    assert len(points) > 20
    nearest = np.linalg.norm(points[:, None] - LATTICE[None], axis=2).min(axis=1)
    assert nearest.max() < 1.0


def test_process_tile_without_oas(tmp_path):
    write_tile(tmp_path / "tile.jpg", true_center=(0.0, 0.0), seed=1)
    tile = process_tile(tmp_path / "tile.jpg", None, (0.0, 0.0), (FOV, FOV), layer=(1, 0))
    assert tile.design_centers.shape == (0, 2)


def test_stitch_tiles_recovers_stage_errors(tmp_path):
    points = []
    for k, (nominal, error) in enumerate(zip(NOMINAL, ERRORS)):
        write_tile(tmp_path / f"{k}.jpg", nominal + error, seed=k)
        tile = process_tile(tmp_path / f"{k}.jpg", None, nominal, (FOV, FOV), layer=(1, 0))
        points.append(tile.sem_points_nm())

    result = stitch_tiles(points, NOMINAL, np.full((4, 2), FOV), max_match_nm=10)

    # Neighbours along x and y overlap in strips; the diagonal pairs only touch at a corner.
    assert sorted((p.i, p.j) for p in result.pairs) == [(0, 1), (0, 2), (1, 3), (2, 3)]
    # A tile seen at nominal + error needs the correction +error (gauge: corrections average 0).
    np.testing.assert_allclose(result.corrections, ERRORS - ERRORS.mean(axis=0), atol=0.3)


def seen_points(errors, noise_nm=0.2, seed=0):
    """Per tile, the lattice contacts it sees (fully inside its true FOV), at nominal placement:
    a tile whose true centre is nominal + error reports contact p at p − error."""
    rng = np.random.default_rng(seed)
    points = []
    for nominal, error in zip(NOMINAL, errors):
        inside = np.all(np.abs(LATTICE - (nominal + error)) < FOV / 2 - 10, axis=1)  # cut-off contacts dropped
        points.append(LATTICE[inside] - error + rng.normal(0, noise_nm, (inside.sum(), 2)))
    return points


def test_default_gate_pairs_offsets_beyond_10_nm():
    errors = np.array([[0.0, 0.0], [12.0, -8.0], [-10.0, 6.0], [4.0, 9.0]])  # B − A up to ~20 nm

    result = stitch_tiles(seen_points(errors), NOMINAL, np.full((4, 2), FOV))

    assert len(result.pairs) == 4 and not any(r.failed for r in result.rejected)
    np.testing.assert_allclose(result.corrections, errors - errors.mean(axis=0), atol=0.3)


def test_unmatched_tile_is_reported_never_silently_zero():
    errors = np.array([[0.0, 0.0], [3.0, -2.0], [-2.0, 1.0], [40.0, 0.0]])  # tile 3: 40 nm off, beyond the gate

    with pytest.warns(UserWarning, match="Stitching incomplete"):
        result = stitch_tiles(seen_points(errors), NOMINAL, np.full((4, 2), FOV))

    assert sorted((r.i, r.j) for r in result.rejected if r.failed) == [(1, 3), (2, 3)]
    assert result.unplaced.tolist() == [3]
    assert np.isnan(result.corrections[3]).all()
    np.testing.assert_allclose(result.corrections[:3], errors[:3] - errors[:3].mean(axis=0), atol=0.3)
