import cv2
import klayout.db as kdb
import numpy as np
import pytest

from affine_ransac.io.design import load_layout
from affine_ransac.features.contact import DetectedContacts
from affine_ransac.pipeline import (DesignContacts, TileResult, choose_design_contacts, inside_frame,
                                    process_tile, stitch_design, stitch_tiles)
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


def write_design(path, centers, size_nm=40, reversed=False, fov=FOV):
    """.oas on layer 1/0 (dbu 1 nm) with square contacts at centers (tile-local nm). Tone reversed:
    the drawn shape is the FOV frame minus the contacts (one polygon with holes)."""
    layout = kdb.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("TOP")
    half = round(size_nm / 2)
    holes = kdb.Region()
    for x, y in np.round(centers).astype(int).tolist():  # KLayout needs Python ints
        holes.insert(kdb.Box(x - half, y - half, x + half, y + half))
    if reversed:
        frame = kdb.Region(kdb.Box(-round(fov / 2), -round(fov / 2), round(fov / 2), round(fov / 2)))
        holes = frame - holes
    top.shapes(layout.layer(1, 0)).insert(holes)
    layout.write(str(path))
    return load_layout(path)


def visible(points, margin=30):
    return points[np.all(np.abs(points) < FOV / 2 - margin, axis=1)]


def test_process_tile_finds_sem_and_design_contacts(tmp_path):
    write_tile(tmp_path / "tile.jpg", true_center=(0.0, 0.0), seed=1)
    write_design(tmp_path / "tile.oas", visible(LATTICE, margin=0))

    tile = process_tile(tmp_path / "tile.jpg", tmp_path / "tile.oas", (5000.0, -2000.0), (FOV, FOV), layer=(1, 0))

    assert tile.pixel_size_nm == FOV / SIZE
    assert len(tile.refined.centers) == len(tile.otsu.centers) > 20
    assert tile.design.ok and not tile.design.reversed
    np.testing.assert_allclose(sorted(map(tuple, tile.design_points_nm() - (5000.0, -2000.0))),
                               sorted(map(tuple, visible(LATTICE, margin=0))), atol=1e-9)
    # SEM points in mask nm: near lattice positions shifted by the tile centre. (render_tile's
    # wide rims leave small background gaps that are detected too; only real contacts count.)
    points = (tile.sem_points_nm() - (5000.0, -2000.0))[tile.refined.areas > 200]
    assert len(points) > 20
    nearest = np.linalg.norm(points[:, None] - LATTICE[None], axis=2).min(axis=1)
    assert nearest.max() < 1.0


def test_process_tile_without_oas(tmp_path):
    write_tile(tmp_path / "tile.jpg", true_center=(0.0, 0.0), seed=1)
    tile = process_tile(tmp_path / "tile.jpg", None, (0.0, 0.0), (FOV, FOV), layer=(1, 0))
    assert tile.design.centers.shape == (0, 2) and not tile.design.ok


@pytest.mark.parametrize("reversed", [False, True])
def test_design_tone_is_chosen_by_matching_the_sem(tmp_path, reversed):
    design = visible(LATTICE, margin=0)
    sem = visible(LATTICE) + (4.0, -3.0)  # stage offset; edge contacts cut off in the SEM
    layout = write_design(tmp_path / "d.oas", design, reversed=reversed)

    found = choose_design_contacts(layout, (1, 0), (FOV, FOV), sem)

    assert found.ok and found.reversed == reversed and found.match_fraction == 1.0
    np.testing.assert_allclose(sorted(map(tuple, found.centers)), sorted(map(tuple, design)), atol=1e-9)


def test_design_tone_flagged_when_neither_tone_matches(tmp_path):
    layout = write_design(tmp_path / "d.oas", visible(LATTICE, margin=0))
    sem = np.random.default_rng(3).uniform(-300, 300, (60, 2))  # no lattice: no shift lines it up

    found = choose_design_contacts(layout, (1, 0), (FOV, FOV), sem)

    assert not found.ok and found.match_fraction < 0.5


def test_design_tone_check_finds_offsets_beyond_the_old_gate(tmp_path):
    # 47 nm offset (beyond the former 40 nm gate; pitch 90 nm, 40 nm < P / 2 along x): found by the shift search.
    layout = write_design(tmp_path / "d.oas", visible(LATTICE, margin=0))
    sem = visible(LATTICE, margin=60) + (40.0, -25.0)

    found = choose_design_contacts(layout, (1, 0), (FOV, FOV), sem)

    assert found.ok and found.match_fraction == 1.0
    np.testing.assert_allclose(found.shift_nm, (40.0, -25.0), atol=1e-6)


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

    with pytest.warns(UserWarning, match="SEM stitching incomplete"):
        result = stitch_tiles(seen_points(errors), NOMINAL, np.full((4, 2), FOV))

    assert sorted((r.i, r.j) for r in result.rejected if r.failed) == [(1, 3), (2, 3)]
    assert result.unplaced.tolist() == [3]
    assert np.isnan(result.corrections[3]).all()
    np.testing.assert_allclose(result.corrections[:3], errors[:3] - errors[:3].mean(axis=0), atol=0.3)


def test_inside_frame_drops_contacts_touching_the_fov():
    centers = np.array([[0.0, 0.0], [340.0, 0.0], [0.0, -330.0]])
    sizes = np.full((3, 2), 40.0)  # half size 20: 340 + 20 reaches the 360 frame edge
    np.testing.assert_array_equal(inside_frame(centers, sizes, (FOV, FOV)), [True, False, True])


def design_tile(nominal, offset, ok=True):
    """A tile whose .oas is drawn `offset` nm away from where it should be (no SEM data needed)."""
    local = LATTICE - nominal
    local = local[np.all(np.abs(local) < FOV / 2 - 10, axis=1)] - offset
    empty = DetectedContacts(np.empty((0, 2)), np.empty(0), [], 0.0)
    return TileResult(image=np.zeros((SIZE, SIZE), np.uint8), center_nm=np.asarray(nominal, float),
                      fov_nm=np.array([FOV, FOV]), pixel_size_nm=FOV / SIZE, otsu=empty, refined=empty,
                      design=DesignContacts(local, False, 1.0, ok))


def test_stitch_design_recovers_offsets_between_oas_files():
    offsets = np.array([[0.0, 0.0], [6.0, -4.0], [-5.0, 3.0], [2.0, 7.0]])

    result = stitch_design([design_tile(n, o) for n, o in zip(NOMINAL, offsets)])

    # A file drawn `offset` too far left needs +offset to line up with its neighbours.
    np.testing.assert_allclose(result.corrections, offsets - offsets.mean(axis=0), atol=1e-9)


def test_stitch_design_leaves_out_flagged_tiles():
    tiles = [design_tile(n, (0.0, 0.0), ok=(k != 3)) for k, n in enumerate(NOMINAL)]
    with pytest.warns(UserWarning, match="design stitching incomplete"):
        result = stitch_design(tiles)
    assert result.unplaced.tolist() == [3]


def test_tile_neighbours_are_the_overlapping_pairs():
    from affine_ransac.pipeline import tile_neighbours
    assert sorted(tile_neighbours(NOMINAL, np.full((4, 2), FOV))) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert tile_neighbours(np.array([[0.0, 0.0], [5000.0, 0.0]]), np.full((2, 2), FOV)) == []
