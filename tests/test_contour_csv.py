import warnings

import numpy as np
import pytest

from affine_ransac.io.contour_csv import COLUMNS, read_contour_csv, read_contour_folder
from affine_ransac.pipeline import stitch_tiles, tile_boxes_from_points
from affine_ransac.registration import merge_observations, paired_errors


def write_csv(path, design, sem, extra_rows=(), bom=False):
    """A contour CSV with extra columns before, between and after the used ones (in the real
    column names) and optional raw extra rows (strings)."""
    lines = ["Index,DesignX_Add,DesignY_Add,Note,SEMX_Add,SEMY_Add,Area"]
    for k, ((dx, dy), (sx, sy)) in enumerate(zip(design, sem)):
        lines.append(f"{k},{float(dx)!r},{float(dy)!r},ok,{float(sx)!r},{float(sy)!r},123")
    lines += list(extra_rows)
    path.write_text(("﻿" if bom else "") + "\n".join(lines) + "\n", encoding="utf-8")


def test_reads_the_four_columns_in_mask_nm(tmp_path):
    design = np.array([[1000.5, -2000.25], [1136.0, -2000.0]])
    sem = design + [[0.3, -0.1], [0.2, 0.4]]
    write_csv(tmp_path / "PRE_tile_01.csv", design, sem, bom=True)

    tile = read_contour_csv(tmp_path / "PRE_tile_01.csv")

    assert tile.name == "PRE_tile_01"
    np.testing.assert_array_equal(tile.design_nm, design)
    np.testing.assert_array_equal(tile.sem_nm, sem)
    assert tile.dropped == 0


def test_rows_with_a_missing_or_text_value_are_left_out_and_counted(tmp_path):
    design = np.array([[0.0, 0.0]])
    write_csv(tmp_path / "a.csv", design, design, extra_rows=["9,5.0,6.0,x,,7.0,1", "10,5.0,abc,x,1.0,7.0,1"])

    tile = read_contour_csv(tmp_path / "a.csv")

    assert len(tile.design_nm) == 1 and tile.dropped == 2


def test_missing_column_names_the_file_and_the_columns(tmp_path):
    (tmp_path / "b.csv").write_text("DesignX_Add,DesignY_Add,SEMX_Add\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"b\.csv.*SEMY_Add"):
        read_contour_csv(tmp_path / "b.csv")


def test_folder_reads_every_csv_sorted_by_name(tmp_path):
    for name in ("c", "a", "b"):
        write_csv(tmp_path / f"{name}.csv", [[0.0, 0.0]], [[0.0, 0.0]])
    (tmp_path / "sub").mkdir()
    write_csv(tmp_path / "sub" / "d.csv", [[0.0, 0.0]], [[0.0, 0.0]])

    assert [t.name for t in read_contour_folder(tmp_path)] == ["a", "b", "c"]
    with pytest.raises(FileNotFoundError):
        read_contour_folder(tmp_path / "sub" / "none")
    assert COLUMNS == ("DesignX_Add", "DesignY_Add", "SEMX_Add", "SEMY_Add")


def test_tile_boxes_are_the_bounding_boxes_of_the_points():
    centers, sizes = tile_boxes_from_points([np.array([[0.0, 0.0], [100.0, 40.0]]), np.array([[5.0, 5.0]])])
    np.testing.assert_allclose(centers, [[50, 20], [5, 5]])
    np.testing.assert_allclose(sizes, [[100, 40], [0, 0]])
    with pytest.raises(ValueError, match="without points"):
        tile_boxes_from_points([np.empty((0, 2))])
    _, grown = tile_boxes_from_points([np.array([[0.0, 0.0], [100.0, 40.0]])], margin_nm=25.0)
    np.testing.assert_allclose(grown, [[150, 90]])


def test_neighbours_sharing_one_column_need_the_margin():
    # Tile A sees columns x = 0..1000, tile B x = 1000..2000: they share only the column x = 1000.
    # B's file is 2 nm to the right, so without a margin the boxes miss each other.
    column = np.arange(0.0, 2001.0, 100.0)
    grid = np.array([(x, y) for x in column for y in np.arange(0.0, 1001.0, 100.0)])
    a, b = grid[grid[:, 0] <= 1000], grid[grid[:, 0] >= 1000] + [2.0, 0.0]

    with pytest.warns(UserWarning, match="could not be stitched"):
        lost = stitch_tiles([a, b], *tile_boxes_from_points([a, b]))
    kept = stitch_tiles([a, b], *tile_boxes_from_points([a, b], margin_nm=25.0))

    assert lost.pairs == [] and lost.rejected == []
    assert len(kept.pairs) == 1 and len(kept.pairs[0].ia) == 11  # the shared column
    np.testing.assert_allclose(kept.corrections, [[1.0, 0.0], [-1.0, 0.0]])  # B - A = +2 nm


# 2 x 2 tiles of 2.88 um, 2.6 um apart: ~280 nm overlap strips; contact pitch 136 nm.
FOV = 2880.0
PITCH = 136.0
LATTICE = np.array([(x, y) for x in np.arange(-3000, 5600, PITCH) for y in np.arange(-5600, 3000, PITCH)])
NOMINAL = np.array([[0.0, 0.0], [0.0, -2600.0], [2600.0, 0.0], [2600.0, -2600.0]])
STAGE = np.array([[1.2, -0.7], [-3.1, 2.4], [0.5, 1.5], [2.0, -2.0]])   # SEM tile placement errors, nm
FILES = np.array([[0.0, 0.0], [4.0, -3.0], [-2.0, 5.0], [1.0, 1.0]])     # offsets between the design files, nm


def write_csv_tiles(folder, noise_nm=0.1, seed=0):
    """Per tile: the contacts fully inside its FOV, design as stored in its file (offset FILES[k]),
    SEM as measured (the tile sits at nominal + STAGE[k], so a contact p is reported at p − STAGE[k])."""
    rng = np.random.default_rng(seed)
    for k, nominal in enumerate(NOMINAL):
        inside = np.all(np.abs(LATTICE - nominal) < FOV / 2 - 30, axis=1)
        contacts = LATTICE[inside]
        sem = contacts - STAGE[k] + rng.normal(0, noise_nm, contacts.shape)
        write_csv(folder / f"PRE_{k}.csv", contacts - FILES[k], sem)


def test_csv_tiles_stitch_like_the_image_pipeline(tmp_path):
    write_csv_tiles(tmp_path)
    tiles = read_contour_folder(tmp_path)
    sem_points = [t.sem_nm for t in tiles]
    design_points = [t.design_nm for t in tiles]

    centers, sizes = tile_boxes_from_points(design_points)
    stitch = stitch_tiles(sem_points, centers, sizes)
    design_stitch = stitch_tiles(design_points, centers, sizes, label="design")

    assert sorted((p.i, p.j) for p in stitch.pairs) == [(0, 1), (0, 2), (1, 3), (2, 3)]
    np.testing.assert_allclose(stitch.corrections, STAGE - STAGE.mean(axis=0), atol=0.1)
    np.testing.assert_allclose(design_stitch.corrections, FILES - FILES.mean(axis=0), atol=1e-6)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing may be skipped
        errors = paired_errors(sem_points, design_points, stitch.corrections, design_stitch.corrections)
    merged = merge_observations(errors, design_stitch.corrections)
    # No mask error: after stitching only the common (gauge) offset is left, the same for every contact.
    common = FILES.mean(axis=0) - STAGE.mean(axis=0)
    np.testing.assert_allclose(merged.error_nm, np.broadcast_to(common, merged.error_nm.shape), atol=0.5)
    # One entry per physical contact: every lattice contact that some tile saw.
    seen = np.unique(np.concatenate([d + FILES[k] for k, d in enumerate(design_points)]), axis=0)
    assert len(merged.count) == len(seen) and merged.count.max() == 4  # corner contacts: 4 tiles
