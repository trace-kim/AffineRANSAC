"""Contract test for io/metadata.py (task T001).

io/metadata.py is written by the remote agent and exists only on the machine with the
real data, so these tests are skipped everywhere else. They check that its interface is
exactly what docs/tasks/T001-tile-index-reader.md specifies (pairing updated in T002), because the rest of the
code is built against that interface without seeing the implementation.

To also check the real data, set AFFINE_RANSAC_DATA_DIR to the data folder:
    AFFINE_RANSAC_DATA_DIR=/path/to/data pytest tests/test_metadata_contract.py
"""

import dataclasses
import inspect
import math
import os
from pathlib import Path

import pytest

metadata = pytest.importorskip("affine_ransac.io.metadata")

EXPECTED_FIELDS = [
    "tile_id", "image_path", "oas_path",
    "center_x_nm", "center_y_nm", "fov_x_nm", "fov_y_nm",
    "image_width_px", "image_height_px", "stage_x_nm", "stage_y_nm", "extra",
]


def test_tile_record_fields():
    assert dataclasses.is_dataclass(metadata.TileRecord)
    names = [f.name for f in dataclasses.fields(metadata.TileRecord)]
    assert names == EXPECTED_FIELDS


def test_function_signatures():
    for name in ("find_metadata_csv", "read_tile_index"):
        params = list(inspect.signature(getattr(metadata, name)).parameters)
        assert params == ["data_dir"], f"{name} must take exactly one argument, data_dir"


@pytest.fixture
def records():
    data_dir = os.environ.get("AFFINE_RANSAC_DATA_DIR")
    if not data_dir:
        pytest.skip("set AFFINE_RANSAC_DATA_DIR to check the real data")
    return metadata.read_tile_index(data_dir)


def test_real_data_records(records):
    assert len(records) > 0
    ids = [r.tile_id for r in records]
    assert len(ids) == len(set(ids)), "tile_id must be unique"

    for r in records:
        assert isinstance(r.tile_id, str) and r.tile_id
        assert isinstance(r.image_path, Path) and r.image_path.is_absolute() and r.image_path.is_file()
        if r.oas_path is not None:
            assert isinstance(r.oas_path, Path) and r.oas_path.is_file()
            assert r.oas_path.parent.name == "ContourCAD"
        for value in (r.center_x_nm, r.center_y_nm, r.fov_x_nm, r.fov_y_nm):
            assert isinstance(value, float) and math.isfinite(value)
        for value in (r.stage_x_nm, r.stage_y_nm):
            assert value is None or (isinstance(value, float) and math.isfinite(value))
        assert isinstance(r.image_width_px, int) and isinstance(r.image_height_px, int)
        assert isinstance(r.extra, dict)


def test_real_data_units_are_nm(records):
    # FOV is 2.88 um = 2880 nm. A value near 2.88 or 2880000 means a unit conversion is wrong.
    for r in records:
        assert 1000 < r.fov_x_nm < 10000, f"{r.tile_id}: fov_x_nm={r.fov_x_nm} is not ~2880 nm"
        assert 1000 < r.fov_y_nm < 10000, f"{r.tile_id}: fov_y_nm={r.fov_y_nm} is not ~2880 nm"
        pixel_nm = r.fov_x_nm / r.image_width_px
        assert 0.1 < pixel_nm < 10, f"{r.tile_id}: pixel size {pixel_nm} nm is implausible"
