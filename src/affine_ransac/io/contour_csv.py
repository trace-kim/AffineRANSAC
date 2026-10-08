"""Read pre-analysed contour CSV files: one file per image, one row per contact (docs/SPEC.md §4.5).

Each row holds the design and the SEM contour centre of the same contact, already paired, in
GLOBAL mask coordinates, nm (the tile centre is already added; y up like the design). Only the
columns in COLUMNS are used; all others are ignored. Rows with a missing or non-numeric value in
one of them are left out and counted. If the file also has the design centre relative to the image
centre (LOCAL_COLUMNS), the image centre in mask nm is global − local (needed by the in-image
distortion map, intrafield.py).
"""

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

COLUMNS = ("DesignX_Add", "DesignY_Add", "SEMX_Add", "SEMY_Add")
LOCAL_COLUMNS = ("DesignX", "DesignY")  # optional: design centre relative to the image centre, nm


@dataclass
class ContourCsvTile:
    name: str              # file name without .csv
    design_nm: np.ndarray  # (N, 2) design contour centres, mask nm
    sem_nm: np.ndarray     # (N, 2) SEM contour centres, mask nm; row k is the same contact as design row k
    dropped: int           # rows left out (missing or non-numeric value)
    # (2,) image centre in mask nm (DesignX_Add − DesignX, DesignY_Add − DesignY); NaN without LOCAL_COLUMNS
    center_nm: np.ndarray = field(default_factory=lambda: np.full(2, np.nan))


def _number(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return np.nan


def read_contour_csv(path: str | Path) -> ContourCsvTile:
    """Read one contour CSV. Raises ValueError if one of COLUMNS is missing."""
    path = Path(path)
    # utf-8-sig drops a byte-order mark; errors="replace" keeps non-UTF-8 text in unused columns harmless.
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        header = [name.strip() for name in (reader.fieldnames or [])]
        missing = [c for c in COLUMNS if c not in header]
        if missing:
            raise ValueError(f"{path.name}: missing column(s) {missing}; found {header}")
        reader.fieldnames = header
        local = all(c in header for c in LOCAL_COLUMNS)
        columns = COLUMNS + (LOCAL_COLUMNS if local else ())
        values = np.array([[_number(row[c]) for c in columns] for row in reader]).reshape(-1, len(columns))

    ok = np.isfinite(values[:, :4]).all(axis=1)
    center = np.full(2, np.nan)
    if local and np.isfinite(values[:, 4:]).any():
        center = np.nanmedian(values[:, :2] - values[:, 4:], axis=0)  # the same for every row of a file
    return ContourCsvTile(path.stem, values[ok, :2], values[ok, 2:4], int((~ok).sum()), center)


def read_contour_folder(folder: str | Path) -> list[ContourCsvTile]:
    """Read every *.csv in folder (not in subfolders) that has the COLUMNS, sorted by file name.
    Other CSVs in the folder (e.g. summary tables written by the measuring tool) are left out
    with a warning that names them."""
    tiles, skipped = [], []
    for path in sorted(Path(folder).glob("*.csv")):
        try:
            tiles.append(read_contour_csv(path))
        except ValueError:  # a column of COLUMNS is missing: not a contour CSV
            skipped.append(path.name)
    if skipped:
        warnings.warn(f"{len(skipped)} .csv file(s) without the columns {list(COLUMNS)}, left out: {skipped}",
                      stacklevel=2)
    if not tiles:
        raise FileNotFoundError(f"No contour .csv files (columns {list(COLUMNS)}) in {folder}")
    return tiles
