"""Read pre-analysed contour CSV files: one file per image, one row per contact (docs/SPEC.md §4.5).

Each row holds the design and the SEM contour centre of the same contact, already paired, in
GLOBAL mask coordinates, nm (the tile centre is already added; y up like the design). Only the
columns in COLUMNS are used; all others are ignored. Rows with a missing or non-numeric value in
one of them are left out and counted.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COLUMNS = ("DesignX_Add", "DesignY_Add", "SEMX_Add", "SEMY_Add")


@dataclass
class ContourCsvTile:
    name: str              # file name without .csv
    design_nm: np.ndarray  # (N, 2) design contour centres, mask nm
    sem_nm: np.ndarray     # (N, 2) SEM contour centres, mask nm; row k is the same contact as design row k
    dropped: int           # rows left out (missing or non-numeric value)


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
        values = np.array([[_number(row[c]) for c in COLUMNS] for row in reader]).reshape(-1, 4)

    ok = np.isfinite(values).all(axis=1)
    return ContourCsvTile(path.stem, values[ok, :2], values[ok, 2:], int((~ok).sum()))


def read_contour_folder(folder: str | Path) -> list[ContourCsvTile]:
    """Read every *.csv in folder (not in subfolders), sorted by file name."""
    paths = sorted(Path(folder).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No .csv files in {folder}")
    return [read_contour_csv(p) for p in paths]
