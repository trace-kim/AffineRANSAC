"""Read an external registration measurement: a txt file with the columns X, Y, dX, dY and no header
(docs/SPEC.md §4.6). A reference result, e.g. from another tool, shown beside ours.

- X, Y: site position in mask µm, the same mask coordinates as the design (e.g. the contour CSVs'
  DesignX_Add, DesignY_Add, which are in nm).
- dX, dY: registration error at the site, nm.
The sign convention of dX, dY is not known (user, 2026-10-08): flip_sign=True negates them.
"""

import io
from pathlib import Path

import numpy as np


def read_external(path: str | Path, flip_sign: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(xy_nm, error_nm), each (N, 2): site positions in mask nm and dX, dY in nm, from the first four
    columns (spaces, tabs or commas between them; further columns are ignored). Raises ValueError
    if a line does not hold at least four numbers."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace").replace(",", " ")
    values = np.loadtxt(io.StringIO(text), ndmin=2)
    if values.shape[1] < 4:
        raise ValueError(f"{Path(path).name}: need 4 columns (X, Y, dX, dY), found {values.shape[1]}")
    sign = -1.0 if flip_sign else 1.0
    return values[:, :2] * 1000.0, sign * values[:, 2:4]
