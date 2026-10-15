"""Basic polygon measures (shoelace formula). Vertices: (N, 2) array, not repeated at the end."""

import numpy as np


def polygon_area(vertices: np.ndarray) -> float:
    """Area of a simple polygon (positive for either vertex order)."""
    x, y = vertices[:, 0], vertices[:, 1]
    return abs((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()) / 2.0


def polygon_centroid(vertices: np.ndarray) -> np.ndarray:
    """Area centroid (x, y) of a simple polygon."""
    x, y = vertices[:, 0], vertices[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0
    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return np.array([cx, cy])


def contact_sizes(outlines: list[np.ndarray], centers: np.ndarray, roi_width: float, spacing: float) -> np.ndarray:
    """(N, 3) area, xCD and yCD of N closed outlines (vertices (M, 2), not repeated at the end).

    xCD: mean length of the horizontal chords through the outline inside a band of height roi_width
    centred on centers[k, 1], one chord every `spacing` (roi_width 0: the single chord through the
    centre). A chord's length is the distance between the outermost crossings of the outline. yCD:
    the same with vertical chords. NaN if no chord crosses the outline. Units: those of the vertices
    (area in their square). All outlines are handled together (one tile at a time is fast).
    """
    n = len(outlines)
    if n == 0:
        return np.empty((0, 3))
    lengths = np.array([len(o) for o in outlines])
    ids = np.repeat(np.arange(n), lengths)
    points = np.concatenate(outlines).astype(np.float64)
    following = np.arange(len(points)) + 1  # index of each vertex's next vertex on its own outline
    following[np.cumsum(lengths) - 1] = np.cumsum(lengths) - lengths
    nxt = points[following]

    cross = points[:, 0] * nxt[:, 1] - nxt[:, 0] * points[:, 1]  # shoelace terms
    area = np.abs(np.bincount(ids, cross, minlength=n)) / 2

    count = int(roi_width // spacing) + 1 if roi_width > 0 else 1
    offsets = (np.arange(count) - (count - 1) / 2) * spacing
    cds = []
    for along in (0, 1):  # xCD: horizontal chords (along x, at fixed y); yCD: vertical chords
        across = 1 - along
        chords = np.full((count, n), np.nan)
        for k, offset in enumerate(offsets):
            line = centers[ids, across] + offset
            s0, s1 = points[:, across] - line, nxt[:, across] - line
            hit = (s0 <= 0) != (s1 <= 0)  # half-open: a vertex on the line counts once
            t = s0[hit] / (s0[hit] - s1[hit])
            position = points[hit, along] + t * (nxt[hit, along] - points[hit, along])
            low, high = np.full(n, np.inf), np.full(n, -np.inf)
            np.minimum.at(low, ids[hit], position)
            np.maximum.at(high, ids[hit], position)
            crossed = high > low
            chords[k, crossed] = high[crossed] - low[crossed]
        found = np.isfinite(chords).sum(axis=0)
        with np.errstate(invalid="ignore"):
            cds.append(np.where(found > 0, np.nansum(chords, axis=0) / found, np.nan))
    return np.column_stack([area, *cds])
