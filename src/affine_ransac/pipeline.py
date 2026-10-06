"""Steps S2–S5 for a set of tiles, built only from the library functions (docs/SPEC.md §5).

- process_tile(): load one SEM image, detect its contacts (Otsu, then edge refinement) and read
  its design contacts in the tone (normal / reversed) that matches the SEM contacts; tiles where
  neither tone matches are flagged (DesignContacts.ok).
- stitch_design(): the same for the per-tile design files, which can be offset against each
  other (the same contact drawn at different positions in neighbouring .oas files).
- stitch_tiles(): match the contacts of every overlapping tile pair, fit each pair robustly
  (translation) and solve one translation correction per tile. Pairs that fail to match and
  tiles that cannot be stitched are reported (result fields, a warning), never dropped silently.

- tile_boxes_from_points(): tile centres and sizes for data without a metadata CSV (e.g. the
  pre-analysed contour CSVs): the bounding box of the contacts each tile reports.

Tile data is passed as plain arguments (paths, nm values), not as metadata records, so this
module works with any metadata reader.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from affine_ransac.features.contact import DetectedContacts, detect_contacts
from affine_ransac.features.edges import refine_edges
from affine_ransac.geometry.frames import pixel_to_tile_nm
from affine_ransac.io.design import load_layout, read_contacts, read_contacts_tone_reversed
from affine_ransac.io.sem_image import load_sem_image
from affine_ransac.matching import match_points
from affine_ransac.overlap import match_overlap, overlapping_pairs, points_in_box
from affine_ransac.overlap_fit import OverlapFit, fit_overlap
from affine_ransac.stitching import solve_tile_shifts


@dataclass
class DesignContacts:
    centers: np.ndarray     # (N, 2) design contact centres, tile-local nm
    reversed: bool          # True if read as tone reversed (holes = space between drawn shapes)
    match_fraction: float   # fraction of the SEM contacts that matched a design contact
    ok: bool                # False: neither tone matched well enough (flagged), or no .oas


def matched_fraction(design: np.ndarray, sem: np.ndarray, max_match_nm: float) -> float:
    """Fraction of the SEM points that pair one-to-one with a design point within the gate."""
    if len(sem) == 0:
        return 0.0
    return len(match_points(design, sem, max_match_nm)[0]) / len(sem)


def inside_frame(centers: np.ndarray, sizes: np.ndarray, fov_nm, margin_nm: float = 1.0) -> np.ndarray:
    """(N,) bool: contacts whose bounding box stays inside the FOV frame (centred on (0, 0)).

    A contact touching the frame may be cut by it, which moves its centre; the SEM detection
    drops border contacts for the same reason.
    """
    half = np.asarray(fov_nm, dtype=float) / 2
    return np.all(np.abs(centers) + sizes / 2 < half - margin_nm, axis=1)


def choose_design_contacts(
    layout,
    layer: tuple[int, int],
    fov_nm,
    sem_local: np.ndarray,
    max_match_nm: float = 40.0,
    min_fraction: float = 0.5,
) -> DesignContacts:
    """Read the design contacts with whichever tone matches the SEM contacts.

    Normal tone first (holes = drawn shapes; contacts touching the FOV frame dropped). If fewer
    than min_fraction of the SEM contacts (sem_local, tile-local nm) match a design contact
    within max_match_nm, try tone reversed (frame-touching holes are dropped by its reader).
    If that does not reach min_fraction either, the better of the two is returned with ok=False.
    """
    normal, sizes = read_contacts(layout, *layer)
    normal = normal[inside_frame(normal, sizes, fov_nm)]
    normal_fraction = matched_fraction(normal, sem_local, max_match_nm)
    if normal_fraction >= min_fraction:
        return DesignContacts(normal, False, normal_fraction, ok=True)

    reversed_, _ = read_contacts_tone_reversed(layout, *layer, frame_nm=tuple(fov_nm))
    reversed_fraction = matched_fraction(reversed_, sem_local, max_match_nm)
    if reversed_fraction >= min_fraction:
        return DesignContacts(reversed_, True, reversed_fraction, ok=True)
    if reversed_fraction > normal_fraction:
        return DesignContacts(reversed_, True, reversed_fraction, ok=False)
    return DesignContacts(normal, False, normal_fraction, ok=False)


@dataclass
class TileResult:
    image: np.ndarray           # uint8, as loaded
    center_nm: np.ndarray       # (2,) nominal mask position of the image centre, nm
    fov_nm: np.ndarray          # (2,) field of view (width, height), nm
    pixel_size_nm: float
    otsu: DetectedContacts      # pixel frame: initial detection (detect_contacts, any method)
    refined: DetectedContacts   # pixel frame: Otsu contours moved to the maximum gradient
    design: DesignContacts      # design contacts in the tone that matches the SEM

    def sem_points_nm(self, refined: bool = True) -> np.ndarray:
        """SEM contact centres in mask nm at the nominal placement."""
        found = self.refined if refined else self.otsu
        return pixel_to_tile_nm(found.centers, self.image.shape, self.pixel_size_nm) + self.center_nm

    def design_points_nm(self) -> np.ndarray:
        """Design contact centres in mask nm (tile-local + nominal centre; the design never moves)."""
        return self.design.centers + self.center_nm


def process_tile(
    image_path: str | Path,
    oas_path: str | Path | None,
    center_nm,
    fov_nm,
    layer: tuple[int, int],
    search_px: float = 5.0,
    design_match_nm: float = 40.0,
    min_match_fraction: float = 0.5,
    detection: str = "otsu",
) -> TileResult:
    """Load one tile and find its SEM and design contacts.

    center_nm: (x, y) mask position of the image centre; fov_nm: (width, height), nm.
    layer: (layer, datatype) of the contacts in the .oas. The design tone is chosen by matching
    against the refined SEM centres (choose_design_contacts). detection: "otsu", "otsu3" or
    "band" (method of features.contact.detect_contacts).
    """
    image = load_sem_image(image_path)
    fov_nm = np.asarray(fov_nm, dtype=float)
    pixel_size_nm = fov_nm[0] / image.shape[1]
    otsu = detect_contacts(image, method=detection)
    refined = refine_edges(image, otsu, search_px=search_px)

    if oas_path is None:
        design = DesignContacts(np.empty((0, 2)), False, 0.0, ok=False)
    else:
        sem_local = pixel_to_tile_nm(refined.centers, image.shape, pixel_size_nm)
        design = choose_design_contacts(load_layout(oas_path), layer, fov_nm, sem_local,
                                        design_match_nm, min_match_fraction)

    return TileResult(
        image=image,
        center_nm=np.asarray(center_nm, dtype=float),
        fov_nm=fov_nm,
        pixel_size_nm=pixel_size_nm,
        otsu=otsu,
        refined=refined,
        design=design,
    )


@dataclass
class PairResult:
    i: int            # tile A (index into the tile list)
    j: int            # tile B
    box: tuple        # nominal overlap box (x_min, x_max, y_min, y_max), mask nm
    ia: np.ndarray    # matched contacts: indices into tile A's points
    ib: np.ndarray    # ... and into tile B's points (same physical contacts)
    fit: OverlapFit   # robust B − A translation; fit.inliers marks the contacts used


@dataclass
class RejectedPair:
    i: int
    j: int
    box: tuple       # nominal overlap box, mask nm
    in_box: int      # contacts in the overlap box: the smaller of the two tiles' counts
    matched: int     # contacts that paired up (fewer than min_matched)
    # True: both tiles had at least min_matched contacts in the overlap, yet they did not pair up
    # (e.g. a B − A offset beyond the match gate). False: the overlap holds too few contacts to
    # be used (e.g. a corner-only overlap).
    failed: bool


@dataclass
class StitchResult:
    pairs: list[PairResult]        # pairs used in the solve
    rejected: list[RejectedPair]   # overlapping pairs not used (fewer than min_matched matches)
    corrections: np.ndarray        # (n_tiles, 2) translation added to each nominal centre, nm;
                                   # NaN for tiles that could not be stitched to the others

    @property
    def unplaced(self) -> np.ndarray:
        """Indices of the tiles that could not be stitched (NaN correction)."""
        return np.flatnonzero(np.isnan(self.corrections[:, 0]))


def stitch_tiles(
    points: list[np.ndarray],
    centers: np.ndarray,
    fovs: np.ndarray,
    max_match_nm: float = 25.0,
    outlier_factor: float = 3.0,
    min_matched: int = 5,
    label: str = "SEM",
) -> StitchResult:
    """Pairwise overlap fits and the per-tile translation corrections.

    points: per tile, contact centres (SEM or design) in mask nm at the nominal placement.
    label: what is being stitched, for the warning text.
    centers, fovs: (n_tiles, 2) nominal tile centres and fields of view, nm.
    max_match_nm: largest expected B − A offset (stage error); must stay below half the pitch.
    Pairs with fewer than min_matched matched contacts are not used; they are kept in
    `rejected`. Each used pair is weighted by its number of inlier contacts.
    Warns if a pair failed to match or a tile could not be stitched (see StitchResult).
    """
    pairs, rejected = [], []
    for i, j, box in overlapping_pairs(centers, fovs):
        ia, ib = match_overlap(points[i], points[j], box, max_match_nm)
        if len(ia) >= min_matched:
            fit = fit_overlap(points[i][ia], points[j][ib], outlier_factor=outlier_factor)
            pairs.append(PairResult(i, j, box, ia, ib, fit))
        else:
            in_box = min(len(points_in_box(points[i], box)), len(points_in_box(points[j], box)))
            rejected.append(RejectedPair(i, j, box, in_box, len(ia), failed=in_box >= min_matched))

    corrections = solve_tile_shifts(
        len(centers),
        [(p.i, p.j) for p in pairs],
        [p.fit.shift for p in pairs],
        weights=[p.fit.inliers.sum() for p in pairs],
    )
    result = StitchResult(pairs, rejected, corrections)
    _warn_about_failures(result, label)
    return result


def tile_boxes_from_points(points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per tile, the centre and size (n_tiles, 2 each, mask nm) of the bounding box of its points.

    For tiles without a known centre and FOV, e.g. contour CSVs in mask coordinates. Passed to
    stitch_tiles as centres and FOVs, two tiles' overlap box is then where BOTH report contacts
    (grown by the match gate when matching), which is all stitching needs. Every tile needs at
    least one point (leave empty tiles out first).
    """
    empty = [k for k, p in enumerate(points) if len(p) == 0]
    if empty:
        raise ValueError(f"Tiles without points (leave them out): {empty}")
    lows = np.array([p.min(axis=0) for p in points]).reshape(-1, 2)
    highs = np.array([p.max(axis=0) for p in points]).reshape(-1, 2)
    return (lows + highs) / 2, highs - lows


def stitch_design(
    tiles: list[TileResult],
    max_match_nm: float = 25.0,
    outlier_factor: float = 3.0,
    min_matched: int = 5,
) -> StitchResult:
    """Stitch the per-tile design files exactly like the SEM tiles (translation per tile).

    The same contact drawn in two neighbouring .oas files should sit at the same mask position;
    where it does not, the files are offset against each other. Tiles whose design tone was
    flagged are left out (their correction is NaN, reported as not stitched).
    """
    points = [t.design_points_nm() if t.design.ok else np.empty((0, 2)) for t in tiles]
    centers = np.array([t.center_nm for t in tiles])
    fovs = np.array([t.fov_nm for t in tiles])
    return stitch_tiles(points, centers, fovs, max_match_nm, outlier_factor, min_matched, label="design")


def _warn_about_failures(result: StitchResult, label: str):
    failed = [(r.i, r.j, r.matched) for r in result.rejected if r.failed]
    if failed or len(result.unplaced):
        warnings.warn(
            f"{label} stitching incomplete: {len(failed)} overlapping pair(s) did not match "
            f"(tile i, tile j, matched contacts): {failed}; {len(result.unplaced)} tile(s) could not be "
            f"stitched (NaN correction): {result.unplaced.tolist()}",
            stacklevel=3,
        )
