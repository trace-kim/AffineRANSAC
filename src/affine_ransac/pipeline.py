"""Steps S2–S5 for a set of tiles, built only from the library functions (docs/SPEC.md §5).

- process_tile(): load one SEM image, detect its contacts (Otsu, then edge refinement) and read
  its design contacts.
- stitch_tiles(): match the contacts of every overlapping tile pair, fit each pair robustly
  (translation) and solve one translation correction per tile. Pairs that fail to match and
  tiles that cannot be stitched are reported (result fields, a warning), never dropped silently.

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
from affine_ransac.overlap import match_overlap, overlapping_pairs, points_in_box
from affine_ransac.overlap_fit import OverlapFit, fit_overlap
from affine_ransac.stitching import solve_tile_shifts


@dataclass
class TileResult:
    image: np.ndarray           # uint8, as loaded
    center_nm: np.ndarray       # (2,) nominal mask position of the image centre, nm
    fov_nm: np.ndarray          # (2,) field of view (width, height), nm
    pixel_size_nm: float
    otsu: DetectedContacts      # pixel frame
    refined: DetectedContacts   # pixel frame: Otsu contours moved to the maximum gradient
    design_centers: np.ndarray  # (N, 2) design contact centres, tile-local nm (empty if no .oas)

    def sem_points_nm(self, refined: bool = True) -> np.ndarray:
        """SEM contact centres in mask nm at the nominal placement."""
        found = self.refined if refined else self.otsu
        return pixel_to_tile_nm(found.centers, self.image.shape, self.pixel_size_nm) + self.center_nm


def process_tile(
    image_path: str | Path,
    oas_path: str | Path | None,
    center_nm,
    fov_nm,
    layer: tuple[int, int],
    tone_reversed: bool = False,
    search_px: float = 5.0,
) -> TileResult:
    """Load one tile and find its SEM and design contacts.

    center_nm: (x, y) mask position of the image centre; fov_nm: (width, height), nm.
    layer: (layer, datatype) of the contacts in the .oas.
    """
    image = load_sem_image(image_path)
    fov_nm = np.asarray(fov_nm, dtype=float)
    otsu = detect_contacts(image)
    refined = refine_edges(image, otsu, search_px=search_px)

    design_centers = np.empty((0, 2))
    if oas_path is not None:
        layout = load_layout(oas_path)
        if tone_reversed:
            design_centers, _ = read_contacts_tone_reversed(layout, *layer, frame_nm=tuple(fov_nm))
        else:
            design_centers, _ = read_contacts(layout, *layer)

    return TileResult(
        image=image,
        center_nm=np.asarray(center_nm, dtype=float),
        fov_nm=fov_nm,
        pixel_size_nm=fov_nm[0] / image.shape[1],
        otsu=otsu,
        refined=refined,
        design_centers=design_centers,
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
) -> StitchResult:
    """Pairwise overlap fits and the per-tile translation corrections.

    points: per tile, SEM contact centres in mask nm at the nominal placement.
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
    _warn_about_failures(result)
    return result


def _warn_about_failures(result: StitchResult):
    failed = [(r.i, r.j, r.matched) for r in result.rejected if r.failed]
    if failed or len(result.unplaced):
        warnings.warn(
            f"Stitching incomplete: {len(failed)} overlapping pair(s) did not match "
            f"(tile i, tile j, matched contacts): {failed}; {len(result.unplaced)} tile(s) could not be "
            f"stitched (NaN correction): {result.unplaced.tolist()}",
            stacklevel=3,
        )
