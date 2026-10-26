"""Steps S2–S5 for a set of tiles, built only from the library functions (docs/SPEC.md §5).

- process_tile(): load one SEM image, detect its contacts (Otsu, then edge refinement) and read
  its design contacts in the tone (normal / reversed) that matches the SEM contacts; tiles where
  neither tone matches are flagged (DesignContacts.ok).
- find_contacts(): the SEM contacts of one image (detection, contrast check, edge refinement), as process_tile and
  the release's contour diagnosis find them.
- stitch_design(): the same for the per-tile design files, which can be offset against each
  other (the same contact drawn at different positions in neighbouring .oas files).
- stitch_tiles(): match the contacts of every overlapping tile pair, fit each pair robustly
  (translation) and solve one translation correction per tile. Pairs that fail to match and
  tiles that cannot be stitched are reported (result fields, a warning), never dropped silently.
- stitch_tiles_affine(): one affine per tile instead, from the same pairs and tie contacts.
- tie_residuals(): how well the tie contacts agree after either stitching.
- overlap_checks(): whether each overlap lines up at a placement, whatever paired its contacts (D137).

- tile_boxes_from_points(): tile centres and sizes for data without a metadata CSV (e.g. the
  pre-analysed contour CSVs): the bounding box of the contacts each tile reports.
- tile_stripes(), measurement_pairs(): the stripes of a serpentine scan and the pairs of images
  measured one after the other (stitch_tiles(only_pairs=...) stitches only those).

Tile data is passed as plain arguments (paths, nm values), not as metadata records, so this
module works with any metadata reader.
"""

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from affine_ransac.features.contact import (BAND_MIN_SOLIDITY, DetectedContacts, contact_contrast, denoised_image,
                                            detect_contacts, flatten_background, in_image_pixels, keep_contacts,
                                            upscaled_image)
from affine_ransac.features.edges import RULES, refine_edges
from affine_ransac.geometry.frames import pixel_to_tile_nm
from affine_ransac.geometry.polygon import contact_sizes, polygon_centroid
from affine_ransac.io.design import contact_outlines, load_layout, read_contacts, read_contacts_tone_reversed
from affine_ransac.io.sem_image import load_sem_image
from affine_ransac.matching import match_points, match_with_shift, scored_shifts_together
from affine_ransac.overlap import match_overlap, overlapping_pairs, points_in_box
from affine_ransac.overlap_fit import OverlapFit, fit_overlap
from affine_ransac.stitching import apply_correction, solve_tile_affines, solve_tile_rigid, solve_tile_shifts


REFINEMENTS = RULES + ("none",)  # process_tile refinement: an edge rule, or none (the detected contours, D124)


@dataclass
class DesignContacts:
    centers: np.ndarray     # (N, 2) design contact centres, tile-local nm
    reversed: bool          # True if read as tone reversed (holes = space between drawn shapes)
    match_fraction: float   # fraction of the SEM contacts that matched a design contact
    ok: bool                # False: neither tone matched well enough (flagged), or no .oas
    # SEM − design shift found by the tone check (tile-local, nm; NaN if nothing matched), and
    # whether other shifts matched equally well (tied, e.g. a fully periodic tile), in which case
    # the smallest was taken (matching.match_with_shift).
    shift_nm: np.ndarray = field(default_factory=lambda: np.full(2, np.nan))
    ambiguous: bool = False


def matched_fraction(design: np.ndarray, sem: np.ndarray, search_nm: float, tolerance_nm: float):
    """(fraction of the SEM points paired with a design point, (ShiftMatch, tied)) after the shift
    search."""
    match, tied = match_with_shift(design, sem, search_nm, tolerance_nm)
    return (match.score / len(sem) if len(sem) else 0.0), (match, tied)


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
    search_nm: float = 100.0,
    min_fraction: float = 0.5,
    tolerance_nm: float = 10.0,
    seconds: dict | None = None,
) -> DesignContacts:
    """Read the design contacts with whichever tone matches the SEM contacts.

    Normal tone first (holes = drawn shapes; contacts touching the FOV frame dropped). Matching
    allows an unknown tile shift up to search_nm and pairs within tolerance_nm after it
    (matching.match_with_shift). If fewer than min_fraction of the SEM contacts (sem_local,
    tile-local nm) match a design contact, try tone reversed (frame-touching holes are dropped by
    its reader). If that does not reach min_fraction either, the better of the two is returned
    with ok=False. seconds, if given, gets the time of each step: contacts (normal tone read), shift (its
    shift search), reversed (the tone-reversed read and search; 0 if not tried).
    """
    seconds = {} if seconds is None else seconds
    t_start = time.perf_counter()
    def result(centers, reversed_tone, fraction, match, ok):
        shift_match, tied = match
        return DesignContacts(centers, reversed_tone, fraction, ok, shift_match.shift, tied)

    normal, sizes = read_contacts(layout, *layer)
    normal = normal[inside_frame(normal, sizes, fov_nm)]
    t_contacts = time.perf_counter()
    normal_fraction, normal_match = matched_fraction(normal, sem_local, search_nm, tolerance_nm)
    t_shift = time.perf_counter()
    seconds.update(contacts=t_contacts - t_start, shift=t_shift - t_contacts, reversed=0.0)
    if normal_fraction >= min_fraction:
        return result(normal, False, normal_fraction, normal_match, ok=True)

    reversed_, _ = read_contacts_tone_reversed(layout, *layer, frame_nm=tuple(fov_nm))
    reversed_fraction, reversed_match = matched_fraction(reversed_, sem_local, search_nm, tolerance_nm)
    seconds["reversed"] = time.perf_counter() - t_shift
    if reversed_fraction >= min_fraction:
        return result(reversed_, True, reversed_fraction, reversed_match, ok=True)
    if reversed_fraction > normal_fraction:
        return result(reversed_, True, reversed_fraction, reversed_match, ok=False)
    return result(normal, False, normal_fraction, normal_match, ok=False)


def design_outlines(layout, layer: tuple[int, int], fov_nm, tone_reversed: bool) -> list[np.ndarray]:
    """Outlines ((M, 2) tile-local nm) of a tile's design contacts in the tone chosen by
    choose_design_contacts, in the order of its centres: the same merge and the same frame filter, from
    one merge of the file (centre and bounding box taken from the outline; a contact has no holes)."""
    fov_nm = np.asarray(fov_nm, dtype=float)
    if tone_reversed:
        return contact_outlines(layout, *layer, tone_reversed=True, frame_nm=tuple(fov_nm))
    return outlines_inside_frame(contact_outlines(layout, *layer), fov_nm)


def outlines_inside_frame(outlines: list[np.ndarray], fov_nm) -> list[np.ndarray]:
    """The normal-tone outlines (tile-local nm) whose bounding box stays inside the FOV frame (inside_frame): the
    contacts choose_design_contacts keeps in normal tone."""
    if not outlines:
        return []
    centers = np.array([polygon_centroid(o) for o in outlines])
    sizes = np.array([o.max(axis=0) - o.min(axis=0) for o in outlines])
    return [o for o, keep in zip(outlines, inside_frame(centers, sizes, np.asarray(fov_nm, dtype=float))) if keep]


@dataclass
class TileResult:
    image: np.ndarray           # uint8, as loaded
    center_nm: np.ndarray       # (2,) nominal mask position of the image centre, nm
    fov_nm: np.ndarray          # (2,) field of view (width, height), nm
    pixel_size_nm: float
    otsu: DetectedContacts      # pixel frame: initial detection (detect_contacts, any method)
    refined: DetectedContacts   # pixel frame: detected contours moved to their edges (refine_edges; the
                                # detected ones with refinement "none"): what the tile is measured with
    design: DesignContacts      # design contacts in the tone that matches the SEM
    oas_path: Path | None = None  # the tile's design file (read again to measure contact sizes)
    seconds: dict[str, float] = field(default_factory=dict)  # time per step of process_tile (logged)
    low_contrast: int = 0         # detected contacts dropped by the contrast check (process_tile min_contrast)
    # Mask position of the .oas file's (0, 0) when it is not center_nm: a realigned tile (realign.realign_tiles),
    # whose design contacts come from all files, not from its own (only drawn, never measured).
    oas_center_nm: np.ndarray | None = None

    def sem_points_nm(self, refined: bool = True) -> np.ndarray:
        """SEM contact centres in mask nm at the nominal placement."""
        found = self.refined if refined else self.otsu
        return pixel_to_tile_nm(found.centers, self.image.shape, self.pixel_size_nm) + self.center_nm

    def design_points_nm(self) -> np.ndarray:
        """Design contact centres in mask nm (tile-local + nominal centre; the design never moves)."""
        return self.design.centers + self.center_nm

    def contact_sizes(self, layer: tuple[int, int], roi_width_nm: float,
                      refined: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(N, 3) area (nm²), xCD and yCD (nm) of the SEM contacts, in the order of sem_points_nm, and
        (K, 3) of the design contacts, in the order of design_points_nm, outlines read again from the
        .oas (design_outlines; none without one). geometry.polygon.contact_sizes: chords through the
        centre, averaged over a band roi_width_nm wide, one chord per SEM pixel."""
        found = self.refined if refined else self.otsu
        outlines = [pixel_to_tile_nm(c, self.image.shape, self.pixel_size_nm) for c in found.contours]
        centers = pixel_to_tile_nm(found.centers, self.image.shape, self.pixel_size_nm)
        sem = contact_sizes(outlines, centers, roi_width_nm, self.pixel_size_nm)
        if self.oas_path is None or self.oas_center_nm is not None or len(self.design.centers) == 0:
            return sem, np.full((len(self.design.centers), 3), np.nan)
        design_shapes = design_outlines(load_layout(self.oas_path), layer, self.fov_nm, self.design.reversed)
        if len(design_shapes) != len(self.design.centers):  # the same file read the same way: should not happen
            raise ValueError(f"{self.oas_path.name}: {len(design_shapes)} design outlines for "
                             f"{len(self.design.centers)} design contacts")
        design = contact_sizes(design_shapes, self.design.centers, roi_width_nm, self.pixel_size_nm)
        return sem, design


def prepared_image(image: np.ndarray, background_px: float = 0.0, dark_contacts: bool = True, denoise: str = "none",
                   median_px: int = 3, nlm_h: float = 10.0, tv_weight: float = 0.1, upscale: int = 1) -> np.ndarray:
    """The image the contacts are found in (find_contacts): with its background flattened over background_px
    (features.contact.flatten_background) if that is > 0, then denoised (denoised_image: method `denoise` with its
    option), then upscaled `upscale` times (upscaled_image); with none of them, the image itself. Denoised before it
    is upscaled: interpolation spreads each noisy pixel over several, which no longer look like noise (D132)."""
    flat = flatten_background(image, background_px, dark_contacts) if background_px > 0 else image
    return upscaled_image(denoised_image(flat, denoise, median_px, nlm_h, tv_weight), upscale)


def find_contacts(
    image: np.ndarray,
    *,  # 19 options: by name only
    background_px: float = 0.0,
    denoise: str = "none",
    median_px: int = 3,
    nlm_h: float = 10.0,
    tv_weight: float = 0.1,
    upscale: int = 1,
    detection: str = "otsu",
    dark_contacts: bool = True,
    detection_blur_px: float = 1.5,
    min_area_px: int = 20,
    max_area_px: int = 0,
    min_solidity: float = BAND_MIN_SOLIDITY,
    min_contrast: float = 0.0,
    refinement: str = "mid-level",
    search_px: float = 5.0,
    edge_step_px: float = 0.25,
    edge_blur_px: float = 1.0,
    normal_span: int = 3,
    seconds: dict | None = None,
) -> tuple[DetectedContacts, DetectedContacts, int]:
    """The contacts of one SEM image (uint8), as every run finds them: (detected, refined, how many detected contacts
    the contrast check dropped). The options are the settings of the same names (analysis.CONTOUR_SETTINGS, D128).

    Every step works on prepared_image (background_px > 0: the background flattened first, D130; denoise: then
    denoised, D132; upscale > 1: then upscaled that many times, D131). The options in pixels, and normal_span (contour
    points), are those of the image as given: in the upscaled image they are `upscale` times larger (areas its square),
    and the contacts are returned in the image's own pixels.
    detected: features.contact.detect_contacts with method `detection` ("otsu", "otsu3" or "band"), blur, area limits
    (max_area_px 0: no limit) and, for "band" only, min_solidity; then, if min_contrast > 0, without the contacts whose
    contrast against a ring around them is below this many noise sigma (contact_contrast, on the image not upscaled,
    denoised if asked: its own noise; false contacts in images without pattern). refined: features.edges.refine_edges
    with edge rule `refinement` ("mid-level", "steepest gradient" or "threshold"), or with "none" the detected contacts
    themselves: either way what the tile is measured with. seconds: if given, gets the time of each step ("detect" includes the
    flattening, the denoising and the upscaling, "refine").
    """
    t_start = time.perf_counter()
    image = prepared_image(image, background_px, dark_contacts, denoise, median_px, nlm_h, tv_weight)
    f = upscale
    big = upscaled_image(image, f)
    detected = detect_contacts(big, dark_contacts, detection_blur_px * f, min_area_px * f ** 2,
                               max_area_px * f ** 2 or None, detection, min_solidity if detection == "band" else None)
    found = len(detected.centers)
    if min_contrast > 0:
        detected = keep_contacts(detected, contact_contrast(image, in_image_pixels(detected, f)) >= min_contrast)
    t_detect = time.perf_counter()
    refined = detected if refinement == "none" else refine_edges(
        big, detected, dark_contacts, search_px * f, edge_step_px * f, edge_blur_px * f, normal_span * f, refinement)
    if seconds is not None:
        seconds.update(detect=t_detect - t_start, refine=time.perf_counter() - t_detect)
    return in_image_pixels(detected, f), in_image_pixels(refined, f), found - len(detected.centers)


def process_tile(
    image_path: str | Path,
    oas_path: str | Path | None,
    center_nm,
    fov_nm,
    layer: tuple[int, int],
    search_px: float = 5.0,
    design_search_nm: float = 100.0,
    min_match_fraction: float = 0.5,
    detection: str = "otsu",
    design_tolerance_nm: float = 10.0,
    match_design: bool = True,
    min_contrast: float = 0.0,
    refinement: str = "mid-level",
    **contour_options,
) -> TileResult:
    """Load one tile and find its SEM and design contacts.

    center_nm: (x, y) mask position of the image centre; fov_nm: (width, height), nm.
    layer: (layer, datatype) of the contacts in the .oas. The design tone is chosen by matching
    against the refined SEM centres (choose_design_contacts: shift search up to design_search_nm,
    pairs within design_tolerance_nm). match_design False: the design is read in normal tone without matching it
    to the SEM (ok, NaN fraction), for images that do not show their own design (realign.realign_tiles).
    The SEM contacts: find_contacts with search_px, detection, min_contrast, refinement and contour_options (its
    other options; their defaults if not given). `refined` holds what the tile is measured with: the design tone is
    chosen with its centres, and sem_points_nm() gives them.
    """
    t_start = time.perf_counter()
    image = load_sem_image(image_path)
    fov_nm = np.asarray(fov_nm, dtype=float)
    pixel_size_nm = fov_nm[0] / image.shape[1]
    seconds = {"read": time.perf_counter() - t_start, "detect": 0.0, "refine": 0.0,  # in the order of the steps
               "oas": 0.0, "contacts": 0.0, "shift": 0.0, "reversed": 0.0}
    otsu, refined, low_contrast = find_contacts(image, detection=detection, min_contrast=min_contrast,
                                                refinement=refinement, search_px=search_px, seconds=seconds,
                                                **contour_options)
    t_refine = time.perf_counter()
    if oas_path is None:
        design = DesignContacts(np.empty((0, 2)), False, 0.0, ok=False)
    else:
        layout = load_layout(oas_path)
        seconds["oas"] = time.perf_counter() - t_refine
        sem_local = pixel_to_tile_nm(refined.centers, image.shape, pixel_size_nm)
        if not match_design:
            centers, sizes = read_contacts(layout, *layer)
            design = DesignContacts(centers[inside_frame(centers, sizes, fov_nm)], False, np.nan, ok=True)
            seconds["contacts"] = time.perf_counter() - t_refine - seconds["oas"]
        else:
            design = choose_design_contacts(layout, layer, fov_nm, sem_local,
                                            design_search_nm, min_match_fraction, design_tolerance_nm, seconds)

    return TileResult(
        image=image,
        center_nm=np.asarray(center_nm, dtype=float),
        fov_nm=fov_nm,
        pixel_size_nm=pixel_size_nm,
        otsu=otsu,
        refined=refined,
        design=design,
        oas_path=None if oas_path is None else Path(oas_path),
        seconds=seconds,
        low_contrast=low_contrast,
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
    only_pairs: set | None = None,
    guide: np.ndarray | None = None,
) -> StitchResult:
    """Pairwise overlap fits and the per-tile translation corrections.

    points: per tile, contact centres (SEM or design) in mask nm at the nominal placement.
    label: what is being stitched, for the warning text.
    centers, fovs: (n_tiles, 2) nominal tile centres and fields of view, nm.
    max_match_nm: largest expected B − A offset (stage error); must stay below half the pitch.
    Pairs with fewer than min_matched matched contacts are not used; they are kept in
    `rejected`. Each used pair is weighted by its number of inlier contacts.
    only_pairs: if given, only these overlapping pairs (i, j), i < j, are stitched (e.g.
    measurement_pairs); the other overlaps are ignored.
    guide: optional (n_tiles, 2) corrections of a first stitching (e.g. in measurement order): the contacts
    of a pair are paired after moving tile j by the B − A shift it predicts (guide_i − guide_j), so an
    offset of about one pitch (drift between images taken far apart in time) cannot pair them one contact
    row off; the shift is then fitted on the paired contacts as they are. A pair with a tile the guide
    did not place is paired as without a guide.
    Warns if a pair failed to match or a tile could not be stitched (see StitchResult).
    """
    pairs, rejected = [], []
    for i, j, box in overlapping_pairs(centers, fovs):
        if only_pairs is not None and (i, j) not in only_pairs:
            continue
        predicted = predicted_shift(guide, i, j)
        ia, ib = match_overlap(points[i], points[j] - predicted, box, max_match_nm)
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


def predicted_shift(guide: np.ndarray | None, i: int, j: int) -> np.ndarray:
    """(2,) the B − A shift of pair (i, j) that the corrections `guide` predict (guide_i − guide_j, from
    t_i − t_j = d_ij), nm; 0 without a guide or if it did not place both tiles."""
    if guide is None:
        return np.zeros(2)
    shift = guide[i] - guide[j]
    return shift if np.isfinite(shift).all() else np.zeros(2)


def slipped_pairs(stitch: StitchResult, guide: np.ndarray, tolerance_nm: float) -> list[tuple[int, int]]:
    """Used pairs (i, j) of stitch whose fitted shift differs from the one `guide` predicts
    (predicted_shift) by more than tolerance_nm, e.g. contacts paired one row off; pairs with a tile
    the guide did not place are not checked."""
    return [(p.i, p.j) for p in stitch.pairs if np.isfinite(guide[[p.i, p.j]]).all()
            and np.linalg.norm(p.fit.shift - predicted_shift(guide, p.i, p.j)) > tolerance_nm]


LINED_UP = 0.5  # placed_image_by_image: a shift lines up with at least this fraction of the most pairs of any shift


def placed_image_by_image(points: list[np.ndarray], centers: np.ndarray, fovs: np.ndarray, search_nm: float,
                          tolerance_nm: float, min_matched: int = 5) -> np.ndarray:
    """(n_tiles, 2) corrections that place the tiles one image at a time, as the guide of stitch_tiles (stitching
    method "image by image", D133).

    points: per tile, contact centres in mask nm at the nominal placement; centers, fovs: (n_tiles, 2), nm.
    The image with the most contacts in its overlaps stays where it is (correction 0). Then, again and again, the
    image sharing the most overlap contacts with the placed ones is matched against all of them at once: its
    contacts in those overlaps (nominal) against theirs (placed), with one shift for all
    (matching.scored_shifts_together: candidate shifts of any size up to search_nm, pairs within tolerance_nm). A
    shift lines them up if it leaves no more contacts without a partner than it pairs (counted in each overlap's
    own box, D50) and pairs at least LINED_UP times as many as the shift pairing the most. Of those it takes the
    smallest, which puts the image nearest its stage position (the user's rule, D48): on a repeating pattern every
    shift by a whole pitch along the overlaps lines up as well, while on a pattern that does not repeat only the true
    shift pairs many. Because an image is matched against all its placed neighbours together, its overlaps cannot
    disagree by a pitch, as overlaps paired one by one at the stage position do once the images lie more than half a
    pitch from it. (Mismatches alone cannot choose here: an overlap one or two rows thick often has no contact
    inside its box shrunk by tolerance_nm, which scores the true shift worst.)
    An image without a shift that lines up, or one pairing fewer than min_matched contacts, waits for more placed
    neighbours; one never placed gets NaN.
    """
    n = len(points)
    shared = {}  # (k, other): indices of tile k's points in its overlap with tile other
    for i, j, box in overlapping_pairs(centers, fovs):
        shared[i, j] = points_in_box(points[i], box, margin=tolerance_nm)
        shared[j, i] = points_in_box(points[j], box, margin=tolerance_nm)
    neighbours = [[] for _ in range(n)]
    for i, j in shared:
        neighbours[i].append(j)
    corrections = np.full((n, 2), np.nan)
    if n == 0:
        return corrections
    gain = np.zeros(n)  # contacts each tile shares with the placed tiles (the fewer of the two sides per overlap)
    tried = np.full(n, -1.0)  # its gain when its match last failed: tried again only once that has grown

    def place(k, correction):
        corrections[k] = correction
        for o in neighbours[k]:
            gain[o] += min(len(shared[k, o]), len(shared[o, k]))

    place(int(np.argmax([sum(len(shared[k, o]) for o in neighbours[k]) for k in range(n)])), 0.0)
    while True:
        ready = np.isnan(corrections[:, 0]) & (gain >= min_matched) & (gain != tried)
        if not ready.any():
            return corrections
        k = int(np.argmax(np.where(ready, gain, -1)))
        parts = [(points[o][shared[o, k]] + corrections[o], points[k][shared[k, o]])  # (theirs placed, mine)
                 for o in neighbours[k] if not np.isnan(corrections[o, 0])]
        matches = [m for m in scored_shifts_together(parts, search_nm, tolerance_nm) if m.mismatches <= m.score]
        most = max((m.score for m in matches), default=0)
        lined_up = [m for m in matches if m.score >= max(min_matched, LINED_UP * most)]
        if lined_up:
            place(k, -min(lined_up, key=lambda m: np.linalg.norm(m.shift)).shift)  # mine + correction = theirs
        else:
            tried[k] = gain[k]


PAIRING_GATE = 0.25  # overlap_checks: contacts pair within this fraction of the contact spacing (well below half)
MIN_AGREEMENT = 0.5  # overlap_checks: an overlap lines up only if at least this share of its contacts pair


def contact_spacing(points: list[np.ndarray]) -> float:
    """The typical distance between neighbouring contacts, nm (the pitch of a contact array): over the tiles, the
    median of each tile's median distance from a contact to its nearest one. NaN if no tile has two contacts."""
    spacings = [np.median(cKDTree(p).query(p, k=2)[0][:, 1]) for p in points if len(p) >= 2]
    return float(np.median(spacings)) if spacings else np.nan


@dataclass
class OverlapCheck:
    i: int
    j: int
    agreement: float  # of the contacts both images should see, the share of the image with fewer that pair up
    offset_nm: float  # length of B − A of those pairs at the placement (robust fit, fit_overlap), nm; NaN: no pairs
    lines_up: bool    # agreement >= MIN_AGREEMENT and offset_nm <= the tolerance


def overlap_checks(points: list[np.ndarray], centers: np.ndarray, fovs: np.ndarray, corrections: np.ndarray,
                   tolerance_nm: float, outlier_factor: float = 3.0, min_matched: int = 5) -> list[OverlapCheck]:
    """Whether the overlaps line up at the placement `corrections` ((n_tiles, 2), e.g. StitchResult.corrections),
    whatever paired them (D137). points: per tile, contact centres in mask nm at the nominal placement; centers, fovs:
    (n_tiles, 2), nm.

    Every overlap of two placed tiles is checked. The contacts of one image that the other should see are those
    inside the box of all the other's contacts, so the rows an image loses at its own frame are not expected of it.
    They pair within PAIRING_GATE times contact_spacing: wide against the noise, well below half the pitch, so a
    contact never pairs with its neighbour. agreement counts the pairs per contact of the image with fewer, so
    contacts one image failed to detect do not count. An image placed wrong by less than the gate shows in the
    offset, one placed wrong by more in the agreement; by a whole pitch on a repeating pattern it lines up (and that
    does not matter, D105). Overlaps where either image has fewer than min_matched such contacts (e.g. corners) are
    not checked.
    """
    gate = PAIRING_GATE * contact_spacing(points)
    placed = [p + c for p, c in zip(points, corrections)]
    extents = [(p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()) if len(p) else None for p in placed]
    checks = []
    for i, j, _ in overlapping_pairs(centers, fovs):
        if np.isnan(corrections[[i, j]]).any() or extents[i] is None or extents[j] is None:
            continue
        a = placed[i][points_in_box(placed[i], extents[j], margin=gate)]
        b = placed[j][points_in_box(placed[j], extents[i], margin=gate)]
        if min(len(a), len(b)) < min_matched:
            continue
        ia, ib = match_points(a, b, gate)
        agreement = len(ia) / min(len(a), len(b))
        offset = (float(np.linalg.norm(fit_overlap(a[ia], b[ib], outlier_factor=outlier_factor).shift)) if len(ia)
                  else np.nan)
        checks.append(OverlapCheck(i, j, agreement, offset, agreement >= MIN_AGREEMENT and offset <= tolerance_nm))
    return checks


def stitch_ties(points: list[np.ndarray], stitch: StitchResult) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per used pair of stitch, (a, b): its inlier contacts (overlap_fit) at nominal placement in
    tile i and in tile j, (K, 2) mask nm each."""
    return [(points[p.i][p.ia[p.fit.inliers]], points[p.j][p.ib[p.fit.inliers]]) for p in stitch.pairs]


def stitch_tiles_affine(points: list[np.ndarray], centers: np.ndarray, stitch: StitchResult) -> np.ndarray:
    """Per-tile affines (n_tiles, 3, 3), nominal mask nm -> stitched (stitching.solve_tile_affines).

    Uses the pairs and tie contacts of the translation stitching `stitch` of the same points
    (stitch_tiles: its used pairs and their inlier contacts), so both stitchings see the same ties
    and differ only in the tile model. The same tiles are left out (NaN).
    """
    return solve_tile_affines(len(points), [(p.i, p.j) for p in stitch.pairs], stitch_ties(points, stitch), centers)


def stitch_tiles_rigid(points: list[np.ndarray], centers: np.ndarray, stitch: StitchResult) -> np.ndarray:
    """Per-tile translation + rotation (n_tiles, 3, 3), nominal mask nm -> stitched
    (stitching.solve_tile_rigid), from the pairs and tie contacts of the translation stitching
    `stitch` of the same points, like stitch_tiles_affine. The same tiles are left out (NaN)."""
    return solve_tile_rigid(len(points), [(p.i, p.j) for p in stitch.pairs], stitch_ties(points, stitch), centers)


def pair_residuals(points: list[np.ndarray], stitch: StitchResult, corrections: np.ndarray) -> list[np.ndarray]:
    """Per used pair of stitch, (K, 2) T_i(a) − T_j(b), nm, of its tie contacts (stitch_ties) after
    the corrections: shifts (n_tiles, 2) or affines (n_tiles, 3, 3). ~0 = the tiles agree."""
    return [apply_correction(a, corrections[p.i]) - apply_correction(b, corrections[p.j])
            for p, (a, b) in zip(stitch.pairs, stitch_ties(points, stitch))]


def tie_residuals(points: list[np.ndarray], stitch: StitchResult, corrections: np.ndarray) -> np.ndarray:
    """(K, 2) the residuals of all pairs together (pair_residuals)."""
    parts = pair_residuals(points, stitch, corrections)
    return np.concatenate(parts) if parts else np.empty((0, 2))


def residual_summary(points: list[np.ndarray], stitch: StitchResult, corrections: np.ndarray,
                     centers: np.ndarray) -> dict:
    """How well the tiles agree after a stitching: the RMS length (nm) of the tie residuals
    (pair_residuals) of the overlaps within stripes ("vertical" pairs), between stripes ("horizontal")
    and of all overlaps, the largest residual, and the number of overlaps of each kind."""
    residuals = pair_residuals(points, stitch, corrections)
    kinds = np.array([pair_kind(centers, p.i, p.j) for p in stitch.pairs])

    def rms(selected):
        parts = [r for r, chosen in zip(residuals, selected) if chosen]
        return float(np.sqrt((np.concatenate(parts) ** 2).sum(axis=1).mean())) if parts else np.nan

    every = np.ones(len(kinds), bool)
    return {"within stripes RMS (nm)": rms(kinds == "vertical"), "between stripes RMS (nm)": rms(kinds == "horizontal"),
            "all overlaps RMS (nm)": rms(every),
            "largest (nm)": float(max((np.linalg.norm(r, axis=1).max() for r in residuals if len(r)), default=np.nan)),
            "overlaps within stripes": int((kinds == "vertical").sum()),
            "overlaps between stripes": int((kinds == "horizontal").sum())}


def pair_kind(centers: np.ndarray, i: int, j: int) -> str:
    """"vertical" (one image above the other, e.g. within a stripe), "horizontal" (side by side,
    between stripes) or "corner", from the tile centres."""
    step = np.abs(centers[j] - centers[i])
    if step[1] > 2 * step[0]:
        return "vertical"
    return "horizontal" if step[0] > 2 * step[1] else "corner"


STRIPE_BREAK = 10.0  # a jump between stripes is at least this many times the largest x scatter within a stripe


def tile_stripes(centers: np.ndarray, image_centers: np.ndarray | None = None) -> np.ndarray:
    """(n_tiles,) stripe of each tile, 0 for the smallest x. Tiles of one stripe share their centre x up to the
    stage scatter (metadata: equal up to rounding, user); neighbouring stripes are one step apart (tile width −
    overlap, whatever the overlap). So the gaps between the sorted centre x are small (scatter) or large (a new
    stripe): the stripes break at the largest ratio between two consecutive sorted gaps (gaps below 1 nm taken as
    1 nm: equal x), if that ratio is at least STRIPE_BREAK; otherwise all tiles are one stripe.
    image_centers: the exact image centres (contour CSVs: DesignX_Add − DesignX), used instead of centers when
    every tile has one: centres of contact bounding boxes vary by up to a pitch within a stripe, which split
    stripes on the contour-CSV test data."""
    if image_centers is not None and np.isfinite(image_centers).all():
        centers = image_centers
    order = np.argsort(centers[:, 0])
    gaps = np.diff(centers[order, 0])
    sorted_gaps = np.sort(np.maximum(gaps, 1.0))
    ratio = sorted_gaps[1:] / sorted_gaps[:-1]
    if len(ratio) and ratio.max() >= STRIPE_BREAK:
        new_stripe = gaps > sorted_gaps[np.argmax(ratio)]
    else:
        new_stripe = np.zeros(len(gaps), bool)
    stripe = np.empty(len(centers), int)
    stripe[order] = np.concatenate([[0], np.cumsum(new_stripe)])
    return stripe


def stripe_ranges(centers: np.ndarray, fovs: np.ndarray, stripe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(low, high), each (n_stripes,): the x range (nm) covered by the tiles (centres, sizes) of each stripe
    (stripe: per tile, tile_stripes)."""
    stripes = range(stripe.max() + 1)
    low = np.array([(centers[stripe == s, 0] - fovs[stripe == s, 0] / 2).min() for s in stripes])
    high = np.array([(centers[stripe == s, 0] + fovs[stripe == s, 0] / 2).max() for s in stripes])
    return low, high


def contact_stripes(x_nm: np.ndarray, centers: np.ndarray, fovs: np.ndarray, stripe: np.ndarray) -> np.ndarray:
    """(N, n_stripes) bool: whether each contact (x, nm) lies in the x range of each stripe (stripe_ranges). A
    contact in the overlap of two stripes is in both; one outside every stripe is put in the nearest."""
    low, high = stripe_ranges(centers, fovs, stripe)
    inside = (x_nm[:, None] >= low) & (x_nm[:, None] <= high)
    outside = np.flatnonzero(~inside.any(axis=1))
    inside[outside, np.argmin(np.abs(x_nm[outside, None] - (low + high) / 2), axis=1)] = True
    return inside


def measurement_pairs(centers: np.ndarray, stripe: np.ndarray, first_upward: bool = True) -> set:
    """Tile pairs (i, j), i < j, measured one after the other in a serpentine scan: the stripes are
    measured in order of x, alternately upward and downward (the first from its bottom if
    first_upward), so the consecutive images are the neighbours within a stripe and, between stripe
    s and s + 1, the image where s ends and the one where s + 1 starts (both at the top after an
    upward stripe, both at the bottom after a downward one)."""
    pairs, ends = set(), []
    for s in range(stripe.max() + 1):
        tiles = np.flatnonzero(stripe == s)
        tiles = tiles[np.argsort(centers[tiles, 1])]  # bottom to top
        pairs |= {(min(a, b), max(a, b)) for a, b in zip(tiles[:-1], tiles[1:])}
        ends.append((tiles[0], tiles[-1]))
    for s in range(len(ends) - 1):
        upward = (s % 2 == 0) == first_upward
        a, b = (ends[s][1], ends[s + 1][1]) if upward else (ends[s][0], ends[s + 1][0])
        pairs.add((min(a, b), max(a, b)))
    return {(int(i), int(j)) for i, j in pairs}


def stripe_directions(centers: np.ndarray, stripe: np.ndarray, order: np.ndarray) -> list[bool | None]:
    """Per stripe, whether it was measured upward: its last image in measurement order (`order`, e.g. the file
    numbers) lies higher than its first. None for a stripe of one image."""
    directions = []
    for s in range(stripe.max() + 1):
        tiles = np.flatnonzero(stripe == s)
        if len(tiles) < 2:
            directions.append(None)
            continue
        first, last = tiles[np.argmin(order[tiles])], tiles[np.argmax(order[tiles])]
        directions.append(bool(centers[last, 1] > centers[first, 1]))
    return directions


def tile_neighbours(centers: np.ndarray, fovs: np.ndarray) -> list[tuple[int, int]]:
    """(i, j) pairs of tiles whose fields of view overlap (corners included), for
    design_errors(neighbours=...)."""
    return [(i, j) for i, j, _ in overlapping_pairs(centers, fovs)]


def tile_boxes_from_points(points: list[np.ndarray], margin_nm: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Per tile, the centre and size (n_tiles, 2 each, mask nm) of the bounding box of its points,
    grown by margin_nm on every side.

    For tiles without a known centre and FOV, e.g. contour CSVs in mask coordinates; passed to
    stitch_tiles as centres and FOVs. Without a margin, two neighbours that share only ONE row
    (or column) of contacts get boxes that just touch or, with a few nm of file offset, miss each
    other, and the pair is lost; a margin of about the match gate keeps it. Contacts that the
    margin adds to an overlap box find no partner and are harmless. Every tile needs at least one
    point (leave empty tiles out first).
    """
    empty = [k for k, p in enumerate(points) if len(p) == 0]
    if empty:
        raise ValueError(f"Tiles without points (leave them out): {empty}")
    lows = np.array([p.min(axis=0) for p in points]).reshape(-1, 2) - margin_nm
    highs = np.array([p.max(axis=0) for p in points]).reshape(-1, 2) + margin_nm
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
    """Incomplete only if tiles were left without a position; overlaps not matched alone are left out of the solve,
    and every tile is still placed through its other overlaps (D109)."""
    failed = [(r.i, r.j, r.matched) for r in result.rejected if r.failed]
    if len(result.unplaced):
        warnings.warn(
            f"{label} stitching incomplete: {len(result.unplaced)} tile(s) could not be stitched (NaN correction): "
            f"{result.unplaced.tolist()}; {len(failed)} overlapping pair(s) did not match "
            f"(tile i, tile j, matched contacts): {failed}",
            stacklevel=3,
        )
    elif failed:
        warnings.warn(
            f"{label} stitching: every tile stitched; {len(failed)} overlapping pair(s) did not match and are left "
            f"out (tile i, tile j, matched contacts): {failed}",
            stacklevel=3,
        )
