"""The release's analysis of one folder (docs/SPEC.md D77): only what the Stitching and the Registration,
moving window tabs show. No corrections (in-image map, drift curves), no RANSAC.

Steps: load the tiles (images: detection + design; contour CSVs: the paired centres), stitch the SEM and the
design (translation per tile; settings.stitch_method), the error of every contact (measured SEM − design)
at the stitched placement, merged over the overlaps, then the moving-window affine.
progress(fraction, stage) reports how far the run is, 0..1, for the status bar.

On request (the Contact size tab, hidden unless turned on): an image run keeps its loaded tiles in memory
(keep_tiles), and measure_contact_sizes measures the area, xCD and yCD of every contact on them. For display on
request: the SEM images (read_display_images), the design pattern (read_design_images) and every contact's contour
from the detection and after refinement (read_contours).

A run that stitched the tiles but measured too few contacts raises AnalysisFailed, which carries what was found
(the stitching, every tile's centres) and why each tile was not measured, so the app can show it.
"""

import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from scipy.spatial import cKDTree

from affine_ransac import __version__
from affine_ransac.analysis import (ContactSizes, Settings, check_settings, detect_layer, load_image_tiles,
                                   merged_contact_sizes, report_stitching, worker_count)
from affine_ransac.fitting.moving_window import MovingWindowResult, moving_window_affine
from affine_ransac.geometry.frames import pixel_to_tile_nm, tile_nm_to_pixel
from affine_ransac.geometry.polygon import polygon_centroid
from affine_ransac.io.design import contact_outlines, load_layout
from affine_ransac.io.contour_csv import read_contour_folder
from affine_ransac.io.sem_image import load_sem_image
from affine_ransac.pipeline import (StitchResult, TileResult, design_outlines, measurement_pairs,
                                    outlines_inside_frame, process_tile, slipped_pairs, stitch_design, stitch_tiles,
                                    stripe_directions, tile_boxes_from_points, tile_neighbours, tile_stripes)
from affine_ransac.registration import DesignErrors, MergedErrors, design_errors, merge_observations, paired_errors
from affine_ransac.stitching import placement_corrections

# Raised by one whenever the same folder and parameters would give other numbers than before (docs/SPEC.md,
# "Versions and run files"; tests/test_release_versions.py holds the results of each number). A saved run of a
# lower number is shown as outdated.
ANALYSIS = 3
DEFAULTS = Settings(max_match_nm=100.0)  # the release's defaults, where they differ from the research app's
# The settings a user can change (Parameters dialog), per kind of folder. The others keep their defaults. The
# contact layer is not one of them: it is detected in the .oas files (detect_layer); nor is first_stripe_upward:
# it is detected from the file numbers, which follow the measurement order (user; scan_direction).
COMMON = ["max_match_nm", "design_max_match_nm", "outlier_factor", "min_matched", "group_radius_nm", "row_gap_nm",
          "moving_window_um", "moving_step_um", "stitch_method"]
PARAMETERS = {"contour CSV": COMMON + ["tile_margin_nm"],
              "images": COMMON + ["detection", "search_px", "design_search_nm",
                                  "design_tolerance_nm", "min_match_fraction", "stitch_refined", "min_contrast", "realign"]}
LOAD_SHARE = 0.8  # fraction of the progress bar for loading the tiles (images: by far the longest step)
IMAGE_BUDGET_PX = 4e8  # SEM images shown on request: at most this many pixels in memory (~400 MB)


@dataclass
class LoadedTiles:
    """What measuring the contact sizes needs from an image run: its loaded tiles and how their contacts were
    merged. Kept in memory only when asked (analyse_folder keep_tiles; the tiles hold their images), never saved."""
    tiles: list[TileResult]
    observations: DesignErrors  # every tile's contacts paired with its design
    contacts: MergedErrors      # the observations merged into the result's contacts (same rows)


@dataclass
class ReleaseResult:
    folder: str
    kind: str                         # "contour CSV" or "images"
    settings: Settings
    tile_ids: list[str]
    centers: np.ndarray               # (n_tiles, 2) tile centres, mask nm
    sizes: np.ndarray                 # (n_tiles, 2) tile sizes (images: FOV), nm
    stripe: np.ndarray                # (n_tiles,) stripe of each tile, 0 = smallest x (pipeline.tile_stripes)
    sem_points: list[np.ndarray]      # per tile, SEM centres at the nominal placement, mask nm
    design_points: list[np.ndarray]   # per tile, design centres at the nominal placement, mask nm
    stitch: StitchResult              # SEM stitching
    design_stitch: StitchResult
    sem_nm: np.ndarray                # (M, 2) merged contacts, stitched SEM, mask nm
    design_nm: np.ndarray             # (M, 2) merged contacts, stitched design, mask nm
    moving: MovingWindowResult | None  # residuals: G_w(SEM) − design per contact (None: a failed run)
    image_paths: list[str]            # per tile its SEM image ([] for contour CSVs), shown on request
    log: list[str]
    oas_paths: list[str] = field(default_factory=list)  # per tile its .oas ("" if none; [] for contour CSVs)
    # (n_tiles, 2) mask position of each .oas file's (0, 0) at the nominal placement: the tile centre, or for
    # realigned tiles (settings.realign) where the file itself sits, stitched (its contacts are not that tile's)
    oas_centers: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    contact_sizes: ContactSizes | None = None  # images, once measured (measure_contact_sizes); saved
    tiles: LoadedTiles | None = None           # images run with keep_tiles; not saved
    version: str = __version__                 # the MaskReg that made the run ("": one before versions were saved)
    analysis: int = ANALYSIS                   # and its ANALYSIS (0: before they were saved)
    # SEM overlaps fitted more than max_match_nm from the shift the stitching in measurement order predicts, (i, j):
    # paired one contact row off? (stitch_sem; None: a run saved before this was kept, D109)
    slipped: list[tuple[int, int]] | None = field(default_factory=list)


class AnalysisFailed(RuntimeError):
    """A run that stopped after the stitching. result: what was found up to there (tiles, centres, stitchings; no
    contacts, moving None), to show on the Stitching tab, never saved; not_measured: tile index -> why no error
    was computed for that tile (registration.design_errors)."""

    def __init__(self, message: str, result: ReleaseResult, not_measured: dict[int, str]):
        super().__init__(message)
        self.result, self.not_measured = result, not_measured


def analyse_folder(folder, kind: str, settings: Settings, records: list | None = None,
                   log: Callable[[str], None] = print,
                   progress: Callable[[float, str], None] = lambda fraction, stage: None,
                   keep_tiles: bool = False) -> ReleaseResult:
    """Analyse a folder of this kind. records: the image folder's tiles (io/metadata.read_tile_index).
    keep_tiles: an image folder's result keeps its tiles in memory (result.tiles), to measure the contact sizes."""
    s = settings
    lines = []

    def record(text: str):
        lines.append(text)
        log(text)

    check_settings(s, kind)
    progress(0.0, "Reading the tiles")
    if kind == "images":
        progress(0.0, "Detecting the contact layer")
        layer, datatype = detect_layer(records)
        s = replace(s, layer=layer, datatype=datatype)
        record(f"Contact layer {layer}/{datatype} (the only layer with shapes in the .oas files)")
        tile_ids, tiles = load_image_tiles(
            records, s, record, lambda done, total: progress(LOAD_SHARE * done / total, f"Loading tile {done} of {total}"))
        centers = np.array([t.center_nm for t in tiles])
        sizes = np.array([t.fov_nm for t in tiles])
        stripe = tile_stripes(centers)
        sem_points = [t.sem_points_nm(s.stitch_refined) for t in tiles]
        design_points = [t.design_points_nm() for t in tiles]
        design_ok = [t.design.ok for t in tiles]
        neighbours = tile_neighbours(centers, sizes)
        design_stitch = stitch_design(tiles, s.design_max_match_nm, s.outlier_factor, s.min_matched)
        image_paths = [str(r.image_path) for r in records]
        oas_paths = [str(t.oas_path) if t.oas_path is not None else "" for t in tiles]
        oas_centers = np.array([t.center_nm if t.oas_center_nm is None else t.oas_center_nm for t in tiles])

        def errors(sem_corrections, design_corrections):  # each tile matched to its own design
            return design_errors(sem_points, design_points, sem_corrections, design_corrections, design_ok,
                                 s.design_search_nm, s.design_tolerance_nm, neighbours)
    else:
        all_tiles = read_contour_folder(folder)
        tiles = [t for t in all_tiles if len(t.design_nm)]
        if not tiles:
            raise ValueError(f"None of the {len(all_tiles)} contour CSV file(s) holds a contact")
        if len(tiles) < len(all_tiles):
            record(f"!! {len(all_tiles) - len(tiles)} file(s) without any contact, left out")
        record(f"{len(tiles)} files, {sum(len(t.design_nm) for t in tiles)} contacts")
        tile_ids = [t.name for t in tiles]
        sem_points = [t.sem_nm for t in tiles]
        design_points = [t.design_nm for t in tiles]
        centers, sizes = tile_boxes_from_points(design_points, s.tile_margin_nm)
        stripe = tile_stripes(centers, np.array([t.center_nm for t in tiles]))  # image centres, if in the files
        design_stitch = stitch_tiles(design_points, centers, sizes, s.design_max_match_nm, s.outlier_factor,
                                     s.min_matched, label="design")
        image_paths, oas_paths, oas_centers = [], [], np.empty((0, 2))

        def errors(sem_corrections, design_corrections):  # CSV rows are already paired
            return paired_errors(sem_points, design_points, sem_corrections, design_corrections)

    names = [Path(r.image_path).stem for r in records] if kind == "images" else tile_ids
    record(f"{stripe.max() + 1} stripe(s)")
    s = replace(s, first_stripe_upward=scan_direction(names, centers, stripe, s.first_stripe_upward, record))
    progress(LOAD_SHARE, "Stitching")
    stitch, slipped = stitch_sem(sem_points, centers, sizes, stripe, s, record, tile_ids)
    report_stitching(record, "Design", design_stitch, tile_ids)

    progress(LOAD_SHARE + 0.1, "Comparing with the design")
    sem_corrections, design_corrections = placement_corrections(stitch.corrections, design_stitch.corrections)["mean"]
    observations = errors(sem_corrections, design_corrections)
    if observations.skipped:
        record(f"!! {len(observations.skipped)} tile(s) NOT MEASURED: "
               f"{ {tile_ids[k]: reason for k, reason in observations.skipped.items()} }")
    contacts = merge_observations(observations, design_stitch.corrections, s.group_radius_nm)
    record(f"{len(observations.tile)} observations -> {len(contacts.count)} contacts")

    def result(moving, kept=None):
        return ReleaseResult(str(folder), kind, s, list(tile_ids), np.asarray(centers), np.asarray(sizes), stripe,
                             sem_points, design_points, stitch, design_stitch, contacts.sem_nm, contacts.design_nm,
                             moving, image_paths, lines, oas_paths, oas_centers, tiles=kept, slipped=slipped)
    if len(contacts.count) < 3:
        raise AnalysisFailed(f"Only {len(contacts.count)} measured contact(s). The Stitching tab shows the tiles, "
                             f"their SEM images and design, and why each tile was not measured",
                             result(None), observations.skipped)

    progress(LOAD_SHARE + 0.15, "Moving-window affine")
    moving = moving_window_affine(contacts.sem_nm, contacts.design_nm, s.moving_window_um * 1000,
                                  s.moving_step_um * 1000)
    progress(1.0, "Done")
    return result(moving, LoadedTiles(tiles, observations, contacts) if keep_tiles and kind == "images" else None)


def measure_contact_sizes(result: ReleaseResult, roi_width_nm: float = 0.0,
                          progress: Callable[[int, int], None] | None = None) -> ContactSizes:
    """Area, xCD and yCD of every contact (rows as result.design_nm) on the SEM contours and the .oas outlines,
    averaged over the tiles that saw it (analysis.merged_contact_sizes). Needs the run's tiles (keep_tiles).
    progress(done, total) after every tile."""
    if result.tiles is None:
        raise ValueError("Measuring the contact sizes needs the tiles of an image run in memory: restart the "
                         "analysis with the Contact size tab on")
    kept = result.tiles
    return merged_contact_sizes(kept.tiles, kept.contacts, kept.observations, result.settings, roi_width_nm, progress)


def file_numbers(names: list[str]) -> np.ndarray | None:
    """The number at the end of every name (e.g. CD000123 -> 123); None if a name has none or two are equal."""
    found = [re.search(r"(\d+)$", name) for name in names]
    if not all(found):
        return None
    numbers = np.array([int(f.group(1)) for f in found])
    return numbers if len(set(numbers.tolist())) == len(numbers) else None


def scan_direction(names: list[str], centers, stripe, fallback: bool, log) -> bool:
    """Whether the first stripe was measured upward, from the file numbers (measurement order); logged. If the
    names have no distinct numbers, or the first stripe has one image, `fallback`. Stripes that do not alternate
    (not a serpentine) are logged with !!: the pairs between stripes in measurement order are then wrong.
    stripe: per tile, tile_stripes."""
    numbers = file_numbers(names)
    directions = stripe_directions(centers, stripe, numbers) if numbers is not None else [None]
    if directions[0] is None:
        log(f"!! Scan direction not known from the file names: first stripe taken as measured "
            f"{'upward' if fallback else 'downward'}")
        return fallback
    upward = directions[0]
    log(f"Scan: first stripe measured {'upward' if upward else 'downward'} (file numbers)")
    odd = [s + 1 for s, d in enumerate(directions) if d is not None and d != ((s % 2 == 0) == upward)]
    if odd:
        log(f"!! Stripe(s) {odd} do not alternate in direction: not a serpentine scan, so the pairs measured one after "
            f"the other between stripes are not known (matters only for 'measurement order first' and the slipped-pair "
            f"check)")
    return upward


def stitch_sem(sem_points, centers, sizes, stripe, s: Settings, log,
               tile_ids) -> tuple[StitchResult, list[tuple[int, int]]]:
    """The SEM stitching over all overlaps, guided by the stitching in measurement order (stripe: per tile,
    tile_stripes) if s.stitch_method says so, and its pairs fitted one row off by that stitching (slipped_pairs;
    logged)."""
    order = stitch_tiles(sem_points, centers, sizes, s.max_match_nm, s.outlier_factor, s.min_matched,
                         label="SEM, measurement order",
                         only_pairs=measurement_pairs(centers, stripe, s.first_stripe_upward))
    guided = s.stitch_method == "measurement order first"
    stitch = stitch_tiles(sem_points, centers, sizes, s.max_match_nm, s.outlier_factor, s.min_matched,
                          label="SEM, all overlaps", guide=order.corrections if guided else None)
    log(f"Stitching method: {s.stitch_method}")
    report_stitching(log, "SEM", stitch, tile_ids)
    slipped = slipped_pairs(stitch, order.corrections, s.max_match_nm)
    if slipped:
        log(f"!! {len(slipped)} SEM overlap(s) fitted more than max_match_nm away from the shift the stitching in "
            f"measurement order predicts (contacts paired one row off?): "
            f"{[(tile_ids[i], tile_ids[j]) for i, j in slipped]}"
            + ("" if guided else "; the stitching method 'measurement order first' avoids this"))
    return stitch, slipped



def read_design_images(result: ReleaseResult, progress: Callable[[float, str], None] = lambda fraction, stage: None
                       ) -> list[np.ndarray | None]:
    """The design pattern for display: per tile, its .oas contacts (drawn_outlines) filled into a uint8 mask
    (1 = contact, rows down, like an SEM image) covering the tile's field of view; None for a tile without an .oas.
    Realigned tiles (their design is cut from every file) show their own file in normal tone, as the realignment
    reads it. Pixels per tile side from IMAGE_BUDGET_PX, at most 1024."""
    s, n = result.settings, len(result.oas_paths)
    side = int(min(1024, np.sqrt(IMAGE_BUDGET_PX / max(n, 1))))
    masks = []
    for k, (path, fov) in enumerate(zip(result.oas_paths, result.sizes)):
        if not path:
            masks.append(None)
        else:
            pixel_nm = fov[0] / side
            shape = (max(1, round(fov[1] / pixel_nm)), side)
            mask = np.zeros(shape, np.uint8)
            layout, layer = load_layout(path), (s.layer, s.datatype)
            outlines = contact_outlines(layout, *layer) if s.realign else \
                drawn_outlines(layout, layer, fov, result.design_points[k] - result.oas_centers[k])
            polygons = [np.round(tile_nm_to_pixel(o, shape, pixel_nm) * 16).astype(np.int32) for o in outlines]
            if polygons:
                cv2.fillPoly(mask, polygons, 1, shift=4)  # 4 fractional bits: sub-pixel edges
            masks.append(mask)
        progress((k + 1) / n, f"Reading design {k + 1} of {n}")
    return masks


def drawn_outlines(layout, layer: tuple[int, int], fov_nm, centres_nm: np.ndarray) -> list[np.ndarray]:
    """The contacts of one tile's .oas to draw, as outlines in tile-local nm: its merged shapes as drawn (normal
    tone), or, if the analysis read the file tone reversed, its holes. A reversed file draws the area around the
    holes, and filling that shape gave a solid block (D106). The tone is the one whose contacts are the tile's
    design centres (centres_nm, tile-local nm: what the analysis kept; choose_design_contacts picked the tone)."""
    drawn = contact_outlines(layout, *layer)
    if same_contacts(outlines_inside_frame(drawn, fov_nm), centres_nm):
        return drawn
    holes = design_outlines(layout, layer, fov_nm, tone_reversed=True)
    return holes if same_contacts(holes, centres_nm) else drawn


def same_contacts(outlines: list[np.ndarray], centres_nm: np.ndarray) -> bool:
    """Whether the outlines are the contacts at centres_nm: as many, every centre within 1 nm of an outline's
    centroid."""
    if len(outlines) != len(centres_nm):
        return False
    if not outlines:
        return True
    distance, _ = cKDTree([polygon_centroid(o) for o in outlines]).query(centres_nm)
    return bool((distance < 1.0).all())


@dataclass
class TileContours:
    """Every contact's contour from the detection and after edge refinement, for the contour layers (D113). Contact k
    lies in tile tile[k]; its contour is rows start[k]:start[k + 1] of `detected` and of `refined` (the same number of
    points: refinement moves each detected point along its normal), tile-local nm, closed (its first point again) and
    followed by a NaN row, so that all of them can be drawn as one line with connect="finite"."""
    tile: np.ndarray              # (n,) tile of each contact
    start: np.ndarray             # (n + 1,) its first row in detected / refined
    detected: np.ndarray          # (rows, 2) float32, tile-local nm
    refined: np.ndarray           # (rows, 2) float32
    detected_centres: np.ndarray  # (n, 2) tile-local nm: the centres a run uses with stitch_refined off
    refined_centres: np.ndarray   # (n, 2): with it on


def _tile_contours(arguments: tuple) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """process_tile(*arguments) reduced to what TileContours needs, tile-local nm: (detected contours, refined
    contours, detected centres, refined centres); at module level so that worker processes can run it."""
    tile = process_tile(*arguments)

    def local(points):
        return pixel_to_tile_nm(np.asarray(points, dtype=float).reshape(-1, 2), tile.image.shape, tile.pixel_size_nm)
    return ([local(c) for c in tile.otsu.contours], [local(c) for c in tile.refined.contours],
            local(tile.otsu.centers), local(tile.refined.centers))


def read_contours(result: ReleaseResult, progress: Callable[[float, str], None] = lambda fraction, stage: None,
                  log: Callable[[str], None] = lambda text: None) -> TileContours:
    """The contours of an image run (TileContours), found again as the run found them: each SEM image detected and
    refined with the run's settings (process_tile without its design), in worker processes like the tiles. The
    centres it used (refined or detected, settings.stitch_refined) must be the run's; tiles where they are not (an
    image changed since the run?) are logged with !!."""
    s = result.settings
    arguments = [(path, None, center, size, (s.layer, s.datatype), s.search_px, s.design_search_nm,
                  s.min_match_fraction, s.detection, s.design_tolerance_nm, False, s.min_contrast)
                 for path, center, size in zip(result.image_paths, result.centers, result.sizes)]
    tiles, detected, refined, detected_centres, refined_centres, differ = [], [], [], [], [], []
    with ProcessPoolExecutor(max_workers=worker_count()) as pool:
        for k, (found, moved, found_centres, moved_centres) in enumerate(pool.map(_tile_contours, arguments)):
            used = (moved_centres if s.stitch_refined else found_centres) + result.centers[k]
            if used.shape != result.sem_points[k].shape or not np.allclose(used, result.sem_points[k], atol=0.01):
                differ.append(result.tile_ids[k])
            tiles.append(np.full(len(found), k))
            detected += found
            refined += moved
            detected_centres.append(found_centres)
            refined_centres.append(moved_centres)
            progress((k + 1) / len(arguments), f"Contours of tile {k + 1} of {len(arguments)}")
    if differ:
        log(f"!! The contours of {len(differ)} tile(s) do not give the centres of the run (an image changed since the "
            f"run?): {differ}")

    def drawable(contours):  # each closed and followed by a NaN row
        nan = np.full((1, 2), np.nan)
        return np.concatenate([np.vstack([c, c[:1], nan]) for c in contours]).astype(np.float32) if contours \
            else np.empty((0, 2), np.float32)
    start = np.concatenate([[0], np.cumsum([len(c) + 2 for c in detected])]).astype(np.int64)
    return TileContours(np.concatenate(tiles) if tiles else np.empty(0, int), start, drawable(detected),
                        drawable(refined), np.concatenate(detected_centres).reshape(-1, 2),
                        np.concatenate(refined_centres).reshape(-1, 2))


def read_display_images(paths: list[str], progress: Callable[[float, str], None] = lambda fraction, stage: None
                        ) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """The SEM images for display: per tile (image, shape of the full image). Downsampled by a whole factor
    if all of them at full size would exceed IMAGE_BUDGET_PX (judged from the first image)."""
    first = load_sem_image(paths[0])
    factor = max(1, int(np.ceil(np.sqrt(first.size * len(paths) / IMAGE_BUDGET_PX))))
    images = []
    for k, path in enumerate(paths):
        image = first if k == 0 else load_sem_image(path)
        small = image if factor == 1 else cv2.resize(image, (image.shape[1] // factor, image.shape[0] // factor),
                                                     interpolation=cv2.INTER_AREA)
        images.append((small, image.shape))
        progress((k + 1) / len(paths), f"Loading SEM image {k + 1} of {len(paths)}")
    return images
