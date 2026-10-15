"""The release's analysis of one folder (docs/SPEC.md D77): only what the Stitching and the Registration,
moving window tabs show. No corrections (in-image map, drift curves), no RANSAC.

Steps: load the tiles (images: detection + design; contour CSVs: the paired centres), stitch the SEM and the
design (translation per tile; settings.stitch_method), the error of every contact (measured SEM − design)
at the stitched placement, merged over the overlaps, then the moving-window affine.
progress(fraction, stage) reports how far the run is, 0..1, for the status bar.
"""

from dataclasses import dataclass, replace
from typing import Callable

import cv2
import numpy as np

from affine_ransac.analysis import Settings, check_settings, detect_layer, load_image_tiles, report_stitching
from affine_ransac.fitting.moving_window import MovingWindowResult, moving_window_affine
from affine_ransac.io.contour_csv import read_contour_folder
from affine_ransac.io.sem_image import load_sem_image
from affine_ransac.pipeline import (StitchResult, measurement_pairs, slipped_pairs, stitch_design, stitch_tiles,
                                    tile_boxes_from_points, tile_neighbours, tile_stripes)
from affine_ransac.registration import design_errors, merge_observations, paired_errors
from affine_ransac.stitching import placement_corrections

DEFAULTS = Settings(max_match_nm=100.0)  # the release's defaults, where they differ from the research app's
# The settings a user can change (Parameters dialog), per kind of folder. The others keep their defaults. The
# contact layer is not one of them: it is detected in the .oas files (detect_layer).
COMMON = ["max_match_nm", "design_max_match_nm", "outlier_factor", "min_matched", "group_radius_nm", "row_gap_nm",
          "moving_window_um", "moving_step_um", "stitch_method", "first_stripe_upward"]
PARAMETERS = {"contour CSV": COMMON + ["tile_margin_nm"],
              "images": COMMON + ["detection", "search_px", "design_search_nm",
                                  "design_tolerance_nm", "min_match_fraction", "stitch_refined"]}
LOAD_SHARE = 0.8  # fraction of the progress bar for loading the tiles (images: by far the longest step)
IMAGE_BUDGET_PX = 4e8  # SEM images shown on request: at most this many pixels in memory (~400 MB)


@dataclass
class ReleaseResult:
    folder: str
    kind: str                         # "contour CSV" or "images"
    settings: Settings
    tile_ids: list[str]
    centers: np.ndarray               # (n_tiles, 2) tile centres, mask nm
    sizes: np.ndarray                 # (n_tiles, 2) tile sizes (images: FOV), nm
    sem_points: list[np.ndarray]      # per tile, SEM centres at the nominal placement, mask nm
    design_points: list[np.ndarray]   # per tile, design centres at the nominal placement, mask nm
    stitch: StitchResult              # SEM stitching
    design_stitch: StitchResult
    sem_nm: np.ndarray                # (M, 2) merged contacts, stitched SEM, mask nm
    design_nm: np.ndarray             # (M, 2) merged contacts, stitched design, mask nm
    moving: MovingWindowResult        # residuals: G_w(SEM) − design per contact
    image_paths: list[str]            # per tile its SEM image ([] for contour CSVs), shown on request
    log: list[str]


def analyse_folder(folder, kind: str, settings: Settings, records: list | None = None,
                   log: Callable[[str], None] = print,
                   progress: Callable[[float, str], None] = lambda fraction, stage: None) -> ReleaseResult:
    """Analyse a folder of this kind. records: the image folder's tiles (io/metadata.read_tile_index)."""
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
        sem_points = [t.sem_points_nm(s.stitch_refined) for t in tiles]
        design_points = [t.design_points_nm() for t in tiles]
        design_ok = [t.design.ok for t in tiles]
        neighbours = tile_neighbours(centers, sizes)
        design_stitch = stitch_design(tiles, s.design_max_match_nm, s.outlier_factor, s.min_matched)
        image_paths = [str(r.image_path) for r in records]

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
        design_stitch = stitch_tiles(design_points, centers, sizes, s.design_max_match_nm, s.outlier_factor,
                                     s.min_matched, label="design")
        image_paths = []

        def errors(sem_corrections, design_corrections):  # CSV rows are already paired
            return paired_errors(sem_points, design_points, sem_corrections, design_corrections)

    progress(LOAD_SHARE, "Stitching")
    stitch = stitch_sem(sem_points, centers, sizes, s, record, tile_ids)
    report_stitching(record, "Design", design_stitch, tile_ids)

    progress(LOAD_SHARE + 0.1, "Comparing with the design")
    sem_corrections, design_corrections = placement_corrections(stitch.corrections, design_stitch.corrections)["mean"]
    observations = errors(sem_corrections, design_corrections)
    if observations.skipped:
        record(f"!! {len(observations.skipped)} tile(s) NOT MEASURED: "
               f"{ {tile_ids[k]: reason for k, reason in observations.skipped.items()} }")
    contacts = merge_observations(observations, design_stitch.corrections, s.group_radius_nm)
    record(f"{len(observations.tile)} observations -> {len(contacts.count)} contacts")
    if len(contacts.count) < 3:
        raise RuntimeError(f"Only {len(contacts.count)} measured contact(s): see the tiles not measured above")

    progress(LOAD_SHARE + 0.15, "Moving-window affine")
    moving = moving_window_affine(contacts.sem_nm, contacts.design_nm, s.moving_window_um * 1000,
                                  s.moving_step_um * 1000)
    progress(1.0, "Done")
    return ReleaseResult(str(folder), kind, s, list(tile_ids), np.asarray(centers), np.asarray(sizes), sem_points,
                         design_points, stitch, design_stitch, contacts.sem_nm, contacts.design_nm, moving,
                         image_paths, lines)


def stitch_sem(sem_points, centers, sizes, s: Settings, log, tile_ids) -> StitchResult:
    """The SEM stitching over all overlaps, guided by the stitching in measurement order if
    s.stitch_method says so; pairs fitted one row off (by that stitching) are logged."""
    stripe = tile_stripes(centers, sizes)
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
    return stitch



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
