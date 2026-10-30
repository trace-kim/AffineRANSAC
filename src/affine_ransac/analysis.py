"""End-to-end analysis of one data folder, the flow of the notebooks, for the standalone app (app.py).

Two kinds of folder (docs/SPEC.md §4):
- pre-analysed contour CSVs (csv_stitch.ipynb): analyse_contour_folder(folder, settings);
- SEM images + .oas + metadata CSV (stitch_viewer.ipynb): analyse_images(folder, records, settings),
  with the records of the metadata reader (io/metadata.read_tile_index, which exists only on the data
  machine, so the caller passes them in).
Both run the same steps on the tiles' points (_analyse): stitching (translation; all overlaps at once, or
guided by a first stitching in measurement order or by placing the images one at a time: settings.stitch_method), errors
per placement and merged contacts, the in-image map, the drift curve, the drift per stripe in
measurement order, RANSAC and the moving window for every set of contacts, and the stitching residuals.
Before any work, the settings and (images) the records are checked and a few sample tiles are matched
to their design (check_settings, check_image_records), so a wrong setting or folder fails in seconds.
On request after an image run, measure_contact_sizes measures every contact's area, xCD and yCD on the
SEM contour and on its .oas outline (ContactSizes; contour CSVs have no outlines).
Both corrections together (the in-image corrected points with their drift per stripe removed) are
stitched for the residuals and, placed with translation + rotation per tile, are one more set.
restitch(result, method) computes the stitching and everything after it again with another stitching
method, from what is in memory (contour CSVs: the folder is read again).
The AnalysisResult holds what the analysis window shows (view_analysis.analysis_window). No UI code.
`log` receives what the notebooks print (counts, failures).
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, fields, replace
from itertools import repeat
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from affine_ransac.features.contact import BAND_MIN_SOLIDITY, DENOISERS
from affine_ransac.fitting.drift import DriftCurve, StripeDrift, drift_curve, stripe_drift
from affine_ransac.fitting.moving_window import MovingWindowResult, moving_window_affine
from affine_ransac.fitting.ransac import RansacResult, ransac_affine
from affine_ransac.intrafield import IntrafieldMap, correct_points, estimate_formula
from affine_ransac.io.contour_csv import read_contour_folder
from affine_ransac.io.design import layers_with_shapes, list_layers, load_layout, read_polygons
from affine_ransac.pipeline import (REFINEMENTS, FingerprintChoice, OverlapCheck, StitchResult, TileResult,
                                    measurement_pairs, overlap_checks, placed_by_fingerprint, placed_image_by_image,
                                    process_tile, slipped_pairs, stitch_design, stitch_image_by_image, stitch_tiles,
                                    stitch_tiles_rigid, tile_boxes_from_points, tile_neighbours, tile_stripes)
from affine_ransac.realign import realign_tiles
from affine_ransac.registration import (DesignErrors, MergedErrors, design_errors, merge_observations, merge_values,
                                        paired_errors)
from affine_ransac.stitching import placement_corrections


def setting(default, help_text: str):
    """A Settings field with its help text (the notebooks' comment), shown as a tooltip by the app."""
    return field(default=default, metadata={"help": help_text})


@dataclass
class Settings:
    """The notebooks' settings cells: same names in lower case, same units."""
    max_match_nm: float = setting(25.0, "SEM overlaps: pair contacts closer than this, the largest B - A stage "
                                        "offset (< pitch / 2). Stitching image by image or fingerprint: the largest "
                                        "offset between neighbouring images, any size")
    overlap_tolerance_nm: float = setting(2.0, "Stitching image by image or fingerprint: after the stitching, an "
                                                "overlap whose two images lie further apart than this is an error. "
                                                "(The stitching pairs contacts within a quarter of the contact "
                                                "spacing and fits them robustly; it does not use this.)")
    design_max_match_nm: float = setting(25.0, "Design overlaps: the largest offset between neighbouring files "
                                               "(< pitch / 2)")
    outlier_factor: float = setting(3.0, "Overlap outlier if its residual > this x the median residual")
    min_matched: int = setting(5, "Overlaps with fewer matched contacts are not used")
    group_radius_nm: float = setting(5.0, "Observations from different tiles whose stitched design positions are "
                                          "this close = one contact")
    ransac_placement: str = setting("mean", "RANSAC on the merged contacts of this placement: mean (stitched) or "
                                            "first (first tile fixed)")
    ransac_threshold_nm: float = setting(0.5, "RANSAC inlier if |G(SEM) - design| < this (fixed)")
    ransac_seed: int = setting(0, "RANSAC random samples are reproducible for a given seed")
    ransac_delay_ms: int = setting(300, "RANSAC monitor: pause between steps")
    row_gap_nm: float = setting(10.0, "Rows: a new row starts where the sorted design y jumps by more than this")
    moving_window_um: float = setting(40.0, "Moving-window affine: window height along y")
    moving_step_um: float = setting(5.0, "Moving-window affine: the window moves up this much each step")
    moving_window_per_stripe: bool = setting(False, "Moving-window affine fitted per stripe: every image corrected by "
                                                    "the windows of its own stripe (stripes are measured at different "
                                                    "times and drift differently); off: every window spans all stripes")
    in_image_correction: bool = setting(False, "Remove the distortion shared by all SEM images (the same at the same "
                                               "place in every image, e.g. of the scan) from the measured centres "
                                               "before stitching: one smooth formula over the image, every power of x "
                                               "and y from 2 to IN_IMAGE_ORDER, measured against the design in all "
                                               "images at once, each with its own affine; left out (the log says so) "
                                               "if the contacts cannot determine it. Mask errors that repeat at the "
                                               "image spacing would be removed with it")
    in_image_order: int = setting(3, "In-image formula: the highest power of x and y (>= 2; 3: 14 numbers, the "
                                     "barrel, pincushion and spiral distortion of the beam's deflection and lenses, "
                                     "and an S-shaped scan; 4: 24, 5: 36). Higher follows finer patterns but swings "
                                     "more at the image edges")
    drift_window_um: float = setting(40.0, "Drift curve: height of the local straight-line fit along y")
    first_stripe_upward: bool = setting(True, "Serpentine scan: the first stripe (smallest x) was measured from its "
                                              "bottom up, then alternately")
    stitch_method: str = setting("all overlaps", "all overlaps: every overlap paired at the nominal placement. "
                                 "measurement order first: stitch the images taken one after the other first, then "
                                 "pair every overlap at the offset that predicts (an offset near one pitch between "
                                 "images taken far apart in time cannot pair contacts one row off). image by image: "
                                 "place the images one at a time, each where its contacts line up with those of all "
                                 "its placed neighbours at once (offsets up to max_match_nm; on a repeating pattern "
                                 "the whole-pitch shift nearest its stage position), then pair every overlap there "
                                 "(overlaps cannot disagree by a pitch). fingerprint: image by image, each image's "
                                 "whole-pitch shift chosen by the residual of its pairs: at the true one each pair is "
                                 "one contact seen twice, a pitch off it is two neighbours, which differ also by the "
                                 "mask's own placement errors. Needs placement errors larger than the noise; where "
                                 "the residuals cannot tell the shifts apart, as image by image (the log says how "
                                 "many)")
    # Contour CSVs only
    tile_margin_nm: float = setting(25.0, "Tile box = bounding box of its design centres grown by this")
    # Images only
    layer: int = setting(100, "Layer of the contacts in the .oas")
    datatype: int = setting(0, "Datatype of the contacts in the .oas")
    background_px: float = setting(0.0, "Flatten the background first, so that one threshold fits the whole image "
                                        "when its brightness varies (charging, shading): the brightness around the "
                                        "contacts (filled in over a disk this many pixels across, then smoothed) is "
                                        "subtracted. Wider than a contact, narrower than the brightness changes; "
                                        "0 = off")
    denoise: str = setting("none", "Remove pixel noise before the contacts are found (after the flattening, before "
                                   "the upscaling). median: each pixel the median of a window MEDIAN_PX wide: removes "
                                   "isolated bright or dark pixels and small spots, keeps edges. non-local means: each "
                                   "pixel a mean of the pixels whose surroundings look alike (NLM_H); can wipe out "
                                   "contacts only a few pixels across. total variation: flat regions with sharp edges "
                                   "between them (TV_WEIGHT). none: off")
    median_px: int = setting(3, "Denoise median: window width in pixels (odd); removes spots narrower than half of it")
    nlm_h: float = setting(10.0, "Denoise non-local means: strength in grey levels (OpenCV h); larger removes more "
                                 "noise, and more detail")
    tv_weight: float = setting(0.1, "Denoise total variation: weight (on grey levels / 255); larger gives flatter "
                                    "regions and removes more detail")
    upscale: int = setting(1, "Interpolate the image (bicubic) to this many times as many pixels per side, then find "
                              "the contacts in it (after the flattening): finer threshold outlines and centres for "
                              "contacts only a few pixels across. The other settings in pixels stay pixels of the "
                              "image as loaded. Time and memory grow with its square; 1 = off")
    detection: str = setting("band", "How contacts are found (each a region with a first, pixel outline; REFINEMENT "
                                      "then places its edge). band: the regions a bright band encloses, however grey "
                                      "the hole. otsu: the dark regions, for images without a bright band. otsu3: the "
                                      "regions darker than the background; finds contacts whose band noise breaks "
                                      "up, but more false regions")
    dark_contacts: bool = setting(True, "Contacts darker than the background (holes, with a bright band); off: "
                                        "brighter. Detection and edge refinement")
    detection_blur_px: float = setting(1.5, "Detection: Gaussian blur (sigma, px) of the image before thresholding; "
                                            "0 = none")
    min_area_px: int = setting(20, "Detection: regions of fewer pixels are left out (noise)")
    max_area_px: int = setting(0, "Detection: regions of more pixels are left out; 0 = no limit")
    min_solidity: float = setting(BAND_MIN_SOLIDITY, "Detection, band only: regions less solid than this (area / "
                                                     "convex hull area) are left out. Contacts are ~1, background "
                                                     "pinched off between nearly touching bands 0.7-0.88; 0 = off")
    search_px: float = setting(5.0, "Edge refinement searches this many pixels in/out along the contour normal")
    refinement: str = setting("mid-level", "Edge of each contact, used for all results. mid-level: where the smoothed "
                                           "image is half-way between the contact's interior and its bright band. "
                                           "threshold: where it crosses the detection's threshold, one level for the "
                                           "whole image: for images without a bright band (flatten the background if "
                                           "the brightness varies). steepest gradient: where it rises fastest "
                                           "(noisier; before D122). none: no refinement, the detected regions' centres")
    edge_step_px: float = setting(0.25, "Edge refinement: the image is sampled along each normal every this many "
                                        "pixels (<= search_px)")
    edge_blur_px: float = setting(1.0, "Edge refinement: Gaussian blur (sigma, px) of the image the edges are found "
                                       "on; 0 = none")
    normal_span: int = setting(3, "Edge refinement: each contour point's normal runs across the contour points this "
                                  "many before and after it (smooths the pixel staircase)")
    design_search_nm: float = setting(100.0, "Design-SEM matching: largest offset of one image's SEM contacts "
                                             "against its own .oas, before stitching (its stage error); for a "
                                             "regular array also > pitch / 2")
    design_tolerance_nm: float = setting(10.0, "Design-SEM matching: pair contacts within this after removing the "
                                               "tile shift (< pitch / 2)")
    min_match_fraction: float = setting(0.5, "A design tone is accepted if at least this fraction of the SEM "
                                             "contacts match it")
    min_contrast: float = setting(0.0, "Drop detected contacts whose mean grey level differs from a 3 px ring "
                                       "around them by less than this many noise sigma (false contacts in images "
                                       "without pattern); 0 = off. Noisy images give real contacts only a few "
                                       "sigma: check the log that it keeps them")
    realign: bool = setting(False, "Images all taken at one wrong offset from their design (failed SEM alignment): "
                                   "stitch the SEM alone, find the offset to the design of all .oas files together "
                                   "(any size; on a repeating pattern, of the equally fitting ones the one pairing the "
                                   "most contacts), give every image the design in its shifted field (normal tone only)")
    spread_flag_nm: float = setting(2.0, "Stitch viewer: ring merged contacts whose SEM observations differ by more")
    arrow_scale: float = setting(10.0, "Stitch viewer: error lines are drawn this many times longer than the error")


CSV_ONLY = {"tile_margin_nm"}
RELEASE_ONLY = {"moving_window_per_stripe", "in_image_correction"}  # used by the release's analysis only
                                                                     # (release/analysis.py)
IMAGES_ONLY = {"layer", "datatype", "detection", "search_px", "refinement", "design_search_nm", "design_tolerance_nm",
               "min_match_fraction", "spread_flag_nm", "arrow_scale", "min_contrast", "realign", "dark_contacts",
               "detection_blur_px", "min_area_px", "max_area_px", "min_solidity", "edge_step_px", "edge_blur_px",
               "normal_span", "background_px", "upscale", "denoise", "median_px", "nlm_h", "tv_weight"}
# The settings that find the contacts in an SEM image (pipeline.find_contacts takes them by these names; D128)
CONTOUR_SETTINGS = ("background_px", "denoise", "median_px", "nlm_h", "tv_weight", "upscale", "detection",
                    "dark_contacts", "detection_blur_px", "min_area_px", "max_area_px", "min_solidity", "min_contrast",
                    "refinement", "search_px", "edge_step_px", "edge_blur_px", "normal_span")
STITCH_METHODS = ("all overlaps", "measurement order first", "image by image", "fingerprint")
BY_IMAGE = ("image by image", "fingerprint")  # stitched image by image after a placement (D133, D144)
CHOICES = {"ransac_placement": ("mean", "first"), "detection": ("otsu", "otsu3", "band"), "refinement": REFINEMENTS,
           "stitch_method": STITCH_METHODS, "denoise": DENOISERS}
# What runs saved before a setting existed did, where that differs from the setting's default: a saved run is read
# with these (saved_settings), so that it says how it was made. Remembered parameters take the defaults instead.
BEFORE = {"refinement": "steepest gradient"}  # D122


POSITIVE = {"max_match_nm", "overlap_tolerance_nm", "design_max_match_nm", "outlier_factor", "group_radius_nm", "ransac_threshold_nm",
            "row_gap_nm", "moving_window_um", "moving_step_um", "drift_window_um", "search_px", "design_search_nm",
            "design_tolerance_nm", "spread_flag_nm", "arrow_scale"}  # must be > 0
MAX_UPSCALE = 8  # a 2048² image upscaled 8 times holds 268 million pixels
SAMPLE_TILES = 3  # tiles matched to their design before the run (first, middle, last with an .oas)


def current_names(values: dict) -> dict:
    """Stored setting values (a saved run's, or remembered parameters) as today's settings: stitch_refined, a check
    box until D124, unticked is refinement "none" (ticked: the run's refinement)."""
    values = dict(values)
    if values.pop("stitch_refined", True) is False:
        values["refinement"] = "none"
    return values


def saved_settings(values: dict) -> Settings:
    """The Settings of a saved run from its stored values (current_names): names unknown to this code are ignored; a
    setting the run does not have takes what runs did before it existed (BEFORE), else its default."""
    known = {f.name for f in fields(Settings)}
    return Settings(**{**BEFORE, **{name: value for name, value in current_names(values).items() if name in known}})


def settings_for(kind: str) -> list[str]:
    """Names of the settings this analysis uses for a folder of this kind ("contour CSV" or "images")."""
    unused = (IMAGES_ONLY if kind == "contour CSV" else CSV_ONLY) | RELEASE_ONLY
    return [f.name for f in fields(Settings) if f.name not in unused]


def check_settings(s: Settings, kind: str):
    """ValueError listing every setting of this kind of folder that is out of range."""
    used = set(settings_for(kind))
    problems = [f"{name} must be > 0 (is {getattr(s, name)})" for name in sorted(POSITIVE & used)
                if not getattr(s, name) > 0]
    limits = {"min_matched": (s.min_matched >= 1, "must be >= 1"),
              "in_image_order": (s.in_image_order >= 2, "must be >= 2"),
              "ransac_delay_ms": (s.ransac_delay_ms >= 0, "must be >= 0"),
              "tile_margin_nm": (s.tile_margin_nm >= 0, "must be >= 0"),
              "min_contrast": (s.min_contrast >= 0, "must be >= 0"),
              "min_match_fraction": (0 < s.min_match_fraction <= 1, "must be in (0, 1]"),
              "background_px": (s.background_px == 0 or s.background_px >= 3, "must be 0 (off) or >= 3"),
              "upscale": (1 <= s.upscale <= MAX_UPSCALE, f"must be 1 (off) to {MAX_UPSCALE}"),
              "median_px": (s.median_px >= 3 and s.median_px % 2 == 1, "must be odd and >= 3"),
              "nlm_h": (s.nlm_h > 0, "must be > 0"),
              "tv_weight": (s.tv_weight > 0, "must be > 0"),
              "detection_blur_px": (s.detection_blur_px >= 0, "must be >= 0"),
              "min_area_px": (s.min_area_px >= 0, "must be >= 0"),
              "max_area_px": (s.max_area_px >= 0, "must be >= 0 (0: no limit)"),
              "min_solidity": (0 <= s.min_solidity <= 1, "must be in [0, 1]"),
              "edge_step_px": (0 < s.edge_step_px <= s.search_px, "must be > 0 and <= search_px"),
              "edge_blur_px": (s.edge_blur_px >= 0, "must be >= 0"),
              "normal_span": (s.normal_span >= 1, "must be >= 1")}
    problems += [f"{name} {rule} (is {getattr(s, name)})" for name, (ok, rule) in limits.items()
                 if name in used and not ok]
    problems += [f"{name} must be one of {choices} (is {getattr(s, name)!r})" for name, choices in CHOICES.items()
                 if name in used and getattr(s, name) not in choices]
    if problems:
        raise ValueError("Invalid settings: " + "; ".join(problems))


def contour_options(s: Settings) -> dict:
    """The settings that find the contacts in an SEM image, by name (pipeline.find_contacts, process_tile)."""
    return {name: getattr(s, name) for name in CONTOUR_SETTINGS}


def check_image_records(records: list, layer: tuple[int, int]):
    """ValueError if the image folder cannot be analysed: no records, image or .oas files missing, no image
    with an .oas, or the contact layer missing from the (first) .oas."""
    if not records:
        raise ValueError("The metadata CSV lists no images")
    missing = [r.tile_id for r in records if not Path(r.image_path).is_file()]
    if missing:
        raise ValueError(f"{len(missing)} image file(s) not found, e.g. {missing[:3]}")
    with_oas = [r for r in records if r.oas_path is not None]
    if not with_oas:
        raise ValueError("No image has a paired .oas file (check the ContourCAD folder and the file names)")
    missing = [r.tile_id for r in with_oas if not Path(r.oas_path).is_file()]
    if missing:
        raise ValueError(f"{len(missing)} .oas file(s) not found, e.g. for {missing[:3]}")
    same = same_position_records(records)
    if same:
        raise ValueError(f"{len(same)} pair(s) of images at the same position (the same point measured twice?), "
                         f"e.g. {same[:3]}: the metadata reader must keep one image per measure point (task T007)")
    layers = list_layers(load_layout(with_oas[0].oas_path))
    if tuple(layer) not in layers:
        raise ValueError(f"Layer {layer[0]}/{layer[1]} is not in {Path(with_oas[0].oas_path).name}; "
                         f"its layers: {layers}")


def same_position_records(records: list, tolerance_nm: float = 1.0) -> list[tuple[str, str]]:
    """(tile_id, tile_id) of every two records whose image centres are closer than tolerance_nm. Two images of
    one point (a measurement repeated after failed pattern matching) would be stitched as two tiles, D94."""
    centers = np.array([(r.center_x_nm, r.center_y_nm) for r in records], dtype=float).reshape(-1, 2)
    pairs = cKDTree(centers).query_pairs(tolerance_nm, output_type="ndarray")
    return [(records[i].tile_id, records[j].tile_id) for i, j in sorted(pairs.tolist())]


def detect_layer(records: list) -> tuple[int, int]:
    """The one (layer, datatype) holding shapes in the .oas files of the records (every file is read). ValueError
    if no file has shapes on any layer, or if they hold more than one (it names the layers and a file of each)."""
    found = {}  # (layer, datatype) -> the first file that holds it
    for r in records:
        if r.oas_path is not None and Path(r.oas_path).is_file():
            for layer in layers_with_shapes(load_layout(r.oas_path)):
                found.setdefault(layer, Path(r.oas_path).name)
    if not found:
        raise ValueError("No .oas file holds any shape: no contact layer to use")
    if len(found) > 1:
        raise ValueError(f"The .oas files hold {len(found)} layers, so the contact layer is not known: "
                         + ", ".join(f"{layer}/{datatype} (e.g. {name})" for (layer, datatype), name in sorted(found.items())))
    return next(iter(found))


@dataclass
class ContactSet:
    """Merged contacts (one per physical contact) and their affine fits: one set of lines in the window."""
    sem_nm: np.ndarray                 # (M, 2) stitched SEM centres, mask nm (no affine)
    design_nm: np.ndarray              # (M, 2) stitched design centres, mask nm
    ransac: RansacResult
    moving: MovingWindowResult | None  # None: no moving-window fit for this set


@dataclass
class ContactSizes:
    """Area (nm²), xCD and yCD (nm) of every contact of the "uncorrected" set (same rows), the mean over
    the tiles that saw it; NaN where no chord crossed the outline (measure_contact_sizes)."""
    sem: np.ndarray      # (M, 3) measured on the SEM contour (TileResult.contact_sizes)
    design: np.ndarray   # (M, 3) measured on the .oas outline
    roi_width_nm: float  # the CDs are the mean of the chords in a band this wide (0: one chord)


@dataclass
class Stitching:
    """Points stitched over all overlaps, for the stitching residuals."""
    points: list[np.ndarray]  # per tile, (N, 2) mask nm at the nominal placement
    stitch: StitchResult      # translation per tile: used pairs, their tie contacts, corrections
    rigid: np.ndarray         # (n_tiles, 3, 3) translation + rotation per tile from the same ties


@dataclass
class ImageTiles:
    """What the image stitch viewer needs besides the points; not saved (the images are large)."""
    tiles: list[TileResult]
    design_polygons: list[list[np.ndarray]]
    merged: dict[str, MergedErrors]  # per placement
    observations: dict[str, DesignErrors]  # per placement: the observations that were merged


@dataclass
class AnalysisResult:
    folder: str
    kind: str                          # "contour CSV" or "images"
    settings: Settings
    tile_ids: list[str]
    centers: np.ndarray                # (n_tiles, 2) tile box centres, mask nm (images: metadata centres)
    sizes: np.ndarray                  # (n_tiles, 2) tile box sizes, nm (images: FOV)
    design_points: list[np.ndarray]    # per tile, design centres at the nominal placement, mask nm
    design_stitch: StitchResult
    stitchings: dict[str, Stitching]   # "raw", "in-image corrected", "stripe drift corrected",
                                       # "in-image + stripe drift corrected" (both, in that order)
    sets: dict[str, ContactSet]        # "uncorrected" (translation stitching), then the corrected sets; the last
                                       # one is placed with translation + rotation per tile
    drift: DriftCurve                  # its bend was subtracted from the "drift corrected" set
    stripe_drift: StripeDrift          # its curves were subtracted from the "stripe drift corrected" set
    log: list[str]                     # what the run reported
    contact_sizes: ContactSizes | None = None  # images only, when measured (measure_contact_sizes)
    images: ImageTiles | None = None   # images only, in memory after a run


def analyse_contour_folder(folder, settings: Settings, log: Callable[[str], None] = print) -> AnalysisResult:
    """The csv_stitch.ipynb flow on a folder of contour CSVs (io.contour_csv)."""
    s = settings
    check_settings(s, "contour CSV")
    log, lines = _recording(log)
    all_tiles = read_contour_folder(folder)
    empty = [t.name for t in all_tiles if len(t.design_nm) == 0]
    if empty:
        log(f"!! {len(empty)} file(s) without any contact, left out: {empty}")
    tiles = [t for t in all_tiles if len(t.design_nm)]
    if not tiles:
        raise ValueError(f"None of the {len(all_tiles)} contour CSV file(s) holds a contact")
    log(f"{len(tiles)} of {len(all_tiles)} files used, {sum(len(t.design_nm) for t in tiles)} contacts, "
        f"{sum(t.dropped for t in all_tiles)} rows left out (missing value)")
    sem_points = [t.sem_nm for t in tiles]
    design_points = [t.design_nm for t in tiles]
    centers, sizes = tile_boxes_from_points(design_points, s.tile_margin_nm)
    design_stitch = stitch_tiles(design_points, centers, sizes, s.design_max_match_nm, s.outlier_factor,
                                 s.min_matched, label="design")

    def errors(points, sem_corrections, design_corrections):  # CSV rows are already paired
        return paired_errors(points, design_points, sem_corrections, design_corrections)

    image_centers = np.array([t.center_nm for t in tiles])  # NaN without the DesignX, DesignY columns
    result, _, _ = _analyse(str(folder), "contour CSV", s, log, [t.name for t in tiles], sem_points, design_points,
                         centers, sizes, image_centers, design_stitch, errors)
    result.log = lines
    return result


def analyse_images(folder, records: list, settings: Settings, log: Callable[[str], None] = print) -> AnalysisResult:
    """The stitch_viewer.ipynb flow on the tiles of an image folder. records: per tile, an object with the
    TileRecord fields of io/metadata.read_tile_index (tile_id, image_path, oas_path, center_x_nm,
    center_y_nm, fov_x_nm, fov_y_nm)."""
    s = settings
    log, lines = _recording(log)
    tile_ids, tiles = load_image_tiles(records, s, log)
    polygons = [read_polygons(load_layout(r.oas_path), s.layer, s.datatype)
                if t.oas_path and t.oas_center_nm is None else []
                for r, t in zip(records, tiles)]  # a realigned tile's own .oas is not its design: no outlines
    result = _analyse_tiles(str(folder), s, log, tile_ids, tiles, polygons)
    result.log = lines
    return result


def worker_count() -> int:
    """Worker processes for loading tiles: half the logical cores, about one per physical core (D87).
    One per logical core ran out of memory on large tiles (each worker holds a whole tile's arrays)."""
    return max(1, (os.cpu_count() or 2) // 2)


def _process_tile(arguments: tuple, options: dict) -> TileResult:
    """process_tile(*arguments, **options); at module level so that worker processes can run it."""
    return process_tile(*arguments, **options)


def load_image_tiles(records: list, settings: Settings, log: Callable[[str], None] = print,
                     progress: Callable[[int, int], None] | None = None) -> tuple[list[str], list[TileResult]]:
    """The tile ids and loaded tiles (process_tile) of an image folder's records, after the checks (settings,
    records, sample tiles matched to their design). progress(done, total) after every tile.
    settings.realign: the design is read without matching, then the tiles are realigned
    (realign.realign_tiles) before the flagged tiles are counted."""
    s = settings
    layer = (s.layer, s.datatype)
    check_settings(s, "images")
    check_image_records(records, layer)
    tile_ids = [r.tile_id for r in records]
    realign = s.realign
    if realign:
        log("!! Realignment on: images are not matched to their own .oas")
    without_oas = [r.tile_id for r in records if r.oas_path is None]
    if without_oas:
        log(f"!! {len(without_oas)} image(s) without an .oas: {without_oas}")

    def arguments(r):  # plain values only, so they can be sent to a worker process
        return r.image_path, r.oas_path, (r.center_x_nm, r.center_y_nm), (r.fov_x_nm, r.fov_y_nm), layer
    options = dict(contour_options(s), design_search_nm=s.design_search_nm, min_match_fraction=s.min_match_fraction,
                   design_tolerance_nm=s.design_tolerance_nm, match_design=not realign)

    # A few tiles first: if none matches its design, the settings or the data are wrong; stop now.
    with_oas = [k for k, r in enumerate(records) if r.oas_path is not None]
    sample = sorted({with_oas[i] for i in np.linspace(0, len(with_oas) - 1, SAMPLE_TILES).round().astype(int)})
    start = time.perf_counter()
    loaded = {}
    for k in sample:
        loaded[k] = process_tile(*arguments(records[k]), **options)
        if progress is not None:
            progress(len(loaded), len(records))
    if not realign and not any(t.design.ok for t in loaded.values()):
        fractions = {tile_ids[k]: round(t.design.match_fraction, 2) for k, t in loaded.items()}
        raise ValueError(f"No design tone matched the SEM in the sample tiles {fractions} (fraction of SEM contacts "
                         f"matched; min_match_fraction {s.min_match_fraction}): check layer / datatype, detection, "
                         f"design_search_nm and design_tolerance_nm")
    if not realign:
        log(f"Sample tiles matched to their design: {sum(t.design.ok for t in loaded.values())} of {len(loaded)}")

    # The other tiles in parallel, about one per physical core (each tile is independent); map keeps their order.
    rest = [k for k in range(len(records)) if k not in loaded]
    done = len(loaded)
    workers = worker_count()
    log(f"Loading the other {len(rest)} tiles in {workers} worker processes")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, tile in zip(rest, pool.map(_process_tile, [arguments(records[k]) for k in rest], repeat(options))):
            loaded[k] = tile
            done += 1
            if progress is not None:
                progress(done, len(records))
            if done % max(1, len(records) // 10) == 0 or done == len(records):
                log(f"{done} of {len(records)} tiles loaded and detected")
    tiles = [loaded[k] for k in range(len(records))]
    log(timing_summary(tiles, time.perf_counter() - start, sample, workers))
    log(contrast_summary(tiles, tile_ids, s.min_contrast))
    if realign:
        tiles = realign_tiles(tiles, s, log)
    flagged = [tile_ids[k] for k, t in enumerate(tiles) if not t.design.ok]
    if flagged:
        reason = "no design contact in its realigned field" if realign else             "neither design tone matches the SEM (or no .oas)"
        log(f"!! {len(flagged)} tile(s) FLAGGED: {reason}: {flagged}")
    if len(flagged) == len(tiles):
        raise RuntimeError("No tile matched its design: nothing can be measured (see the flagged tiles above)")
    return tile_ids, tiles


def contrast_summary(tiles: list[TileResult], tile_ids: list[str], min_contrast: float) -> str:
    """One log line: the contacts the contrast check dropped and the tiles left without any contact; with !! if it
    removed every contact of a tile (right for an image without pattern, wrong for a noisy image of contacts)."""
    if min_contrast <= 0:
        return "Contrast check off (min_contrast 0)"
    dropped = sum(t.low_contrast for t in tiles)
    empty = [tile_ids[k] for k, t in enumerate(tiles) if len(t.refined.centers) == 0]
    emptied = sum(t.low_contrast > 0 and len(t.refined.centers) == 0 for t in tiles)
    text = (f"Contrast check (min_contrast {min_contrast:g} noise sigma): {dropped} detected contact(s) dropped in "
            f"{sum(t.low_contrast > 0 for t in tiles)} tile(s); {len(empty)} tile(s) without any contact: {empty}")
    if emptied:
        return (f"!! {text}. It removed every contact of {emptied} tile(s): right for images without pattern, but if "
                f"they show contacts, their noise is too high for min_contrast {min_contrast:g}: lower it (0 = off)")
    return text


def timing_summary(tiles: list[TileResult], wall_s: float, sample: list[int], workers: int) -> str:
    """One log line: the tile loading's wall time, the median time per step of process_tile, and the wall
    time the steps alone would take (the sample tiles one by one, the rest spread over the workers).
    A wall time well above that is time outside the steps (worker start-up, sending tiles back, waiting)."""
    steps = list(tiles[0].seconds)
    totals = np.array([sum(t.seconds.values()) for t in tiles])
    one_by_one = np.isin(np.arange(len(tiles)), sample)
    expected = totals[one_by_one].sum() + totals[~one_by_one].sum() / workers
    medians = ", ".join(f"{step} {np.median([t.seconds[step] for t in tiles]):.3f}" for step in steps)
    reversed_tried = sum(t.seconds["reversed"] > 0 for t in tiles)
    return (f"Timing: {len(tiles)} tiles loaded in {wall_s:.1f} s; per tile, median s: {medians} "
            f"(total {np.median(totals):.3f}); the steps alone would take {expected:.1f} s with {workers} workers; "
            f"the reversed tone was tried on {reversed_tried} tile(s)")


def _analyse_tiles(folder: str, s: Settings, log, tile_ids, tiles: list[TileResult], polygons) -> AnalysisResult:
    """The steps of an image run after its tiles are loaded (also restitch); polygons: per tile, its .oas
    polygons for the viewer."""
    centers = np.array([t.center_nm for t in tiles])
    fovs = np.array([t.fov_nm for t in tiles])
    sem_points = [t.sem_points_nm() for t in tiles]
    design_points = [t.design_points_nm() for t in tiles]
    design_ok = [t.design.ok for t in tiles]
    neighbours = tile_neighbours(centers, fovs)  # overlapping tiles: their matching shifts must agree
    design_stitch = stitch_design(tiles, s.design_max_match_nm, s.outlier_factor, s.min_matched)

    def errors(points, sem_corrections, design_corrections):  # each tile matched to its own design
        # Every set pairs on the nominal SEM points (D86): the same contacts, so the same pairs in every set; image by
        # image, tied tiles choose with the set's own corrections (D133), so sets can differ there by whole pitches.
        return design_errors(points, design_points, sem_corrections, design_corrections, design_ok,
                             s.design_search_nm, s.design_tolerance_nm, neighbours, pair_points=sem_points,
                             follow_stitching=s.stitch_method in BY_IMAGE)

    result, observations, merged = _analyse(folder, "images", s, log, tile_ids, sem_points, design_points,
                                            centers, fovs, centers, design_stitch, errors)
    result.images = ImageTiles(tiles, polygons, merged, observations)
    return result


def restitch(result: AnalysisResult, method: str, log: Callable[[str], None] = print) -> AnalysisResult:
    """A new result of the same folder with settings.stitch_method = method: the stitching and every step
    after it (errors, corrections, RANSAC, moving window) computed again. Image runs reuse their tiles in
    memory (result.images; a reopened image result cannot be restitched); contour CSVs read the folder
    again. Contact sizes are not kept (their rows are the contacts of the new stitching). The new log
    is the old one, a line naming the method, then what this run reported."""
    if method not in STITCH_METHODS:
        raise ValueError(f"Stitching method must be one of {STITCH_METHODS} (is {method!r})")
    s = replace(result.settings, stitch_method=method)
    log, lines = _recording(log)
    log(f"=== Stitching method: {method}; the stitching and every step after it computed again")
    if result.kind == "contour CSV":
        new = analyse_contour_folder(result.folder, s, log)
    elif result.images is None:
        raise ValueError("Restitching an image run needs its tiles in memory: analyse the image folder again")
    else:
        new = _analyse_tiles(result.folder, s, log, result.tile_ids, result.images.tiles, result.images.design_polygons)
    new.log = result.log + lines
    return new


def measure_contact_sizes(result: AnalysisResult, roi_width_nm: float = 0.0,
                          progress: Callable[[int, int], None] | None = None) -> ContactSizes:
    """Area, xCD and yCD of every contact of the "uncorrected" set (rows as its contacts), on the SEM contours
    and the .oas outlines (read again, one merge per file) of an image run still in memory, averaged over the
    tiles that saw each contact. roi_width_nm: see ContactSizes. progress(done, total) after every tile."""
    if result.images is None:
        raise ValueError("Contact sizes need the SEM images of an image run: analyse the image folder again")
    placement = result.settings.ransac_placement
    return merged_contact_sizes(result.images.tiles, result.images.merged[placement],
                                result.images.observations[placement], result.settings, roi_width_nm, progress)


def merged_contact_sizes(tiles: list[TileResult], merged: MergedErrors, observations: DesignErrors, s: Settings,
                         roi_width_nm: float = 0.0, progress: Callable[[int, int], None] | None = None
                         ) -> ContactSizes:
    """Area, xCD and yCD of the merged contacts (rows as `merged`, merge_observations of `observations`) on
    the loaded tiles' SEM contours and .oas outlines (read again, one merge per file), averaged over the tiles
    that saw each contact. Also used by the release. roi_width_nm: see ContactSizes. progress(done, total)
    after every tile."""
    if roi_width_nm < 0:
        raise ValueError(f"The ROI width must be >= 0 (is {roi_width_nm})")
    sem_sizes, design_sizes = [], []
    for k, tile in enumerate(tiles):
        sem, design = tile.contact_sizes((s.layer, s.datatype), roi_width_nm)
        sem_sizes.append(sem)
        design_sizes.append(design)
        if progress is not None:
            progress(k + 1, len(tiles))
    sem, design = merge_values(merged, observations, sem_sizes, design_sizes)
    return ContactSizes(sem, design, float(roi_width_nm))


def _recording(log):
    """A log function that also keeps every line, and the list that keeps them."""
    lines = []

    def record(text: str):
        lines.append(text)
        log(text)
    return record, lines


def _analyse(folder, kind, s: Settings, log, tile_ids, sem_points, design_points, centers, sizes, image_centers,
             design_stitch, errors) -> tuple[AnalysisResult, dict, dict]:
    """The steps both kinds share, on the tiles' points at the nominal placement (mask nm). sizes: tile
    boxes (FOVs); image_centers: (n_tiles, 2) image centres for the in-image map (NaN: no map);
    errors(points, sem_corrections, design_corrections) -> DesignErrors of the SEM points against the
    design. Returns the result, and the observations and merged contacts of every placement."""
    stripe = tile_stripes(centers, image_centers)
    order_pairs = measurement_pairs(centers, stripe, s.first_stripe_upward)
    guided = s.stitch_method == "measurement order first"
    by_image = s.stitch_method in BY_IMAGE

    def stitch(points, label, only_pairs=None, guide=None):
        """All overlaps (only_pairs None), or those pairs; with the chosen method, all overlaps are guided
        by a first stitching in measurement order (stitch_tiles(guide=...)), or stitched image by image after the
        placement (stitch_image_by_image; guide: the placement, if found already). label: which stitching, for its
        warnings."""
        if only_pairs is None and by_image:
            if guide is None:
                guide = placement_guide(points, centers, sizes, s, log, label)
            return stitch_image_by_image(points, centers, sizes, guide, s.outlier_factor, s.min_matched, label=label)
        if only_pairs is None and guide is None and guided:
            guide = stitch(points, label.replace("all overlaps", "measurement order (guide)"),
                           only_pairs=order_pairs).corrections
        return stitch_tiles(points, centers, sizes, s.max_match_nm, s.outlier_factor, s.min_matched,
                            label=label, only_pairs=only_pairs, guide=guide)

    def merged_contacts(points, corrections):  # shifts (n, 2) or translation + rotation affines (n, 3, 3)
        return merge_observations(errors(points, corrections, design_stitch.corrections),
                                  design_stitch.corrections, s.group_radius_nm)

    order_stitch = stitch(sem_points, "SEM, measurement order", only_pairs=order_pairs)
    reference = (placement_guide(sem_points, centers, sizes, s, log, "SEM") if by_image  # the SEM stitching is
                 else order_stitch.corrections)                                            # checked on it
    raw_stitch = stitch(sem_points, "SEM, all overlaps", guide=reference if guided or by_image else None)
    log(f"Stitching method: {s.stitch_method}" + (f" ({len(order_stitch.pairs)} pairs in measurement order, "
                                                   f"{len(order_stitch.unplaced)} tile(s) not placed by them: "
                                                   f"paired at their nominal position)" if guided else "")
        + (f" ({placement_text(reference)})" if by_image else ""))
    for name, result in (("SEM", raw_stitch), ("Design", design_stitch)):
        report_stitching(log, name, result, tile_ids)
    checked_overlaps(sem_points, centers, sizes, raw_stitch, reference, s, log, tile_ids)

    # Raw errors per placement (no affine), one entry per physical contact.
    observations = {placement: errors(sem_points, sem_c, design_c) for placement, (sem_c, design_c)
                    in placement_corrections(raw_stitch.corrections, design_stitch.corrections).items()}
    merged = {placement: merge_observations(o, design_stitch.corrections, s.group_radius_nm)
              for placement, o in observations.items()}
    for placement, o in observations.items():
        if o.skipped:
            log(f"!! {placement}: {len(o.skipped)} tile(s) NOT MEASURED: "
                f"{ {tile_ids[k]: reason for k, reason in o.skipped.items()} }")
    contacts = merged[s.ransac_placement]
    log(f"{len(observations[s.ransac_placement].tile)} observations -> {len(contacts.count)} contacts "
        f"({s.ransac_placement} placement)")
    if len(contacts.count) < 3:
        raise RuntimeError(f"Only {len(contacts.count)} measured contact(s): see the tiles not measured above")
    sets = {"uncorrected": _fitted(contacts.sem_nm, contacts.design_nm, s, log, "uncorrected")}
    stitchings = {"raw": (sem_points, raw_stitch)}

    # In-image map, from the matched contacts at the nominal placement, relative to each image centre.
    ic_points = None
    if np.isnan(image_centers).any():
        log("!! No image centres (contour CSVs without DesignX, DesignY): the in-image correction is left out")
    elif (intrafield := in_image_map(observations["nominal"], image_centers, s, log)) is not None:
        log(f"In-image map from {intrafield.images} of {len(sem_points)} images, "
            f"up to {np.abs(intrafield.shift_nm).max():.3f} nm")
        ic_points = [correct_points(p, c, intrafield) for p, c in zip(sem_points, image_centers)]
        ic_stitch = stitch(ic_points, "In-image corrected SEM, all overlaps")
        ic = merged_contacts(ic_points, ic_stitch.corrections)
        sets["in-image corrected"] = _fitted(ic.sem_nm, ic.design_nm, s, log, "in-image corrected")
        stitchings["in-image corrected"] = (ic_points, ic_stitch)

    # Drift curve of the stitched contacts, its bend subtracted before RANSAC.
    drift = drift_curve(contacts.design_nm[:, 1], contacts.error_nm, s.drift_window_um * 1000, s.row_gap_nm)
    drift_sem = contacts.sem_nm - drift.correction(contacts.design_nm[:, 1])
    sets["drift corrected"] = _fitted(drift_sem, contacts.design_nm, s, log, "drift corrected", moving=False)

    # Drift per stripe: stitch in measurement order, subtract each stripe's curve, stitch everything.
    log(f"{stripe.max() + 1} stripes of {np.bincount(stripe).tolist()} tiles")

    def without_stripe_drift(points, name, first_stitch=None):
        """points minus each stripe's drift curve, measured after stitching them in measurement order
        (first_stitch, if already done); and the drift."""
        if first_stitch is None:
            first_stitch = stitch(points, f"{name}, measurement order", only_pairs=order_pairs)
        lost = len(first_stitch.unplaced)
        log(("!! " if lost else "") + f"{name} stitched in measurement order: {len(first_stitch.pairs)} pairs used, "
            f"{lost} tile(s) not stitched" + (" (the chain broke: these stay at their nominal position, and the "
                                              "stripe drift is measured without them)" if lost else ""))
        first = errors(points, first_stitch.corrections, design_stitch.corrections)
        drifts = stripe_drift(first.design_nm[:, 1], first.error_nm, stripe[first.tile], s.drift_window_um * 1000,
                              s.row_gap_nm)
        placed = np.nan_to_num(first_stitch.corrections)  # a tile not stitched in measurement order stays nominal
        return [p + placed[k] - drifts.correction(stripe[k], p[:, 1] + placed[k, 1]) for k, p in enumerate(points)], drifts

    sd_points, stripe_drifts = without_stripe_drift(sem_points, "SEM points", order_stitch)
    sd_stitch = stitch(sd_points, "Stripe drift corrected SEM, all overlaps")
    sd = merged_contacts(sd_points, sd_stitch.corrections)
    sets["stripe drift corrected"] = _fitted(sd.sem_nm, sd.design_nm, s, log, "stripe drift corrected")
    stitchings["stripe drift corrected"] = (sd_points, sd_stitch)
    if ic_points is not None:  # both corrections before stitching, in that order
        icsd_points, _ = without_stripe_drift(ic_points, "In-image corrected points")
        stitchings["in-image + stripe drift corrected"] = (icsd_points, stitch(
            icsd_points, "In-image + stripe drift corrected SEM, all overlaps"))

    # Translation + rotation per tile, from the ties of each translation stitching (stitching residuals).
    stitchings = {name: Stitching(points, st, stitch_tiles_rigid(points, centers, st))
                  for name, (points, st) in stitchings.items()}
    if "in-image + stripe drift corrected" in stitchings:  # both corrections, placed with translation + rotation
        both = stitchings["in-image + stripe drift corrected"]
        m = merged_contacts(both.points, both.rigid)
        name = "in-image + stripe drift corrected, translation + rotation"
        sets[name] = _fitted(m.sem_nm, m.design_nm, s, log, name)

    result = AnalysisResult(
        folder=folder, kind=kind, settings=s, tile_ids=list(tile_ids), centers=centers, sizes=sizes,
        design_points=design_points, design_stitch=design_stitch, stitchings=stitchings,
        sets=sets, drift=drift, stripe_drift=stripe_drifts, log=[])
    return result, observations, merged


def in_image_map(observations: DesignErrors, image_centers: np.ndarray, s: Settings, log) -> IntrafieldMap | None:
    """The in-image distortion from every tile's contacts paired with its design at the nominal placement
    (observations), relative to its image centre (image_centers: (n_tiles, 2), mask nm): intrafield.estimate_formula
    of order s.in_image_order (D145), None (logged with !!) if the contacts cannot determine it. Also used by the
    release (D141)."""
    order = np.argsort(observations.tile, kind="stable")  # each tile's contacts together, in their order
    starts = np.searchsorted(observations.tile[order], np.arange(1, len(image_centers)))
    design_local = [d - c for d, c in zip(np.split(observations.design_nm[order], starts), image_centers)]
    sem_local = [p - c for p, c in zip(np.split(observations.sem_nm[order], starts), image_centers)]
    m = estimate_formula(design_local, sem_local, s.in_image_order)
    if m is None:
        log(f"!! The paired contacts cannot determine the in-image formula of order {s.in_image_order} (too few "
            f"contacts, or all in a few rows or columns; a lower IN_IMAGE_ORDER needs fewer): the in-image correction "
            f"is left out")
    return m


def _fitted(sem_nm, design_nm, s: Settings, log, name: str, moving: bool = True) -> ContactSet:
    """RANSAC (and the moving window) of one set of contacts."""
    ransac = ransac_affine(sem_nm, design_nm, s.ransac_threshold_nm, seed=s.ransac_seed)
    window = moving_window_affine(sem_nm, design_nm, s.moving_window_um * 1000, s.moving_step_um * 1000) if moving else None
    residual = ransac.residuals
    log(f"RANSAC, {name}: {ransac.inliers.sum()} of {len(residual)} inliers ({ransac.inliers.mean():.1%}), "
        f"3σ ({3 * residual[:, 0].std():.3f}, {3 * residual[:, 1].std():.3f}) nm")
    return ContactSet(sem_nm, design_nm, ransac, window)


def report_stitching(log, name: str, result: StitchResult, tile_ids):
    """The stitching's counts; !! only for tiles left without a position (not measured). Overlaps not matched are
    listed plainly: left out, their tiles are placed through their other overlaps (D109)."""
    failed = [r for r in result.rejected if r.failed]
    log(f"{name} stitching: {len(result.pairs)} overlaps used, {len(failed)} not matched, "
        f"{len(result.rejected) - len(failed)} skipped (too few contacts)")
    if len(result.unplaced):
        log(f"!! {name.upper()} STITCHING INCOMPLETE: {len(result.unplaced)} tile(s) not stitched, so not measured: "
            f"{[tile_ids[k] for k in result.unplaced]}")
    if failed:
        log(f"{name} overlaps not matched (left out of the stitching): "
            f"{[(tile_ids[r.i], tile_ids[r.j]) for r in failed]}")


def placement_guide(points, centers, sizes, s: Settings, log, label: str) -> np.ndarray:
    """The placement the stitching image by image follows (D133): pipeline.placed_image_by_image, or with the method
    fingerprint pipeline.placed_by_fingerprint (D144), how it chose logged (fingerprint_text; label: which points).
    Also used by the release."""
    if s.stitch_method == "fingerprint":
        guide, choices = placed_by_fingerprint(points, centers, sizes, s.max_match_nm, s.min_matched)
        log(fingerprint_text(choices, label))
        return guide
    return placed_image_by_image(points, centers, sizes, s.max_match_nm, s.min_matched)


def fingerprint_text(choices: list[FingerprintChoice], label: str) -> str:
    """How the placement by fingerprint chose each image's shift, in one log line: by the residual of its pairs, by the
    contacts paired (several fitting as well), tied (several agreeing as well too: the smallest correction, as image by
    image) or as its only option; how clearly the residuals differ (second best over best) and the residual taken."""
    if not choices:
        return f"Fingerprint, {label}: no image placed against others"
    one = sum(c.options == 1 for c in choices)
    residual = sum(c.options > 1 and c.fitting == 1 for c in choices)
    tied = sum(c.tied > 1 for c in choices)
    ratios = [c.ratio for c in choices if c.options > 1]
    clearly = f"; second best residual / best: median {np.median(ratios):.2f}" if ratios else ""
    return (f"Fingerprint, {label}: of {len(choices)} images placed against others, {residual} chosen by the residual "
            f"of their pairs, {len(choices) - one - residual - tied} by the contacts paired, {tied} tied (the smallest "
            f"correction, as image by image: the mask's placement errors not larger than the noise?), {one} with one "
            f"option{clearly}; residual of the shifts taken: median {np.median([c.rms_nm for c in choices]):.2f} nm")


def placement_text(guide: np.ndarray) -> str:
    """How many images the placement image by image (pipeline.placed_image_by_image) placed, for the log."""
    missing = int(np.isnan(guide[:, 0]).sum())
    if not missing:
        return f"all {len(guide)} images placed image by image"
    return (f"{len(guide) - missing} of {len(guide)} images placed image by image; the other {missing} paired at their "
            f"nominal position")


def checked_overlaps(points, centers, sizes, stitch: StitchResult, reference: np.ndarray, s: Settings, log,
                     tile_ids) -> list[tuple[int, int]]:
    """The SEM overlaps (i, j) the stitching got wrong, logged with !!. Image by image or fingerprint: those that do
    not line up where the stitching placed their images (pipeline.overlap_checks, D137, fingerprint by its residuals,
    D144; its numbers logged too). Otherwise: those fitted more than max_match_nm from the shift the stitching in
    measurement order (reference) predicts (pipeline.slipped_pairs, D75)."""
    if s.stitch_method in BY_IMAGE:
        checks = overlap_checks(points, centers, sizes, stitch.corrections, s.overlap_tolerance_nm, s.max_match_nm,
                                s.outlier_factor, s.min_matched, by_fit=s.stitch_method == "fingerprint")
        log(overlap_check_text(checks, s))
        found = [(c.i, c.j) for c in checks if not c.lines_up]
    else:
        found = slipped_pairs(stitch, reference, s.max_match_nm)
    if found:
        log(f"!! {len(found)} SEM overlap(s) {slip_text(s)}: {[(tile_ids[i], tile_ids[j]) for i, j in found]}")
    return found


def overlap_check_text(checks: list[OverlapCheck], s: Settings) -> str:
    """The overlap checks in one log line: how many, the share of their contacts that pair up, and how far apart their
    two images lie (the median and the worst of each), and how many have a clearly better shift (better_text)."""
    if not checks:
        return "SEM overlaps where the stitching placed their images: none checked (too few contacts)"
    agreement = np.array([c.agreement for c in checks])
    offset = np.array([c.offset_nm for c in checks])
    offset = offset[np.isfinite(offset)]
    apart = f"median {np.median(offset):.2f} nm, largest {offset.max():.2f} nm" if offset.size else "no contact paired"
    return (f"SEM overlaps where the stitching placed their images: {len(checks)} checked; contacts paired up: median "
            f"{np.median(agreement):.0%}, lowest {agreement.min():.0%}; the two images apart: {apart}; "
            f"{sum(c.better for c in checks)} where another shift {better_text(s)}")


def better_text(s: Settings) -> str:
    """What makes another shift of one image of an overlap clearly better (pipeline.clearly_better), by method."""
    if s.stitch_method == "fingerprint":
        return "fits clearly better (or fits as well and pairs clearly more)"
    return "pairs clearly more"


def slip_text(s: Settings) -> str:
    """What the SEM overlaps that checked_overlaps finds are, for the log line that lists them, by stitching method."""
    if s.stitch_method in BY_IMAGE:
        return (f"do not line up where the stitching placed their images (fewer than half their contacts pair up, "
                f"another shift of one image {better_text(s)}, or the two images lie more than overlap_tolerance_nm "
                f"apart)")
    return ("fitted more than max_match_nm away from the shift the stitching in measurement order predicts (contacts "
            "paired one row off?)" + ("" if s.stitch_method == "measurement order first"
                                      else "; the stitching method 'measurement order first' avoids this"))


def tile_edges_y(centers: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """y of every tile box edge (mask nm), for the row pitch plot."""
    return np.unique(np.round(np.concatenate([centers[:, 1] - sizes[:, 1] / 2, centers[:, 1] + sizes[:, 1] / 2])))
