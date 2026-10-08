"""End-to-end analysis of one data folder, the flow of the notebooks, for the standalone app (app.py).

Two kinds of folder (docs/SPEC.md §4):
- pre-analysed contour CSVs (csv_stitch.ipynb): analyse_contour_folder(folder, settings);
- SEM images + .oas + metadata CSV (stitch_viewer.ipynb): analyse_images(folder, records, settings),
  with the records of the metadata reader (io/metadata.read_tile_index, which exists only on the data
  machine, so the caller passes them in).
Both run the same steps on the tiles' points (_analyse): stitching (translation, all overlaps), errors
per placement and merged contacts, the in-image map, the drift curve, the drift per stripe in
measurement order, RANSAC and the moving window for every set of contacts, and the stitching residuals
(also of the in-image corrected points with their drift per stripe removed).
The AnalysisResult holds what the analysis window shows (view_analysis.analysis_window). No UI code.
`log` receives what the notebooks print (counts, failures).
"""

from dataclasses import dataclass, field, fields
from typing import Callable

import numpy as np

from affine_ransac.fitting.drift import DriftCurve, StripeDrift, drift_curve, stripe_drift
from affine_ransac.fitting.moving_window import MovingWindowResult, moving_window_affine
from affine_ransac.fitting.ransac import RansacResult, ransac_affine
from affine_ransac.intrafield import correct_points, estimate_map
from affine_ransac.io.contour_csv import read_contour_folder
from affine_ransac.io.design import load_layout, read_polygons
from affine_ransac.pipeline import (StitchResult, TileResult, measurement_pairs, process_tile, stitch_design,
                                    stitch_tiles, stitch_tiles_rigid, tile_boxes_from_points, tile_neighbours,
                                    tile_stripes)
from affine_ransac.registration import MergedErrors, design_errors, merge_observations, paired_errors
from affine_ransac.stitching import placement_corrections


def setting(default, help_text: str):
    """A Settings field with its help text (the notebooks' comment), shown as a tooltip by the app."""
    return field(default=default, metadata={"help": help_text})


@dataclass
class Settings:
    """The notebooks' settings cells: same names in lower case, same units."""
    max_match_nm: float = setting(25.0, "SEM overlaps: pair contacts closer than this, the largest B - A stage "
                                        "offset (< pitch / 2)")
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
    intrafield_nodes: int = setting(9, "In-image distortion map: nodes per axis over the image, edge to edge")
    drift_window_um: float = setting(40.0, "Drift curve: height of the local straight-line fit along y")
    first_stripe_upward: bool = setting(True, "Serpentine scan: the first stripe (smallest x) was measured from its "
                                              "bottom up, then alternately")
    # Contour CSVs only
    tile_margin_nm: float = setting(25.0, "Tile box = bounding box of its design centres grown by this")
    # Images only
    layer: int = setting(100, "Layer of the contacts in the .oas")
    datatype: int = setting(0, "Datatype of the contacts in the .oas")
    detection: str = setting("band", "otsu (plain), otsu3 (3 classes) or band (regions enclosed by the bright band)")
    search_px: float = setting(5.0, "Edge refinement searches this many pixels in/out along the contour normal")
    design_search_nm: float = setting(100.0, "Design-SEM matching: largest expected shift of a tile's SEM contacts "
                                             "against its design (> pitch / 2)")
    design_tolerance_nm: float = setting(10.0, "Design-SEM matching: pair contacts within this after removing the "
                                               "tile shift (< pitch / 2)")
    min_match_fraction: float = setting(0.5, "A design tone is accepted if at least this fraction of the SEM "
                                             "contacts match it")
    stitch_refined: bool = setting(True, "Stitch with the refined centres (off: Otsu centres)")
    spread_flag_nm: float = setting(2.0, "Stitch viewer: ring merged contacts whose SEM observations differ by more")
    arrow_scale: float = setting(10.0, "Stitch viewer: error lines are drawn this many times longer than the error")


CSV_ONLY = {"tile_margin_nm"}
IMAGES_ONLY = {"layer", "datatype", "detection", "search_px", "design_search_nm", "design_tolerance_nm",
               "min_match_fraction", "stitch_refined", "spread_flag_nm", "arrow_scale"}
CHOICES = {"ransac_placement": ("mean", "first"), "detection": ("otsu", "otsu3", "band")}


def settings_for(kind: str) -> list[str]:
    """Names of the settings a folder of this kind ("contour CSV" or "images") uses."""
    unused = IMAGES_ONLY if kind == "contour CSV" else CSV_ONLY
    return [f.name for f in fields(Settings) if f.name not in unused]


@dataclass
class ContactSet:
    """Merged contacts (one per physical contact) and their affine fits: one set of lines in the window."""
    sem_nm: np.ndarray                 # (M, 2) stitched SEM centres, mask nm (no affine)
    design_nm: np.ndarray              # (M, 2) stitched design centres, mask nm
    ransac: RansacResult
    moving: MovingWindowResult | None  # None: no moving-window fit for this set


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
    sets: dict[str, ContactSet]        # "uncorrected" (translation stitching), then the corrected sets
    drift: DriftCurve                  # its bend was subtracted from the "drift corrected" set
    stripe_drift: StripeDrift          # its curves were subtracted from the "stripe drift corrected" set
    log: list[str]                     # what the run reported
    images: ImageTiles | None = None   # images only, in memory after a run


def analyse_contour_folder(folder, settings: Settings, log: Callable[[str], None] = print) -> AnalysisResult:
    """The csv_stitch.ipynb flow on a folder of contour CSVs (io.contour_csv)."""
    s = settings
    log, lines = _recording(log)
    all_tiles = read_contour_folder(folder)
    empty = [t.name for t in all_tiles if len(t.design_nm) == 0]
    if empty:
        log(f"!! {len(empty)} file(s) without any contact, left out: {empty}")
    tiles = [t for t in all_tiles if len(t.design_nm)]
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
    result, _ = _analyse(str(folder), "contour CSV", s, log, [t.name for t in tiles], sem_points, design_points,
                         centers, sizes, image_centers, design_stitch, errors)
    result.log = lines
    return result


def analyse_images(folder, records: list, settings: Settings, log: Callable[[str], None] = print) -> AnalysisResult:
    """The stitch_viewer.ipynb flow on the tiles of an image folder. records: per tile, an object with the
    TileRecord fields of io/metadata.read_tile_index (tile_id, image_path, oas_path, center_x_nm,
    center_y_nm, fov_x_nm, fov_y_nm)."""
    s = settings
    log, lines = _recording(log)
    layer = (s.layer, s.datatype)
    tiles = []
    for k, r in enumerate(records):
        tiles.append(process_tile(r.image_path, r.oas_path, (r.center_x_nm, r.center_y_nm), (r.fov_x_nm, r.fov_y_nm),
                                  layer, s.search_px, s.design_search_nm, s.min_match_fraction, s.detection,
                                  s.design_tolerance_nm))
        if (k + 1) % max(1, len(records) // 10) == 0 or k + 1 == len(records):
            log(f"{k + 1} of {len(records)} tiles loaded and detected")
    tile_ids = [r.tile_id for r in records]
    flagged = [tile_ids[k] for k, t in enumerate(tiles) if not t.design.ok]
    if flagged:
        log(f"!! {len(flagged)} tile(s) FLAGGED: neither design tone matches the SEM (or no .oas): {flagged}")
    centers = np.array([t.center_nm for t in tiles])
    fovs = np.array([t.fov_nm for t in tiles])
    sem_points = [t.sem_points_nm(s.stitch_refined) for t in tiles]
    design_points = [t.design_points_nm() for t in tiles]
    design_ok = [t.design.ok for t in tiles]
    neighbours = tile_neighbours(centers, fovs)  # overlapping tiles: their matching shifts must agree
    design_stitch = stitch_design(tiles, s.design_max_match_nm, s.outlier_factor, s.min_matched)

    def errors(points, sem_corrections, design_corrections):  # each tile matched to its own design
        return design_errors(points, design_points, sem_corrections, design_corrections, design_ok,
                             s.design_search_nm, s.design_tolerance_nm, neighbours)

    result, merged = _analyse(str(folder), "images", s, log, tile_ids, sem_points, design_points, centers, fovs,
                              centers, design_stitch, errors)
    polygons = [read_polygons(load_layout(r.oas_path), *layer) if r.oas_path else [] for r in records]
    result.images = ImageTiles(tiles, polygons, merged)
    result.log = lines
    return result


def _recording(log):
    """A log function that also keeps every line, and the list that keeps them."""
    lines = []

    def record(text: str):
        lines.append(text)
        log(text)
    return record, lines


def _analyse(folder, kind, s: Settings, log, tile_ids, sem_points, design_points, centers, sizes, image_centers,
             design_stitch, errors) -> tuple[AnalysisResult, dict]:
    """The steps both kinds share, on the tiles' points at the nominal placement (mask nm). sizes: tile
    boxes (FOVs); image_centers: (n_tiles, 2) image centres for the in-image map (NaN: no map);
    errors(points, sem_corrections, design_corrections) -> DesignErrors of the SEM points against the
    design. Returns the result and the merged contacts of every placement."""
    def stitch(points, only_pairs=None):
        return stitch_tiles(points, centers, sizes, s.max_match_nm, s.outlier_factor, s.min_matched,
                            only_pairs=only_pairs)

    def merged_contacts(points, stitched):
        return merge_observations(errors(points, stitched.corrections, design_stitch.corrections),
                                  design_stitch.corrections, s.group_radius_nm)

    raw_stitch = stitch(sem_points)
    for name, result in (("SEM", raw_stitch), ("Design", design_stitch)):
        _report_stitching(log, name, result, tile_ids)

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
    else:
        nominal = observations["nominal"]
        n = len(sem_points)
        design_local = [nominal.design_nm[nominal.tile == k] - image_centers[k] for k in range(n)]
        sem_local = [nominal.sem_nm[nominal.tile == k] - image_centers[k] for k in range(n)]
        intrafield = estimate_map(design_local, sem_local, s.intrafield_nodes)
        log(f"In-image map from {intrafield.images} of {n} images, up to {np.abs(intrafield.shift_nm).max():.3f} nm")
        ic_points = [correct_points(p, c, intrafield) for p, c in zip(sem_points, image_centers)]
        ic_stitch = stitch(ic_points)
        ic = merged_contacts(ic_points, ic_stitch)
        sets["in-image corrected"] = _fitted(ic.sem_nm, ic.design_nm, s, log, "in-image corrected")
        stitchings["in-image corrected"] = (ic_points, ic_stitch)

    # Drift curve of the stitched contacts, its bend subtracted before RANSAC.
    drift = drift_curve(contacts.design_nm[:, 1], contacts.error_nm, s.drift_window_um * 1000, s.row_gap_nm)
    drift_sem = contacts.sem_nm - drift.correction(contacts.design_nm[:, 1])
    sets["drift corrected"] = _fitted(drift_sem, contacts.design_nm, s, log, "drift corrected", moving=False)

    # Drift per stripe: stitch in measurement order, subtract each stripe's curve, stitch everything.
    stripe = tile_stripes(centers, sizes)
    order_pairs = measurement_pairs(centers, stripe, s.first_stripe_upward)
    log(f"{stripe.max() + 1} stripes of {np.bincount(stripe).tolist()} tiles")

    def without_stripe_drift(points, name):
        """points minus each stripe's drift curve, measured after stitching them in measurement order; and
        the drift."""
        first_stitch = stitch(points, only_pairs=order_pairs)
        log(f"{name} stitched in measurement order: {len(first_stitch.pairs)} pairs used, "
            f"{len(first_stitch.unplaced)} tile(s) not stitched")
        first = errors(points, first_stitch.corrections, design_stitch.corrections)
        drifts = stripe_drift(first.design_nm[:, 1], first.error_nm, stripe[first.tile], s.drift_window_um * 1000,
                              s.row_gap_nm)
        placed = np.nan_to_num(first_stitch.corrections)  # a tile not stitched in measurement order stays nominal
        return [p + placed[k] - drifts.correction(stripe[k], p[:, 1] + placed[k, 1]) for k, p in enumerate(points)], drifts

    sd_points, stripe_drifts = without_stripe_drift(sem_points, "SEM points")
    sd_stitch = stitch(sd_points)
    sd = merged_contacts(sd_points, sd_stitch)
    sets["stripe drift corrected"] = _fitted(sd.sem_nm, sd.design_nm, s, log, "stripe drift corrected")
    stitchings["stripe drift corrected"] = (sd_points, sd_stitch)
    if ic_points is not None:  # both corrections before stitching, in that order: for the stitching residuals
        icsd_points, _ = without_stripe_drift(ic_points, "In-image corrected points")
        stitchings["in-image + stripe drift corrected"] = (icsd_points, stitch(icsd_points))

    result = AnalysisResult(
        folder=folder, kind=kind, settings=s, tile_ids=list(tile_ids), centers=centers, sizes=sizes,
        design_points=design_points, design_stitch=design_stitch,
        stitchings={name: Stitching(points, st, stitch_tiles_rigid(points, centers, st))
                    for name, (points, st) in stitchings.items()},
        sets=sets, drift=drift, stripe_drift=stripe_drifts, log=[])
    return result, merged


def _fitted(sem_nm, design_nm, s: Settings, log, name: str, moving: bool = True) -> ContactSet:
    """RANSAC (and the moving window) of one set of contacts."""
    ransac = ransac_affine(sem_nm, design_nm, s.ransac_threshold_nm, seed=s.ransac_seed)
    window = moving_window_affine(sem_nm, design_nm, s.moving_window_um * 1000, s.moving_step_um * 1000) if moving else None
    residual = ransac.residuals
    log(f"RANSAC, {name}: {ransac.inliers.sum()} of {len(residual)} inliers ({ransac.inliers.mean():.1%}), "
        f"3σ ({3 * residual[:, 0].std():.3f}, {3 * residual[:, 1].std():.3f}) nm")
    return ContactSet(sem_nm, design_nm, ransac, window)


def _report_stitching(log, name: str, result: StitchResult, tile_ids):
    failed = [r for r in result.rejected if r.failed]
    log(f"{name} stitching: {len(result.pairs)} overlaps used, {len(failed)} not matched, "
        f"{len(result.rejected) - len(failed)} skipped (too few contacts)")
    if failed or len(result.unplaced):
        log(f"!! {name.upper()} STITCHING INCOMPLETE. Tiles not stitched: {[tile_ids[k] for k in result.unplaced]}; "
            f"overlaps not matched: {[(tile_ids[r.i], tile_ids[r.j]) for r in failed]}")


def tile_edges_y(centers: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """y of every tile box edge (mask nm), for the row pitch plot."""
    return np.unique(np.round(np.concatenate([centers[:, 1] - sizes[:, 1] / 2, centers[:, 1] + sizes[:, 1] / 2])))
