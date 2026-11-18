"""Temporary release method for horizontal lines and spaces separated by bright SEM edges."""

from contextlib import closing, nullcontext
from dataclasses import replace

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu

from affine_ransac.analysis import Settings, helper_pool, line_options
from affine_ransac.contour_memory import bounded_results, contour_workers
from affine_ransac.features.contact import DetectedContacts, threshold_image
from affine_ransac.features.lines import in_image_lines, lines_in_mask, refine_line_edges
from affine_ransac.geometry.frames import pixel_to_tile_nm
from affine_ransac.geometry.polygon import polygon_area, polygon_centroid
from affine_ransac.io.design import horizontal_band_outlines, load_layout
from affine_ransac.io.sem_image import load_sem_image
from affine_ransac.matching import match_points_tree, match_with_shift, mutual_nearest
from affine_ransac.overlap import overlapping_pairs
from affine_ransac.overlap_fit import fit_overlap
from affine_ransac.pipeline import PairResult, RejectedPair, find_lines, prepared_image, solved_stitching
from affine_ransac.registration import design_errors_after_stitching

SLICES = 10


def band_settings(s: Settings) -> Settings:
    """Use the brightest grey-level class for separators. Either interior tone can lie below it."""
    return replace(s, detection="band", dark_contacts=True)


def band_mask(image, s):
    """Threshold the prepared image once, including the two-level and uniform cases."""
    pixels, lut = threshold_image(image, s.detection_blur_px * s.upscale, s.equalize, s.clahe_clip, s.clahe_tiles)
    hist = cv2.calcHist([pixels], [0], None, [256], [0, 256]).ravel()
    levels = np.flatnonzero(hist)
    high = threshold_multiotsu(classes=3, hist=hist)[-1] if len(levels) >= 3 else levels[0]
    level = np.flatnonzero(lut <= high)[-1] if lut is not None else high
    return (pixels <= high).astype(np.uint8), float(level), pixels


def find_bands(image: np.ndarray, s: Settings) -> tuple[DetectedContacts, DetectedContacts]:
    """Prepare the whole image once, then close and measure the interiors in ten x slices."""
    big = prepared_image(image, s.background_px, True, s.denoise, s.median_px, s.nlm_h, s.tv_weight, s.upscale)
    mask, level, _ = band_mask(big, s)
    return bands_from_mask(big, mask, level, s)


def closed_band(x, top, bottom):
    return np.vstack((np.column_stack((x, top)), np.column_stack((x, bottom))[::-1]))


def bands_from_mask(big, mask, level, s):
    f = s.upscale
    width = big.shape[1] // f
    cuts = np.linspace(-0.5, width - 0.5, SLICES + 1)  # physical pixel edges; adjacent polygons share each cut
    found, moved = [], []
    left_out = {"not crossing or joined": 0, "clipped": 0, "small": 0, "large": 0}
    for left, right in zip(cuts, cuts[1:]):
        start, stop = int(np.ceil(left)), int(np.ceil(right))
        if start == stop:  # no native pixel centre in this slice
            continue
        part = slice(start * f, stop * f)
        edges = lines_in_mask(mask[:, part], level, connectivity=4, single_at_sides=True)
        left_out["not crossing or joined"] += edges.left_out
        refined = edges if s.refinement == "none" else refine_line_edges(
            big[:, part], edges, True, s.search_px * f, s.edge_step_px * f, s.edge_blur_px * f, s.refinement)
        detected, refined = in_image_lines(edges, f), in_image_lines(refined, f)
        x = np.r_[left, np.arange(start, stop), right]
        for top, bottom, upper, lower in zip(detected.upper, detected.lower, refined.upper, refined.lower):
            # Pixel faces enclose the detected interior, including a single row. Refined edges are crossings.
            top, bottom = top - 0.5 / f, bottom + 0.5 / f
            if not np.all(bottom > top):  # a boundary leaves the top or bottom of the field
                left_out["clipped"] += 1
                continue
            # The saved area limits describe a full-width band. Apply the equivalent limit to its local thickness.
            area_px = np.mean(bottom - top) * width
            if area_px < s.min_area_px:
                left_out["small"] += 1
                continue
            if s.max_area_px and area_px > s.max_area_px:
                left_out["large"] += 1
                continue
            if s.refinement == "none":
                upper, lower = top, bottom
            else:
                valid = (upper < lower) & (upper >= -0.5) & (lower <= big.shape[0] / f - 0.5)
                upper, lower = np.where(valid, upper, top), np.where(valid, lower, bottom)
            for contours, a, b in ((found, top, bottom), (moved, upper, lower)):
                contours.append(closed_band(x, np.r_[a[0], a, a[-1]], np.r_[b[0], b, b[-1]]))

    def regions(contours):
        centers = np.asarray([polygon_centroid(c) for c in contours]).reshape(-1, 2)
        return DetectedContacts(centers, np.array([polygon_area(c) for c in contours]), contours,
                                level, left_out)
    return regions(found), regions(moved)


def band_slope(contour):
    """Slope of the least-squares line through the filled slice, used to compare heights at a common x."""
    moments = cv2.moments((contour - contour[0]).astype(np.float32))
    return moments["mu11"] / moments["mu20"] if moments["mu20"] else 0.0


def tile_centers(path, oas_path, fov_nm, s: Settings):
    """Return each slice's centroid (tile-local nm) and centreline slope (dy/dx in the mask frame)."""
    image = load_sem_image(path)
    _, bands = find_bands(image, s)
    pixel_nm = np.asarray(fov_nm) / image.shape[::-1]
    sem = pixel_to_tile_nm(bands.centers, image.shape, pixel_nm)
    sem_slope = np.array([band_slope(c) for c in bands.contours]) * (-pixel_nm[1] / pixel_nm[0])
    outlines = horizontal_band_outlines(load_layout(oas_path), s.layer, s.datatype, fov_nm, SLICES) if oas_path else []
    design = np.asarray([polygon_centroid(c) for c in outlines]).reshape(-1, 2)
    return sem, design, sem_slope, np.array([band_slope(c) for c in outlines])


def load_centers(records, centers, sizes, s, log, progress, pool, detailed_log=False):
    """Read SEM and design centres in tile order, limiting simultaneous image work as for general contours."""
    workers = contour_workers([r.image_path for r in records], s.upscale, log, pool=pool)
    args = ((r.image_path, r.oas_path, size, s) for r, size in zip(records, sizes))
    sem_points, design_points, sem_slopes, design_slopes = [], [], [], []
    with helper_pool(workers) if pool is None else nullcontext(pool) as helpers:
        with closing(bounded_results(helpers, tile_centers, args, workers)) as results:
            for k, (sem, design, sem_slope, design_slope) in enumerate(results):
                sem_points.append(sem + centers[k])
                design_points.append(design + centers[k])
                sem_slopes.append(sem_slope)
                design_slopes.append(design_slope)
                if not len(sem) or not len(design):
                    log(f"!! {records[k].tile_id} has {len(sem)} SEM and {len(design)} design L/S centres")
                if detailed_log:
                    cuts = slice_edges(0, sizes[k, 0])
                    counts = [np.histogram(p[:, 0], bins=cuts)[0].tolist() for p in (sem, design)]
                    log(f"{records[k].tile_id} L/S centres per x slice, left to right: SEM {counts[0]}, design {counts[1]}")
                progress(k + 1, len(records))
    return sem_points, design_points, sem_slopes, design_slopes


def tile_outlines(path, fov_nm, s: Settings, full_width=False):
    """The same closed contours for Load contours, in tile-local nm."""
    image = load_sem_image(path)
    pixel_nm = np.asarray(fov_nm) / image.shape[::-1]
    if full_width:  # reproduce the outlines of a saved analysis 9 run
        edges, moved = find_lines(image, **line_options(replace(s, detection="otsu", dark_contacts=True)),
                                  connectivity=4)
        x = np.arange(image.shape[1])
        found, refined = [], []
        for a, b, u, v in zip(edges.upper, edges.lower, moved.upper, moved.lower):
            area = (b - a + 1).sum()
            if (len(x) > 1 and all(np.all(t >= 0) and np.any(t > 0) for t in (b - a, v - u))
                    and s.min_area_px <= area and (not s.max_area_px or area <= s.max_area_px)):
                found.append(closed_band(x, a, b))
                refined.append(closed_band(x, u, v))
    else:
        detected, measured = find_bands(image, s)
        found, refined = detected.contours, measured.contours
    return ([pixel_to_tile_nm(c, image.shape, pixel_nm) for c in found],
            [pixel_to_tile_nm(c, image.shape, pixel_nm) for c in refined], np.ones(len(found), bool))


def slice_edges(center_x, width):
    return center_x + np.linspace(-width / 2, width / 2, SLICES + 1)


def slice_rows(points, edges):
    ids = np.searchsorted(edges, points[:, 0], side="right") - 1
    return [np.flatnonzero(ids == i) for i in range(SLICES)]


def common_slices(a, b):
    """Intersect the two sets of cuts, widest common interval first."""
    left = np.maximum(a[:-1, None], b[None, :-1])
    right = np.minimum(a[1:, None], b[None, 1:])
    shared = right - left
    for i, j in sorted(zip(*np.nonzero(shared > 0)), key=lambda ij: -shared[ij]):
        yield i, j, (left[i, j] + right[i, j]) / 2


def stitch_bands(points, slopes, centers, sizes, gate, s, label):
    """Compare slice centres at a common x in the overlap, then solve the usual robust y translations."""
    edges = [slice_edges(c[0], size[0]) for c, size in zip(centers, sizes)]
    rows = [slice_rows(p, cuts) for p, cuts in zip(points, edges)]
    pairs, rejected = [], []
    for i, j, box in overlapping_pairs(centers, sizes):
        ia, ib, sample_x, heights_a, heights_b = [], [], [], [], []
        used = {k: np.zeros(len(points[k]), bool) for k in (i, j)}
        inside = {i: set(), j: set()}
        for si, sj, x in common_slices(edges[i], edges[j]):
            samples = []
            for k, strip in ((i, si), (j, sj)):
                indices = rows[k][strip]
                indices = indices[~used[k][indices]]
                p = points[k][indices]
                y = p[:, 1] + slopes[k][indices] * (x - p[:, 0])
                inside[k].update(indices[(y >= box[2]) & (y <= box[3])])
                near = (y >= box[2] - gate) & (y <= box[3] + gate)
                samples.append((indices[near], y[near]))
            (a, ya), (b, yb) = samples
            u, v = match_points_tree(ya[:, None], yb[:, None], gate)
            ia.extend(a[u])
            ib.extend(b[v])
            used[i][a[u]] = used[j][b[v]] = True
            sample_x.extend([x] * len(u))
            heights_a.extend(ya[u])
            heights_b.extend(yb[v])
        if len(ia) < s.min_matched:
            in_box = min(len(inside[i]), len(inside[j]))
            rejected.append(RejectedPair(i, j, box, in_box, len(ia), failed=in_box >= s.min_matched))
            continue
        fit = fit_overlap(np.column_stack((sample_x, heights_a)), np.column_stack((sample_x, heights_b)),
                          outlier_factor=s.outlier_factor)
        pairs.append(PairResult(i, j, box, np.asarray(ia), np.asarray(ib), fit))
    return solved_stitching(len(points), pairs, rejected, label)


def _pairs_in_slices(parts, design, sem, slopes, design_rows, dy, s):
    design_ids, sem_ids = [], []
    for d, m in parts:
        if not len(m) or not len(d):
            continue
        if dy is None:  # seed the tile from the nominal pairs in all its slices
            match, _ = match_with_shift(design[d, 1:], sem[m, 1:], s.design_search_nm, s.design_tolerance_nm)
            design_ids.extend(d[match.ia])
            sem_ids.extend(m[match.ib])
            continue
        # Prefer this tile's design. A neighbour can supply rows outside its clipped design.
        own = design_rows[d] >= 0
        for rows in (d[own], d[~own]):
            if len(rows) and len(m):
                dx = design[rows, 0] - sem[m, 0, None]
                y = sem[m, 1, None] + slopes[m, None] * dx  # SEM height at each possible design x
                u, v = mutual_nearest(np.abs(y - design[rows, 1] - dy), s.design_tolerance_nm)
                design_ids.extend(rows[v])
                sem_ids.extend(m[u])
                m = np.delete(m, u)
    ia, ib = np.asarray(design_ids, int), np.asarray(sem_ids, int)
    heights = sem[ib, 1] + slopes[ib] * (design[ia, 0] - sem[ib, 0])
    return ia, ib, np.column_stack((design[ia, 0], heights))


def paired_rows(sem_points, design_points, sem_slopes, sem_shift, design_shift, centers, sizes, neighbours, s):
    """Follow one y offset per tile. Match heights within each slice and report them at the design sample's x."""
    def match_slices(k, design, sem, design_rows, offset):
        cuts = slice_edges(centers[k, 0], sizes[k, 0])
        parts = list(zip(slice_rows(design, cuts), slice_rows(sem, cuts)))
        ia, ib, measured = _pairs_in_slices(parts, design, sem, sem_slopes[k], design_rows,
                                            None if offset is None else offset[1], s)
        if len(ia) and offset is not None:
            dy = np.median(measured[:, 1] - design[ia, 1])
            return _pairs_in_slices(parts, design, sem, sem_slopes[k], design_rows, dy, s)
        return ia, ib, measured

    return design_errors_after_stitching(
        sem_points, design_points, sem_shift, design_shift, [True] * len(design_points), neighbours,
        s.design_search_nm, s.design_tolerance_nm, s.group_radius_nm, match_slices=match_slices)
