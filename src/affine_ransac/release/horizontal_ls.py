"""Temporary release method for horizontal lines and spaces separated by bright SEM edges."""

from contextlib import closing, nullcontext
from dataclasses import replace

import numpy as np

from affine_ransac.analysis import Settings, helper_pool, line_options
from affine_ransac.contour_memory import bounded_results, contour_workers
from affine_ransac.features.contact import DetectedContacts
from affine_ransac.geometry.frames import pixel_to_tile_nm
from affine_ransac.geometry.polygon import polygon_area, polygon_centroid
from affine_ransac.io.design import horizontal_band_centers, load_layout
from affine_ransac.io.sem_image import load_sem_image
from affine_ransac.pipeline import find_lines
from affine_ransac.registration import design_errors_after_stitching


def band_settings(s: Settings) -> Settings:
    """Two grey-level classes: the white separators and everything between them, in either design tone."""
    return replace(s, detection="otsu", dark_contacts=True)


def find_bands(image: np.ndarray, s: Settings) -> tuple[DetectedContacts, DetectedContacts]:
    """Close each pair of horizontal edges along the sides of the image. Return area centroids in native pixels."""
    # A one-pixel white separator can step diagonally. Its two interiors share no pixel side.
    detected, refined = find_lines(image, **line_options(band_settings(s)), connectivity=4)
    x = np.arange(image.shape[1])
    found, moved = [], []
    left_out = {"not crossing": detected.left_out, "clipped or flat": 0, "small": 0, "large": 0}
    for top, bottom, new_top, new_bottom in zip(detected.upper, detected.lower, refined.upper, refined.lower):
        # A missing edge at the top or bottom of the field would put the centre at the wrong height.
        thickness, refined_thickness = bottom - top, new_bottom - new_top
        if len(x) < 2 or not all(np.all(t >= 0) and np.any(t > 0) for t in (thickness, refined_thickness)):
            left_out["clipped or flat"] += 1
            continue
        area_px = np.sum(thickness + 1)
        if area_px < s.min_area_px:
            left_out["small"] += 1
            continue
        if s.max_area_px and area_px > s.max_area_px:
            left_out["large"] += 1
            continue
        for contours, upper, lower in ((found, top, bottom), (moved, new_top, new_bottom)):
            upper_edge = np.column_stack((x, upper))
            lower_edge = np.column_stack((x, lower))
            contours.append(np.vstack((upper_edge, lower_edge[::-1])))

    def regions(contours):
        centers = np.asarray([polygon_centroid(c) for c in contours]).reshape(-1, 2)
        return DetectedContacts(centers, np.array([polygon_area(c) for c in contours]), contours,
                                detected.threshold, left_out)
    return regions(found), regions(moved)


def tile_centers(path, oas_path, fov_nm, s: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Read one image and design clip. Return only the two sets of centres from the worker, in tile-local nm."""
    image = load_sem_image(path)
    _, bands = find_bands(image, s)
    pixel_nm = np.asarray(fov_nm) / image.shape[::-1]
    sem = pixel_to_tile_nm(bands.centers, image.shape, pixel_nm)
    design = (horizontal_band_centers(load_layout(oas_path), s.layer, s.datatype, fov_nm)
              if oas_path else np.empty((0, 2)))
    return sem, design


def load_centers(records, centers, sizes, s, log, progress, pool):
    """Read SEM and design centres in tile order, limiting simultaneous image work as for general contours."""
    workers = contour_workers([r.image_path for r in records], s.upscale, log, pool=pool)
    args = ((r.image_path, r.oas_path, size, s) for r, size in zip(records, sizes))
    sem_points, design_points = [], []
    with helper_pool(workers) if pool is None else nullcontext(pool) as helpers:
        with closing(bounded_results(helpers, tile_centers, args, workers)) as results:
            for k, (sem, design) in enumerate(results):
                sem_points.append(sem + centers[k])
                design_points.append(design + centers[k])
                if not len(sem) or not len(design):
                    log(f"!! {records[k].tile_id} has {len(sem)} SEM and {len(design)} design L/S centres")
                progress(k + 1, len(records))
    return sem_points, design_points


def tile_outlines(path, fov_nm, s: Settings):
    """The same closed contours for Load contours, in tile-local nm."""
    image = load_sem_image(path)
    detected, refined = find_bands(image, s)
    pixel_nm = np.asarray(fov_nm) / image.shape[::-1]

    def local(bands):
        return [pixel_to_tile_nm(c, image.shape, pixel_nm) for c in bands.contours]
    return local(detected), local(refined), np.ones(len(detected.centers), bool)


def paired_rows(sem_points, design_points, sem_shift, design_shift, centers, neighbours, s):
    """Follow the stitched SEM–design pairs as in a contact run, comparing only y. x locates the sampled strip."""
    errors = design_errors_after_stitching(
        [p[:, 1:] for p in sem_points], [p[:, 1:] for p in design_points], sem_shift[:, 1:], design_shift[:, 1:],
        [True] * len(design_points), neighbours, s.design_search_nm, s.design_tolerance_nm, s.group_radius_nm)
    x = centers[errors.tile, 0]
    errors.sem_nm = np.column_stack((x, errors.sem_nm.ravel()))
    errors.design_nm = np.column_stack((x, errors.design_nm.ravel()))
    errors.design_nominal_nm = np.column_stack((x, errors.design_nominal_nm.ravel()))
    errors.error_nm = errors.sem_nm - errors.design_nm
    dy = errors.match_shift_nm[:, 0]
    errors.match_shift_nm = np.column_stack((np.where(np.isfinite(dy), 0.0, np.nan), dy))
    return errors
