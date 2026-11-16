"""CPU-only synthetic SEM benchmark requested by the user, 2026-11-16.

Example (from the repository root):
    python benchmarks/contour_stitching.py --counts 8 16 32 64 --repeats 2

Outputs stay under build/ (gitignored). No real images or metadata are needed. Images are generated once;
each independent noisy acquisition is decoded and processed by MaskReg's own _edge_tile worker. Image
generation is timed separately from contour extraction, placement, and the final overlap checks.
"""

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

# Set before NumPy/OpenCV or spawned workers import numerical libraries. All operations below use CPU arrays.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

import cv2
import numpy as np
import scipy
from scipy.special import ndtr

from affine_ransac.analysis import any_shape_options, helper_pool, worker_count
from affine_ransac.contour_stitching import overlap_distances, placed_by_contours
from affine_ransac.release.analysis import DEFAULTS, _edge_tile
from affine_ransac.step_log import StepLog

cv2.ocl.setUseOpenCL(False)

SIZE = 2048
PERIODS = 6
STEP_PX = round(SIZE / 3)
PIXEL_NM = 1.0
SEED = 20261116


def synthetic_image(center_y_px: float, index: int) -> np.ndarray:
    """Six horizontal bright lines/spaces, 50% duty, optical edge blur and independent detector noise.

    World y is up; increasing the acquisition centre moves a fixed line downward in the pixel frame.
    Integrating the Gaussian edge profile preserves subpixel stage shifts without resizing an image.
    """
    pitch = SIZE / PERIODS
    world_y = center_y_px - (np.arange(SIZE) - (SIZE - 1) / 2)
    phase = (world_y + pitch / 4) % pitch
    distance = np.minimum(phase, pitch - phase) - pitch / 4
    profile = 40.0 + 150.0 * ndtr(-distance / 1.2)
    rng = np.random.default_rng(SEED + index)
    image = profile[:, None] + rng.normal(0, 6.0, (SIZE, SIZE))
    return np.rint(image).clip(0, 255).astype(np.uint8)


def write_images(folder: Path, count: int, errors: np.ndarray) -> tuple[list[Path], float]:
    folder.mkdir(parents=True, exist_ok=True)
    start = last = time.perf_counter()
    paths = []
    for k in range(count):
        path = folder / f"synthetic_{k:04d}.png"
        image = synthetic_image(k * STEP_PX + errors[k, 1] / PIXEL_NM, k)
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise RuntimeError("Could not encode the synthetic image")
        encoded.tofile(path)
        paths.append(path)
        if time.perf_counter() - last >= 5:
            print(f"Generated {k + 1}/{count} images", flush=True)
            last = time.perf_counter()
    return paths, time.perf_counter() - start


def extract_tile(path, options):
    """Production extraction, with its CPU time and wall time recorded inside the worker."""
    start, cpu = time.perf_counter(), time.process_time()
    result = _edge_tile(path, (SIZE * PIXEL_NM, SIZE * PIXEL_NM), options)
    return result, time.perf_counter() - start, time.process_time() - cpu


def extract_images(paths, centers, settings, counts):
    start, edges, records, milestones = time.perf_counter(), [], [], {}
    with helper_pool() as pool:
        futures = [pool.submit(extract_tile, str(path), any_shape_options(settings)) for path in paths]
        last = start
        for k, future in enumerate(futures):
            (points, closed, opened, pixel_nm), wall, cpu = future.result()
            edges.append(points.moved(centers[k]))
            records.append(dict(image=k, wall_s=wall, cpu_s=cpu, closed=closed, opened=opened,
                                edge_points=len(points.xy), pixel_nm=pixel_nm))
            elapsed = time.perf_counter() - start
            if k + 1 in counts:
                milestones[k + 1] = elapsed
            if time.perf_counter() - last >= 5 or k + 1 == len(paths):
                print(f"Extracted {k + 1}/{len(paths)} images in {elapsed:.2f} s", flush=True)
                last = time.perf_counter()
    return edges, dict(wall_s=time.perf_counter() - start, milestones_s=milestones, tiles=records)


def stitch_sample(edges, centers, sizes, errors, search_nm, repeat):
    n, timing = len(edges), StepLog()
    start, cpu, last = time.perf_counter(), time.process_time(), [time.perf_counter()]

    def progress(done, total):
        if time.perf_counter() - last[0] >= 10 or done == total:
            print(f"Stitch {n} images, repeat {repeat}: {done}/{total}, {time.perf_counter() - start:.2f} s", flush=True)
            last[0] = time.perf_counter()

    corrections, choices = placed_by_contours(edges, centers, sizes, search_nm, PIXEL_NM, progress, log=timing)
    placement_s, cpu_s = time.perf_counter() - start, time.process_time() - cpu
    start = time.perf_counter()
    distances = overlap_distances(edges, centers, sizes, corrections, PIXEL_NM)
    check_s = time.perf_counter() - start
    placed = np.isfinite(corrections).all(axis=1)
    anchors = np.flatnonzero(placed & ~np.isin(np.arange(n), [c.tile for c in choices]))
    anchor = int(anchors[0])
    residual = corrections[placed] - (errors[placed] - errors[anchor])
    paired_rms = np.array([d.rms_nm for d in distances if np.isfinite(d.rms_nm)])
    return dict(images=n, repeat=repeat, placement_s=placement_s, placement_cpu_s=cpu_s,
                overlap_check_s=check_s, stitching_s=placement_s + check_s, placed=int(placed.sum()),
                anchor=anchor, across_error_rms_nm=float(np.sqrt(np.mean(residual[:, 1] ** 2))),
                across_error_max_nm=float(np.abs(residual[:, 1]).max()),
                along_max_shift_nm=float(np.abs(corrections[placed, 0]).max()),
                ambiguous_choices=sum(c.positions > 1 for c in choices),
                directions_1=sum(c.directions == 1 for c in choices),
                directions_2=sum(c.directions == 2 for c in choices),
                overlap_rms_median_nm=float(np.median(paired_rms)) if len(paired_rms) else None,
                overlap_rms_max_nm=float(paired_rms.max()) if len(paired_rms) else None,
                phases_s=timing.seconds, work=timing.counts)


def write_results(path, results):
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--search-nm", type=float, default=DEFAULTS.max_match_nm)
    parser.add_argument("--out", type=Path, default=Path("build/contour-benchmark"))
    args = parser.parse_args()
    counts = sorted(set(args.counts))
    if counts[0] < 2 or args.repeats < 1:
        parser.error("counts must be >= 2 and repeats must be >= 1")
    args.out.mkdir(parents=True, exist_ok=True)
    n = counts[-1]
    settings = replace(DEFAULTS, pattern="contours", dark_contacts=False, detection="otsu",
                       max_match_nm=args.search_nm)
    rng = np.random.default_rng(SEED)
    errors = np.column_stack([np.zeros(n), rng.uniform(-3, 3, n)]) * PIXEL_NM
    centers = np.column_stack([np.zeros(n), np.arange(n) * STEP_PX]) * PIXEL_NM
    sizes = np.full((n, 2), SIZE * PIXEL_NM)
    results = dict(revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   machine=dict(platform=platform.platform(), processor=platform.processor(),
                                logical_cpus=os.cpu_count(), workers=worker_count(), python=platform.python_version(),
                                numpy=np.__version__, scipy=scipy.__version__, opencv=cv2.__version__),
                   synthetic=dict(size_px=SIZE, periods=PERIODS, step_px=STEP_PX,
                                  overlap_fraction=1 - STEP_PX / SIZE, pixel_nm=PIXEL_NM,
                                  seed=SEED, stage_error_bound_px=3, blur_sigma_px=1.2, noise_sigma=6,
                                  background=40, foreground=190, gpu=False),
                   settings=asdict(settings), stitching=[])
    print(json.dumps(dict(counts=counts, repeats=args.repeats, search_nm=args.search_nm,
                          workers=worker_count(), output=str(args.out))), flush=True)
    paths, results["generation_s"] = write_images(args.out / "images", n, errors)
    edges, results["extraction"] = extract_images(paths, centers, settings, counts)
    # Retain measured edge points so a larger geometry-only scaling check can avoid regenerating images.
    np.savez_compressed(args.out / "edges.npz", centers=centers, sizes=sizes, errors=errors,
                        **{f"xy_{k}": e.xy for k, e in enumerate(edges)},
                        **{f"normal_{k}": e.normals for k, e in enumerate(edges)},
                        step_nm=np.array([e.step_nm for e in edges]))
    result_path = args.out / "results.json"
    write_results(result_path, results)
    for count in counts:
        for repeat in range(1, args.repeats + 1):
            found = stitch_sample(edges[:count], centers[:count], sizes[:count], errors[:count],
                                  args.search_nm, repeat)
            results["stitching"].append(found)
            write_results(result_path, results)
            print(json.dumps(found), flush=True)
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()
