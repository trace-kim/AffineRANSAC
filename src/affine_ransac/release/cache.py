"""Autosaved analysis of a folder (docs/SPEC.md D77): one .npz of plain arrays (results_file's layout helpers,
loaded with allow_pickle=False), written after every run and opened instead of a new run when the folder is
opened again.

Where: CACHE_NAME in the data folder; if that cannot be written (read-only share), the same file under the
user's own cache folder, named after the data folder's path. Loading tries both, data folder first.
"""

import hashlib
import json
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from affine_ransac.analysis import Settings
from affine_ransac.fitting.moving_window import MovingWindowResult
from affine_ransac.release.analysis import ReleaseResult
from affine_ransac.results_file import _get_fields, _get_parts, _get_stitch, _put_fields, _put_parts, _put_stitch

CACHE_NAME = "MaskReg_analysis.npz"
FORMAT = 3  # 2: the stripe of each tile (D89); 3: the .oas paths and positions (D93)


def cache_paths(folder, user_cache_dir) -> list[Path]:
    """Where the cache of `folder` is looked for and written, in that order."""
    key = hashlib.sha1(str(Path(folder).resolve()).lower().encode()).hexdigest()[:16]
    return [Path(folder) / CACHE_NAME, Path(user_cache_dir) / f"{key}_{CACHE_NAME}"]


def save_cache(result: ReleaseResult, user_cache_dir) -> Path:
    """Write the result to the first cache path that can be written; returns it."""
    arrays = {"centers": result.centers, "sizes": result.sizes, "stripe": result.stripe, "sem_nm": result.sem_nm, "design_nm": result.design_nm,
              "oas_centers": result.oas_centers}
    _put_parts(arrays, "sem_points", result.sem_points, np.empty((0, 2)))
    _put_parts(arrays, "design_points", result.design_points, np.empty((0, 2)))
    _put_stitch(arrays, "stitch", result.stitch)
    _put_stitch(arrays, "design_stitch", result.design_stitch)
    _put_fields(arrays, "moving", result.moving)
    info = {"format": FORMAT, "saved": time.strftime("%Y-%m-%d %H:%M"), "folder": result.folder, "kind": result.kind,
            "settings": asdict(result.settings), "tile_ids": result.tile_ids, "image_paths": result.image_paths,
            "oas_paths": result.oas_paths,
            "log": result.log}
    arrays["info"] = np.array(json.dumps(info))
    errors = []
    for path in cache_paths(result.folder, user_cache_dir):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:  # a file object: np.savez would add .npz to a name without it
                np.savez_compressed(f, **arrays)
            return path
        except OSError as error:
            errors.append(f"{path}: {error}")
    raise OSError("The analysis could not be saved: " + "; ".join(errors))


def load_cache(folder, user_cache_dir) -> tuple[ReleaseResult, str] | None:
    """The saved analysis of `folder` and when it was saved, or None if there is none (or of another format)."""
    for path in cache_paths(folder, user_cache_dir):
        if path.is_file():
            with np.load(path, allow_pickle=False) as arrays:
                info = json.loads(str(arrays["info"]))
                if info.get("format") != FORMAT:
                    return None
                known = {f.name for f in fields(Settings)}  # settings added since keep their default
                settings = Settings(**{k: v for k, v in info["settings"].items() if k in known})
                result = ReleaseResult(
                    folder=str(folder), kind=info["kind"], settings=settings, tile_ids=info["tile_ids"],
                    centers=arrays["centers"], sizes=arrays["sizes"],
                    stripe=arrays["stripe"], sem_points=_get_parts(arrays, "sem_points"),
                    design_points=_get_parts(arrays, "design_points"), stitch=_get_stitch(arrays, "stitch"),
                    design_stitch=_get_stitch(arrays, "design_stitch"), sem_nm=arrays["sem_nm"],
                    design_nm=arrays["design_nm"], moving=_get_fields(arrays, "moving", MovingWindowResult),
                    image_paths=info["image_paths"], log=info["log"], oas_paths=info["oas_paths"],
                    oas_centers=arrays["oas_centers"])
            return result, info["saved"]
    return None
