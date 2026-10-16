"""Autosaved analysis of a folder (docs/SPEC.md D77): one .npz of plain arrays (results_file's layout helpers,
loaded with allow_pickle=False), written after every run and opened instead of a new run when the folder is
opened again.

The contact sizes, once measured (Contact size tab), are optional arrays: a file without them loads with none.
The tiles kept to measure them are not saved (the images are large).

Where: CACHE_NAME in the data folder; if that cannot be written (read-only share), the same file under the
user's own cache folder, named after the data folder's path. Loading tries both, data folder first.

Backups (D98): a new run moves the run it replaces into the BACKUPS folder next to it, named by when that run was
saved; the newest KEEP_BACKUPS are kept. A file is written as <name>.part first and renamed when complete, so a
failed save leaves the previous run in place.

Versions (D99): every run file holds the MaskReg version and the ANALYSIS number that made it. Opening checks its
FORMAT: an older one (or a damaged file) cannot be read, so the folder is analysed again (the old file becomes a
backup); a newer one raises NewerRunFile, so it is never replaced unless the user restarts the analysis. A run of
another ANALYSIS is shown, and the window flags it.
"""

import contextlib
import hashlib
import json
import os
import time
import zipfile
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from affine_ransac import __version__
from affine_ransac.analysis import ContactSizes, Settings
from affine_ransac.fitting.moving_window import MovingWindowResult
from affine_ransac.release.analysis import ReleaseResult
from affine_ransac.results_file import _get_fields, _get_parts, _get_stitch, _put_fields, _put_parts, _put_stitch

CACHE_NAME = "MaskReg_analysis.npz"
# Raised by one whenever a file saved before can no longer be read (tests/test_release_versions.py holds the
# layout of each number). 2: the stripe of each tile (D89); 3: the .oas paths and positions (D93).
FORMAT = 3
BACKUPS = "MaskReg_backups"  # folder of the earlier runs, next to the saved run
KEEP_BACKUPS = 5


class NewerRunFile(Exception):
    """The saved run was written by a newer MaskReg, in a file format this one cannot read."""


def made_by(version: str) -> str:
    """Who made a run, for messages: 'MaskReg 1.2.0', or for runs saved before versions were: 'MaskReg before 1.0.0'."""
    return f"MaskReg {version}" if version else "MaskReg before 1.0.0"


def cache_paths(folder, user_cache_dir) -> list[Path]:
    """Where the cache of `folder` is looked for and written, in that order."""
    key = hashlib.sha1(str(Path(folder).resolve()).lower().encode()).hexdigest()[:16]
    return [Path(folder) / CACHE_NAME, Path(user_cache_dir) / f"{key}_{CACHE_NAME}"]


def save_cache(result: ReleaseResult, user_cache_dir, backup: bool = True, log=lambda text: None) -> Path:
    """Write the result to the first cache path that can be written; returns it. backup: the result is a new run,
    so the run it replaces is kept as a backup; False: the same run saved again (e.g. with its contact sizes)."""
    arrays = {"centers": result.centers, "sizes": result.sizes, "stripe": result.stripe, "sem_nm": result.sem_nm, "design_nm": result.design_nm,
              "oas_centers": result.oas_centers}
    _put_parts(arrays, "sem_points", result.sem_points, np.empty((0, 2)))
    _put_parts(arrays, "design_points", result.design_points, np.empty((0, 2)))
    _put_stitch(arrays, "stitch", result.stitch)
    _put_stitch(arrays, "design_stitch", result.design_stitch)
    _put_fields(arrays, "moving", result.moving)
    if result.contact_sizes is not None:
        arrays["contact_sizes/sem"] = result.contact_sizes.sem
        arrays["contact_sizes/design"] = result.contact_sizes.design
        arrays["contact_sizes/roi_width_nm"] = np.asarray(result.contact_sizes.roi_width_nm)
    info = {"format": FORMAT, "version": result.version, "analysis": result.analysis,
            "saved": time.strftime("%Y-%m-%d %H:%M"), "folder": result.folder, "kind": result.kind,
            "settings": asdict(result.settings), "tile_ids": result.tile_ids, "image_paths": result.image_paths,
            "oas_paths": result.oas_paths,
            "log": result.log}
    arrays["info"] = np.array(json.dumps(info))
    errors = []
    for path in cache_paths(result.folder, user_cache_dir):
        part = path.with_name(path.name + ".part")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(part, "wb") as f:  # a file object: np.savez would add .npz to a name without it
                np.savez_compressed(f, **arrays)
            if backup and path.is_file():
                log(f"Previous run kept as {back_up(path)}")
            os.replace(part, path)
            return path
        except OSError as error:
            errors.append(f"{path}: {error}")
            with contextlib.suppress(OSError):
                part.unlink(missing_ok=True)
    raise OSError("The analysis could not be saved: " + "; ".join(errors))


def back_up(path: Path) -> Path:
    """Move the saved run `path` into the BACKUPS folder next to it, named by when it was saved; delete all but the
    newest KEEP_BACKUPS there. Returns where it went."""
    folder = path.parent / BACKUPS
    folder.mkdir(exist_ok=True)
    saved = path.stat().st_mtime
    # A backup saved in the same second already: take the next free second, so the names stay in saving order.
    while (target := folder / f"{path.stem}_{time.strftime('%Y%m%d-%H%M%S', time.localtime(saved))}.npz").exists():
        saved += 1
    os.replace(path, target)
    for old in sorted(folder.glob(f"{path.stem}_*.npz"))[:-KEEP_BACKUPS]:
        old.unlink()
    return target


def load_cache(folder, user_cache_dir, log=lambda text: None) -> tuple[ReleaseResult, str] | None:
    """The saved analysis of `folder` and when it was saved, or None if there is none this MaskReg can read (an
    older format, or a damaged file: logged). A file of a newer format raises NewerRunFile."""
    for path in cache_paths(folder, user_cache_dir):
        if path.is_file():
            try:
                return read_run(path, folder, log)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                log(f"!! The saved analysis {path} cannot be read ({type(error).__name__}: {error}): analysing again "
                    f"(the file is kept as a backup)")
                return None
    return None


def read_run(path: Path, folder, log) -> tuple[ReleaseResult, str] | None:
    """One saved run (load_cache), with its version and ANALYSIS number."""
    with np.load(path, allow_pickle=False) as arrays:
        info = json.loads(str(arrays["info"]))
        made, file_format = made_by(info.get("version", "")), info.get("format", 0)
        if file_format > FORMAT:
            raise NewerRunFile(f"The saved analysis of this folder was made by {made}, in a newer file format "
                               f"({file_format}) than MaskReg {__version__} reads ({FORMAT}). Update MaskReg to open "
                               f"it, or restart the analysis (the newer run is then kept as a backup).")
        if file_format < FORMAT:
            log(f"!! The saved analysis was made by {made}, in an older file format ({file_format}) that this MaskReg "
                f"cannot read: analysing again (the old file is kept as a backup)")
            return None
        known = {f.name for f in fields(Settings)}  # settings added since keep their default
        settings = Settings(**{k: v for k, v in info["settings"].items() if k in known})
        sizes = None
        if "contact_sizes/sem" in arrays:
            sizes = ContactSizes(arrays["contact_sizes/sem"], arrays["contact_sizes/design"],
                                 arrays["contact_sizes/roi_width_nm"].item())
        result = ReleaseResult(
            folder=str(folder), kind=info["kind"], settings=settings, tile_ids=info["tile_ids"],
            centers=arrays["centers"], sizes=arrays["sizes"],
            stripe=arrays["stripe"], sem_points=_get_parts(arrays, "sem_points"),
            design_points=_get_parts(arrays, "design_points"), stitch=_get_stitch(arrays, "stitch"),
            design_stitch=_get_stitch(arrays, "design_stitch"), sem_nm=arrays["sem_nm"],
            design_nm=arrays["design_nm"], moving=_get_fields(arrays, "moving", MovingWindowResult),
            image_paths=info["image_paths"], log=info["log"], oas_paths=info["oas_paths"],
            oas_centers=arrays["oas_centers"], contact_sizes=sizes,
            version=info.get("version", ""), analysis=info.get("analysis", 0))
    return result, info["saved"]
