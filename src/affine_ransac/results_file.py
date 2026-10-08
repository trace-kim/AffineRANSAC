"""Save and load an analysis.AnalysisResult as one .npz file (docs/SPEC.md D63).

Plain numpy arrays only, loaded with allow_pickle=False: the file opens without this code
(np.load(path) lists the arrays) and loading it never runs code. Array names are paths, e.g.
"sets/uncorrected/ransac/residuals". A list of arrays (per tile, per overlap) is one concatenated
array plus the length of each part ("<name>/data", "<name>/lengths"). The folder, kind, settings, tile
ids and log are one JSON text, "info". The images of an image run (AnalysisResult.images) are not
saved (gigabytes): a reopened result shows the points stitch view.
FORMAT changes whenever the layout does; a file of another format is refused with a message.
"""

import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from affine_ransac.analysis import AnalysisResult, ContactSet, Settings, Stitching
from affine_ransac.fitting.drift import DriftCurve, StripeDrift
from affine_ransac.fitting.moving_window import MovingWindowResult
from affine_ransac.fitting.ransac import RansacResult
from affine_ransac.overlap_fit import OverlapFit
from affine_ransac.pipeline import PairResult, RejectedPair, StitchResult

FORMAT = 1


def save_result(result: AnalysisResult, path: str | Path):
    arrays = {"centers": result.centers, "sizes": result.sizes}
    _put_parts(arrays, "design_points", result.design_points, np.empty((0, 2)))
    _put_stitch(arrays, "design_stitch", result.design_stitch)
    for name, st in result.stitchings.items():
        _put_parts(arrays, f"stitchings/{name}/points", st.points, np.empty((0, 2)))
        _put_stitch(arrays, f"stitchings/{name}/stitch", st.stitch)
        arrays[f"stitchings/{name}/rigid"] = st.rigid
    for name, c in result.sets.items():
        arrays[f"sets/{name}/sem_nm"], arrays[f"sets/{name}/design_nm"] = c.sem_nm, c.design_nm
        _put_fields(arrays, f"sets/{name}/ransac", c.ransac)
        if c.moving is not None:
            _put_fields(arrays, f"sets/{name}/moving", c.moving)
    _put_fields(arrays, "drift", result.drift)
    _put_fields(arrays, "stripe_drift/common", result.stripe_drift.common)
    for stripe, curve in result.stripe_drift.stripes.items():
        _put_fields(arrays, f"stripe_drift/stripes/{stripe}", curve)
    info = {"format": FORMAT, "folder": result.folder, "kind": result.kind, "settings": asdict(result.settings),
            "tile_ids": result.tile_ids, "log": result.log, "stitchings": list(result.stitchings),
            "sets": {name: c.moving is not None for name, c in result.sets.items()},  # name: has a moving window
            "stripes": sorted(result.stripe_drift.stripes)}
    arrays["info"] = np.array(json.dumps(info))
    np.savez_compressed(path, **arrays)  # about half the size; saving takes a few seconds for 200k contacts


def load_result(path: str | Path) -> AnalysisResult:
    with np.load(path, allow_pickle=False) as arrays:
        info = json.loads(str(arrays["info"]))
        if info["format"] != FORMAT:
            raise ValueError(f"{Path(path).name} has results format {info['format']}; this version reads format "
                             f"{FORMAT}. Analyse the folder again.")
        known = {f.name for f in fields(Settings)}  # settings added since the file was saved keep their default
        settings = Settings(**{k: v for k, v in info["settings"].items() if k in known})
        stitchings = {name: Stitching(_get_parts(arrays, f"stitchings/{name}/points"),
                                      _get_stitch(arrays, f"stitchings/{name}/stitch"),
                                      arrays[f"stitchings/{name}/rigid"])
                      for name in info["stitchings"]}
        sets = {name: ContactSet(arrays[f"sets/{name}/sem_nm"], arrays[f"sets/{name}/design_nm"],
                                 _get_fields(arrays, f"sets/{name}/ransac", RansacResult),
                                 _get_fields(arrays, f"sets/{name}/moving", MovingWindowResult) if moving else None)
                for name, moving in info["sets"].items()}
        stripe_drift = StripeDrift({s: _get_fields(arrays, f"stripe_drift/stripes/{s}", DriftCurve) for s in info["stripes"]},
                                   _get_fields(arrays, "stripe_drift/common", DriftCurve))
        return AnalysisResult(
            folder=info["folder"], kind=info["kind"], settings=settings, tile_ids=info["tile_ids"],
            centers=arrays["centers"], sizes=arrays["sizes"], design_points=_get_parts(arrays, "design_points"),
            design_stitch=_get_stitch(arrays, "design_stitch"), stitchings=stitchings, sets=sets,
            drift=_get_fields(arrays, "drift", DriftCurve), stripe_drift=stripe_drift, log=info["log"])


def _put_parts(arrays: dict, name: str, parts: list, empty: np.ndarray):
    """parts (arrays of one dtype and row shape) as one array and their lengths; `empty` if no parts."""
    arrays[f"{name}/data"] = np.concatenate(parts) if parts else empty
    arrays[f"{name}/lengths"] = np.array([len(p) for p in parts], dtype=np.int64)


def _get_parts(arrays, name: str) -> list[np.ndarray]:
    lengths = arrays[f"{name}/lengths"]
    return np.split(arrays[f"{name}/data"], np.cumsum(lengths)[:-1]) if len(lengths) else []


def _put_fields(arrays: dict, name: str, record):
    """Every field of a dataclass of arrays and numbers (RansacResult, MovingWindowResult, DriftCurve)."""
    for f in fields(record):
        arrays[f"{name}/{f.name}"] = np.asarray(getattr(record, f.name))


def _get_fields(arrays, name: str, cls):
    values = {f.name: arrays[f"{name}/{f.name}"] for f in fields(cls)}
    return cls(**{key: value.item() if value.ndim == 0 else value for key, value in values.items()})


def _put_stitch(arrays: dict, name: str, stitch: StitchResult):
    """A StitchResult: corrections; per used pair its tiles, box, matched contacts and overlap fit; per
    rejected pair its tiles, box, counts and whether it failed."""
    pairs, rejected = stitch.pairs, stitch.rejected
    arrays[f"{name}/corrections"] = stitch.corrections
    arrays[f"{name}/pairs/ij"] = np.array([(p.i, p.j) for p in pairs], dtype=np.int64).reshape(-1, 2)
    arrays[f"{name}/pairs/box"] = np.array([p.box for p in pairs], dtype=float).reshape(-1, 4)
    _put_parts(arrays, f"{name}/pairs/ia", [p.ia for p in pairs], np.empty(0, np.int64))
    _put_parts(arrays, f"{name}/pairs/ib", [p.ib for p in pairs], np.empty(0, np.int64))
    arrays[f"{name}/pairs/shift"] = np.array([p.fit.shift for p in pairs], dtype=float).reshape(-1, 2)
    arrays[f"{name}/pairs/rotation"] = np.array([p.fit.rotation for p in pairs], dtype=float)
    arrays[f"{name}/pairs/center"] = np.array([p.fit.center for p in pairs], dtype=float).reshape(-1, 2)
    _put_parts(arrays, f"{name}/pairs/inliers", [p.fit.inliers for p in pairs], np.empty(0, bool))
    _put_parts(arrays, f"{name}/pairs/residuals", [p.fit.residuals for p in pairs], np.empty((0, 2)))
    arrays[f"{name}/rejected/ij"] = np.array([(r.i, r.j) for r in rejected], dtype=np.int64).reshape(-1, 2)
    arrays[f"{name}/rejected/box"] = np.array([r.box for r in rejected], dtype=float).reshape(-1, 4)
    arrays[f"{name}/rejected/counts"] = np.array([(r.in_box, r.matched) for r in rejected], dtype=np.int64).reshape(-1, 2)
    arrays[f"{name}/rejected/failed"] = np.array([r.failed for r in rejected], dtype=bool)


def _get_stitch(arrays, name: str) -> StitchResult:
    ia, ib = _get_parts(arrays, f"{name}/pairs/ia"), _get_parts(arrays, f"{name}/pairs/ib")
    inliers, residuals = _get_parts(arrays, f"{name}/pairs/inliers"), _get_parts(arrays, f"{name}/pairs/residuals")
    shift, rotation, center = (arrays[f"{name}/pairs/{key}"] for key in ("shift", "rotation", "center"))
    pairs = [PairResult(int(i), int(j), tuple(box), ia[k], ib[k],
                        OverlapFit(shift[k], float(rotation[k]), center[k], inliers[k], residuals[k]))
             for k, ((i, j), box) in enumerate(zip(arrays[f"{name}/pairs/ij"], arrays[f"{name}/pairs/box"].tolist()))]
    rejected = [RejectedPair(int(i), int(j), tuple(box), int(in_box), int(matched), bool(failed))
                for (i, j), box, (in_box, matched), failed in zip(arrays[f"{name}/rejected/ij"],
                                                                  arrays[f"{name}/rejected/box"].tolist(),
                                                                  arrays[f"{name}/rejected/counts"],
                                                                  arrays[f"{name}/rejected/failed"])]
    return StitchResult(pairs, rejected, arrays[f"{name}/corrections"])
