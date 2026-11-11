"""MaskReg for an AI agent, from the command line, without a window (docs/SPEC.md D154).

    python -m affine_ransac.release.agent run <folder> [--set name=value ...] [--out file.npz]
    python -m affine_ransac.release.agent review <file.npz>

run: MaskReg's analysis of the folder (release.analysis.analyse_folder, so the same numbers), with MaskReg's default
parameters except those given with --set (only the ones of the Parameters dialog, PARAMETERS). The run is saved as a
run file of its own (--out; default maskreg_runs/<folder name>_<time>.npz under the current folder), never as the
folder's saved run, which stays the user's. review: the same summary of a run file saved before.

Standard output is one JSON object, the review of the run (review); the log and the progress go to standard error.
Exit code 1 with {"error": ...} if it could not run, or with the review of what was found and "failed" if the run
stopped after the stitching (AnalysisFailed; nothing is saved, as in MaskReg).
"""

import argparse
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from affine_ransac.analysis import Settings
from affine_ransac.registration import error_summary
from affine_ransac.release.analysis import DEFAULTS, PARAMETERS, AnalysisFailed, ReleaseResult, analyse_folder
from affine_ransac.release.cache import NewerRunFile, read_run, run_arrays, write_run

MAX_LISTED = 20  # names listed per finding; the count is always given


def changed_settings(kind: str, pairs: list[str]) -> Settings:
    """MaskReg's defaults with the name=value pairs applied. ValueError for a name that is not a parameter of this
    kind of folder or a value of the wrong type (ranges are checked by the analysis)."""
    values = {}
    for pair in pairs:
        name, _, text = pair.partition("=")
        if name not in PARAMETERS[kind]:
            raise ValueError(f"{name!r} is not a parameter of {kind} folders. They are: {', '.join(PARAMETERS[kind])}")
        values[name] = parsed(name, text, getattr(DEFAULTS, name))
    return replace(DEFAULTS, **values)


def parsed(name: str, text: str, default):
    """The text as a value of the default's type (true / false for an on-off parameter)."""
    if isinstance(default, bool):
        if text.lower() not in ("true", "false"):
            raise ValueError(f"{name} must be true or false (is {text!r})")
        return text.lower() == "true"
    try:
        return type(default)(text)
    except ValueError:
        raise ValueError(f"{name} must be a {type(default).__name__} (is {text!r})") from None


def listed(names: list[str]) -> dict:
    return {"count": len(names), "names": names[:MAX_LISTED]}


def stitching_review(r: ReleaseResult, stitch) -> dict:
    """One stitching in numbers: its overlaps used, not matched (enough contacts, too few pairs) and skipped (too few
    contacts), and the tiles it could not place."""
    def pair_names(found):
        return listed([f"{r.tile_ids[o.i]} / {r.tile_ids[o.j]}" for o in found])
    return {"overlaps_used": len(stitch.pairs),
            "overlaps_not_matched": pair_names([o for o in stitch.rejected if o.failed]),
            "overlaps_skipped_too_few_contacts": sum(not o.failed for o in stitch.rejected),
            "tiles_not_stitched": listed([r.tile_ids[k] for k in stitch.unplaced])}


def review(r: ReleaseResult, saved: str = "") -> dict:
    """What an agent needs to judge a run, as plain values: positions in µm, errors in nm."""
    from affine_ransac.release.views import stitching_verdict  # imports Qt: only when a review is made
    changed = {name: getattr(r.settings, name) for name in PARAMETERS[r.kind]
               if getattr(r.settings, name) != getattr(DEFAULTS, name)}
    slipped = None if r.slipped is None else listed([f"{r.tile_ids[i]} / {r.tile_ids[j]}" for i, j in r.slipped])
    found = {"folder": r.folder, "kind": r.kind, "saved": saved, "maskreg_version": r.version, "analysis": r.analysis,
             "tiles": len(r.tile_ids), "stripes": int(r.stripe.max()) + 1,
             "parameters_changed_from_default": changed,
             "stitching": {"verdict": [re.sub(r"<[^>]+>", "", line) for line in stitching_verdict(r)],
                           "measured": stitching_review(r, r.stitch), "design": stitching_review(r, r.design_stitch),
                           "overlaps_that_do_not_line_up": slipped},
             "contacts_measured": len(r.design_nm),
             "problems_logged": [line for line in r.log if line.startswith("!!")]}
    if r.moving is not None:  # None: a run that stopped after the stitching
        # + 0.0: no "-0.0"
        errors = {name: round(value, 3) + 0.0 for name, value in error_summary(r.moving.residuals).items()}
        found["registration_error_after_moving_window_nm"] = errors
        found["moving_windows"] = {"count": len(r.moving.counts), "fewest_contacts": int(r.moving.counts.min())}
    return found


def run(folder: Path, pairs: list[str], out: Path | None) -> dict:
    """Analyse the folder, save the run to `out`, return its review with the file's path."""
    from affine_ransac.app import folder_kind, image_records  # imports Qt; the records: only on the data machine
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a folder")
    folder = folder.resolve()  # the run file names it: in full, wherever the command was given
    kind = folder_kind(folder)
    settings = changed_settings(kind, pairs)
    records = image_records(folder) if kind == "images" else None
    tenths = [0]

    def progress(fraction: float, stage: str):  # one line per tenth of the run
        if int(fraction * 10) > tenths[0]:
            tenths[0] = int(fraction * 10)
            print(f"[{tenths[0] * 10} %] {stage}", file=sys.stderr, flush=True)

    result = analyse_folder(folder, kind, settings, records, lambda text: print(text, file=sys.stderr, flush=True),
                            progress)
    out = out or Path.cwd() / "maskreg_runs" / f"{folder.name}_{time.strftime('%Y%m%d-%H%M%S')}.npz"
    write_run(run_arrays(result), out)
    return review_file(out)  # read back from the file: what a later review will say


def review_file(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"{path} is not a file")
    read = read_run(path, None, lambda text: print(text, file=sys.stderr))
    if read is None:
        raise ValueError(f"{path} was saved in an older file format that this MaskReg cannot read: run the folder again")
    return {"run_file": str(path.resolve()), **review(*read)}


def plain(value):
    """numpy numbers as JSON numbers."""
    return value.item() if isinstance(value, np.generic) else str(value)


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m affine_ransac.release.agent", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run_command = commands.add_parser("run", help="analyse a folder, save the run, print its review")
    run_command.add_argument("folder", type=Path)
    run_command.add_argument("--set", action="append", default=[], metavar="name=value", dest="pairs",
                             help="a parameter of the Parameters dialog other than its default; may be repeated")
    run_command.add_argument("--out", type=Path, help="the run file to write (default: under maskreg_runs/)")
    review_command = commands.add_parser("review", help="print the review of a run file")
    review_command.add_argument("run_file", type=Path)
    args = parser.parse_args(arguments)
    sys.stderr.reconfigure(errors="replace")  # a console that cannot show a character of the log must not stop the run
    code = 0
    try:
        found = run(args.folder, args.pairs, args.out) if args.command == "run" else review_file(args.run_file)
    except AnalysisFailed as failure:
        names = failure.result.tile_ids
        found = {"failed": str(failure), **review(failure.result),
                 "tiles_not_measured": {names[k]: reason for k, reason in failure.not_measured.items()}}
        code = 1
    except (ValueError, RuntimeError, OSError, NewerRunFile) as error:
        found, code = {"error": str(error)}, 1
    print(json.dumps(found, indent=1, default=plain))
    return code


if __name__ == "__main__":
    sys.exit(main())
