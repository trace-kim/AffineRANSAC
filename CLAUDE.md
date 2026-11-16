# CLAUDE.md

Photomask registration-error measurement: OASIS design vs. stitched SEM images, with RANSAC-robust affine calibration.

**Before any work, read the parts of `docs/SPEC.md` the task touches**, not the whole file (it is long): find them with `grep -n '^#' docs/SPEC.md`, then read the Decision Log (§12) rows they cite and the open questions (§13). It is the source of truth for architecture, coordinate conventions, algorithms, decisions (§12) and open questions (§13).

## The real data: what you can and cannot know
- The real data is only on the user's PCs. You never see an image, a result or a log of theirs.
- Your only evidence is what the user tells you. They type findings from the logs by hand when you ask: ask for a few named values in one message and say what each answer would mean. Never ask for pasted output, and never hand back a list of settings to toggle.
- What the user says they saw or checked is a fact. Don't ask them to check it again; work out what it implies.
- Synthetic images show that the code does what you intended, nothing more: a test may pin down a bug you found by reading the code. They never show that a method works on the real images, how long it takes there, or what causes a problem there. Don't rank, reject or tune methods on them, and don't answer a real-data problem with a synthetic reproduction or synthetic timings. (A timing on this PC holds only for work whose cost does not depend on what is in the image; say so.)
- Never state as fact anything about the user's images, datasets, PCs, settings or intentions that they did not tell you. Write a guess as a guess.

## How to work, by kind of request
- **A problem seen on the real data** (wrong result, failure, slow run): change no behaviour until the cause is known.
  1. Start from what the user reported: what is wrong, how large, where. Read the whole code path that produces it, every stage and not only the latest commits, and work out what each step does and costs with data like theirs.
  2. Say what the code makes certain. For the rest, list the possible causes with the one fact that tells each apart, and ask for those facts in one message. Don't rank causes by which one you managed to reproduce.
  3. If reading the code does not give the cause, add verbose logs that pinpoint the exact step causing the problem: one change that covers every step of the path (its time, its counts, the values each decision used), so that a single run answers it, never one small probe per run. Make the answer short to type: one line per step and a last line naming the step that stands out. Commit, push, and say which lines to read.
  4. Fix only once the cause is known, from the code or from the user's findings.
- **A request that leaves the method to you, or data of a kind not handled before:** proposal first, in a few lines: what is reused from the pipeline (by its spec name), what is new, and what you assume about the images. Build nothing until the user says go.
- **An approved plan:** build every step, one commit each. Stop only where a step cannot be built without a fact from the real data, and ask for that fact.
- **Any other feature or fix:** build it as in "Development approach" below. Several unrelated things in one message: one at a time, and wait for the user's result on each before the next.
- **Always:**
  - Commit and push every change (the user tests by `git pull`), then say what to check on the real data. Passing tests is not "tested" here; the user's run on the real data is.
  - End every reply that builds or diagnoses with **Assumed, not known:** what you took for granted about the data, the PCs or what the user wants.
  - Edit files with the Edit and Write tools, not through shell heredocs or one-line scripts: on this Windows shell they lose backslashes and line breaks.

## Development approach — never overengineer
- Build the smallest useful version of one feature or fix at a time. New behaviour is an option, off by default, unless the user asked for it to replace the old behaviour.
- Synthetic data: write the tests the step needs. No benchmarks, variant comparisons or tuning on synthetic data beyond them.
- Build only what the current step needs. The spec describes the full target; don't build ahead of it (no speculative abstractions, options or layers "for later").
- Write each piece so a human reviewer can quickly read it and understand what it does: short functions, plain names, straightforward logic, brief comments where the intent isn't obvious.
- Keep changes small and reviewable. Prefer several small steps over one large one.

## Key rules
- If code deviates from the spec, update `docs/SPEC.md` in the same change and add a Decision Log entry.
- Don't guess answers to the open questions in §13; ask the user.
- Internal units: nm, float64. **Display** (notebooks, plots, tables): positions in **µm**, position errors/differences in **nm**. Design frame is y-up; image pixel frame `(x, y)` px is y-down (origin = centre of the top-left pixel). All frame conversions go through `geometry/frames.py`.
- RANSAC is used only for the SEM→design fit, never for tile stitching. Its outliers are excluded from fitting the affine, never from reporting.
- Viewers in the package (`view_*.py`) must be interactive and use **pyqtgraph** (PySide6). matplotlib is only for quick plots in notebooks.
- Notebooks (`notebooks/`) are for the user to test implemented functions on **real data** in VS Code. A notebook is a settings cell (paths/params) plus short cells that call library functions and print results. **No demo modes or synthetic-data code.** Commit notebooks with outputs cleared.
- Every new dependency goes in `pyproject.toml`: in `dependencies` if production code needs it, otherwise in the `dev` extra. When dependencies change, tell the user to re-run `pip install -e ".[dev]"`.
- Always seed the RNG. Tests run on synthetic data (spec §10); they check the code, not the method.
- MaskReg (`release/`) and the library code it runs: a change that alters its results raises `ANALYSIS`, one that makes older run files unreadable raises `FORMAT`; record the new values in `tests/test_release_versions.py` and add a `CHANGELOG.md` line under *Unreleased* in the same commit (spec "Versions and run files", D99). Change `__version__` only when the user releases a version.

## Remote agent (one-way handoff)
The real data (SEM images, `.oas`, metadata CSV) is only on the user's remote machine. Data-specific work there is done by a **remote agent**: **opencode** with an unknown model, *not* Claude Code. Its standing rules are in `AGENTS.md`.
- **One-way:** you write tasks in `docs/tasks/TNNN-<name>.md` and push them; the user pulls them on the remote. The remote agent **never commits or pushes**, and you **never see its code, files or reports**. **Nothing leaves the remote machine:** never ask the user to paste or forward reports, code or data. What you learn comes only from what the user chooses to tell you in their own words. When you need a fact, ask the user a short, specific question; they may already know the answer (e.g. the image orientation).
- **Assume its work is correct.** If something comes back wrong, fix it with a new or updated task file.
- **Task files must be self-contained and exact:** list the files it may create; give the exact interface (names, types, units, frames), the required tests, the checks to run on the real data, and what the summary must contain. Never rely on it reading `CLAUDE.md`.
- **Build against the interface, never the implementation.** Guard each interface with a contract test (`tests/test_*_contract.py`) that skips where the module is missing, so it runs only on the remote. Production code you write must not import remote-only modules at import time. Take their data as plain arguments; the notebooks (run on the remote) glue them together.
- **Keep `git pull` on the remote working:** never create or modify the paths reserved for the remote agent:
  - `src/affine_ransac/io/metadata.py`, `tests/test_metadata.py` (T001)
  - `docs/remote/` (gitignored; its reports and scratch scripts)
- The task's `**Status:**` line is maintained by you, from what the user reports.
- Never commit data files or real data values.
