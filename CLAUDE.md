# CLAUDE.md

Photomask registration-error measurement: OASIS design vs. stitched SEM images, with RANSAC-robust affine calibration.

**Read `docs/SPEC.md` before any work.** It is the source of truth for architecture, coordinate conventions, algorithms, decisions (§12) and open questions (§13).

## Development approach — never overengineer
- Start small. Take one step at a time and implement bit by bit, testing each piece before moving on.
- Build only what the current step needs. The spec describes the full target; don't build ahead of it (no speculative abstractions, options or layers "for later").
- Write each piece so a human reviewer can quickly read it and understand what it does: short functions, plain names, straightforward logic, brief comments where the intent isn't obvious.
- Keep changes small and reviewable. Prefer several small steps over one large one.

## Key rules
- If code deviates from the spec, update `docs/SPEC.md` in the same change and add a Decision Log entry.
- Don't guess answers to the open questions in §13; ask the user.
- Internal units: nm, float64. Design frame is y-up; image pixel frame `(x, y)` px is y-down (origin = centre of the top-left pixel). All frame conversions go through `geometry/frames.py`.
- RANSAC is used only for the SEM→design fit, never for tile stitching. Its outliers are excluded from fitting the affine, never from reporting.
- Viewers in the package (`view_*.py`) must be interactive and use **pyqtgraph** (PySide6). matplotlib is only for quick plots in notebooks.
- Notebooks (`notebooks/`) are for the user to test implemented functions on **real data** in VS Code. A notebook is a settings cell (paths/params) plus short cells that call library functions and print results. **No demo modes or synthetic-data code.** Commit notebooks with outputs cleared.
- Every new dependency goes in `pyproject.toml`: in `dependencies` if production code needs it, otherwise in the `dev` extra. When dependencies change, tell the user to re-run `pip install -e ".[dev]"`.
- Always seed the RNG. Validate with synthetic data (spec §10).

## Task handoff: local agent ↔ remote agent
The real data (SEM images, `.oas`, metadata CSV) is only on a remote machine. Data-specific work (parsing, checks on real data) is done there by a **remote agent**, instructed through task files.
- Tasks live in `docs/tasks/TNNN-<name>.md` with a `**Status:**` line (`OPEN`, `DONE`, `BLOCKED: <reason>`). Reports go in `docs/tasks/TNNN-report.md`.
- **If you can see the real data (remote agent):** do the `OPEN` tasks in number order, exactly as written. Then set the status, write the report, commit with a message starting `TNNN:`, and push.
- **If you can't see the data (local agent):** write tasks with an exact interface (names, types, units), required tests, and the facts to report. Build only on what the reports state; don't guess data formats.
- Never commit data files or real data values. Docs and tests use made-up values in the real format.
- The remote agent changes only the files its task names. It reports needed changes elsewhere, with evidence, instead of making them.
