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
- Internal units: nm, float64. **Display** (notebooks, plots, tables): positions in **µm**, position errors/differences in **nm**. Design frame is y-up; image pixel frame `(x, y)` px is y-down (origin = centre of the top-left pixel). All frame conversions go through `geometry/frames.py`.
- RANSAC is used only for the SEM→design fit, never for tile stitching. Its outliers are excluded from fitting the affine, never from reporting.
- Viewers in the package (`view_*.py`) must be interactive and use **pyqtgraph** (PySide6). matplotlib is only for quick plots in notebooks.
- Notebooks (`notebooks/`) are for the user to test implemented functions on **real data** in VS Code. A notebook is a settings cell (paths/params) plus short cells that call library functions and print results. **No demo modes or synthetic-data code.** Commit notebooks with outputs cleared.
- Every new dependency goes in `pyproject.toml`: in `dependencies` if production code needs it, otherwise in the `dev` extra. When dependencies change, tell the user to re-run `pip install -e ".[dev]"`.
- Always seed the RNG. Validate with synthetic data (spec §10).

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
