# CLAUDE.md

Photomask registration-error measurement: OASIS design vs. stitched SEM images, with RANSAC-robust affine calibration.

**Before any work, read the parts of `docs/SPEC.md` the task touches**, not the whole file (it is long): find them with `grep -n '^#' docs/SPEC.md`, then read the open questions (§13) and the decision rows the text cites (D123: `grep -n '^| D123 ' docs/DECISIONS.md`). The spec is the source of truth for what the code does: architecture, coordinate conventions, algorithms. `docs/DECISIONS.md` is the Decision Log, the history with the reasons; it also holds decisions that later rows replaced, so never read it as the current behaviour. `docs/OPEN.md` lists what waits for the user's decision, what is built but not yet run on the real data, and the known problems: read it too, and keep it current.

## The real data: what you can and cannot know
- The real images and datasets are on the user's remote PC, which local Codex cannot access (reaffirmed
  2026-11-16). Do not search this development PC for them or ask for a local data-folder path. You never see an
  image, a result or a log of theirs; local checks cannot establish real-data timings or stitching quality.
- Your only evidence is what the user tells you. They type findings from the logs by hand when you ask: ask for a few named values in one message and say what each answer would mean. Never ask for pasted output, and never hand back a list of settings to toggle.
- What the user says they saw or checked is a fact. Don't ask them to check it again; work out what it implies.
- Synthetic images show that the code does what you intended, nothing more: a test may pin down a bug you found by reading the code. They never show that a method works on the real images, how long it takes there, or what causes a problem there. Don't rank, reject or tune methods on them, and don't answer a real-data problem with a synthetic reproduction or synthetic timings. (A timing on this PC holds only for work whose cost does not depend on what is in the image; say so.)
- Never state as fact anything about the user's images, datasets, PCs, settings or intentions that they did not tell you. Write a guess as a guess.
- Before asking the user anything, look for the answer in the spec (§13), `docs/notes/`, `docs/OPEN.md` and your memory note `user-stated-facts`; if it may have been asked before, search the session transcripts (`~/.claude/projects/E--PythonProjects-AffineRANSAC/*.jsonl`). Add every fact the user states about their data or PCs to that note, with the date.

## How to work, by kind of request
- **A problem seen on the real data** (wrong result, failure, slow run): change no behaviour until the cause is known.
  1. Start from what the user reported: what is wrong, how large, where. Read the whole code path that produces it, every stage and not only the latest commits, and work out what each step does and costs with data like theirs.
  2. Say what the code makes certain. For the rest, list the possible causes with the one fact that tells each apart, and ask for those facts in one message. Don't rank causes by which one you managed to reproduce.
  3. If reading the code does not give the cause, add verbose logs that pinpoint the exact step causing the problem: one change that covers every step of the path (its time, its counts, the values each decision used), so that a single run answers it, never one small probe per run. Make the answer short to type: one line per step and a last line naming the step that stands out. Commit, push, and say which lines to read.
  4. Fix only once the cause is known, from the code or from the user's findings.
- **A request that leaves the method to you, or data of a kind not handled before:** proposal first, in a few lines: what is reused from the pipeline (by its spec name), what is new, and what you assume about the images. Build nothing until the user says go. Offer the methods others use on such images (search; sources from SEM and e-beam work), justified by how the SEM forms the image, not by an analogy from another field. Where several fit, build each as an off-by-default option with the library's standard implementation; the user compares them on the real images.
- **An approved plan:** build every step, one commit each. Stop only where a step cannot be built without a fact from the real data, and ask for that fact.
- **Any other feature or fix:** build it as in "Development approach" below. Several unrelated things in one message: one at a time, and wait for the user's result on each before the next.
- **Always:**
  - Commit and push every change (the user tests by `git pull`), then say what to check on the real data. Passing tests is not "tested" here; the user's run on the real data is.
  - End every reply that builds or diagnoses with **Assumed, not known:** what you took for granted about the data, the PCs or what the user wants.
  - Edit files with the Edit and Write tools, not through shell heredocs or one-line scripts: on this Windows shell they lose backslashes and line breaks.

## Talking to the user
- Conclusion first, then plain words the user can picture: one idea per short paragraph, no file or function names and no jargon inside the explanation, a number only as proof. A method explained "step by step" needs one picture per step on one running example, not a numbered list of sentences.
- Say which parts of an explanation were measured (and on what) and which are inferred. No digits you did not measure.
- Asked "is this just X?", answer yes or no first, then only the differences. Rank options by what they answer and how surely, not by what already exists in the code or is cheaper to build.
- A decision the user made covers the case it was made for. Before using it elsewhere, restate that case and check the new one against it.
- For choices about how something reaches users (packaging, installer, updates), show the standard complete solution too, not only the one with the least setup: "never overengineer" is about code structure.

## Development approach — never overengineer
- Build the smallest useful version of one feature or fix at a time. New behaviour is an option, off by default, unless the user asked for it to replace the old behaviour.
- Synthetic data: write the correctness tests the step needs. When speed measurement is requested, carry it
  through to a measured before/after result and compare numerical outputs without waiting for separate prompts.
  Stop benchmarks when cancelled; a later request for the new speed reauthorizes the necessary measurement.
  A pipeline benchmark must include preprocessing already known to be required, especially total-variation
  denoising, and account for the configured upscaling. Never switch those off to make a speed estimate smaller,
  or present a stage-only measurement as the whole run's time. Distinguish measured local speed, extrapolation,
  intended equivalence and observed output differences explicitly.
- Build only what the current step needs. The spec describes the full target; don't build ahead of it (no speculative abstractions, options or layers "for later").
- Write each piece so a human reviewer can quickly read it and understand what it does: short functions, plain names, straightforward logic, brief comments where the intent isn't obvious.
- Keep changes small and reviewable. Prefer several small steps over one large one.
- Build for any dataset, not the one at hand: no tuning to features of the current data. There are two patterns only: contacts, and arbitrary contours (any shape, open where a region leaves the image). Don't add a mode per pattern type (line & space is one case of contours; the existing `lines` pattern stays as it is), and assume no direction, extent or shape: ask before a rule relies on one.
- A step that drops data (a filter, a threshold) ships off unless its value was checked on the real data, and the log says loudly (`!!`) when it empties a tile.

## Key rules
- If code deviates from the spec, update `docs/SPEC.md` in the same change and add a row to `docs/DECISIONS.md`.
- Don't guess answers to the open questions in §13; ask the user.
- Internal units: nm, float64. **Display** (notebooks, plots, tables): positions in **µm**, position errors/differences in **nm**. Design frame is y-up; image pixel frame `(x, y)` px is y-down (origin = centre of the top-left pixel). All frame conversions go through `geometry/frames.py`.
- RANSAC is used only for the SEM→design fit, never for tile stitching. Its outliers are excluded from fitting the affine, never from reporting. Its threshold τ stays 0.5 nm: never change or tune it. Drift correction comes before any affine: never fit an affine to the data, in any role, before the drift step.
- Viewers in the package (`view_*.py`) must be interactive and use **pyqtgraph** (PySide6). matplotlib is only for quick plots in notebooks.
- UI requests are literal and narrow: change only the view or control asked about. Never resize or regroup other graphs or controls, never merge per-plot controls into one, never change a default unasked. If a new tab, window or layout seems needed, propose it in one line first.
- A new result goes into the existing plots as extra lines or sets, named in the legend. A value measured on a graph is drawn on the graph (lines, circles, shading), not listed as a table of numbers or as coordinates in text.
- Viewers start quiet: every layer off, failure markers too, with failures reported as text. One click selects one thing; evidence is shown only where something is wrong; verdicts are words, not a symbol code; normal behaviour is never drawn in an alarm colour. Pop-ups and warnings: at most two short sentences (what it does, the one cost). Check a UI change on a realistic mosaic (several stripes, an inner tile with 8 neighbours), not the 2 × 2 test folder.
- Notebooks (`notebooks/`) are for the user to test implemented functions on **real data** in VS Code. A notebook is a settings cell (paths/params) plus short cells that call library functions and print results. **No demo modes or synthetic-data code.** Commit notebooks with outputs cleared.
- Every new dependency goes in `pyproject.toml`: in `dependencies` if production code needs it, otherwise in the `dev` extra. When dependencies change, tell the user to re-run `pip install -e ".[dev]"`.
- Always seed the RNG. Tests run on synthetic data (spec §10); they check the code, not the method.
- MaskReg (`release/`) and the library code it runs: a change that alters its results raises `ANALYSIS`, one that makes older run files unreadable raises `FORMAT`; record the new values in `tests/test_release_versions.py` and add a `CHANGELOG.md` line under *Unreleased* in the same commit (spec "Versions and run files", D99). Change `__version__` only when the user releases a version.
- The user runs MaskReg from source on their PCs (`python -m affine_ransac.release.app`); an update is `git pull`. Never tell them to rebuild, and keep what they need out of frozen-only paths (`packaging/maskreg.py`).
- Never stop processes by name: `taskkill /IM python.exe` closes the user's own Python programs (MaskReg, notebook kernels). Stop only your own jobs, by task id or PID.

## Remote agent (one-way handoff)
The real data (SEM images, `.oas`, metadata CSV) is only on the user's remote machine. Data-specific work there is done by a **remote agent**: **opencode** with an unknown model, *not* Claude Code. Its standing rules are in `AGENTS.md`.
`AGENTS.md` governs that remote task-file role, not an agent doing local development. For local development,
follow this file; reviewing and fixing the existing code does not require a remote task file.
- **One-way:** you write tasks in `docs/tasks/TNNN-<name>.md` and push them; the user pulls them on the remote. The remote agent **never commits or pushes**, and you **never see its code, files or reports**. **Nothing leaves the remote machine:** never ask the user to paste or forward reports, code or data. What you learn comes only from what the user chooses to tell you in their own words. When you need a fact, ask the user a short, specific question; they may already know the answer (e.g. the image orientation).
- **Assume its work is correct.** If something comes back wrong, fix it with a new or updated task file.
- **Task files must be self-contained and exact:** list the files it may create; give the exact interface (names, types, units, frames), the required tests, the checks to run on the real data, and what the summary must contain. Never rely on it reading `CLAUDE.md`. One small step per task file, on a small sample of tiles; never bundle analyses to save round trips.
- **Build against the interface, never the implementation.** Guard each interface with a contract test (`tests/test_*_contract.py`) that skips where the module is missing, so it runs only on the remote. Production code you write must not import remote-only modules at import time. Take their data as plain arguments; the notebooks (run on the remote) glue them together.
- **Keep `git pull` on the remote working:** never create or modify the paths reserved for the remote agent:
  - `src/affine_ransac/io/metadata.py`, `tests/test_metadata.py` (T001)
  - `docs/remote/` (gitignored; its reports and scratch scripts)
- The task's `**Status:**` line is maintained by you, from what the user reports.
- Never commit data files or real data values.

## Keeping these rules
- When the user corrects how you work, change the matching rule in this file in the same turn: replace it, don't stack a second rule on it, and don't write a memory note for it.
- Where things go: what is open → `docs/OPEN.md`; facts the user states about their data and PCs → your memory note `user-stated-facts`; decisions and their reasons → `docs/DECISIONS.md`. Beyond that, memory notes are only for facts about this PC and its tools.
