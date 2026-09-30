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
- Internal units: nm, float64. Design frame is y-up; image pixel frame is v-down. All frame conversions go through `geometry/frames.py`.
- RANSAC is used only for the SEM→design fit, never for tile stitching. Its outliers are excluded from fitting the affine, never from reporting.
- All viewers must be interactive and use **pyqtgraph** (PySide6). Don't use matplotlib.
- Always seed the RNG. Validate with synthetic data (spec §10).
