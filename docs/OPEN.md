# Open items

What waits for a decision of the user, what is built but not yet confirmed on the real data, known problems that
are not fixed, and what is queued. Only the user's run on the real data confirms anything.

- Each item has the date of the last word on it. An old date means "ask before acting on it", not "still true".
- Change an item in the same commit that changes its state. Delete it once the user confirms or drops it: the
  history is in git and in `docs/DECISIONS.md`.
- No values or file names from the real data here.

Started 2026-11-16 from the working notes kept until then. Where it says "code checked", the code was read that
day; nothing here was confirmed again by the user that day.

## Waiting for the user's decision

| Item | Last word | The decision |
|---|---|---|
| Load contours: the "same centres" check is too loose | 2026-10-30 | `release/analysis.read_contours` compares with `np.allclose(..., atol=0.01)` and the default relative tolerance, which allows about 10 nm per mm of coordinate; the spec says within 0.01 nm (D141). Code checked: still so. The fix is `rtol=0` and a test at mask-scale coordinates; it may flag runs that pass today. Waiting for the go. |
| Stripe drift: one unmatched tile breaks the chain (research app) | 2026-10-09 | A tile whose overlaps do not match breaks the stitching in measurement order: only the largest piece is placed, the rest stays nominal, and the drift-corrected sets lose most tiles. Code checked: a tile not stitched in measurement order still stays nominal. Options told to the user: bridge the breaks through the all-overlap placement, or flag tiles with large offsets. Waiting for the choice. |
| Single tiles paired one row off on a repeating array | 2026-10-28 | `docs/notes/2026-10-28-single-tile-row-off.md`: two flaws in `matching.consistent_choice`, found on synthetic staggered mosaics, not confirmed on the real data; the note ends with four questions to the user. Spec S5 *Known problem* 1 and 2 are earlier synthetic diagnoses that the user did not accept. D143 and D144 (below) were built the same day. |

## Built, waiting for the user's run on the real data

| What | Built | State and what to check |
|---|---|---|
| Reuse contour preprocessing and shared region statistics (D163) | 2026-11-16 | Diagnosis reuses the native-resolution flattened/denoised image while only downstream settings change; new images or active preprocessing changes invalidate it. Numba warm-up runs once per process; contours sharing a region reuse its band level. Exact-output and invalidation tests pass; real-data timing is not measured. Check the log's reuse entry while adjusting area or refinement settings. The first denoising of a new image still uses the D159 solver. |
| Memory allocation during contour refinement (D166) | 2026-11-18 | User reported an allocation failure during refinement. Code review found redundant full-image float buffers, whole-contour sampling maps, CPU-only worker sizing and consumed futures retaining original coordinates alongside translated copies. D166 removes that waste and limits active contour work using actual image sizes and RAM/commit headroom, including kept pools. Denoising/upscaling and output calculations remain. Synthetic full-size extraction and the normal eight-image run path match prior outputs exactly; process memory results and limits are in `docs/notes/2026-11-18-contour-memory.md`. Check the normal run after restarting MaskReg: the `Contour memory` line names the selected worker limit. No claim yet that this resolves the user's remote allocation failure. |
| Equalization before the threshold (D167, off by default) | 2026-11-18 | User asked for histogram equalization before Otsu, suspecting it as the cause of wrong contours, to check in the contour diagnosis window. Built: `equalize` global (one threshold on the image, chosen on the equalized histogram) and adaptive (CLAHE, `clahe_tiles`, `clahe_clip`; the threshold mask only, refinement on the image itself). Check in the window, threshold mask on: where the contours were wrong, does global or adaptive put the mask on the pattern? Not known: what was wrong with the contours (missing parts, extra regions, or edges in the wrong place), and where in the image. |
| Stitching by contours (D156), pattern `contours` | 2026-11-17 | The user's runs took hours and left many images not placed. Why they were not placed is not known. D165 extends D164 with bounded distance painting, run sums where cheaper than FFTs, coverage on the shift grid, refinement only for the next selected candidate, and grouped interior medians. After cancelling the earlier benchmark, the user asked for the new speed and output comparison. On 32 synthetic 2048² images, TV denoising and 4× upscaling, two workers: D164 → D165 extraction plus stitching 199.10 → 46.52 s (4.28×); stitching alone 37.69 → 19.39 s (1.94×). From the original stitcher before either optimization: 62.30 → 19.39 s (3.21×). Extracted edges and placements/diagnostics were identical in the comparisons. Details and limits: `docs/notes/2026-11-17-contour-tv4-comparison.md`. The earlier estimate omitting preprocessing remains withdrawn. No remote timing/quality claim: the speedup there, TV's remaining cost and the unplaced-image problem are still unconfirmed. |
| SEM–design pairs after the stitching (D143); stitching method `fingerprint` (D144, off by default) | 2026-10-28 | Not run on the real data as of 2026-10-28. Order: the defaults first (D143 alone), then `stitch_method = fingerprint`. If D143 alone removes the staircase, the fingerprint can be reverted. Parts added only for problems seen on synthetic mosaics, to trim if the run is good: in D143 `_design_around` with the −1 rows of `merge_values`, and `MIN_FOLLOWED` / `_followed`; in D144 `clearly_better` / `by_fit` in `overlap_checks`, and `capped_rms` (code checked: the contour stitching uses `capped_rms` too now). The log lines (`offset_steps`, `fingerprint_text`) stay: the user judges by them. Gap: D143 has no setting to go back to the old pairing, though the user asked for fixes to be optional. |
| Parameters window in sections (D147) | 2026-10-30 | Layout only; waiting for the user's look. |
| High-performance mode (D151, off by default) | 2026-11-04 | On the workstation, a second run should load without the long start-up wait. |
| MaskReg from the command line for an agent, step 1: `run`, `review` (D154) | 2026-11-11 | A run on real images. |

Settled, kept so it is not asked again: pattern `contours` (D155) outlines the user's lines with a large enough
`min_area_px` (user, 2026-11-16). Pattern `lines` (D153) found none on the real images, whose lines are vertical;
it stays as built at the user's wish (2026-11-12) and is not to be extended.

## Asked for, not built

- **A faster proper denoising for contours, and a reliable contour extraction** (user, 2026-11-16). The method was
  left open, so a proposal comes first. D161 (regions found in a binned image) was built for it without one and
  reverted the same day: averaging blocks of pixels does not denoise, and total variation is needed for a proper
  threshold mask on their images. Noted with D159, not built: a total variation solver that needs fewer
  rounds (it changes the results, so the user judges it). What was built and left out among the denoisers, and
  why: D132. D163 removes repeated preparation during diagnosis and repeated per-region work without changing the
  method. It does not settle this request for a faster first denoising pass. Before further kernel or solver changes,
  use the D159 rounds/time log on the real images. D165 removes the interior-median sort over the whole upscaled
  image and carries extraction timings into the run log, while preserving the denoiser. Duplicate
  upscale/threshold work for the diagnosis display remains.
- **Contours, stage 3:** pair the contours with the design, fit, report, as step 3 of
  `docs/notes/2026-11-08-line-space-plan.md`. What is reported is settled there and is not to be asked again. It
  needs a stitching that works on the real images first, and how the `.oas` clips hold the lines: look at what the
  reader finds, ask only if it cannot be built without.
- **Parameters tidy-up**, agreed 2026-10-30, one at a time: (a) `min_matched` default 2 (code checked: still 5;
  check whether `ANALYSIS` must rise); (b) a "Default" set that can be restored (not settled whether an entry in the
  saved-parameters list or a restore button: ask); (c) options that do nothing with the current choices greyed out;
  (d) plain labels, the setting's name in the tooltip; (e) whether the contour settings stay folded there or live
  only in the contour diagnosis window (the user has not answered).
- **MaskReg for an agent, later steps** (D154): listing the parameters, comparing runs, pictures for the agent, the
  skill file.

## Known problems, not fixed

- The user's requested synthetic horizontal-line benchmark (2026-11-16) placed all 128 images with at most
  0.0714 nm vertical error, but moved them along the unmeasured horizontal direction by up to 86 pixels despite
  zero true horizontal error. All choices reported one measured direction. The 16-image example gave identical
  42-pixel sideways shifts before and after D164, so it predates the optimization. No real-data claim or fix;
  settings and reproduction are in `docs/notes/2026-11-16-contour-cpu-benchmark.md`.
- A moving-window position holding fewer than 3 contacts stops the whole run (`fitting/moving_window.py`, code
  checked). The user met it on 2026-10-27.

## Parked: not asked for, here so they are not invented twice

- In-image formula (D145, D146; the user found it better on the real data, 2026-10-30): if a pattern stays along
  the line-start edge of every image, one scan-start term; elsewhere order 4 or 5. The user decides.
- Contacts lost at low contrast with noise (D122): finding them again by correlation with the image's own mean
  contact was tried on synthetic images only and needs robust seeds first. To propose only if the user reports
  missing contacts.
- The sawtooth left by translation stitching: the user asked to leave it alone (2026-10-07).
- The slow start-up of the tile-loading workers on the workstation has a cause outside MaskReg that was never
  confirmed (2026-11-04); D151 works around it.
