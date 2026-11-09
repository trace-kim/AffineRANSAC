# Line & space patterns in MaskReg: implementation plan

Written 2026-11-08 for a fresh session. The big idea only; the details are the implementer's.
Read first: `CLAUDE.md`, then in `docs/SPEC.md` the sections S2, S4, S5, S7 and the decisions D105,
D133, D138, D143, D144 (find them with `grep -n '^#' docs/SPEC.md`). The principles, with pictures, are in
`docs/notes/2026-11-08-line-space-method.html`; where that page and this plan differ (the order of the two
row tests), this plan wins.

## Goal

MaskReg measures registration on contact-hole masks: contact centres found per SEM image, images stitched
by their overlaps, centres paired with the design, a robust affine, errors reported. The next dataset is
lines and spaces, horizontal first, any orientation later. Today an image of lines yields nothing: every
band touches the image border and is dropped as a cut contact, and every later stage assumes a feature
with two coordinates.

## The idea

A line edge is a feature that knows its position across itself and nothing along itself. Treat it as such
everywhere.

- **Detect** edges with the side they face. Per edge: its height across, averaged along the whole edge
  (so sub-pixel), and its profile along, the edge roughness.
- **Stitch** image by image as today, in one dimension: candidates one pitch apart around the stage
  position. The row and the position along the lines are decided together by correlating the paired
  edges' roughness profiles over the along-line lag, summed over the paired edges: the roughness is the
  mask's own fingerprint and every real edge has it. No clear peak: the row comes from the height
  residuals of the lines seen twice (the D144 rule in 1-D) and the along position from the stage, said so
  in the log. Still tied: the smallest correction, logged.
- **Pair** measured edges with design edges of the same polarity after the stitching, as D143 does. Per
  line a centre error and a width error.
- **Fit and report** only what the edges can see; the other terms are "not measured", never zero.

Contact holes are the special case where edges face every way, so today's method stays untouched.
A row offset of the whole mosaic against the design is harmless (D105); a slip between neighbours is not,
it averages different lines' errors, which is why the row decision above exists.

## Steps, one per turn, each off by default until the user has checked it on real images

Status: step 1 built 2026-11-09 (spec D153), awaiting the user's check on real images.

1. **Line edges as a feature.** A pattern-type setting (contacts, the default | lines). For lines, keep
   the border-touching bands; per edge: polarity, height across (the existing mid-level refinement),
   extent along, profile along. Show them in the contour diagnosis window and print per image the
   roughness (rms of the profile) and the per-column noise, so the chances of step 2 are known before it
   is built. User checks: the edges on real images, the two numbers.
2. **Stitching across the lines.** Image by image in 1-D; `pipeline.placed_by_fingerprint` is the model
   (same order of placement, same tie rule, same log style). Deciders in the order above; the along
   position from the lag or from the stage. Shown on the Stitching tab; one log line per image saying how
   its row and its along position were decided. User checks: Stitching tab, log.
3. **Pairing, fit, report.** Design edges of the same polarity from the `.oas` rectangles, which also give
   the line direction; centre and width error per line; the RANSAC fit with one-direction observations and
   the unobservable terms reported as not measured; map arrows across only; the moving window and the
   in-image formula on the same observations. User checks: Registration tab against expectation.

## Facts to ask the user before step 1 (spec §13, question 14)

Pitch, line and space widths, line tone; pixel size, image size, overlap widths within and between
stripes; anything non-periodic in the measured area; whether the `.oas` clips hold rectangles cut at the
FOV; the largest stage error and whether consecutive images drift by more than half a pitch; the wanted
output per line (one across error per image, or a profile along the line).

## Rules that bind this work

- One step per turn: build, commit, push, stop, wait for the user's real-data result. Synthetic tests only
  for what the step needs; no ranking of methods on synthetic images.
- Nothing changes for contact datasets: the pattern-type setting defaults to contacts. `ANALYSIS` and
  `FORMAT` (D99) apply as soon as results or run files change.
- Spec update and Decision Log entry in the same commit as the code.
- Plain words in the UI and the log; UI messages at most two short sentences.

## Where the pieces live

`features/contact.py`, `features/edges.py` (detection, refinement); `pipeline.py`
(`placed_image_by_image`, `placed_by_fingerprint`, `placement_options`, `overlap_checks`); `matching.py`
(`candidate_shifts`); `registration.py` (`design_errors`); `realign.py` (D105); `release/analysis.py`
(the run and its settings, `stitch_method` among them); `release/contour_dialog.py` (the contour diagnosis
window); `release/views.py` (the tabs).
