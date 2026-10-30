# MaskReg changelog

What changed in MaskReg, the release app, per version. The rules for the version number and for the run-file
numbers `FORMAT` and `ANALYSIS` are in [`docs/SPEC.md`](docs/SPEC.md), "Versions and run files" (D99); how to
release a version is in [`README.md`](README.md), "Releasing MaskReg".

Every change goes under **Unreleased** in the commit that makes it. Say when saved runs are affected:
- **Format N:** `FORMAT` went up; runs saved by older versions are analysed again when opened (kept as backups).
- **Analysis N:** `ANALYSIS` went up; runs saved by older versions are shown flagged as outdated, with an offer to
  analyse them again.

## Unreleased (will be 1.0.0)

First numbered version. Saved runs: format 3, analysis 9. Runs saved by earlier copies of MaskReg carry no version:
they open, flagged as outdated.

- *File → Folders...*: choose the folder of `maskreg.ini` and the folder where FTP downloads are kept (D96).
- *File → Open recent*: the last 10 folders (D97).
- Analysing a folder again keeps the run it replaces in `MaskReg_backups/` (the last 5); an interrupted save no
  longer leaves a damaged file (D98).
- Every run records the MaskReg version and analysis that made it, and opening a folder checks them (D99). A run
  saved by a newer MaskReg is never replaced by an older one without the user's restart.
- MaskReg comes as an installer, `MaskReg-setup-X.Y.Z.exe`: a program of its own, Python included, installed per
  user without admin rights, with a Start menu entry and an uninstall entry (D100). `MaskReg <folder>` opens that
  folder at start.
- Automatic updates: at every start MaskReg looks for a newer version on the update server and offers it; Yes installs
  it and restarts MaskReg, No asks again next time. *Help → Check for updates...* (D101).
- A run that fails for too few measured contacts now shows what it found on the Stitching tab: the tiles, their
  centres, the SEM images and design on request, and the tiles not measured with the reason (layer *Tiles not
  measured*; hover a tile). Nothing is saved (D102).
- **Analysis 3.** Fix: realignment (`realign`) no longer limits the offset to half the field. Across a single stripe
  that was half an image, so offsets of several µm could not be found, and a run could move every image to the limit
  (e.g. −166 µm). It now searches offsets of any size. On a repeating pattern, where many offsets fit equally and give
  the same errors (the design is ideal), it takes the one that pairs the most contacts and logs that the others fit as
  well. It stops only when no offset fits (D104, D105). Runs saved with an earlier analysis open flagged as outdated;
  analysed again, only those made with `realign` on give other numbers.
- The contrast check (`min_contrast`, 5 by default since D93) is now off by default (0): on noisy images it can drop
  every real contact (seen on noisy synthetic images). When turned on, the log warns (`!!`) if it removes every
  contact of a tile. Parameters saved before keep their value (D103).
- Fix: *Load design* showed a solid block of colour for `.oas` files drawn tone reversed (the area around the
  holes). It now shows their contacts, in the tone the analysis read each file. Runs saved before show it too (D106).
- Stitching tab: the failure markers can be told apart, also when drawn with the GPU, where dashed lines came out
  solid. Overlaps not matched are filled boxes, orange for the measured stitching and blue for the design; tiles not
  stitched have a thick red outline. Each has its own check box (D107).
- Stitching tab: tile names are dark text on white labels, readable over the SEM images (D108).
- Stitching tab: the status line says how serious a stitching problem is, worst first. *Error*: overlaps that
  look paired one contact row off, so the errors around them can be wrong by up to a pitch. *Warning*: tiles not
  stitched, whose contacts are missing from the Registration tab. Otherwise *Stitching OK*, with a note when
  overlaps were not matched (every tile is still stitched and measured), saying how many are at corners. The log
  says "STITCHING INCOMPLETE" only when tiles are not stitched. Each run keeps its check for overlaps paired one row
  off (D109).
- Stitching tab: click a tile to see which image it is: it alone is outlined and named, its SEM image comes to the
  front, and a panel beside the plot says in words what is wrong with it, with a *show why* link for each overlap
  not matched. With *Overlaps not matched* shown, click a box (or *show why*) to see only its two images and each
  one's contacts in it, filled and hollow, named in the legend, with the verdict in words. The legend lists only
  the layers shown (D112).
- Stitching tab: *Load contours* finds every contact's contour again, from the detection and after edge
  refinement, as the analysis did, and shows them as layers like the others: *Detected contours*, *Refined
  contours*, *Detected centres* and *Refined centres*, to compare the two methods that *stitch_refined* switches
  between. Contours are drawn for the contacts in view; zoom in if there are too many (D113).
- Registration tab: at start and on *View all* in any map, the maps show the full height of their contacts, centred,
  at the width the window gives them; also after a change of stripes or values and when the window is resized
  (unless you zoomed or panned: then the area is kept). *View all* in the dx or dy map no longer zooms out a little
  more each time, and the figure no longer runs past the right edge of a narrow window. The maps' titles are now
  *Δx (nm)* etc.; the colour bars say *measured − design* (D114, D117).
- Registration tab: each mean per row (dr, dx, dy) has its own range, in boxes just below it: type *from* and *to*
  (Enter applies) for a fixed range, kept when you change stripes, or tick *Auto*. A zoom with the mouse there shows
  its range in the boxes; *View all* there goes back to *Auto* (D115, D119).
- Registration tab: *Copy image* and *Save image* → *dx and dy: maps, profiles, colour bar*: the dx and dy maps
  with their means per row and per column and their colour bar, as on screen, without the dr graphs (D116).
- Registration tab: the means per column are hidden, and the maps take their height; tick *Mean per column* to
  show them (D118).
- Registration tab: a copy button (the two-sheets icon) over the top-left corner of every map and graph copies it
  to the clipboard, a map with its colour bar (D120).
- Registration tab: *Symbol size* sets the size of the contacts on the dx and dy maps in screen pixels (*Auto*: as
  before, a disk of 0.8 × the pitch that grows as you zoom in); *Line width* sets the width of the profiles' lines
  (D121).
- **Analysis 4.** Edge refinement now puts each contour point where the image rises half-way between the contact's
  interior and its bright band (*mid-level*), each contact with its own levels, instead of at the steepest rise. On
  noisy, low-contrast or low-resolution images the steepest rise jumped to noise point by point, so some refined
  contours did not follow the hole; on synthetic images the centres scatter 1.4–3.2× less. *Analysis → Parameters*,
  `refinement` *steepest gradient*, gives the previous results. Runs saved with an earlier analysis open flagged as
  outdated; analysed again, image folders give other numbers (contour CSV folders do not change). The *Refined
  contours* and *Refined centres* layers name the rule (D122).
- *Analysis → Parameters*, `moving_window_per_stripe`: the moving-window affine fitted per stripe, every image
  corrected by the windows of its own stripe (stripes are measured at different times and drift differently); a
  contact seen by two stripes gets the mean of both. The Registration tab's summary says how it was fitted. Off
  by default: runs give the same numbers as before (D123).
- **Analysis 5.** *Analysis → Parameters*: the check box `stitch_refined` is gone; `refinement` has a third choice,
  *none* (no refinement: the detected centres, as the unticked check box gave). Parameters and runs saved with it
  unticked read as *none*. With *none* the design tone is now chosen with the detected centres too (before: the
  refined ones), so such runs can give other numbers; refined runs do not change. The help texts of `detection`
  and `refinement` say what each choice does (D124).
- Registration tab: with *Values* → *Grid average* the means per row and per column follow the grid too: one point
  per row (column) of cells, the mean of the contacts in it, so the larger the cell, the smoother the lines. *Copy
  profiles* and *Export profiles CSV* give these values, headed *grid row* / *grid column* (D125).
- Registration tab: under each mean per row (dr, dx, dy), its range (max − min) three ways: *all* rows, its largest *ramp* (where the profile turns, wiggles smaller than *Ramp wiggles up to* ignored; *Auto* = 8 × the profile's noise) and *manual*, between two lines you drag on the graph. Tick one to draw its min and max as lines on the graph, with the rows they come from circled and the ramp's rows shaded. View only: saved runs are unaffected (D126, D127).
- *Analysis → Parameters* lists every option of the contact detection and the edge refinement: besides
  `detection`, `min_contrast`, `refinement` and `search_px`, now `dark_contacts`, `detection_blur_px`, `min_area_px`,
  `max_area_px`, `min_solidity` (band only), `edge_step_px`, `edge_blur_px` and `normal_span`. Their defaults are the
  values used so far, so runs give the same numbers (D128).
- *Analysis → Contour diagnosis...* (Ctrl+D): open one SEM image and try the contour settings on it. The contours
  and centres they find, from the detection and after refinement, are drawn on the image and found again as you
  change a setting; a line says how many contacts were found and why the other regions were left out; the
  threshold mask on request. *Apply to parameters* keeps the settings for the next run, *Cancel* leaves them (D129).
- For images without a bright band (e.g. wafer ADI), two new contour settings, both off by default so runs give the
  same numbers: `background_px` flattens the background before the contacts are found, so that one threshold fits an
  image whose brightness varies across it (charging, shading); `refinement` *threshold* places each contour where the
  smoothed image crosses the detection's threshold, sub-pixel. *Contour diagnosis* shows the flattened image (D130).
- Contour setting `upscale` (1 = off, the default, so runs give the same numbers): the image is interpolated to that
  many times as many pixels per side before the contacts are found in it, for contacts only a few pixels across; the
  contacts come back in the image's own pixels, and the other settings stay in pixels of the image as loaded. *Contour
  diagnosis* shows the upscaled image (D131).
- Contour setting `denoise` (none by default, so runs give the same numbers): removes pixel noise before the
  contacts are found, after the flattening and before the upscaling: *median* (window `median_px`), *non-local means*
  (strength `nlm_h`) or *total variation* (`tv_weight`). *Contour diagnosis* shows the denoised image and greys the
  options of the methods not chosen (D132).
- New stitching method *image by image* (`stitch_method`, with the new setting `overlap_tolerance_nm`, 2 nm): the
  images are placed one at a time, each where its contacts line up with those of all its placed neighbours at once
  (offsets up to `max_match_nm`; on a repeating pattern the whole-pitch shift nearest its stage position), then every
  overlap is paired there. For contact grids whose pitch is small against the offsets between neighbouring images
  (e.g. 50 nm): pairing every overlap at the stage position there gives each overlap its own whole-pitch error, which
  put images off by part of a pitch while the stitching said OK. The default method is unchanged, so runs give the
  same numbers (D133).
- Stitching *image by image*: on a repeating pattern each tile is now matched to its design the way it was placed, so
  a tile placed a whole pitch from its stage position no longer gets errors of about one pitch (D134).
- *Analysis → Parameters*: save the parameters under a name of your choice (*Save as...*) and load them again
  (*Load*, then OK), e.g. one set per dataset; *Delete* removes one. The dialog opens on the set in use. Each set
  holds the parameters of both image and contour CSV folders (D136).
- Stitching *image by image*: a check that can fail. After the stitching, every overlap is checked where its two
  images were placed: an overlap where fewer than half the contacts pair up, or whose images lie more than
  `overlap_tolerance_nm` apart, is an *Error* on the Stitching tab and listed in the log, which also gives the median
  and worst of both numbers. The check it replaces could never report anything, so badly stitched images showed as
  "Stitching OK". Numbers unchanged (D137).
- **Analysis 6.** Stitching *image by image* places each image where the most of its overlap contacts pair up, no
  longer where a smaller shift pairs only most of them: where the pattern does not fully repeat (e.g. missing
  contacts) images could be placed a whole pitch off at any `overlap_tolerance_nm`. Images whose overlaps cannot tell
  the shifts apart are placed last. The placement no longer uses `overlap_tolerance_nm`, so contact noise above it
  no longer leaves images unplaced. Runs stitched image by image open flagged as outdated; other runs give the same
  numbers (D138).
- **Analysis 7.** Stitching *image by image* pairs each overlap where the placement puts its images, within a quarter
  of the contact spacing, uses it only if at least half its contacts pair up there, and fits its shift robustly.
  `overlap_tolerance_nm` no longer pairs anything (only the check after the stitching uses it), so contact noise above
  it no longer leaves overlaps not matched or tiles not stitched. An overlap left out says on the Stitching tab how
  many of its contacts paired (D139).
- Stitching *image by image*: the check after the stitching also flags an overlap where another shift of one image
  pairs clearly more of its contacts than where it is, e.g. an image a whole pitch off where the pattern does not fully
  repeat (most contacts still pair there). Numbers unchanged (D140).
- New parameter `in_image_correction` (off by default, so runs give the same numbers), with `intrafield_nodes`: removes
  the distortion shared by all SEM images (the same at the same place in every image, e.g. of the scan) from the
  measured centres before the stitching, measured against the design (D53). For datasets whose images are distorted:
  the moving window cannot remove what repeats in every tile. The log gives the map's largest shift, and the
  Registration tab's summary says the distortion was removed. A run file of a corrected run keeps the map (optional
  arrays; files saved before open as before) (D141).
- Stitching tab: placement *Stitched + affine* shows the measured side (SEM images, centres, contours) also corrected
  by the moving-window affine, as the registration errors are, over the design at its stitched placement. Layer
  *SEM–design pairs* (off at start): an arrow from each contact's design centre to its measured centre, in this
  placement its registration error at true length; red where longer than half the contact spacing, so the measured
  contact was paired with a neighbouring design contact. Numbers unchanged (D142).
- **Analysis 8.** Image folders: the SEM–design pairs are chosen where the stitching put the images, each image
  following the pairs of its neighbours, against its own `.oas` contacts and its neighbours'. Before, each image was
  paired with its own design at its stage position; where the stitching moved images whole pitches from there (a
  repeating pattern), their errors stepped by whole pitches from image to image, and the moving-window map showed
  rainbow-like bands. An image the stitching put more than `design_tolerance_nm` off its neighbours is paired on its
  own as before, and the log names it (`!!`). Folders whose images sit where their own design pairs them give the same
  numbers (D143).
- New stitching method *fingerprint* (`stitch_method`; the default is unchanged, so runs give the same numbers): image
  by image, with each image's whole-pitch shift chosen by how well its contacts fit those of its neighbours. At the
  true shift each pair is one contact seen twice, so they differ only by the SEM noise; a pitch off, each pair is two
  neighbouring contacts, which differ also by the mask's own placement errors. For repeating arrays where image by
  image places images or whole regions whole pitches off: it works where the mask's placement errors are clearly
  larger than the noise; where they are not, it places as image by image. The log says how many images their fit
  decided and how clearly; the check after the stitching judges the overlaps the same way (D144).
- New parameters `in_image_model` and `in_image_order` (the default is unchanged, so runs give the same numbers): with
  `in_image_correction`, *formula* measures the distortion shared by all images as one smooth formula over the whole
  image (every power of x and y from 2 to `in_image_order`, 3 by default: the barrel, pincushion and spiral
  distortion of the SEM's deflection and lenses, and an S-shaped scan), fitted to all images at once, each with its
  own affine, instead of a value per node (*nodes*). Unlike a node, it is not thrown off by the few contacts near one
  corner (red tile corners with *nodes* on some datasets). When the paired contacts cannot determine it, the
  correction is left out and the log says so (`!!`). The log
  gives its order, the images used and its largest shift; run files are unchanged (D145).
- The in-image correction is always the formula: the node map and its parameters `intrafield_nodes` and
  `in_image_model` are removed. **Analysis 9:** a run made with `in_image_correction` on and the node map now gives
  the formula's numbers; runs without it give the same numbers. Saved runs and saved parameters that still name the
  removed parameters open as before (D146).
- Everything before (SPEC D77–D95): Stitching and Registration tabs, FTP browser and download, autosave, contact
  size tab.
