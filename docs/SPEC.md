# AffineRANSAC — Software Specification

> **Audience:** human developers and AI coding agents working on this repository.
> **Status:** Design spec, v0.1 (2026-09-30). No code exists yet.
> **Rule for agents:** Treat this document as the source of truth for intent and architecture.
> If the code has to deviate from it, update this document in the same change and record the
> reason in §12 (Decision Log). Unresolved items live in §13 (Open Questions). Do not silently
> guess answers to those; ask the user.
>
> **This spec is the long-term target, not a build list.** It describes swappable interfaces
> (`Feature`, `OverlapRegistration`, `TileMetadataReader`, the RANSAC model type) and
> configurable options (tile models, gauge choice, sub-pixel methods, MSAC scoring, etc.).
> **Do not build these up front.** Start with the simplest concrete implementation: plain
> functions and a single hard-wired method. Introduce an interface or option only when a
> second real implementation is actually needed. Follow the development approach in
> `CLAUDE.md`: small steps, each tested and easy for a human to review.

---

## 1. Purpose

Measure the **registration error** of photomask patterns. Registration error is the positional
deviation of each real, as-fabricated pattern from where the design says it should be.

The program:

1. Reads the **design** from an OASIS (`.oas`) layout file.
2. Reads a set of **SEM images** (JPEG) covering a large area as overlapping tiles, plus
   per-image metadata (stage position and pixel size).
3. Extracts pattern coordinates from both sources.
4. **Stitches** the SEM tiles into one common coordinate space. Contacts that appear in the
   overlap of adjacent tiles serve as tie points.
5. Calibrates the SEM coordinates with **Affine transforms**, fitted robustly with **RANSAC**
   so that erroneous patterns do not corrupt the calibration.
6. Compares the calibrated SEM coordinates with the design coordinates and reports per-pattern
   registration error, plus summary statistics.

### 1.1 Scope

| In scope (v1) | Future (design for it, don't build it) |
|---|---|
| Isolated **contact holes**, one (x, y) point per feature | **Line & space** patterns, which give a 1-D position constraint per feature |
| JPEG SEM images with stage coords + pixel size | Other image formats (TIFF/PNG, 16-bit) |
| Pluggable metadata reader (format still unknown) | Vendor-specific metadata parsers |
| Per-tile affine + global affine | Higher-order distortion models (polynomial intrafield) |
| Overlap stitching from matched contact coordinates | **Pixel-to-pixel (intensity-based) affine** registration of the overlap regions |
| Batch/CLI processing | GUI, interactive review |

---

## 2. Glossary

| Term | Meaning |
|---|---|
| **Design / layout** | Intended pattern geometry from the `.oas` file. |
| **Tile** | One SEM image (one field of view, FOV). |
| **Overlap** | Region imaged by two adjacent tiles. |
| **Feature** | A measurable pattern instance. In v1, a contact hole. |
| **Feature point** | The (x, y) position representing a feature, e.g. a centroid. |
| **Correspondence** | A pair of feature points believed to be the same physical feature (SEM↔design, or tile A↔tile B). |
| **Inlier / outlier** | A correspondence consistent / inconsistent with the RANSAC model, within threshold τ. |
| **Tile affine** `T_i` | Affine that maps tile *i*'s nominal stage coordinates into the common stitched frame. |
| **Global affine** `G` | Affine that maps the stitched frame onto the design frame. |
| **Registration error** | Residual `r = G(T_i(p)) − d` for SEM point *p* matched to design point *d*. |
| **Gauge freedom** | A transform that can be applied to all tiles without changing any overlap agreement, which makes it unobservable from overlaps alone. |

---

## 3. Coordinate Systems and Conventions

Getting coordinate conventions wrong is the most likely source of silent bugs. **All
conversions must go through one module (`geometry/frames.py`) and be unit-tested.**

| Frame | Origin / axes | Units | Notes |
|---|---|---|---|
| **Pixel** `(x, y)` px | Centre of the top-left pixel, **y points down** | px | x = column, y = row. Pixel centres at integer coordinates. Named `x_px, y_px` where it could be confused with nm frames. |
| **Tile-local** `(x_t, y_t)` | Image centre, **y points up**. Same frame as the per-tile `.oas` (centred on (0, 0)) | nm | `x_t = (x_px − x_c)·s_x`, `y_t = −(y_px − y_c)·s_y`, where `(x_c, y_c)` is the image centre in px and `s` is the pixel size. |
| **Nominal stage** `(X, Y)` | SEM stage origin | nm | `(X, Y) = R_stage · (x_t, y_t) + (X_i, Y_i)`. `R_stage` is a configurable orientation (rotation/flip) between the image axes and the stage axes. |
| **Stitched** | Same as nominal stage after the tile affines are applied | nm | Output of stitching. |
| **Design** | OASIS coordinates, **y up** | nm internally | Converted from database units (DBU) on load. |

Rules:
- Internal unit is **nanometres, float64**, everywhere. Convert at I/O boundaries only.
- Points are stored as `numpy.ndarray` of shape `(N, 2)`.
- Affines are stored as 3×3 homogeneous matrices (last row `[0, 0, 1]`), applied to column
  vectors: `p' = A @ [x, y, 1]ᵀ`.
- **Image ↔ design orientation is known (user, 2026-10-01): no flip, no rotation.** Image
  right = design +x, image up = design +y. So `pixel_to_tile_nm` (y flip of the pixel frame
  only) maps SEM points straight into the `.oas` frame (D19).
- The image→stage orientation (`R_stage`, sign flips) and any stage→design offset are
  **configuration**, not hard-coded. They are unknown until the SEM metadata format is known.

---

## 4. Inputs

### 4.1 Design: OASIS files (one per SEM tile)
- **Each SEM image comes with its own small, pre-processed `.oas` file** covering that image.
  It is in **local coordinates centred on (0, 0) = the image centre**, i.e. the tile-local
  frame (§3). The global mask position of each file's (0, 0) is in the metadata CSV (§4.3).
- Folder layout: `<DATA_DIR>/*.jpg` + one metadata `.csv`, and `<DATA_DIR>/Contour/*.oas`.
- **Tone reversal:** some `.oas` files draw the area *around* the holes. For those, use
  `read_contacts_tone_reversed()` (holes = frame − drawn shapes, D17). Which tone a file uses is
  decided per tile by matching against the SEM (D28, `pipeline.choose_design_contacts`): normal
  tone if ≥ 50 % of the SEM contacts match a design contact (tile-local, **40 nm** gate: the
  SEM-to-design offset of a tile exceeds 25 nm, user), else tone reversed if that reaches 50 %,
  else the tile is **flagged** (`DesignContacts.ok = False`). No file property alone identifies
  reversed files (user). Normal-tone contacts whose bounding box touches the FOV frame are dropped
  (`inside_frame`): they may be cut by the frame, like border contacts in the SEM. OASIS cannot store holes, so a
  polygon with holes arrives as one outline with zero-width cut lines.
  We do *not* read one large full-mask layout. The workload is therefore **many small files**
  (about 1,000 tiles × a few hundred contacts). Per-file overhead matters, not large-file
  tricks (see D11).
- Library: **KLayout Python module** (`pip install klayout`, `import klayout.db`), which is the
  reference open-source OASIS implementation. All library use stays inside `io/design.py`, so
  it can be swapped for `gdstk` (Boost licence) if our code is ever shared (D11).
- Pipeline entry point: `read_contacts()` (centres + sizes). `read_polygons()` is for the
  viewer only, because extracting every vertex is ~6× slower.
- Config selects the **top cell** and **layer/datatype**.
- Hierarchy (cell references, arrays) is flattened.
- For contact holes: each polygon → one design feature point = **polygon centroid**. Also keep
  the bounding size (w, h) for classification and matching sanity checks.
- Output: a `DesignFeatures` table with `id, x, y, w, h, kind`.

### 4.2 SEM images: JPEG
- 8-bit grayscale (convert RGB to grayscale if needed). JPEG is **lossy**. Block artifacts can
  bias sub-pixel centroids slightly. Note this in reports. If the SEM can export PNG/TIFF,
  prefer that later.
- SEM images often have an **info/data bar** (scale bar, text). Crop it using a configurable
  crop rectangle, or auto-detect it.
- Loader: `io/sem_image.py::load_sem_image()` using **`opencv-python-headless`**. The regular
  `opencv-python` wheel bundles its own Qt, which clashes with PySide6. It returns a 2-D
  **uint8** array, exactly as stored in the file. Processing steps convert to float
  themselves (D13).
- Files are read with `np.fromfile` + `cv2.imdecode`, **not** `cv2.imread`, which silently
  returns `None` for non-ASCII paths on Windows.
- Viewer: `view_sem.py` (pyqtgraph). It draws pixel centres at integer (x, y) px, with y pointing
  down, matching §3.

### 4.3 SEM metadata
Needed per tile: **stage position (X_i, Y_i)** and **pixel size (s_x, s_y)**. Also useful if
available: FOV, magnification, scan rotation, timestamp, tile row/col.

Known (user, 2026-10-01): one CSV per data folder with per-image metadata, including the
**global mask position of each `.oas` centre**, **FOV = 2.88 µm** and **image size
2048 × 2048 px** → pixel size 1.40625 nm. The reader (`io/metadata.py::read_tile_index`) and the
format description are being produced by the remote agent: task **T001**
(`docs/tasks/T001-tile-index-reader.md`, D18). The module exists only on the remote machine.
Local code is built against the T001 interface, which `tests/test_metadata_contract.py` checks
on the remote.

The file format is **not yet known** (§13). Define an abstract interface:

```text
class TileMetadataReader(Protocol):
    def read(self, image_path) -> TileMetadata
TileMetadata: image_path, stage_x_nm, stage_y_nm, pixel_size_x_nm, pixel_size_y_nm,
              image_shape, (optional) scan_rotation_deg, row, col, extra: dict
```

v1 ships a **CSV manifest reader** (one row per image), so development and testing can
proceed before the vendor format is known. Vendor readers are added later as new
implementations.

### 4.4 Configuration
One YAML/TOML config file holds all tunables (layer, ROI, crop, detection parameters, RANSAC
thresholds, model choices, output paths). No magic numbers in code. Every run writes the
resolved config to its output folder for reproducibility.

---

## 5. Pipeline Overview

```text
 .oas ──► [S1 Design load] ──► design points D (design frame)
                                                    │
 .jpg + metadata                                    │
   │                                                │
   ├─► [S2 Feature extraction] ─► pixel points per tile
   ├─► [S3 Nominal placement]  ─► stage-frame points per tile
   │
   ├─► [S4 Tile↔tile matching in overlaps] ─► pairwise tie points
   │        (no RANSAC; few points per strip)
   ├─► [S5 Global stitching solve] ─► tile affines T_i  (gauge fixed)
   │
   ├─► [S6 SEM↔design matching] ─► correspondences (stitched ↔ design)
   ├─► [S7 Global RANSAC affine] ─► best 3-point model → inliers
   │        └─ least-squares refit on inliers ─► final global affine G
   │
   └─► [S8 Registration error] ─► residuals for ALL matched points
                                  (inliers and outliers), stats, plots
```

Each stage is a pure function (plain data in, plain data out) and can be run and tested in
isolation. Intermediate results can be dumped for debugging (§9).

---

## 6. Stage Details

### S1 — Design loading
See §4.1. Build a **KD-tree** (`scipy.spatial.cKDTree`) over the design points for fast
matching.

### S2 — SEM feature extraction (contact holes)
**Implemented (v1, D14):** `features/contact.py::detect_contacts()`. It applies a Gaussian
blur → Otsu threshold → connected regions → drops border-touching and too-small regions →
centre = pixel centroid of the region, plus its outer contour. On synthetic images it gives
≈0.03 px RMS centre error. Steps 3–4 below (better sub-pixel methods, quality metrics) are
future refinements.

**Edge refinement (D22, D23):** `features/edges.py::refine_edges()`. Every Otsu contour point
is moved along its own **outward normal** (tangent from neighbours ±3 points) within a **fixed
±`search_px`** (default 5 px) to the steepest dark→bright rise (bright→dark for bright
contacts) of the lightly smoothed image, with a sub-pixel parabola fit. No radius or centre is
assumed, so it works for any shape (circles, ellipses, rectangles). The refined centre is the
area centroid of the refined contour. Known limit: at sharply curved edges (curvature radius
of a few px) image blur pulls the steepest gradient slightly inward. Why: on
real data the `.oas` contours sit on the white band, while Otsu contours sit just outside the
dark area. Note: a uniform radial shift of a contour does not move its centroid, so
refinement matters for centres only where the edge bias is not the same all around.

1. Crop the data bar. Optionally denoise (Gaussian or median, configurable σ).
2. Detect candidates: threshold (Otsu or adaptive) + connected components, or
   blob/template matching. Contacts usually appear as dark discs with bright edges, but
   the polarity is configurable.
3. **Sub-pixel centre:** intensity-weighted centroid or ellipse fit to the edge contour. The
   method is configurable. This is the single most accuracy-critical step, so validate it on
   synthetic images with known centres.
4. Per-feature quality metrics: area, equivalent diameter, circularity, contrast, and
   distance to the image border.
5. Reject features that are **truncated by the image border** or fail the quality gates.
   Keep rejected features, flagged, for diagnostics.

Output per tile: `TileFeatures` with `id, x_px, y_px, quality…, flags`.

The feature abstraction must allow a future `LineEdgeFeature` (for line & space patterns),
which constrains only the position perpendicular to the line. Code in S4–S8 should depend on a
`Feature` interface, not assume "always a 2-D point". For v1, only the point implementation
exists.

### S3 — Nominal placement
Apply pixel → tile-local → nominal stage using the metadata and `R_stage` (§3). This is the
starting guess. Stage error is expected to be small relative to the pattern pitch (see the
matching constraint in S4/S6).

### S4 — Tile↔tile matching (overlap)
**Implemented (v1, D21):** `overlap.py`. `overlap_box()` gives the overlap rectangle of two
tiles from their centres and FOVs. `match_overlap(points_a, points_b, box, max_distance)` pairs the
contacts both tiles see inside that box (grown by `max_distance`), using `match_points`. Points
are SEM centres in mask nm at the **nominal** placement (tile-local + CSV centre). The pairs'
B − A differences are the input to S5. Contacts cut by either image's edge are already dropped
by detection, so a narrow strip may keep only one row of contacts.

**Acquisition geometry:** tiles are acquired in **stripes**. Within a stripe, adjacent tiles
overlap in a thin strip at the top/bottom of the image. Once multiple stripes are measured,
adjacent stripes overlap in a thin strip at the left/right of the image. Each overlap strip
therefore contains only **a handful of contacts**.

**RANSAC is NOT used for stitching** (decision D8). With so few points per strip, a robust
sampling approach is meaningless. All matched tie points go straight into the least-squares
solve (S5). Bad tie points are prevented at the feature level instead: S2 quality gates,
especially rejecting contacts truncated by the image border. They are also *flagged* (not
removed) via their residuals after S5.

Overlap registration is a **pluggable strategy** (`OverlapRegistration` interface), which
yields tie points / pairwise constraints between tiles *i* and *j*:

- **v1: `FeatureOverlap`.** Uses contact coordinates.
  1. Collect the features of each tile that lie inside the overlap region (with a margin).
  2. **Coarse offset:** by default, nearest-neighbour matching in stage coordinates.
     Optionally refine it with image cross-correlation (phase correlation) of the overlap
     crops.
  3. **Correspondences:** mutual nearest neighbours within a gate radius `g`. These become the
     tie points for S5.
- **Future: `PixelOverlap`.** A pixel-to-pixel (intensity-based) affine registration of the
  two overlap crops, e.g. ECC maximisation (`cv2.findTransformECC` with `MOTION_AFFINE`) or a
  similar method. It uses the full image content rather than a few centroids, which matters
  when a strip contains too few contacts, or for patterns such as line & space. Its output (a
  pairwise affine plus uncertainty) is converted into equivalent constraints for S5, e.g.
  virtual tie points sampled over the overlap region.

S5 must consume the strategy's output without knowing which strategy produced it.

⚠ **Periodicity / aliasing:** contact arrays are periodic with pitch `P`. If the nominal
misplacement exceeds `P/2`, nearest-neighbour matching locks onto the wrong neighbour. All
matches are then off by exactly one pitch, and RANSAC cannot detect it because they are
mutually consistent. Requirements:
- The gate radius `g < P/2`.
- Stage error must be `< P/2`, **or** the coarse offset must come from something
  non-periodic: image correlation over an aperiodic region, a unique mark, or a pattern
  boundary.
- Log a warning when the matched fraction is low or the offset is close to `P/2`.
- *Deferred idea (user, 2026-10-02):* make matching more robust by ranking several candidate
  matchings (offsets) by matching score and choosing, among the good ones, the one with the
  smallest translation. Not urgent: input images are controlled so offsets stay within 25 nm.

### S5 — Global stitching solve (per-tile affines)
**Pairwise analysis before the global solve (D24):** before choosing the tile model, each
overlapping pair is analysed (`notebooks/overlap_flow.ipynb`):
- `overlap_fit.fit_overlap(a, b, rotation=False|True)`: a robust least-squares B-relative-to-A
  shift (± rotation, no scale) from the matched overlap contacts. Outliers are residuals above
  `outlier_factor` × the median residual, re-estimated over a few iterations from a
  median-difference start. RANSAC is not used (too few points per strip, D8). The inlier RMS is
  the measurement noise floor.
- `overlap_image.image_overlap_shift(...)`: an independent B − A shift from the overlap
  **pixels** (upsampled cross-correlation, scikit-image). Precision on real-like strips is about
  0.1–0.2 px, so it is a check on the contact pipeline, not a replacement.
- `overlap.overlapping_pairs(centers, fovs)`: all overlapping tile pairs.
Decide from these results whether the per-tile model needs rotation (only if it lowers the RMS
clearly and consistently), and whether residuals show a repeating pattern (→ shared intrafield
distortion).

**Global solve v1 (D25):** `stitching.solve_tile_shifts(n_tiles, pairs, shifts, weights)`, one
**translation** correction per tile from all pairwise step-1 shifts (t_i − t_j = d_ij, weighted
least squares, gauge: corrections average zero). `mosaic.build_mosaic()` stitches tile images at
sub-pixel positions (overlaps averaged), to show the result before/after correction. Rotation
or affine tile models are added only if the pairwise analysis shows they are needed.

**Pipeline v1 (D26):** `pipeline.py` composes the library functions, nothing else:
`process_tile()` (S2: load, Otsu, edge refinement; design contacts from the tile's `.oas`) and
`stitch_tiles()` (S4 + S5: `overlapping_pairs` → `match_overlap` → `fit_overlap` per pair →
`solve_tile_shifts`, weighted by inlier count). Tile data comes in as plain arguments, so it
does not depend on the remote-only metadata reader. `view_stitch.StitchViewer` (pyqtgraph, OpenGL
viewport) shows the result for any tile selection: SEM images, design contours/centres, Otsu and
refined contours/centres, overlap boxes and overlap outliers, at nominal or stitched placement,
with layers and single tiles hideable (`notebooks/stitch_viewer.ipynb`).

**Stitching failures are never silent (D27).** Every overlapping pair is solved together (no
stripe-by-stripe order): a tile in stripe 2 is constrained by its top/bottom and left/right
neighbours at once. The overlap match gate defaults to **25 nm** (largest B − A stage offset,
user); it must stay below P/2. `stitch_tiles` keeps every pair it does not use
(`StitchResult.rejected`): *failed* = both tiles have ≥ `min_matched` contacts in the overlap
but they do not pair up; otherwise *skipped* (too few contacts, e.g. corner-only).
`solve_tile_shifts` solves only the largest group of tiles connected through used pairs; every
other tile gets a **NaN** correction (`StitchResult.unplaced`), and `stitch_tiles` warns. The
viewer marks failed overlaps and unstitched tiles in magenta and never moves the latter. The
gauge stays mean-zero over the stitched group (D6); `stitching.fix_tile()` re-expresses the
solution with one tile fixed (viewer: "first tile fixed"), e.g. for later comparison with design
centres.

**Design stitching (D30).** The per-tile `.oas` files can be offset against each other (the same
contact drawn at different mask positions in neighbouring files; user, real data). They are
stitched exactly like the SEM: `pipeline.stitch_design()` runs `stitch_tiles` on the design
centres (translation per tile, no rotation; flagged tiles left out → NaN, reported).
`stitching.placement_corrections()` gives, per placement ("nominal", "mean", "first"), the
(SEM, design) corrections; "first" fixes the first tile stitched in both.

Unknowns: one affine `T_i` per tile (6 parameters each). Observations: every tie point
`(p ∈ tile i, q ∈ tile j)` contributes `T_i(p) − T_j(q) = 0` (2 equations).

- Solve as one **sparse linear least-squares** problem (`scipy.sparse.linalg.lsqr`). Affines
  are linear in their parameters.
- **Gauge fixing (mandatory):** overlaps cannot observe a transform applied to *all* tiles.
  Fix it by constraining the **mean of the tile affines to equal identity**, i.e. the nominal
  calibration. Whatever common error remains is absorbed later by the global affine `G` (S7).
  Alternative: pin one reference tile. This is simpler, but it biases the solution towards
  that tile. Config option `gauge: mean | reference_tile`.
- Weighting: weight each observation by its feature-quality weight (optional).
- After solving, compute the per-tie residuals. Report them, and **flag** ties above a
  threshold as diagnostics. Do not remove them automatically (no RANSAC / outlier rejection in
  stitching, D8).

⚠ **Conditioning:** overlap strips are narrow, so a tile's scale and skew are weakly
determined by points confined to its edges. Errors can accumulate along long chains of
tiles. Mitigations, all configurable:
- `tile_model: affine | similarity | rigid | translation`. The default is `affine`, as
  requested. Check the rank and condition number and warn or fall back when a tile is
  under-constrained.
- `tile_model: shared_intrafield`. A single affine distortion common to all tiles (same SEM,
  same magnification) plus per-tile translation (and rotation). This is physically plausible
  and much better conditioned. Evaluate it against the per-tile affine using synthetic data
  (§10).
- Optional weak regularisation pulling each `T_i` towards identity.

Output: `T_i` for each tile, per-tile diagnostics (number of ties, RMS tie residual, condition
number) and the stitched coordinates of every SEM feature. When a feature appears in multiple
tiles, keep **all** observations. For comparison with the design, use their mean, and report
the spread as a stitching-quality metric.

### S6 — SEM↔design matching
**Implemented (v1, D20):** `matching.py::match_points(a, b, max_distance)` gives one-to-one
pairs of mutual nearest neighbours closer than `max_distance`. Only contacts found in **both**
the design and the SEM are used. Edge contacts that SEM detection drops (cut off by the image
border) are left out. v1 works per tile, by brute-force distances (fine for hundreds of points);
use a KD-tree once whole masks are matched at once.

1. Coarse alignment: stitched frame ≈ design frame, plus a configurable offset/orientation
   (from config or from an initial correlation). The same periodicity caveat as in S4 applies.
2. For each stitched SEM point, find the nearest design point (KD-tree). Accept it if the
   distance is `< g_design` (`< P/2`). Enforce **one-to-one** matching (mutual nearest, or the
   Hungarian algorithm for conflicts).
3. Unmatched SEM points: extra/spurious features. Unmatched design points inside the imaged
   area: missing features. Both are reported, never silently dropped.

### S7 — Global RANSAC affine (the core algorithm)
**Implemented (v1, D32):** `fitting/ransac.py` on the **merged** contacts (`MergedErrors.sem_nm`
→ `design_nm`), τ = 0.5 nm to start (user). `ransac_affine_steps()` yields every search
iteration and refit (sample, model, inliers, best so far, adaptive N) for the live monitor
`view_ransac.RansacMonitor`; `ransac_affine()` runs it to the end without UI and returns the
model, reference point (design centroid), inliers, residuals of **all** contacts, iteration
count and best sample. Affine maths in `geometry/affine.py` (fit, apply, §8.2 decomposition).
Degenerate samples: triangle area < 10⁻³ × mean squared distance from the reference point.
Precision note: with ~0.2 nm centre noise the affine terms are only good to
≈ noise / (field std · √N), e.g. ~100 ppm for one 0.7 µm synthetic tile, ~0.1 ppm for a
35 µm field with 40 000 contacts.

This implements the approach specified by the user:

1. Repeatedly draw **3 correspondences** at random (the minimal sample for a 2-D affine: 6
   unknowns, 6 equations).
2. Compute the **exact affine** mapping the 3 SEM points onto their 3 design points.
3. Apply it to **all** SEM points and count the **inliers**: correspondences whose residual is
   `< τ` (the RANSAC threshold, in nm).
4. Keep the 3-point model with the **most inliers**. Ties are broken by the lower sum of inlier
   residuals.
5. **Refit:** least-squares affine using **all inliers** of the best model → final global
   affine `G`. Optionally iterate: recompute the inliers with `G`, refit, and repeat until the
   inlier set is stable (max `k` iterations).
6. Apply `G` to **all** stitched SEM coordinates.

Details are in §7.

**Interpretation of inliers and outliers (user's intent):**
- **Inliers** are assumed to be correctly patterned features. The affine fitted to them, `G`,
  therefore estimates the SEM **measurement error** (calibration, stage, distortion) and
  nothing else.
- `G` is applied to **all** SEM points, inliers and outliers alike, correcting the measurement
  error everywhere.
- **Outliers** then show large residuals against the design, and those residuals are the
  indication of **mask defects / local registration errors**. Outliers are excluded from
  *fitting* `G` only. They must never be dropped from the transformed output or the report.

RANSAC is used **only here** (SEM↔design), not in stitching (D8).

### S8 — Registration error and reporting
**Final registration error (v1, D34, D35):** `RansacResult.residuals`, i.e. G(SEM) − design for
**every** merged contact (inliers and outliers). `view_registration.RegistrationView` shows two
error maps (dx, dy): every contact a dot coloured by its error (jet), **not binned**, rasterized to
the screen (one image pixel per device pixel, redrawn on zoom, pan and resize; contacts on the
same pixel are averaged when zoomed out; disks of ~0.4 × pitch when zoomed in). Below, the mean
dx and dy **per row of contacts** against the row's y: `registration.group_rows()` groups contacts
by design y (a new row where the sorted y jumps by more than `gap_nm`, default 10 nm) and
`registration.row_means()` averages each row. No inlier/outlier distinction in these views. The
RANSAC monitor also draws the residual after the current model on its dx/dy plots (crosses),
next to the raw error (dots).

**Implemented first (v1, D29): raw error, no affine removed.** `registration.design_errors()`
matches each tile's SEM centres to that tile's design centres, each placed with its own
corrections (one placement of `placement_corrections`; `match_points`, 40 nm gate), and returns
`DesignErrors` (tile, design, SEM, error per contact). Tiles that are not measured (SEM or design
not stitched, design tone flagged, nothing matched) are listed in `skipped` with the reason, plus
a warning; no error is computed for them. `error_summary()` gives count, mean, 3σ and max. Overlap
contacts get one entry per tile (all observations kept, S5). The sign is set only in
`registration_error()`: SEM − design, tentative (§13-8). The viewer shows it as an error map
beside the SEM view with shared zoom, one error set per placement (switching the placement
switches the arrows).

**Merging overlap observations (D31).** `design_errors` gives one *observation* per contact per
tile, so a contact in an overlap has 2–4. `registration.merge_observations(errors,
design_corrections, radius_nm=5)` turns them into one entry per physical contact
(`MergedErrors`): observations from **different tiles** whose **stitched design** positions
(nominal design + `stitch_design` correction, the same for every placement) are within 5 nm are
one contact (KD-tree pairs → `connected_groups`). Its SEM and design positions are averaged; the
**spread** (largest distance between its SEM observations), the count and the member observations
are kept. SEM positions are never used for grouping (they hold the error being measured). A
group with two observations of one tile is warned about. RANSAC (S7) uses the merged contacts.
The viewer draws one line per merged contact; merged overlap contacts can be ringed and a
magenta ring flags a spread above a display threshold (notebook `SPREAD_FLAG_NM`, default 2 nm). The RANSAC affine (S7) and residuals after `G` follow.

**Moving-window affine (D36), for comparison with the global RANSAC affine.**
`fitting/moving_window.moving_window_affine(sem, design, window_nm, step_nm)`: windows of height
`window_nm` slide along the design y in steps of `step_nm`, from the first window with its lower
edge at the lowest contact to the last with its upper edge at the highest. Each window gets a plain
least-squares affine on all its contacts (no RANSAC, no outlier rejection), relative to its design
centroid. Each contact's residual `G_w(SEM) − design` uses the window whose centre is nearest to its
design y (windows overlap for the fits; every contact gets one residual). The notebook shows it
beside the RANSAC result (summaries, per-row means, per-window terms, two registration views).
`view_moving_window.MovingWindowTuner` (D37) sets the window and step with sliders and recomputes
live: the registration view (`RegistrationView.set_errors`, colour scale kept) with the RANSAC row
means dashed, and one affine term per window against its centre y, beside the RANSAC affine
recentred (`geometry.affine.recentre`) on each window's reference point.

- For every matched pair: `r = G(stitched SEM point) − design point` → `(dx, dy, |r|)`, plus
  the inlier/outlier flag, tile id(s) and quality metrics.
- **Reported registration error = residuals after the full affine is removed** (decision D4).
  The fitted affine terms are reported separately (§8.2), because linear mask errors (scale,
  rotation, orthogonality) are indistinguishable from SEM calibration errors and are absorbed
  into `G`.
- Summary stats (per axis and for inliers / all points): mean, σ, **3σ**, max |r|, count.
  Also per-tile stats.
- Viewers (interactive, **pyqtgraph**, D12): residual **vector map** (arrows scaled up),
  histograms of dx/dy, tile layout with tie quality, and the outlier locations.
- Outputs: CSV/Parquet of per-feature results, JSON summary (affine terms, stats, config,
  versions). Static images, if needed, are exported from the pyqtgraph views. The output format
  may change (§13).

---

## 7. RANSAC Specification

Module: `fitting/ransac.py`. It is used by S7 only. Keep it generic over the model type
(affine now; possibly others later) so that it can be tested independently.

| Parameter | Meaning | Default guidance |
|---|---|---|
| `threshold_nm` (τ) | Inlier residual cutoff | Roughly 3× the expected measurement noise (from synthetic tests / repeatability). Configurable. |
| `confidence` (p) | Target probability of drawing ≥1 all-inlier sample | 0.999 |
| `max_iters` | Hard cap | 10 000 |
| `min_inlier_ratio` | Fail/warn below this | configurable, e.g. 0.5 |
| `seed` | RNG seed | **Always set.** Runs must be reproducible. |
| `refine_iters` | Inlier-refit loops after the best model | 3 |

- **Adaptive iterations:** `N = log(1 − p) / log(1 − w^s)`, where `w` is the current best
  inlier ratio and `s = 3` is the sample size. Update `N` whenever a better model is found.
  *Derivation* (Fischler & Bolles 1981; Hartley & Zisserman, *Multiple View Geometry*, 2nd ed.,
  §4.7.1): a random sample is all inliers with probability `w^s`; `N` samples all fail with
  probability `(1 − w^s)^N`; requiring this to be `≤ 1 − p` gives `N ≥ log(1 − p) / log(1 − w^s)`.
  One all-inlier sample suffices because the final model comes from the least-squares refit on
  its inliers. `w` is estimated by the best model so far, which can only underestimate the true
  ratio, so `N` is conservative. The search stops when the iteration count reaches `N` or
  `max_iters` (degenerate samples count as iterations); then up to `refine_iters` refits, stopping
  early when the inlier set no longer changes. Examples (p = 0.999, s = 3): w = 0.9 → 6,
  0.8 → 10, 0.5 → 52, 0.3 → 253, 0.1 → 6905.
- **Degenerate samples:** reject triplets that are (near-)collinear or too close together, i.e.
  triangle area `< min_area` (relative to the point spread). They produce ill-conditioned
  affines. Prefer samples spread over the field. Optionally use stratified sampling across
  tiles/regions.
- **Exact solve:** a 6×6 linear system (or `cv2.getAffineTransform`). **Least-squares refit:**
  `numpy.linalg.lstsq` on the inlier set. Optionally apply Hartley-style normalisation
  (centre and scale the points) before solving, for numerical stability at mask-scale
  coordinates (µm–mm offsets with nm residuals).
- Returns: the model, the inlier mask, the residuals, the iteration count, and the best
  3-point sample indices (for traceability).
- The `pure 3-point inlier-count` scoring is the specified default. MSAC (truncated-quadratic
  scoring) may be offered as an option, but must not replace the default without a
  Decision Log entry.

---

## 8. Affine Model

### 8.1 Form
```
[x']   [a  b  tx] [x]
[y'] = [c  d  ty] [y]
[1 ]   [0  0  1 ] [1]
```

### 8.2 Decomposition for reporting (small-angle, mask-metrology style)
- Translation: `Tx = tx`, `Ty = ty` (nm)
- Magnification: `Mx = a − 1`, `My = d − 1` (report in ppm)
- Rotation terms: `Rx = c`, `Ry = −b` (µrad)
- **Rotation** `θ = (Rx + Ry)/2`, **Orthogonality / skew** `ω = Rx − Ry` (µrad)
- **Display (D33):** `geometry.affine.report_terms()` shows the correction G with translation in
  nm, magnification in ppm (= nm per mm), rotation and orthogonality in **degrees** (user), plus,
  per term, how far that term alone moves the furthest contact (nm at the field edge: |M|·x_max,
  |θ|·r_max, |ω|/2·r_max). `decompose()` keeps µrad internally.

Also provide an exact decomposition (polar/QR) for large transforms. The sign conventions above
must be tested with synthetic transforms of known rotation and scale.

Apply transforms relative to a **reference point** (e.g. the centroid of the design points in
the ROI), so that translation is not coupled to rotation/magnification about a distant origin.

---

## 9. Proposed Code Layout (Python)

```text
AffineRANSAC/
├─ CLAUDE.md                 # agent entry point → points here
├─ docs/SPEC.md              # this file
├─ pyproject.toml
├─ config/example.yaml
├─ src/affine_ransac/
│  ├─ io/
│  │  ├─ design.py           # OASIS → DesignFeatures (wraps klayout)
│  │  ├─ sem_image.py        # JPEG load, crop, grayscale
│  │  └─ metadata.py         # TileMetadataReader protocol + CSV manifest impl
│  ├─ features/
│  │  ├─ base.py             # Feature interface (point now, line-edge later)
│  │  └─ contact.py          # contact-hole detection + sub-pixel centroid
│  ├─ geometry/
│  │  ├─ frames.py           # pixel↔tile↔stage↔design conversions
│  │  └─ affine.py           # fit (exact/LSQ), apply, decompose, normalise
│  ├─ fitting/
│  │  ├─ ransac.py           # generic RANSAC
│  │  └─ moving_window.py    # moving-window affine along y (D36)
│  ├─ matching.py            # KD-tree NN, mutual/one-to-one, gating
│  ├─ stitching/
│  │  ├─ overlap.py          # OverlapRegistration interface; FeatureOverlap (v1), PixelOverlap (future)
│  │  └─ solve.py            # global sparse LSQ solve for tile affines
│  ├─ registration.py        # S6–S8 orchestration, residuals, stats
│  ├─ report.py              # CSV/JSON/plots
│  ├─ pipeline.py            # end-to-end run from config
│  └─ cli.py                 # `affine-ransac run config.yaml`
└─ tests/
   ├─ synthetic/             # generators for layouts, SEM tiles, distortions
   └─ test_*.py
```

Stack: Python ≥ 3.11, numpy, scipy, opencv-python (or scikit-image), klayout, pyyaml,
pyqtgraph + PySide6 (all viewers), pandas; pytest for tests. Use type hints and dataclasses
for data records.

---

## 10. Validation Strategy (synthetic-first)

Real data and the metadata format are not available yet, so **build a synthetic data
generator early** and use it for all tests:

1. Generate a contact array layout (with configurable pitch, size and ROI). Write it as `.oas`
   via klayout, so the reader is tested end-to-end.
2. Inject **known mask registration errors**: random per-contact displacement, plus some
   large-error "defect" contacts, plus an optional linear mask error.
3. Render SEM tiles: a grid with overlap, per-tile **known affine distortion**, stage noise,
   blur, shot noise, edge contrast and JPEG compression. Include missing/extra features and
   truncated border features.
4. Write the metadata manifest.
5. Assertions:
   - Recovered tile affines ≈ injected ones, modulo gauge.
   - After the full pipeline, residuals ≈ injected per-contact errors, minus their affine
     component.
   - Injected defects are flagged as outliers but still reported.
   - RANSAC is reproducible with a fixed seed.
   - Periodicity trap: a stage error `> P/2` without an aperiodic coarse alignment triggers a
     warning or failure, never a silent wrong answer.

Unit tests are also required for: frame conversions (axis flips!), exact 3-point affine,
LSQ refit, decomposition signs, degenerate-sample rejection, and one-to-one matching.

---

## 11. Non-Functional Requirements
- **Accuracy target:** TBD (§13). The target determines the sub-pixel method and noise model.
- **Scale:** it must handle hundreds to thousands of tiles and 10⁵–10⁶ contacts. Use vectorised
  numpy, KD-trees and a sparse stitching solve. Don't use Python loops over all points.
- **Reproducibility:** seeded RNG. The resolved config, package versions and input file hashes
  are written to the output.
- **Traceability:** every reported point can be traced back to its image file and pixel
  location.
- **Platform:** developed on Windows. Use `pathlib` and avoid shell-specific assumptions.

---

## 12. Decision Log

| # | Date | Decision | Rationale |
|---|---|---|---|
| D1 | 2026-09-30 | v1 supports isolated contact holes only; the feature abstraction must allow line & space later. | User requirement. |
| D2 | 2026-09-30 | SEM images are JPEG; metadata gives stage coords + pixel size, format unknown → pluggable reader, CSV manifest first. | User answer. |
| D3 | 2026-09-30 | Transform structure: **per-tile affine** (stitching via overlaps) **+ one global affine** to design. | User choice. |
| D4 | 2026-09-30 | Reported registration error = **residuals after full affine removal**. Affine terms are reported separately. | User choice. Linear mask errors are not separable from SEM calibration errors. |
| D5 | 2026-09-30 | Global calibration uses RANSAC with a 3-point minimal sample and max inlier count, followed by an LSQ refit on the inliers. Outliers are excluded from fitting but still reported. | User-specified algorithm. |
| D6 | 2026-09-30 | Stitching gauge fixed by mean tile affine = identity (default). | Overlaps can't observe a common transform; `G` absorbs it. |
| D7 | 2026-09-30 | Internal units nm, float64; design frame y-up; affines are 3×3 homogeneous matrices. | Consistency. |
| D8 | 2026-09-30 | **No RANSAC in stitching.** Overlap tie points go directly into the LSQ solve. Bad ties are prevented by feature quality gates and flagged via residuals, not removed. | User decision. Overlap strips (top/bottom within a stripe, left/right between stripes) contain only a handful of contacts. |
| D9 | 2026-09-30 | Overlap registration is a pluggable strategy: v1 feature-based (contact coordinates); future pixel-to-pixel affine. | User requirement. Some patterns may need intensity-based overlap matching. |
| D10 | 2026-09-30 | Interfaces and options in this spec are the long-term target. Build the simplest concrete version first, and add an interface or option only when a second real implementation is needed. | User rule: never overengineer; small, reviewable steps. |
| D11 | 2026-09-30 | Keep **KLayout** for OASIS reading. Extract centres without per-vertex Python work (bounding-box/centroid per shape). | Benchmark with 1,000 small files × 400 contacts: KLayout ≈ gdstk ≈ 1.2 ms/file, so speed is not a differentiator, and reading ≪ SEM image processing. KLayout has the strongest OASIS compliance. GPL does not affect the user (sole user) or the vendor, who implements from the spec, not our code. Hand over the spec + test data, not code. The per-vertex reader was 10 ms/file and must be replaced. |
| D12 | 2026-09-30 | **All viewers are interactive, built with pyqtgraph** on PySide6. No matplotlib. | User requirement. PySide6 (LGPL) chosen over PyQt (GPL). |
| D13 | 2026-10-01 | SEM images are loaded as **uint8** (not float32, as first drafted). | Keeps the exact file values and uses 4× less memory across ~1,000 tiles. Conversion to float belongs in the processing step that needs it. |
| D14 | 2026-10-01 | v1 contact detection = blur + **Otsu** threshold + connected regions; centre = region pixel centroid. Regions touching the border or below `min_area_px` are dropped. The data bar is removed by cropping bottom rows (`crop_databar`), which leaves pixel (x, y) unchanged. | User request: start with a simple Otsu-based method. Synthetic test: 176/176 found, 0.03 px RMS. Without cropping, letter interiors in the data-bar text can be detected as contacts. |
| D15 | 2026-10-01 | matplotlib is allowed in **notebooks** for quick plots (in the `dev` extra). Package viewers stay pyqtgraph. | User request. Narrows D12. |
| D16 | 2026-10-01 | SEM contact centres are converted to the **tile-local frame** (origin = image centre, y up, nm) with pixel size = FOV / image width. That is the frame of the (0,0)-centred `.oas`, so the two compare directly. | `.oas` files are local and centred (user). |
| D17 | 2026-10-01 | Tone-reversed `.oas`: contacts = connected empty regions of (frame − drawn shapes), via a KLayout Region boolean. Regions touching the frame are dropped by default. A separate function, `read_contacts_tone_reversed()`. | Some files are tone reversed (user). The boolean handles OASIS keyhole polygons. Automatic detection comes later. |
| D18 | 2026-10-01 | Data-specific work is done by a **remote agent** (opencode, unknown model) with data access. It is **one-way**: task files in `docs/tasks/` go out by `git pull`; the remote agent never commits; findings come back only via the user. Remote-only modules are used through exact interfaces guarded by contract tests (protocol in `CLAUDE.md`, rules for the remote agent in `AGENTS.md`). | The real data and the remote agent's work cannot be sent to the local agent. |
| D19 | 2026-10-01 | SEM image and `.oas` share the same orientation: no flip or rotation between them. | Stated by the user, who knows the data. Replaces the orientation analysis requested in T001 §5e; its result isn't needed. |
| D20 | 2026-10-01 | Only design↔SEM pairs (mutual nearest, within a gate) are used downstream. Unpaired contacts are excluded, e.g. design contacts whose SEM image is cut off at the image edge. | User: "coordinates that only have a corresponding point in both design and image must be detected". Seen on real data: left-edge contacts were in the design but not detected in SEM. |
| D21 | 2026-10-01 | Overlap matching v1: contacts in the nominal overlap box (grown by the match gate) are paired by mutual nearest neighbour. Nominal placement = CSV tile centre. Flat module `overlap.py` (not a `stitching/` package) until more stitching code exists. | User: build overlap matching between two tiles. Synthetic check: recovers a built-in (−6, +4) nm stage difference to ±0.05 nm. |
| D22 | 2026-10-01 | SEM contours are refined after Otsu to the **maximum intensity gradient** along rays (steepest rise for dark contacts). `refine_edges()` is a separate step, so Otsu and refined results can be compared. | User request: Otsu contours sit just outside the dark area, while the `.oas` contours lie on the white band. Whether the steepest rise is the right physical edge is to be judged on real data (contour check in `tile_index.ipynb`). |
| D23 | 2026-10-01 | Refinement searches along each Otsu contour point's **outward normal** within a **fixed pixel distance** (`search_px`), replacing radial rays from the centre with a window of ±50 % of the circle-equivalent radius. | User: radius-based search misbehaves; contacts are elliptical and other shapes will follow; the pixel size is known, so a fixed pixel window is better. Synthetic 24×10 px ellipse: centre error 0.014 px, mean edge distance 0.2 px. |
| D24 | 2026-10-02 | Pairwise overlap analysis in four steps: (1) robust translation, (2) image cross-correlation check, (3) robust translation + rotation (no scale), (4) residual map across pairs. Outliers come from a median-residual threshold, not RANSAC. Image registration uses scikit-image `phase_cross_correlation` (normalization=None, Hann window, upsample 100); `cv2.phaseCorrelate` and the "phase" normalisation were rejected (sub-pixel errors up to 0.38 / 0.6 px). | User: overlap arrows mostly agree but some differ (noise vs outliers); compare translation-only, robust rigid and image-based registration before choosing the stitching model. |
| D25 | 2026-10-02 | First global stitching solve is **translation per tile** (gauge: mean correction 0), plus a mosaic builder for before/after views. `overlap_flow.ipynb` shows every step as diagnostic plots: before/after arrows at separate raw/residual magnifications with 1 nm key arrows, difference scatter with the outlier threshold, an image overlay before/after registration, a rotation trend plot, all-pairs comparison plots, stitched residual maps, and a mosaic with zoom. | User: results must be checkable visually, not just as numbers; a stitched result before/after correction is needed. Translation first, because the rotation model is not yet justified by data. |
| D26 | 2026-10-02 | `pipeline.py` (S2, S4+S5 composed only from library functions) and one interactive stitch viewer for all chosen tiles. Placement is a Nominal/Stitched switch (SEM items move, design stays) rather than two copies of every layer. Each tile is its own image item (no single mosaic image), so ~1,000 tiles fit in memory. Drawing is relative to a local origin in whole µm, because the OpenGL viewport works in float32 (mask coordinates ~10⁷ nm would round to several nm). Opaque tiles: the later tile covers the earlier one in an overlap; hide a tile to see the other. | User: one viewer showing design, SEM before/after stitching, all contour methods, centres and overlaps, with zoom/pan and hideable layers; pipeline must use production functions; OpenGL for speed. |
| D27 | 2026-10-02 | Overlap match gate default 10 → **25 nm**. Pairs that do not match are kept and classified (failed / skipped); tiles not connected to the largest stitched group get **NaN** corrections plus a warning, instead of the previous silent correction of 0. Gauge stays mean-zero; a "first tile fixed" view (`fix_tile`) is offered in the viewer. | User: SEM stage offsets between neighbours reach 25 nm, so the 10 nm gate rejected real pairs; those tiles then silently kept a 0 correction and misaligned in the stitched view. A stitching failure contaminates every later error and must be visible for diagnosis. User chose mean-zero gauge with an optional first-tile-fixed view (for a later global rotation/scale check against the design). |
| D28 | 2026-10-02 | Design tone chosen per tile by matching: normal first, tone reversed if fewer than 50 % of the SEM contacts match, flag the tile if neither reaches 50 %. Replaces the single `TONE_REVERSED` setting. | User: no simple file property identifies reversed files; compare with the SEM contacts using the existing matching. |
| D29 | 2026-10-02 | First design comparison = **raw** error per contact (stitched SEM − design, no affine removed), per tile against its own design, in `registration.py` (`design_errors`, `error_summary`). Tiles without errors are listed with a reason and warned. Sign SEM − design is tentative and lives in one function. The viewer shows an error map beside the SEM view; zoom/pan are linked by copying the visible rectangle, because pyqtgraph's setXLink aligns views by screen position (shifts side-by-side plots). | User: apply the tone choice, compute the error of the stitched centres vs the design, show it side by side with shared zoom, using library functions reusable in other pipelines; sign convention to be checked later. RANSAC (S7) next. |
| D30 | 2026-10-02 | The per-tile design files are **stitched** like the SEM (`stitch_design`, translation per tile). Design errors are computed per placement (nominal / stitched mean-0 / stitched first tile fixed) with each side's own corrections (`placement_corrections`). Design–SEM gate 25 → 40 nm. Normal-tone design contacts touching the FOV frame are dropped. Error-map arrows default ×10; tiles without a computed error are labelled "not measured". | User: `.oas` contours do not coincide in overlaps (file offsets, files cannot be fixed), so the design must be stitched too, translation only; the SEM-to-design offset exceeds 25 nm (40 nm works); arrows must differ between placements; "Tiles without errors" read as zero error. |
| D31 | 2026-10-02 | Overlap observations of the same contact are merged before RANSAC: grouped by stitched design position within **5 nm** (different tiles only), SEM and design positions averaged, spread and count kept, nothing dropped. scipy (`cKDTree`) becomes a declared dependency. Viewer layers start mostly off (only SEM images, design centres, refined centres, overlaps used, error arrows, tile outlines and failure markers on). | User: 2–4 observations per overlap contact would be counted several times in RANSAC; averaging agreed; radius 5 nm because design coordinates have very small residuals; viewer too slow with all layers on. |
| D32 | 2026-10-02 | RANSAC (S7) implemented as a **step generator** (`ransac_affine_steps`) plus a UI-free runner (`ransac_affine`), on the merged contacts, τ = 0.5 nm to start. Affine-only (not generic over model types, D10) and small-angle decomposition only (no polar/QR) until needed. A live pyqtgraph monitor (`view_ransac.RansacMonitor`) shows each iteration's sample, inliers/outliers, residuals, raw-error-vs-position plots with the model's line, inlier counts and the decomposed terms. | User: RANSAC pipeline that can run on its own, plus a UI to watch in real time which points are sampled, the parameter values, the linear-fit scatter and the outliers at every step. |
| D33 | 2026-10-02 | Affine terms are displayed as the correction G (sign as in §8.2) with rotation and orthogonality in degrees, magnification in ppm, plus a "nm at field edge" value per term (`report_terms`). | User: not used to µrad; wants degrees, and a shift in nm comparable to the threshold. Kept the correction-G sign (opposite to the raw-error plots). |
| D34 | 2026-10-02 | Final registration error = residual after the RANSAC affine for every contact, shown as dx/dy heatmaps (binned mean) and dx/dy-vs-y profiles averaged over x, via library functions (`binned_mean_2d`, `profile`). Heatmaps and means include outliers by default (they are the defects); the heatmaps can be rebinned from the inliers only, and the profiles also show the inlier-only mean. | User: the final registration error is the wanted result; wants a heatmap and the x-averaged dx/dy trend along y; the monitor's raw dx/dy plots alone could mislead, so the residual goes on the same plots. |
| D35 | 2026-10-02 | Registration error maps are per-contact coloured dots (jet), rasterized to the screen, replacing the binned heatmaps; the trend along y is the mean error per contact **row** (grouped by design y), drawn as plain lines for dx and dy, replacing the binned profile with scatter, ±1σ and inlier/outlier lines. Colours are computed by the view (opaque RGB) rather than by ImageItem. | User: heatmap must not be binned; a coloured scatter with suitable rasterization for the zoomed-out view; jet colour map; the x-averaged trend must average the contacts of each row; only the mean lines, no inlier/outlier split. |
| D36 | 2026-10-02 | Second correction, **moving-window affine** along y (`moving_window_affine`): window 40 µm, step 5 µm (notebook settings), plain least squares per window, each contact corrected by the window with the nearest centre. Shown side by side with the global RANSAC affine; does not replace it (D3/D4 unchanged). | User: the RANSAC result does not match the known values; the known-value algorithm fits an affine to the points within a window along y and slides it up. The nearest-centre rule and the 5 µm step are my defaults ("shift up a bit"), to be confirmed against the reference algorithm. |
| D37 | 2026-10-02 | Interactive **moving-window tuner** (`MovingWindowTuner`, pyqtgraph): sliders for window and step (0.5 µm resolution), recompute 150 ms after a change, invalid settings reported and the last result kept. `RegistrationView` gains `set_errors` (and a `correction` name for its summary). Per-window terms are compared with the RANSAC affine recentred on each window's reference point, not with its Tx, Ty at the global centroid. | User: wants to change the moving-window parameters with sliders and watch the results in real time. Recentring: a global rotation or scale makes the shift depend on position, so only the shift at the same point is comparable. |

---

## 13. Open Questions (ask the user; don't guess)

1. **Metadata format:** where are the stage coords and pixel size stored (JPEG EXIF/comment,
   sidecar `.txt`/`.xml`/`.csv`)? Which vendor/tool is used?
2. *Partly answered:* the image ↔ design orientation is identical (D19). Still open:
   **Stage ↔ image orientation:** the sign of the stage axes relative to the image, scan
   rotation, and whether the stage coords refer to the image centre or a corner.
3. **Stage ↔ design relationship:** how are stage coordinates related to design coordinates
   (alignment marks, known offset)?
4. **Pattern geometry:** contact pitch and size vs. expected stage error. Is the array fully
   periodic, or are there aperiodic regions/marks for unambiguous coarse alignment?
5. **Tile geometry:** typical FOV, pixel size, overlap width (top/bottom and left/right),
   tiles per stripe, number of stripes, and the typical number of contacts per overlap strip.
6. **Accuracy target / expected noise:** e.g. required 3σ registration precision in nm. This
   sets τ and the sub-pixel method.
7. **Tile model conditioning:** is a full per-tile affine really needed, or would a
   shared-intrafield + per-tile rigid model be acceptable if it proves more stable on
   synthetic data?
8. **Output format** preferred by downstream tools (CSV columns, units, sign convention
   `SEM − design` vs `design − SEM`). *Tentative (user, 2026-10-02):* SEM − design, to be
   checked; set only in `registration.registration_error()` so it can be changed in one place.
9. **Data bar:** do the JPEGs include an info bar that must be cropped? Is it always the same
   size?
10. **Per-tile `.oas` coordinate frame:** *Answered (2026-10-01):* **local, centred on
    (0, 0)** = image centre. The global position of each file's centre is in the metadata CSV.
    Global design position = tile-local + CSV centre (no rotation assumed until T001 §5e
    confirms the orientation).
11. **Per-tile `.oas` extent:** *Tentative:* exactly the image FOV. Contacts in an overlap
    region appear in **both** neighbouring tiles' files, so the design IDs of the same physical
    contact must be matched across tiles, e.g. by identical global design coordinates.
12. **Pairing:** *Answered:* by a **file-naming convention** (exact convention TBD).
13. **How are the per-tile `.oas` clips generated? Is the design an independent reference?**
    *Hypothesis (2026-10-02), to check and discuss with the vendors; not yet confirmed.*
    Each tile's `.oas` was most likely cut **in the SEM image's own frame**: a design-clip tool
    may align each clip to its image (shift, possibly rotation/scale), or clip it using the
    image's FOV and scan geometry.
    - *Evidence (real data, user):* (1) the `.oas` contours of neighbouring tiles do not
      coincide in their overlaps (D30); (2) after translation-only stitching of the SEM and,
      independently, of the design, the overlap contacts flagged for a large SEM spread show a
      **similar spread with a similar trend** in the design copies. The two stitchings share no
      data except the CSV tile centre, which is a pure translation and is removed by stitching,
      so a common residual points to a per-tile geometry (shift, rotation, scale) shared by each
      image and its clip.
    - *Why it matters:* if the clips follow each image, the design is **not an independent
      reference** within a tile. Part of the real per-tile error (offset, rotation, scale)
      would cancel in SEM − design, and stitching the design (D30) would undo the same per-tile
      offsets as stitching the SEM. This affects what the reported registration error measures.
    - *Checks:* (a) per tile, SEM vs design stitching corrections (`stitch_viewer.ipynb` step 2):
      similar if the clip follows the image placement; (b) at a flagged overlap contact,
      SEM(B) − SEM(A) ≈ design(B) − design(A) in direction and size?; (c) along a strip, does
      the spread grow toward the ends with opposite signs (rotation) or toward the tile edges
      (scale), the same way in SEM and design?
    - *Questions for the vendor:* how is each clip's origin chosen (planned site, stage
      reading, or alignment to the image)? Is the clip aligned to the image (pattern matching)?
      Is it rotated/scaled with the image (scan rotation, pixel-size calibration)?
