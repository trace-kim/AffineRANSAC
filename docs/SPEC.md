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
| Standalone analyzer app (D64), notebooks | Batch/CLI processing |

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
- Folder layout: `<DATA_DIR>/*.jpg` + one metadata `.csv`, and `<DATA_DIR>/ContourCAD/*.oas`.
  An `.oas` belongs to an image when its file name starts with **`SA`** and ends in the **same
  number** as the image's name (`CD<number>`; compared as integers); more than one match is an
  error, none gives no `.oas` (D72, task T004; replaces the "contains" rule of D41).
- **Tone reversal:** some `.oas` files draw the area *around* the holes. For those, use
  `read_contacts_tone_reversed()` (holes = frame − drawn shapes, D17). Which tone a file uses is
  decided per tile by matching against the SEM (D28, `pipeline.choose_design_contacts`): normal
  tone if ≥ 50 % of the SEM contacts match a design contact (tile-local, shift-search matching
  `match_with_shift`, D48: the SEM-to-design offset of a tile can exceed 40 nm), else tone reversed
  if that reaches 50 %,
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
- **Fractured design data (2026-10-06, user):** in the `ContourCAD/` files one contact is stored
  as several fractures (e.g. 16 trapezoids) that touch or overlap. `read_contacts()` therefore
  **merges** all shapes on the layer first (KLayout boolean OR) and returns one centre per merged
  pattern (D43). `notebooks/design_inspect.ipynb` shows one file's stored shapes and merged
  patterns (`list_shapes()`, `merge_shapes()`, D42). `read_polygons()` still returns the stored
  fractures (viewer drawing only).
- Config selects the **top cell** and **layer/datatype**.
- Hierarchy (cell references, arrays) is flattened.
- For contact holes: each **merged pattern** (touching/overlapping shapes joined) → one design
  feature point = its **area centroid** (holes subtracted). Also keep the bounding size (w, h) of
  the merged pattern for classification and matching sanity checks.
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
2048 × 2048 px** → pixel size 1.40625 nm. Some folders have **1024 × 1024 px** at the same FOV
(2.8125 nm/px; user, 2026-10-12): the pixel size is always FOV / image width of the loaded image
(`pipeline.process_tile`), never a constant. The CSV may be UTF-8 or cp949 (task T003, D68). A folder may hold **several** metadata CSVs (e.g. one per image): they are merged, exact duplicate rows dropped, and only rows whose *File name* (file-name part) names a `.jpg` in the folder kept (task T005, D84). The reader (`io/metadata.py::read_tile_index`) and the
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

### 4.5 Pre-analysed contour CSVs (alternative input)
Some folders hold **pre-analysed contour CSV files** instead of images: one `*.csv` per image,
no raw data, one row per contact, read with `io/contour_csv.py::read_contour_folder()` (D45).
- Columns used: `DesignX_Add`, `DesignY_Add` (design contour centre) and `SEMX_Add`, `SEMY_Add`
  (SEM contour centre); other columns are ignored. Rows with a missing or non-numeric value in
  one of them are left out and counted. CSVs in the folder without these four columns (e.g. the
  measuring tool's summary tables, `AffineCoefficients_*.csv`, `DiffAllPoint.csv`) are left out
  and named in a warning (D60).
- **Global mask coordinates, nm**, tile centre already added, y up (user, 2026-10-06).
- **One row = one contact:** design and SEM on a row are already paired, so they are not matched
  again (`registration.paired_errors`).
- **Image centre (D53):** if the file also has `DesignX`, `DesignY` (the design centre relative to
  the image centre; present in the real files), the image centre in mask nm is `DesignX_Add − DesignX`,
  `DesignY_Add − DesignY` (`ContourCsvTile.center_nm`; NaN without these columns). It is exact for
  partial tiles too, unlike the box centre; the in-image distortion map needs it.
- **No metadata CSV** in these folders (user): each tile's box (centre and size) is the bounding
  box of its design centres grown by a margin (`pipeline.tile_boxes_from_points`, notebook
  `TILE_MARGIN_NM` = 25 nm), passed to `stitch_tiles` as centres and FOVs, so two tiles overlap
  where both report contacts (D46).
- Viewer: `view_points.PointsStitchView` shows the design and SEM centres, tile boxes, overlaps
  (used / skipped / not matched) and unstitched tiles; it is also the Stitching tab of the
  analysis window for this input.
- After reading, SEM and design are stitched, merged, fitted and shown exactly as for the images
  (`notebooks/csv_stitch.ipynb`).

### 4.6 External reference measurements (D61)
Registration results of the same mask from elsewhere (e.g. another tool), to compare with ours: a
**txt file with four columns X, Y, dX, dY and no header** (spaces, tabs or commas between them),
read with `io/external.py::read_external()`.
- X, Y: site position in **mask µm**, the same mask coordinates as the design (user, 2026-10-08);
  converted to nm on load. dX, dY: registration error in **nm**.
- The sign convention is not known (user): `flip_sign=True` negates dX, dY (the app asks per file).
- It is a **reference residual** that ideally matches our result, whatever corrections it went
  through (user), so it is compared with our residuals after the affine. `AnalysisWindow.add_external()`
  puts it into every registration view (`RegistrationView.add_external`): its mean dx and dy per row
  (rows grouped by the sites' y, as for the contacts) as lines with symbols in the row profile after
  the affine, and on the dx / dy error maps a hollow **ring around every site** in the colour of its
  error (the maps' jet scale, white edges), sized with the zoom to enclose the contact disk there
  (D65): the contact inside shows our error, the ring the reference, so the same colour means
  agreement; hover shows the values. Its dx / dy check box in the Show bar hides its line and its
  rings together. Several files can be shown. The app logs how many sites lie on the analysed field
  (±1 µm), a check of the units and coordinates.

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
`method` (D39, D40; pipeline `detection`, notebook `DETECTION`): `"otsu"` (plain, default),
`"otsu3"` (three-class Otsu, `skimage.filters.threshold_multiotsu`: hole | background | bright
band; threshold between hole and background) or `"band"` (the bright band: pixels above the
background | band threshold; candidates are the non-band regions, 4-connected; the background
touches the border and is dropped, so the regions enclosed by a band remain; regions with solidity
< 0.9 are dropped, removing background pinched off between nearly touching bands). `"band"` does
not need the hole interior to be darker than the background. Wide bands break plain Otsu: it then
separates background from band. `tile_index.ipynb` shows the histogram, thresholds and the mask of
every method for one tile.

**Edge refinement (D22, D23):** `features/edges.py::refine_edges()`. Every Otsu contour point
is moved along its own **outward normal** (tangent from neighbours ±3 points) within a **fixed
±`search_px`** (default 5 px) to the steepest dark→bright rise (bright→dark for bright
contacts) of the lightly smoothed image, with a sub-pixel parabola fit. No radius or centre is
assumed, so it works for any shape (circles, ellipses, rectangles). The refined centre is the
area centroid of the refined contour. A point without a tangent (its ±3 neighbours coincide, at the tip of a 1 px wide spur) is not moved (D69). Known limit: at sharply curved edges (curvature radius
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
viewer marks failed overlaps and unstitched tiles in magenta and never moves the latter; these layers
start off (D76), and the status line above the plot gives the failures' counts in magenta. Every
stitching of `analysis._analyse` warns under its own name (`stitch_tiles(label=...)`, e.g. "SEM, all
overlaps", "SEM, measurement order", "Stripe drift corrected SEM, all overlaps"), and a measurement-order
stitching that leaves tiles unplaced (the chain broke) is logged with `!!` (D76). The
gauge stays mean-zero over the stitched group (D6); `stitching.fix_tile()` re-expresses the
solution with one tile fixed (viewer: "first tile fixed"), e.g. for later comparison with design
centres.

**Stitching method (D75).** On a periodic contact grid, an overlap whose real B − A offset lies within
the match gate of a whole pitch pairs every contact with its neighbour one row over, and the fit looks
perfect: the pair is used with a shift one pitch wrong, without a warning (synthetic, pitch 100 nm:
a real 85 nm offset fitted as +15 nm). It happens to overlaps between images taken far apart in time
(the drift between them adds up), e.g. the bottom images of two stripes of a serpentine. Setting
`stitch_method` (analysis): **all overlaps** (default, as before) pairs every overlap at the nominal
placement; **measurement order first** first stitches only the pairs measured one after the other
(`measurement_pairs`, D57), then pairs every overlap after moving tile j by the shift those corrections
predict (`stitch_tiles(guide=corrections)`, `pipeline.predicted_shift` = guide_i − guide_j; a pair with
a tile the guide did not place is paired at the nominal placement) and fits the shift on the paired
contacts as they are. In `analysis._analyse` the chosen method applies to every stitching over all
overlaps (raw, in-image corrected, stripe drift corrected, both); the design files are always stitched
over all overlaps. With either method, the raw SEM stitching is checked against the stitching in
measurement order (`pipeline.slipped_pairs`): a used pair whose shift differs from the predicted one by
more than `max_match_nm` is logged with `!!` ("paired one row off?"). The method is chosen in the settings
dialog or on the *Stitching* tab (`view_analysis.StitchingTab`); changing it there runs
`analysis.restitch(result, method)` in the background (image runs reuse the tiles in memory; contour
CSVs read the folder again; a reopened image result cannot), and the new result replaces the window, so
every tab after the stitching shows it; external measurements loaded are shown again; contact sizes
are dropped (measured again on request). A failed restitching leaves the window and its method as they
were.

**Design stitching (D30).** The per-tile `.oas` files can be offset against each other (the same
contact drawn at different mask positions in neighbouring files; user, real data). They are
stitched exactly like the SEM: `pipeline.stitch_design()` runs `stitch_tiles` on the design
centres (translation per tile, no rotation; flagged tiles left out → NaN, reported).
`stitching.placement_corrections()` gives, per placement ("nominal", "mean", "first"), the
(SEM, design) corrections; "first" fixes the first tile stitched in both.

**Affine stitching (D51).** `stitching.solve_tile_affines(n_tiles, pairs, ties, centers)`: one full
affine per tile, `T_i(p) = p + D_i (p − c_i) + t_i` (2×2 `D_i` about the tile centre `c_i`), least
squares over every tie contact `T_i(a) − T_j(b)` (weight 1 each; with `D` = 0 this is exactly the
translation solve, whose pair weights are the inlier counts). x and y share one sparse normal matrix
(3 unknowns per tile and axis, `D` in nm/µm for conditioning), solved with the gauge as exact
constraints (KKT, `scipy.sparse.linalg.spsolve`): `D_i` and `t_i` average zero over the largest
connected group (mean tile affine = identity, D6); other tiles NaN. Tiles whose tie contacts lie on
one line (spread across it < 10 nm) are warned about; a tiny pull of `D` towards 0 (10⁻⁶ µm²) keeps
their unmeasured scale nominal and biases measured tiles by ~1 ppb. `pipeline.stitch_tiles_affine()`
uses the **same pairs and inlier tie contacts** as the translation stitching (`stitch_tiles`), so
the two differ only in the tile model; ~0.7 s for 1,500 tiles / 5,200 pairs. Only the SEM is
affine-stitched; the design keeps its translation stitching. Result: (n_tiles, 3, 3) matrices,
nominal mask nm → stitched mask nm. A tile correction is a shift (2,) or an affine (3, 3)
(`stitching.apply_correction`), accepted by `design_errors` and `paired_errors`.
`pipeline.tie_residuals()` gives `T_i(a) − T_j(b)` of every tie after either stitching.
**Not used by the notebooks any more (D53):** on real data (long columns of ~250 tiles) the
per-tile affine turned a consistent apparent relative scale in every overlap, caused by a
sub-nanometre distortion shared by all images, into a tile-scale ramp along each column (several %)
and µm-sized parabolic tile shifts. Overlaps only measure scale relative to the neighbours, so a
per-tile scale integrates any such bias; a synthetic mosaic with one shared bow and no per-tile
differences reproduces it. The functions stay in the library (tested); a per-tile model needs a
constraint before it is used on long stripes.

**Translation + rotation and the stitching residuals (D58).** `stitching.solve_tile_rigid(n_tiles,
pairs, ties, centers)`: per tile T_k(p) = R(θ_k)(p − c_k) + c_k + t_k, least squares over all tie
contacts (weight 1 each), rotation solved to first order (error ≈ θ²/2 · r, 0.0005 nm for 1 mrad at
1 µm), gauge θ and t average zero (exact, KKT), written as (n, 3, 3) affines with the exact rotation;
with no rotation in the data it gives exactly the translation solution. `pipeline.stitch_tiles_rigid()`
uses the pairs and inlier tie contacts of the translation stitching, so the two differ only in the
tile model. **How well the tiles agree:** `pipeline.pair_residuals(points, stitch, corrections)` (per
used pair, T_i(a) − T_j(b) of its tie contacts, for shifts or affines), `pair_kind(centers, i, j)`
("vertical" = one image above the other, within a stripe; "horizontal" = side by side, between
stripes; "corner") and `residual_summary(...)` (RMS over the tie contacts within stripes, between
stripes and all, largest, number of overlaps). Both notebooks compute every input (raw, in-image
corrected, stripe drift corrected, and both corrections in the order they are made: the in-image
corrected points with their own drift per stripe measured and removed, D66) × tile model (translation,
translation + rotation) and show them in the *Stitching residuals* tab
(`view_stitch_residuals.StitchResidualView`: RMS per overlap against its y, within / between stripes,
a check box per stitching, two per row: an input's two tile models). A correction that acts inside the
images (in-image map) shows up there; a slowly varying one (drift curves, tens of µm) hardly does: over
one overlap it is nearly a shift, which the per-tile translations take up anyway. The registration
results use translation stitching, except the set *in-image + stripe drift corrected, translation +
rotation* (D67): both corrections, each tile placed with its translation + rotation (the errors take
the (n, 3, 3) affines as corrections), then RANSAC and the moving window. Along a long stripe the
rotations add up: a tile's rotation moves its overlap contacts sideways, so the translations of the
next tiles follow, and small rotation errors become a bow of the tiles in x that no global affine
removes (the moving window does).

**In-image (intrafield) distortion map (D53).** `intrafield.py`: one distortion shared by all
images, measured against the design and removed from every SEM point **before** stitching.
`estimate_map(design_local, sem_local, nodes=9)`: per image, SEM − design (positions relative to
the image centre) after that image's own affine (fit_affine, SEM → design; the affine part of the
shared distortion is left to the global affine), on nodes × nodes nodes from edge to edge of the
image (the overlaps lie at the edges; nothing is extrapolated there). Each node's value is a plane
fitted to the residuals of all images nearest to it (values beyond 5 median absolute deviations
left out), taken at the node; not a median, because the contacts can sit at a few positions away
from the node (e.g. an image step that is a whole number of pitches). Images with fewer contacts
than 90 % of the median are left out of the estimate (their own affine would take up part of the
distortion) but corrected like the others. `map_shift` is bilinear between nodes;
`correct_points(points, centre, map)` subtracts it. Mask errors at fixed places on the mask fall at
a different place in every image and average out; **mask errors that repeat at the image spacing
(e.g. a writer field of that size) do not, and would be removed as SEM distortion** (user concern:
future datasets may have such fields). Checks: `overlap_slopes(points, stitch, centers)`, how
B − A changes across each vertical / horizontal overlap (ppm; the same contact in two images, so
mask errors cancel and the design is not used), which a correct map brings to ~0; and the map from
the lower vs the upper half of the field (notebooks). Synthetic: a shared bow of ~0.2 nm at the
edge, each image with its own ±300 ppm affine: map within 0.01 nm; overlap slopes −640 / +480 ppm
→ ~0.

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
pairs of mutual nearest neighbours closer than `max_distance`. It is right only while the offset
between the sets is well below half the pitch P (the gate rejects pairs, it never chooses them).
**Design↔SEM matching uses `match_with_shift` (D48):** per tile, (1) every SEM−design difference
within `search_nm` votes for a shift and the strongest peaks are candidates; (2) each candidate is
refined (pairs = mutual nearest within `tolerance_nm` after the shift) and scored by its
**mismatches** (D50): inside the box where both sets are complete at that shift (overlap of their
bounding boxes, shrunk by `tolerance_nm`), the points of either set without a partner. At the true
shift ~0 (failed detections only); one row off, every missing contact of the pattern leaves points
without a partner. (3) The shift is chosen: a candidate with at least `min_extra` (= 2) fewer
mismatches (in points of the best's compared set) than every other is a **clear winner**; otherwise
the candidates are tied (*ambiguous*, fully periodic content). For one tile alone
(`match_with_shift`, used by the tone check): tied → the smallest shift. **For the design errors
(`consistent_choice`, D49/D50):** clear winners keep their shift (neighbours cannot overrule them);
tied tiles take the tied candidate closest to their overlapping neighbours' shifts, spreading out
from the decided tiles and repeated until stable; a connected group of only tied tiles takes, of the
solutions spread from its tile with the most pairs, the fewest total mismatches, equal: the
smallest median shift. Per tile `DesignErrors.match_mismatches` (~0 expected) and
`neighbour_jump_nm` (largest difference from a neighbour's shift) show anything left. The shift only chooses the pairs: each reported
error is the full SEM − design (option a, user). Only contacts found in **both**
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
matches each tile's SEM centres to that tile's design centres and computes each error with both
placed with their own corrections (one placement of `placement_corrections`; `match_with_shift`, D48).
**The pairs are chosen at the nominal placement (D86)**, before the corrections, where a tile is
only its own stage error away from its design: the corrections are translations (or rigid / affine
maps) of the whole tile, so they move the pairs without changing them, while far from the centre of a
large field the stitching can move a tile further from its design than `design_search_nm` (e.g. a
scale error added up over many tiles). `pair_points` names the points to pair on when a set's points
already hold a placement (the research app pairs every set on the nominal SEM points, so all sets
share the same pairs). `design_search_nm` is the largest offset of one image against its own `.oas`
(stage error; for a regular array also > P/2). It returns
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
magenta ring flags a spread above a display threshold (notebook `SPREAD_FLAG_NM`, default 2 nm; layer off at
start, D59). The RANSAC affine (S7) and residuals after `G` follow.

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

**Row-to-row pitch (D38).** `registration.row_pitch(design_y, y, gap_nm)`: rows grouped by design y,
row position = mean y of its contacts, pitch = difference between neighbouring rows, placed at the
rows' mean design y. `view_pitch.PitchView` plots it along y for the design and any SEM sets
(notebook: stitched SEM without affine, SEM after the RANSAC affine), plus SEM − design pitch,
with the tile edges as dotted lines.

**Drift curve along y (D54).** `fitting/drift.py`, the user's method. The mean SEM − design per row
(`row_means`) of the translation-stitched contacts follows a straight line where the error is
affine; drift bends it, and no affine removes a bend. `drift_curve(design_y, error, window_nm)`
smooths the row means with a local straight-line fit (at each row, the weighted least-squares line
through the rows within ±window/2, taken at that row; notebook `DRIFT_WINDOW_UM` = 40 µm) and also
fits one straight line through all rows. `DriftCurve.correction(design_y)` is the **bend**, curve
minus line, per contact; it is subtracted from the SEM positions and RANSAC is fitted again. The
straight part stays for the global affine, so G's terms compare with the uncorrected fit. Real mask
errors that vary smoothly along y on the scale of the window are removed too: they cannot be told
apart from drift. **Both components** are corrected, dx and dy, each as a function of the row's y
only (user's "curve fit of the dX/dY results"). The curve is a local fit, so it has no single set of
coefficients; the notebooks report what was applied (D56): the window, the straight line left for
the global affine (`DriftCurve.line_centre_y_nm`, `line_intercept_nm`, `line_slope`, per component),
and per row the row mean, the curve and the subtracted bend; the *Drift corrected* tab shows the
bend as lines (*drift correction applied*).

**Drift per stripe in measurement order (D57),** the user's flow. The images are measured stripe by
stripe along y in a serpentine (the first stripe, at the smallest x, from its bottom up if
`first_upward` / notebook `FIRST_STRIPE_UPWARD`, then alternately), so neighbouring stripes are
measured at very different times and drift differently. **No affine is fitted before the drift
correction** (user rule): an affine fitted to the initial data, in any role, could absorb real
pattern errors; the global affine is RANSAC (τ = 0.5 nm, fixed) at the end.
1. Stitch in measurement order: `pipeline.tile_stripes(centers, image_centers)` (stripe per tile, D89: the
   gaps between the sorted centre x are rounding within a stripe and one step between stripes, whatever the
   overlap; a new stripe at the gaps above the largest ratio between consecutive sorted gaps, gaps floored at
   1 nm, if that ratio is ≥ `STRIPE_BREAK` = 10, else one stripe; the exact image centres are used when every tile
   has one, as contour CSVs' bounding-box centres vary by up to a pitch) and
   `pipeline.measurement_pairs(centers, stripe, first_upward)` (the neighbours within each stripe and,
   between stripe s and s + 1, the image where s ends and the one where s + 1 starts: the top images
   after an upward stripe, the bottom images after a downward one); `stitch_tiles(..., only_pairs=...)`
   stitches only these.
2. `fitting.drift.stripe_drift(design_y, error, stripe, window_nm)` on the observations of that
   stitching (SEM − design, no affine): a drift curve per stripe from its own contacts (dx and dy, as
   `drift_curve`), and one straight line through the rows of all stripes, left for the global affine;
   `StripeDrift.correction(stripe, y)` = the stripe's curve minus that common line, subtracted from the
   stripe's coordinates (stitched in measurement order). A tile not stitched in step 1 stays nominal.
3. Stitch everything (all overlaps) on the drift-corrected coordinates; observations, merging.
4. RANSAC and the moving window on the merged contacts.
Neighbouring stripes' curves do not meet at their tied ends, and are not expected to: the tie makes
the two end images agree on their shared contacts (to 0.02 nm on the test data), but each curve is
the mean over its own stripe's contacts, about 2 µm apart in x, so any x-dependence of the error
(field-wide x-scale, rotation, skew) separates them by slope × distance (on the test data ~+1.9 nm in
dx, ~−1.4 nm in dy at every joint). The per-stripe correction removes that offset too, and the final
stitch over all overlaps puts it back (the x-slope of the error is the same before and after the flow).
`view_drift.StripeDriftView` (tab *Stripe drift curves*) shows per stripe the mean dx and dy per row
after step 1 (thin) and the subtracted curve (bold), both minus the common line, with a check box per
stripe and the common line's terms; the result is the set *stripe drift corrected* (extra lines and
its own colormap tab).

**Corrected sets in the analysis window (D52, D55).** Both notebooks compute the in-image corrected
contacts (stitched and, for the images, matched to the design again; RANSAC and moving window) and
the drift-corrected contacts (RANSAC), with `error_summary` tables. `AnalysisWindow(..., extra={name:
(sem_nm, design_nm, ransac, moving or None)})` keeps its tabs and adds each set as **extra lines**
(`RegistrationView.add_rows`; 1st set yellow dx / green dy, 2nd magenta / cyan, 3rd purple / lime,
4th red / blue, the same in every tab (D67), named after the set, each with its summary on a line of
its own): the row means after the affine and without affine in
the RANSAC tab, in the moving-window tab if the set has a moving-window result, and the pitch after
its RANSAC affine in the row pitch (`PitchView` also takes a set with its own contacts, `(design_nm,
sem_nm)`). Each set also gets one tab (*In-image corrected*, *Drift corrected*): the registration
view of its RANSAC residuals, on the colour scale of *Registration, RANSAC*. The RANSAC monitor, the
moving-window tuner and the stitch viewers show the uncorrected translation stitching. The row plots
of `RegistrationView` have no legends (D56): a **Show** bar above the plots lists each set of lines
(the view's own contacts under `name`, e.g. "uncorrected", then every `add_rows` set and lines
registered with `add_line`, e.g. the tuner's dashed RANSAC means) with a dx and a dy check box and the
line colour, three sets per row (D67); a box shows or hides that component of the set in both row
plots, which rescale in y.
**Copy rows** / **Export rows CSV...** (D70), beside the summary, give the shown row-profile lines
as a table: per set and plot (after the affine, then without affine) its row y − y_ref (µm), dx and
dy (nm) columns, blank below a set's last row; tab-separated on the clipboard, or a CSV file
(UTF-8 with BOM).

**End-to-end flows as library functions (D62).** `analysis.py` runs the notebooks' flow on one folder,
without UI: `analyse_contour_folder(folder, settings, log)` (csv_stitch.ipynb) and
`analyse_images(folder, records, settings, log)` (stitch_viewer.ipynb; `records` from the remote-only
`read_tile_index`, passed in by the caller). Both share `_analyse`: translation stitching over all
overlaps, errors per placement and merged contacts, the in-image map (left out, with a log line, for
contour CSVs without `DesignX`, `DesignY`), the drift curve, the drift per stripe in measurement order,
RANSAC (+ moving window) of every set, and the translation + rotation stitchings for the residuals.
`Settings` holds the notebooks' settings (same names in lower case, defaults as the notebooks, help
texts as field metadata). The `AnalysisResult` holds what the window shows: the tile boxes and points,
the design stitching, `stitchings` (raw / in-image corrected / stripe drift corrected / in-image +
stripe drift corrected: points,
translation `StitchResult`, rigid affines), `sets` (uncorrected / in-image / drift / stripe drift
corrected / in-image + stripe drift corrected, translation + rotation: merged contacts, RANSAC, moving
window), the drift curves and the run's log; after an image
run also the tiles with their images, the design polygons and the merged errors per placement
(`ImageTiles`, for the image stitch viewer; not saved). `view_analysis.analysis_window(result)` builds
the window of the notebooks' last cell from it (points stitch view when there are no images). On the
local contour-CSV test data the flow gives bit-identical results to the notebook. The notebooks keep
their step-by-step cells (they show the intermediate tables).

**Results file (D63).** `results_file.save_result(result, path)` / `load_result(path)`: one `.npz`
(compressed) of plain arrays, loaded with `allow_pickle=False` (opens without this code; loading
never runs code). Array names are paths (`sets/uncorrected/ransac/residuals`); a list of arrays (per
tile, per overlap) is one concatenated array plus the part lengths; the folder, kind, settings, tile
ids and log are one JSON text (`info`). The images of an image run are not saved: a reopened result
shows the points stitch view. `FORMAT` changes with the layout and other formats are refused; saved
settings unknown to the code are ignored, missing ones take their default. Local contour-CSV test
data (221k contacts): 51 MB, saved in ~6 s, loaded in < 1 s. This is the app's own file, not the
downstream report format (§13-8, still open).

**Standalone analyzer (D64).** `app.py` (`python -m affine_ransac.app`): a main window with the
analysis window in the middle and a log below. *File → Open folder* detects the kind (a `ContourCAD`
folder or `*.jpg` → images, else contour CSVs), shows that kind's settings (name as in the notebooks,
value, help text; the RANSAC threshold is shown but fixed; the last used settings and folder are
remembered in the user's Qt settings), then runs `analyse_contour_folder` / `analyse_images` in a
background thread; image folders get their records from `io/metadata.read_tile_index`, imported only
then (only on the data machine; elsewhere the error says so). The log shows the run's messages and the
library's warnings with the time; a failure shows its last line in a message box and the traceback in
the log. *Open results* / *Save results* use `results_file` (in the background); a reopened result
also shows the log of its run. *Load external measurement* reads a txt reference (§4.6), asking for
the sign of dX, dY. *View*: the log, and OpenGL drawing (off if the plots stay black; used for the next
window). On the local contour-CSV test data (1,494 files, 221k contacts) a run takes ~45 s, building
the window ~9 s.

**Release app MaskReg (D77).** `release/` (`python -m affine_ransac.release.app`, or `maskreg`): the minimal
analyzer given to users; the research app (`app.py`) is unchanged. Two tabs only: **Stitching** (`release/views.
StitchingView`: tile outlines, overlaps used, design and measured centres at the nominal or stitched placement;
stitching failures and tile names off at start, D76; **Load SEM images** reads the images in the background,
downsampled by a whole factor to stay under 4·10⁸ pixels in memory, `release/analysis.read_display_images`; an opacity slider, D80) and
**Registration** (`RegistrationView`, D80, D88: one figure, side by side and showing the same area (same centre and µm per pixel, `link_views`): the **arrow map** (the error vectors, shaft + two head strokes, each coloured by its length dr = |error| in jet on its own colour bar, 32 steps; scale box, default so the 99th-percentile error is ~half a pitch long) and the **dx and dy colour maps** (shared colour bar); every map has its profiles beside it (`MarginalMap`): the mean per row on its right, rotated so the row's y runs down the side aligned with the map's y, and the mean per column (`row_means` over design x, same gap) below it aligned with its x, both following the map's zoom and pan; the colour maps show their own component, the arrow map the mean dr; Copy profiles / Export profiles CSV give rows and columns (dx, dy, dr); no profile without affine; D81: *Values* every contact or **grid average** (`registration.grid_means`: mean per square cell, cell size in µm, default ~2000 cells and at least two pitches; colour maps then one flat square per cell, arrows one per non-empty cell from its centre, coloured by the length of the cell's mean, scale reset to ~half a cell), **Grid lines** on every map (on at start, D88), and **Copy image / Save image** menus with the whole figure or any one graph (a map with its colour bar), also in each graph's right-click menu; a graph is cut from the whole figure rendered at 2×; D88: **Stripes** (`StripeSelector`, a drop-down of tick boxes, all ticked at start, *All*; one stays ticked): the maps, profiles, colour scales (99th percentile of the contacts shown) and summary show only the contacts of the ticked stripes; a contact belongs to every stripe whose x range (`pipeline.stripe_ranges`: its tiles' boxes, stripes `ReleaseResult.stripe` from `tile_stripes`, computed once per run and saved, D89) holds it, so one in the overlap of two stripes is in both (`pipeline.contact_stripes`; outside every range: the nearest); the moving-window errors are not fitted again). The error is **measured (SEM) − design**
(`registration_error`; moving window `G_w(SEM) − design`). `release/analysis.analyse_folder` runs only what these
tabs need: tiles (`analysis.load_image_tiles` with per-tile progress, or the contour CSVs), SEM stitching
(`stitch_method`, slipped-pair check), design stitching, the errors at the stitched ("mean") placement, merging,
the moving window: no in-image map, no drift corrections, no RANSAC. A common shift of all contacts does not
change the moving-window residual, so the placement is not a parameter. Parameters (`release/analysis.PARAMETERS`,
a subset of `Settings`) are hidden: *Analysis → Parameters* (the research app's `SettingsDialog`); after a change
it offers a restart. *Analysis → Restart analysis* (F5) analyses the open folder again. **Autosave**
(`release/cache.py`): after every run the result is written to `MaskReg_analysis.npz` in the data folder (plain
arrays, `allow_pickle=False`, its own `FORMAT`), or, if that cannot be written, under the user's cache folder;
opening a folder shows the saved result instead of a new run (another format → a new run). **Figures**
(`release/style.py`): white background, dark text, light grid, axes in mask µm (drawn relative to a whole-µm
origin, `MaskAxis`), no SI prefixes, line colours dx blue / dy orange (colour-blind-checked palette), error maps
jet (D35) with white where there is no contact; each figure has **Copy image** / **Save image...**, rendered
through the scene at 2× (works with an OpenGL viewport). **GPU / CPU:** drawing uses OpenGL if a context with a
real renderer can be made at start (`opengl_works`; not the Windows "GDI Generic" 1.1 fallback, with which the
plots stay black), else the CPU; *View → Draw with the GPU* overrides it and redraws at once. The computation is
CPU (numpy / scipy) either way. **Defaults and the contact layer (D78):** the release's defaults are `release/analysis.DEFAULTS` (`max_match_nm` = 100 nm; the research app keeps 25 nm). The contact layer is not a parameter: `analysis.detect_layer(records)` reads every `.oas` of the folder and takes the one (layer, datatype) that holds shapes (`io/design.layers_with_shapes`: polygons, boxes, paths; empty layer definitions and texts do not count); more than one, or none, stops the run with a message naming the layers and a file of each. **Scan direction (D79):** `first_stripe_upward` is not a parameter either: the file numbers (image or CSV file name `CD<number>`) follow the measurement order (user), so `release/analysis.scan_direction` takes it from the first stripe (`pipeline.stripe_directions`: its last image in number order above its first → upward) and stores it in the result's settings; stripes that do not alternate are logged with `!!` (not a serpentine: the between-stripe pairs of `measurement_pairs` are then wrong); names without distinct numbers, or a first stripe of one image, fall back to the default (upward) with a `!!` line. **Status bar:** the step (e.g. "Loading tile 312 of 744"), a progress bar with its
percentage and the time so far; when idle, what is shown and whether drawing is on the GPU or the CPU.
**FTP browser (D82).** *File → Browse FTP server...* (Ctrl+Shift+O, `release/ftp_dialog.FtpDialog`): host, port,
user (empty: anonymous) and password, **Connect**; then the remote folder's path, **Up**, and its folders (with a
trailing `/`, first) and files, each sorted; double-click or Enter opens a folder. Plain FTP, passive mode, standard
`ftplib` (`release/ftp.py`: `connect`, `list_folder` by `MLSD`, or by parsing `LIST` in Unix or DOS format when the
server has no `MLSD`), a 15 s timeout per command; commands block the window under a busy cursor. Host, port and user
are remembered in the preferences, a typed password never.
**Download and analyse (D83).** The button takes the selected folder (or the folder shown when none is selected)
and shows where it will be kept: `ftp.local_folder` = `<download folder>/<host>/<remote path>` (names made safe for
Windows, `safe_name`). The window then (`MaskRegWindow.download_and_open`, a background job with its own connection)
runs `ftp.download_folder`: every file under the remote folder, subfolders included (`walk`; an image folder's
`ContourCAD`), each written as `<name>.part` and renamed when complete; a file already there with the same size is
kept; an empty remote folder stops with a message. Then the local folder is opened as in *Open folder*: if no file was
downloaded its saved analysis opens at once, otherwise it is analysed again (a saved result would not hold the new
files). Local files that are gone from the server are not deleted. **Pre-set server** (`release/config.py`):
`maskreg.ini`, `[ftp]` keys `host`, `port`, `user`, `password`, `download_folder` (all optional), read from the path in
the `MASKREG_CONFIG` environment variable, else `%LOCALAPPDATA%/MaskReg/maskreg.ini` (`GenericConfigLocation`);
pre-set values are used over remembered ones, and with a pre-set host the dialog connects when it opens. The
download folder defaults to `Documents/MaskReg data`. A malformed file is reported and ignored.

**Checks before the run (D73).** `analysis.check_settings(settings, kind)` (every setting of that kind
of folder in range: `POSITIVE` ones > 0, `min_matched` ≥ 1, `intrafield_nodes` ≥ 2, `min_match_fraction`
in (0, 1], choices) runs first in both flows; for images `check_image_records(records, layer)` (images
listed and on disk, at least one image with an `.oas`, the `.oas` files on disk, the contact layer in the
first `.oas`), then `SAMPLE_TILES` = 3 tiles (first, middle, last with an `.oas`) are loaded and matched
to their design; if none matches, the run stops with their match fractions. The other tiles are then
loaded **in parallel**, in half as many worker processes as logical cores, about one per physical core
(`concurrent.futures.ProcessPoolExecutor`, `analysis.worker_count`, D85, D87; the count is logged),
in their order; the result is the same as loading them one by one. Once loaded, one `Timing:` line is logged
(`analysis.timing_summary`, D90): the wall time, the median seconds per step of `process_tile` (read, detect,
refine, oas; then `choose_design_contacts`: contacts, shift, reversed; kept in `TileResult.seconds`),
the wall time the steps alone would take, and on how many tiles the reversed tone was tried. After all tiles are loaded,
the run stops at once if no tile matched its design (before stitching). Each failure names its cause.

**Contact size (D71, D73, D74).** Measured **on request** after an image run, not during it: the
*Contact size* tab (`view_contact_size.ContactSizeTab`) has an **ROI width** (nm) and **Run measurement**,
which runs `analysis.measure_contact_sizes(result, roi_width_nm, progress)` in a background thread and
then shows the maps. It needs the run's tiles in memory (`AnalysisResult.images`, which keeps the
observations of every placement); a reopened result shows sizes saved with it, and cannot measure. Per tile
(`TileResult.contact_sizes(layer, roi, refined)`): area, xCD and yCD of the SEM contour (refined if
`stitch_refined`, else Otsu; pixel → tile nm via `frames.py`) and of the `.oas` outline, read again from
the tile's `oas_path` in the tone that matched (`pipeline.design_outlines`: one merge of the file; centre
and bounding box from the outline, the same frame filter as `choose_design_contacts`, so the outlines
follow the design centres; a count mismatch raises). `geometry/polygon.contact_sizes`: area = shoelace;
xCD = the length of the horizontal chord through the contact's centre (area centroid), between the
outermost crossings of the outline; with ROI width > 0 the mean of the chords one SEM pixel apart inside a
band that wide centred on the centre (`roi // pixel + 1` chords; the design uses the same lines); yCD
likewise with vertical chords; NaN where no chord crosses. Averaged per merged contact over the tiles
that saw it (`registration.merge_values`, following each observation's `sem_index` / `design_index` in
the `DesignErrors`), NaN left out. `AnalysisResult.contact_sizes` (`sem`, `design` (M, 3), rows of the
"uncorrected" set, and `roi_width_nm`; saved as optional arrays in the results file, no format change).
Maps (`ContactSizeView`): Show boxes Area / xCD / yCD (Area at start; any combination side by side, linked
zoom), Value **SEM** or **SEM − design**, each map its own jet colour bar from the 1st to the 99th
percentile; a summary line per quantity (mean, 3σ of SEM, design, SEM − design). **Behind the map**
(D74): check boxes *SEM images* and *Design* (off at start; enabled only with the run's tiles in memory)
show every tile's SEM image (below the map) and its `.oas` polygons (above it, as outlines), each tile at
its nominal centre plus the SEM / design correction of `ransac_placement`, the placement the contacts
were merged with (`view_contact_size.tile_backgrounds`; tiles not stitched left out); the items are made
the first time a box is checked; with the SEM images on, the map is drawn RGBA, transparent between the
contacts (`view_registration.colorize(transparent=True)`). Contour-CSV runs have no outlines: the tab
says N/A. Cost (synthetic `.oas`, 400 contacts × 16 fractures): ~36 ms per tile, of
which the KLayout merge ~28 ms, vertex copy ~8 ms, the chord arithmetic ~1–2 ms.

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
│  │  ├─ contour_csv.py      # pre-analysed contour CSVs (§4.5)
│  │  ├─ external.py         # external reference measurements, txt (§4.6)
│  │  └─ metadata.py         # TileMetadataReader protocol + CSV manifest impl
│  ├─ features/
│  │  ├─ base.py             # Feature interface (point now, line-edge later)
│  │  └─ contact.py          # contact-hole detection + sub-pixel centroid
│  ├─ geometry/
│  │  ├─ frames.py           # pixel↔tile↔stage↔design conversions
│  │  ├─ polygon.py          # area, centroid; contact area / xCD / yCD (D71)
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
│  ├─ pipeline.py            # S2–S5 steps for a set of tiles
│  ├─ analysis.py            # end-to-end flow of one folder (contour CSVs / images), D62
│  ├─ results_file.py        # save / load an AnalysisResult (.npz), D63
│  ├─ app.py                 # standalone analyzer: python -m affine_ransac.app, D64
│  ├─ release/               # MaskReg, the minimal app for users: analysis, cache, style, views, app (D77), ftp + ftp_dialog + config (D82, D83)
│  ├─ view_contact_size.py   # contact size tab: measure on request, maps (area / xCD / yCD), D71, D73, D74
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
| D38 | 2026-10-02 | **Row-to-row pitch** along y (`row_pitch`, `PitchView`) for the design, the stitched SEM without affine and the SEM after the RANSAC affine; SEM − design pitch below; tile edges marked. Row position = plain mean y of the row's contacts. | User: the moving-window affine did not find the cause; the relative position of each row (pitch) should be compared along y for design and SEM, including the bare stitched SEM. Tile edges added to tell stitching seams from mask errors. |
| D39 | 2026-10-02 | Contact detection option **three-class Otsu** (`detect_contacts(classes=3)`), threshold hole | background; plain Otsu stays the library default, the notebooks use 3. A band-enclosure method (bright band threshold, holes = regions enclosed by bands) was rejected. | User: on a second dataset with wider white bands Otsu detected regions inside and outside the bands; the hole interior is clearly darker than the background; bands come close but never touch. Synthetic test: 16 px bands, plain Otsu 85 regions for 49 holes, three-class 49 (< 0.1 px). Band enclosure fails the same way as plain Otsu, because the blurred gaps between nearly touching bands rise above a band threshold and close the rings. |
| D40 | 2026-10-02 | Detection `method="band"` added (regions enclosed by the bright band, 4-connected, solidity ≥ 0.9) and made the notebook default; `classes` replaced by `method` (`"otsu"`, `"otsu3"`, `"band"`). Supersedes the rejection in D39: the solidity filter removes the pinched-off background. Detection check cell in `tile_index.ipynb`. | User: three-class Otsu still fails on the second dataset; asked for the white-band method that had been proposed. Synthetic: 16 px bands 49/49 (< 0.2 px), also with the hole interior at background grey; background pieces have solidity 0.73–0.88, holes ~1.04. |
| D41 | 2026-10-06 | Design folder is `ContourCAD/` (was `Contour/`); `.oas` paired with its image when the file name **contains** the T001 key; no match → `None`, several → `ValueError`. Done by the remote agent (T002); contract test checks the `ContourCAD` parent folder. | User: the folder name changed and the `.oas` names now have a prefix. |
| D42 | 2026-10-06 | Inspection helpers `list_shapes()` and `merge_shapes()` in `io/design.py`, plus `notebooks/design_inspect.ipynb` for one `.oas` file: stored shapes (kind, source cell, size, area), exact duplicates, merged patterns (shapes per pattern, overlap area, distance of shape centroids from the merged centroid), and plots of the whole file and one pattern. `read_contacts()` is unchanged until the structure is confirmed. | User: after the `ContourCAD` fix, one contact consists of several overlapping shapes, so a centre is found per fracture; wants to open a single file and check it step by step before the method is chosen. Merging (KLayout boolean OR) is the proposed fix. |
| D43 | 2026-10-06 | `read_contacts()` merges all shapes on the layer (`Region.merged()`) and returns the area centroid (holes subtracted) and bounding size of each merged pattern, replacing one centre per stored shape. The box-only fast path is gone. The tone-reversed reader already worked on the merged region. The stitch viewer still draws every tile's whole `.oas` (stored fractures, not clipped to the FOV); fixing that is deferred. | User: one contact = 16 touching/overlapping trapezoids; the merged patterns in `design_inspect.ipynb` match one SEM contact each, no duplicates. The extra shifted "contacts" seen in the stitch viewer were neighbouring tiles' `.oas` files (they extend beyond the FOV; they disappear when the neighbours are hidden); the viewer is deferred (user). Speed: ~39 ms per file of 400 contacts × 16 trapezoids (merge 11 ms, Python centroids 25 ms), above D11's 1.2 ms but small next to SEM processing; optimise only if it matters. |
| D44 | 2026-10-06 | All analysis viewers in **one tabbed window**, `view_analysis.AnalysisWindow`: an optional stitching tab (the caller's `StitchViewer`), RANSAC monitor, registration error (RANSAC and moving window), moving-window tuner, row pitch. Every registration row profile also gets a **reference plot below it: the mean error per row of the stitched contacts without any affine** (`RegistrationView(raw_error_nm=...)`), in its own plot sharing the row axis because the raw error can be far larger than the residual. `stitch_viewer.ipynb` keeps its calculation and table cells and opens the one window at the end; its row table lists the no-affine row means too. | User: the UIs ran in separate notebook cells; wants one UI for inspection, plus the x-averaged dX/dY of the stitched coordinates without affine correction as a reference next to the existing plots. |
| D45 | 2026-10-06 | **Contour CSV input** (§4.5): `io/contour_csv.py` reads the four `_Add` columns per file; tiles without contacts are left out (listed); tile boxes from the bounding box of the design centres; SEM and design stitched with `stitch_tiles` as for the images; errors from the CSV pairing (`paired_errors`, no re-matching), then `merge_observations`, RANSAC, moving window, pitch and `AnalysisWindow` without a stitching tab (no images). `notebooks/csv_stitch.ipynb`. Synthetic check (2 × 2 tiles, known stage errors and file offsets) recovers both to < 0.1 nm. | User: folders of pre-analysed CSVs (contour centres only, no raw data) should be stitched and analysed like the image pipeline. User answers: global mask nm, rows already paired, no metadata CSV. The bounding-box tile box replaces a centre estimate with a fixed FOV, which could be off by up to half a pitch. |
| D46 | 2026-10-06 | Contour CSV tile boxes are the bounding box of the design centres **grown by a margin** (`tile_boxes_from_points(margin_nm)`, notebook default 25 nm, about the match gate). Points-only stitching viewer `view_points.PointsStitchView` in `csv_stitch.ipynb` section 2 and as its Stitching tab. | User: on real data section 2 gave NaN corrections for (nearly) every tile although 1014 overlaps were used and none failed. Reproduced on a synthetic 23 × 23 mosaic: neighbours sharing only one row/column of contacts get boxes that just touch or, with a few nm of file offset, miss each other, so pairs drop out at random, the mosaic splits into many groups and only the largest is solved (49 of 529 tiles). With the margin all 529 are stitched. Extra contacts inside a grown box find no partner. User asked for a viewer of the design and SEM coordinates in section 2. |
| D47 | 2026-10-06 | `stitch_viewer.ipynb` section 3b: the stitch viewer opens **in the middle**, right after the design errors, when a stitching pair did not match, a tile was not stitched or a tile was not measured in some placement (`DIAGNOSTIC_VIEWER = "auto"`; True / False to force). RANSAC (section 5) stops with a clear message if fewer than 3 contacts were measured. `StitchViewer` is tested with no measured contact at all. | User: on new data every tile reported "no SEM contact matched the design"; with all viewers merged into one window at the end (D44) the failure could not be inspected, because the later steps fail first. |
| D48 | 2026-10-06 | Design↔SEM matching (tone check and `design_errors`) by **shift search** (`matching.match_with_shift`): find the per-tile shift at which most contacts line up within `tolerance_nm` (default 10 nm, notebook `DESIGN_TOLERANCE_NM`), searched up to `search_nm` (default 100 nm, `DESIGN_SEARCH_NM`, replaces `DESIGN_MATCH_NM`); near-ties (within 10 % of the best score) → the smallest shift (`choose_shift`, tile marked ambiguous). Pairs are chosen after removing the shift; the error keeps it (option a). The shift and the ambiguous flag are kept per tile (`DesignContacts.shift_nm/ambiguous`, `DesignErrors.match_shift_nm/ambiguous`) and listed in `stitch_viewer.ipynb`. Tile↔tile overlap matching (`match_overlap`) and `tile_index.ipynb` still use plain `match_points`. | User: on new data the 40 nm gate rejected every pair; at 100 nm contacts were paired with the wrong design contact. Nearest-neighbour pairing only works for offsets below P/2, and a gate cannot choose, only reject. User's method (overlay the patterns, find the matching shift, correct it, the minimum-distance match among several) and the proposed vote are the same idea; combined: votes find the candidates, the score decides, the user's minimum-distance rule breaks near-ties. Contour shapes were not used: identical contacts add no position information over their centres. The user may later replace the tie-break with the smallest change against the adjacent tiles' shifts. Synthetic: offsets of 55 nm (P = 130 nm) paired correctly where a 100 nm gate picked wrong neighbours; a 0.6 P offset is taken as −0.4 P and flagged (documented limit). |
| D49 | 2026-10-06 | Design↔SEM shifts in `design_errors` are chosen **for all tiles together** (`matching.consistent_choice`, `design_errors(neighbours=pipeline.tile_neighbours(...))`): each tile takes the candidate (score ≥ 50 % of its best) closest to its overlapping neighbours' shifts, and the lattice offset of each connected group is decided by the total score over its tiles; `neighbour_jump_nm` per tile (largest difference from a neighbour's shift, not the median, which hides a straight seam). The per-tile smallest-shift rule stays only for single sets (tone check). `stitch_viewer.ipynb` passes the neighbours, lists the jumps and opens the diagnostic viewer for jumps above 2 × `DESIGN_TOLERANCE_NM`. | User: on the real data the smallest-shift tie-break failed; overlaps suddenly matched the next contact row/column. User proposed the smallest change against the adjacent shifts; added: the branch of the whole group by the total score, because each tile's edge-row advantage of the true offset is small (~5 %) but adds up over many tiles, which also resolves a field-wide offset beyond P/2. Synthetic: a shift drifting 45 → 85 nm across P/2 = 65 nm (the reported failure) and a field-wide 80 nm offset are both recovered (< 0.5 nm); 529 tiles: 0.2 s for the choice. |
| D50 | 2026-10-06 | Shift candidates are scored by **mismatches** (points of either set without a partner inside the box where both are complete) instead of the number of pairs; a clear winner needs ≥ 2 fewer mismatches than every other candidate (absolute, not 10 %); `consistent_choice` keeps clear winners and lets only tied tiles follow their neighbours; a group of only tied tiles is decided by the fewest total mismatches, then the smallest median shift (the total-score rule of D49 is dropped). `match_with_shift` returns (match, tied); `DesignErrors.match_mismatches` per tile; `min_extra` replaces `tie_fraction`. | User: on the real data (patterns with missing contacts) D49 still matched the next row in places, although the missing contacts should rule that out. Cause: the missing contacts change the pair count by only a few %, far less than the D48 tie margin (10 %) and the D49 allowance (any candidate with ≥ 50 % of the best score), so the evidence was discarded and the smallest-shift / neighbour rules decided; and the neighbours could overrule a tile that clearly knew its shift. Counting pairs also mixes in how much area the sets share at each shift, which does not tell the rows apart; the D49 claim that the total pair score picks the right lattice offset relied on that and is withdrawn. Synthetic: 8 % missing contacts, offset 80 nm (> P/2) inside the same area: one row off keeps ≥ 90 % of the pairs (a former 'tie') but has > 5 mismatches, the true shift 0; independent per-tile shifts ±85 nm all recovered; periodic tiles follow decided neighbours; 7 ms per tile. |
| D51 | 2026-10-06 | **Affine stitching** per SEM tile (`solve_tile_affines`, `stitch_tiles_affine`) beside the translation stitching, from the same pairs and inlier ties; gauge mean affine = identity as exact KKT constraints; tiny ridge on the linear part (10⁻⁶ µm²) plus a warning for tiles with collinear ties. Corrections may be shifts or affines (`apply_correction`) in `design_errors` / `paired_errors`. Shown in the analysis window (D52). Design stays translation-stitched; the stitch viewer shows translation only; affine side mean gauge only. | User: implement a full affine correction for the SEM stitching and compare it with translation stitching in the final analysis UI of `csv_stitch.ipynb` and `stitch_viewer.ipynb`. Same ties so only the tile model differs. Per-tie least squares instead of pairwise affines: strips are narrow, so pairwise affines are poorly conditioned, while all ties of a tile together (top, bottom, sides) fix its scale and skew. KKT instead of a heavy gauge weight: exact, no weight to tune. Synthetic: per-tile affines recovered up to one common affine (≤ 0.01 ppm), pure shifts reproduce the translation solution, ~1 nm tile scale/rotation errors: tie RMS 0.2 nm (shifts) → < 0.001 nm (affine). |
| D52 | 2026-10-06 | The comparison window of D51 (`view_compare.StitchingComparison`: a Comparison tab plus one `AnalysisWindow` per stitching; `registration.stitching_summary`) is **removed**. The original `AnalysisWindow` stays as it was; the affine-stitched results are extra lines in its registration and row-pitch plots (`affine_stitching=...`, `RegistrationView.add_rows`, `PitchView` sets with their own contacts) plus one tab, *Affine stitching*. The notebooks compare the numbers with `error_summary` tables. | User: did not ask for a new tab and could not understand it; keep the original UI design, add the affine-stitched results as new lines, and maybe one tab for just the affine stitching. The tab holds what cannot be a line: the affine-stitched error maps (with their row profiles), on the RANSAC tab's colour scale so the maps compare. |
| D53 | 2026-10-06 | **In-image distortion map** shared by all images (`intrafield.py`), from the design (per-image affine removed; node grid from edge to edge; plane per node with MAD rejection), subtracted from the SEM points before stitching; checked with the overlap slopes (design-free) and the lower/upper half maps. The contour CSV reader adds the image centre from `DesignX`/`DesignY`. The per-tile affine stitching (D51) leaves the notebooks and the UI (library kept). | Diagnosis (user asked): the per-tile affine stitching gave unrealistic results because every overlap shows the same apparent relative scale (a shared non-affine distortion of the images, sub-nm), which a per-tile scale integrates along each column; each tile fitted alone to its design has the same scale; a synthetic shared bow reproduces the failure. User: the overlaps cannot see the image centre (right: they only measure differences between the edge bands, and an overlap-only parabola underestimated the centre about 2×), so the map is measured from the design. User concern that real, periodic mask errors (writer fields of the image size) would be absorbed: a documented limit, with the design-free overlap check. On the real data the overlap slopes drop to near 0 after the correction. Node grid and plane fit instead of bin medians: bin centres left the overlap bands to extrapolation, and medians sat at the few contact positions of a bin (synthetic tests). |
| D54 | 2026-10-06 | **Drift curve** (`fitting/drift.py`): local straight-line fit through the row means of SEM − design (window 40 µm); its bend (curve minus the straight line through all rows) is subtracted from the SEM positions before RANSAC. | User's method: the row means follow a linear trend that the global affine removes, but drift bends it; fit a curve to the translation-stitched dX/dY, subtract it from the pattern positions, then the global affine. Only the bend is subtracted, so G keeps its (physical) linear terms. Local line fit: follows local bends, no edge bias for a straight trend. Smoothly varying real mask errors are removed with it (documented). Observed on the real data: the largest remaining structure after RANSAC is a sawtooth in dy with a period that is not a whole number of tiles, created by the translation stitching (absent in the unstitched positions); cause not yet known. |
| D55 | 2026-10-06 | `AnalysisWindow(extra={name: (sem, design, ransac, moving or None)})` replaces `affine_stitching=`: any number of corrected sets as extra lines (a colour pair per set) plus one tab each; `PitchView` shows each set after its RANSAC affine. | User: make the method available in the analysis UI; keep the original design (D52). |
| D56 | 2026-10-06 | `RegistrationView` row plots: a **Show** bar of check boxes (per set: name, dx and dy with the line colour) replaces the pyqtgraph legends; a box hides or shows that set's component in both row plots and the y axis rescales. Drift report: `DriftCurve` keeps the window and the straight line's centre, intercept and slope; the notebooks print them and a per-row table (row mean, curve, bend), and the *Drift corrected* tab shows the subtracted bend. | User: with all corrections in one graph the legend overflowed and the lines could not be chosen interactively (pyqtgraph toggles a curve only by clicking its small legend sample). User asked whether the drift correction also acts in x (yes: dx and dy, both as functions of y, as described in the request) and how to see which correction was applied (it is a local fit: shown as its parts and per-row values, not polynomial coefficients). |
| D57 | 2026-10-07 | **Drift per stripe in measurement order:** stitch only images measured one after the other (`measurement_pairs`: within each stripe along y, plus the end/start images between consecutive stripes of a serpentine; `stitch_tiles(only_pairs=...)`), a drift curve per stripe for dx and dy from those coordinates (`stripe_drift`: the stripe's curve minus one straight line through all stripes' rows, which is left for the global affine), stitch everything on the drift-corrected coordinates, then RANSAC and the moving window. **No affine of any kind before the drift correction.** UI: *Stripe drift curves* tab (`StripeDriftView`) and the result as the extra set *stripe drift corrected*. | User's specification: the stripes are measured at very different times, so their drifts differ; tie only consecutively measured images first, so each stripe's drift is seen on its own; remove it, then use all overlaps; the intermediate curves must be visible for diagnosis. User rule: drift correction comes before any affine, because an affine applied to the initial data can remove real pattern errors (a proposal to measure per-stripe curves from the residuals of a least-squares affine was rejected for this reason). The serpentine start (first stripe from the bottom) is an assumption (setting); on the test data the derived end/start pairs are exactly the consecutive files. |
| D58 | 2026-10-07 | **Translation + rotation stitching** (`solve_tile_rigid`, `stitch_tiles_rigid`, first-order rotation, same ties as the translation stitching) and the **stitching residuals** of every input × tile model (`pair_residuals`, `pair_kind`, `residual_summary`; tab *Stitching residuals*, `StitchResidualView`). `intrafield.overlap_slopes` uses `pair_kind`. Registration results stay on translation stitching. | User: with corrections applied before stitching, a tab showing the stitching residual should tell whether they are valid (lower residual than the raw input); translation + rotation added to diagnose all combinations of stitching. User asked why the stripe drift curves do not meet at the joints: verified that the tied images agree on their shared contacts and that the gap equals the error's x-slope times the stripe distance. |
| D59 | 2026-10-08 | Stitch viewer: the *large spread* rings start **off**, like the other heavy layers; only the failure markers (overlaps not matched, tiles not stitched, tiles not measured) stay on. | User: the viewer opens on the nominal placement, where nearly every overlap contact has a large spread (nothing is stitched yet), so the magenta rings flooded the view and slowed the start; flags should be off unless there is a severe error such as a stitching failure. |
| D60 | 2026-10-08 | `read_contour_folder` reads every `*.csv` that has the four contour columns and leaves out (with a warning naming them) the other CSVs of the folder, instead of failing on them. | The real folders also hold the measuring tool's summary tables (`AffineCoefficients_*.csv`, `AvgDiff*_ByDesign*.csv`, `DiffAllPoint.csv`); the user had narrowed the file pattern to `CD*.csv` locally. Checking the columns works for any naming. |
| D61 | 2026-10-08 | **External reference measurements** (§4.6): `read_external` (X, Y mask µm, dX, dY nm, no header; optional sign flip) and `AnalysisWindow.add_external` / `RegistrationView.add_external`: row means as lines with symbols in the row profile after the affine; on the 2-D maps, each site as a marker filled in the maps' colour scale with a white ring, hover for values, toggled with the set's dx / dy boxes. | User: external txt results should be overlaid on ours, literally in the row plots; the map design was left to me. Markers in the same colour scale let the eye compare at every site while the contact map stays visible around them; a separate difference map would need SEM values interpolated at the sites. Units and frame from the user; sign unknown, so it is chosen per file; the data are a reference residual to compare with our result after the affine (user). |
| D62 | 2026-10-08 | The notebooks' flows become library functions: `analysis.py` (`Settings`, `analyse_contour_folder`, `analyse_images`, shared `_analyse`, `AnalysisResult`) and `view_analysis.analysis_window(result)`. Image records are passed in (the metadata reader stays remote-only). A contour-CSV folder without image centres skips the in-image correction instead of failing. | User: a standalone analyzer UI that loads folders, instead of the notebooks holding the whole pipeline in memory; it must stay compatible with the notebook UI (the notebooks and `AnalysisWindow`'s signature are unchanged). One shared core keeps the two input kinds from drifting apart; a missing optional correction should not cost the user the whole run. |
| D63 | 2026-10-08 | **Results file**: an `AnalysisResult` saved as one compressed `.npz` of named plain arrays plus a JSON `info` text (`results_file.py`), with a format number; no pickle; images not saved. | User: results should be saved and loaded later instead of living only in notebook memory. Pickle would be shorter but runs code on load and breaks silently when the result classes change; named arrays stay readable with numpy alone. Compression halves the file (114 → 51 MB on the test data) for ~6 s of saving, done in the background by the app. |
| D64 | 2026-10-08 | **Standalone analyzer app** (`app.py`): open a contour-CSV or image folder (settings dialog per kind, background run with a log), open / save results, load external measurements; the image metadata reader is imported only when an image folder is opened. | User: the analysis UI of the notebooks as the first version of a standalone UI where folders are loaded for analysis; both kinds of folder (user's choice); results saved and reloaded; external measurements overlaid. A daemon thread rather than a QThread, so quitting during a run cannot crash on a running thread. |
| D65 | 2026-10-08 | External reference on the error maps: a hollow ring in the error's colour around the contact at each site, white edges, growing with the zoom like the contact disks, replacing the filled markers of D61. | Seen on the local test data (a field of 16 × 650 µm): zoomed out, 12 px filled markers hid the whole contact column; zoomed in, a marker was the size of a contact disk and could not be found. A ring keeps our contact visible inside the reference colour. |
| D66 | 2026-10-08 | Stitching residuals of **both corrections before stitching**: the in-image corrected points, stitched in measurement order, their own drift per stripe measured and removed (D57), then stitched over all overlaps; translation and translation + rotation (`stitchings["in-image + stripe drift corrected"]`, both notebooks and `analysis._analyse`; only for the stitching residuals, no registration set). The *Stitching residuals* tab's check boxes are laid out two per row. | User asked for the cases "in-image correction, drift correction, translation" and "... translation + rotation". The drift is the drift per stripe: the global drift curve (D54) is applied after stitching and has no stitching of its own. The drift is measured again on the corrected points, since the in-image correction changes what the stitching in measurement order sees. Two per row because eight boxes in one row needed ~2000 px with Windows fonts, wider than a full-HD screen, and the tab sets the whole window's minimum width. On the local test data the run takes ~8 s longer (43 → 51 s); the other results are unchanged (bit-identical). |
| D67 | 2026-10-08 | Set **in-image + stripe drift corrected, translation + rotation**: the stitching of D66 with each tile placed by its translation + rotation (`stitch_tiles_rigid` affines as the corrections of `paired_errors` / `design_errors`), merged, then RANSAC and the moving window; an extra set (lines, row pitch, own tab) in `analysis._analyse` and both notebooks (sections 11b / 12b). Window: a set has the same colours in every tab (`add_rows(colors=...)`, 4th pair red / blue); the `RegistrationView` Show bar holds three sets per row; `PitchView` cycles its colours (it dropped sets beyond its five). | User asked for the methods "in-image, stripe drift correction, translation + rotation" with the global RANSAC and with the moving-window affine (overriding D58's translation-only registration for this set). On the local test data the tile rotations are small (tens of µrad) but add up along the long stripes into a bow of the tile centres in x, the same in all stripes; the global RANSAC cannot remove it (its dx error about doubles against the translation sets), the moving window does (on par with them); dy improves slightly. The same integration made the per-tile affine stitching fail (D53). Colours were assigned per tab by order, so a set without a moving-window result shifted the colours of the next sets in the moving-window tab; a fourth set would have reused the first set's pair; five sets in one Show row needed 1431 px with Windows fonts. |
| D68 | 2026-10-12 | Metadata CSV: read as `utf-8-sig`, then `cp949`, else `ValueError`; no lossy fallback. Done by the remote agent (task T003, `io/metadata.py`). 1024 × 1024 px images need no code change: the pixel size comes from FOV / image width. | User: a new folder failed with a UTF-8 decode error in the app; its images are 1024 px at the same FOV. A `latin-1` or `errors="replace"` fallback never fails, so it would garble non-ASCII file names silently. |
| D69 | 2026-10-12 | `outward_normals`: where the tangent has length 0 the normal is (0, 0), so `refine_edges` leaves that contour point where Otsu put it. | User: "invalid value encountered in divide" at `edges.py:26` on a 1024 px dataset. Reproduced with a disk plus a 1 px wide spur: the contour runs out and back along the spur, so at its tip points k ± 3 coincide; the 0/0 normal made that point, and with it the contact's refined centre, NaN. The spur encloses no area, so keeping its tip unrefined does not move the centroid. |
| D76 | 2026-10-15 | Stitch viewers: the **magenta failure layers** (overlaps not matched, tiles not stitched, tiles not measured) also start **off** (overrides D59); the status line still reports the failures in magenta. Each stitching's warning names the stitching (`label`), and a measurement-order stitching that leaves tiles unplaced logs a `!!` line saying they stay nominal and the stripe drift is measured without them. | User: a run warned "284 tiles could not be stitched" at one line of `analysis.py` while the Stitching tab looked stitched and its magenta check boxes did nothing; they had wanted the boxes off by default, not on. Checked here: the boxes do hide and show their items (synthetic run with one tile unplaced); they had nothing to show because the warning came from another stitching, most likely the measurement-order chain of the stripe drift correction (open chain-break problem, seen before on Chip1-1-3), while the tab shows the all-overlaps stitching, which routes around a bad tile (inference: the user's log was not seen). Every stitching warned from the same line with the same "SEM" label, so the warning could not tell which one failed. |
| D75 | 2026-10-15 | **Stitching method** `stitch_method`: *all overlaps* (default, unchanged) or *measurement order first* (pairs measured one after the other stitched first; every overlap then paired at the offset they predict, `stitch_tiles(guide=...)`); a check of the raw SEM stitching against the measurement-order stitching logs pairs fitted more than `max_match_nm` off (`slipped_pairs`); the method is chosen on the *Stitching* tab too, where a change computes the stitching and everything after it again (`analysis.restitch`) and replaces the window. Contour CSVs without any contact now fail with a message. | User: on one dataset the last image of stripe 2 was stitched to the first image of stripe 1 one contact row off. Diagnosis: pairing by nearest contact within the 25 nm gate cannot tell rows apart on a periodic grid; reproduced (pitch 100 nm, real offset 85 nm → fitted +15 nm, used without warning); that pair is the one taken furthest apart in time between the two stripes, so the drift between them is largest (inference; the real offset was not measured). The user asked for the fix to be optional, chosen on the first (Stitching) tab, with the later results computed again for the chosen method. Images measured one after the other are seconds apart, so their offsets stay far below half a pitch; their stitching predicts the other overlaps' offsets to a few nm. A chain broken by an unmatched consecutive pair leaves the tiles beyond it unguided (paired at the nominal placement, as before; D57 known limit). Synthetic run (2 stripes × 3 tiles, 17 nm drift per image): all overlaps off by up to 48 nm with the pair flagged; measurement order first 0.1 nm. |
| D74 | 2026-10-15 | *Contact size* tab: check boxes **SEM images** and **Design** show the tiles' SEM images below the maps and their `.oas` outlines above them, placed with the corrections of `ransac_placement` like the contacts; off at start; with the images on, the map is transparent between the contacts. | User asked for the design and the actual SEM image behind the colour map, each switched on / off with a check box. Off at start (quiet defaults; one image item per tile and map is slow to make, so they are made on first use). The map was opaque black between contacts on purpose (an old note: transparent pixels drew as grey blocks); synthetic run with the software renderer: no grey blocks, SEM centres within 0.4 nm of their map dots; the OpenGL viewport could not be captured here, so it is not checked. |
| D73 | 2026-10-15 | Contact sizes **measured on request** (tab *Contact size*: ROI width + *Run measurement*, background thread, `measure_contact_sizes`), no longer during the run; the `.oas` outlines are read only then, one merge per file (`pipeline.design_outlines`); `DesignContacts.outlines` and the `cd_roi_width_nm` setting are gone (the ROI width is chosen in the tab and stored in `ContactSizes`). **Checks before the run**: `check_settings`, `check_image_records`, 3 sample tiles matched first, and a stop right after loading if no tile matched. No GPU. | User: processing became much slower after D71, and a run that could not work (0 measured contacts) failed only after all tiles were loaded; asked for fast checks first, a faster measurement (GPU to be considered), and the measurement optional, run from the tab. Measured on a synthetic `.oas` (400 contacts × 16 fractures): D71 added a second merge plus a Python vertex copy to every tile load (~35 ms per tile, next to 57 ms for `read_contacts`); the chord arithmetic is ~1–2 ms per tile, so a GPU has nothing to speed up (the cost is in KLayout and in Python, not in the math). Threads gave no gain (1.02× with 4 or 8 threads: KLayout holds the GIL); processes might, not built. Measuring on request costs the run nothing; one merge per file instead of two. Not measured on the real data. |
| D72 | 2026-10-15 | Image ↔ `.oas` pairing by the **trailing number**: the `.oas` in `ContourCAD/` whose name starts with `SA` and ends in the same number (as an integer) as the image key (`CD<number>`); several matches → `ValueError`, none → `None`. Replaces the "contains" rule of D41; `io/metadata.py`, task T004 (remote agent). | User: the analyzer failed because no `.oas` was found; in every folder the design files have the prefix `SA` and the same trailing number as the `CD` image name, so a name containing the image key never exists. The prefix is a constant in one place; the number is compared as an integer so a different zero padding cannot break the pairing (to be confirmed on the real data, T004 §4). |
| D71 | 2026-10-15 | **Contact size** (area, xCD, yCD) for image runs (see "Contact size (D71)" in §6): SEM from the contacts' contours, design from the `.oas` outlines, chord CDs through the centre with an optional ROI band (`cd_roi_width_nm`, chords one SEM pixel apart, averaged); merged per contact through the overlaps; a *Contact size* tab with selectable maps and SEM / SEM − design; N/A for contour CSVs. `RegistrationView.rasterize_maps` drawing moved to `view_registration.draw_map`, shared by both views. | User asked for a contour area map in the analyzer UI, with a choice of area / xCD / yCD or all three. User answers: images and .oas compute the values, contour CSVs N/A; CDs as the chord through the centre with an optional ROI width and averaging (the proposed band of chords one pixel apart, 0 = single chord); both SEM value and SEM − design, switchable. Its own tab, since a map of another quantity cannot be a line on the existing plots. The colour scale is a percentile range in both modes: a size bias between SEM and design is expected (synthetic run: SEM disc 1279 nm² vs design square 1600 nm²) and a scale centred on 0 showed one uniform colour. Chord span between the outermost crossings, so a notch inside a contour does not shorten the CD (choice made here, not asked). All contours of a tile in one vectorised pass: ~3 ms per tile side (single chord), ~8 ms with a 10 nm band, for 400 contacts. |
| D90 | 2026-10-16 | Tile loading logs one `Timing:` line (`analysis.timing_summary`): wall time, the median seconds per step of `process_tile` (read, detect, refine, oas, and the design matching split into contacts = normal-tone read, shift = its shift search, reversed = the tone-reversed read and search, 0 if not tried; `TileResult.seconds`), the wall time the steps alone would take (sample tiles one by one, the rest ÷ `worker_count()`) and on how many tiles the reversed tone was tried; a wall time well above it is time outside the steps (worker start-up, sending the tiles back). Always logged, no setting. | User: turning *Stitch with the refined centres* off did not speed up the release; it only selects which centres the stitching uses, the refinement always runs (the design matching uses the refined centres). User wanted the step times in the analysis log rather than a remote task (T006, reverted). First real run (1494 tiles, 8 workers): 157.0 s wall, steps alone 142.4 s; per tile, median s: read 0.043, detect 0.093, refine 0.236, oas 0.019, match 0.363 (total 0.756). So matching, not refinement, is the largest step; the match was then split into its parts to find out why. |
| D89 | 2026-10-16 | Stripe detection (`pipeline.tile_stripes`) no longer uses the tile width: a new stripe at the gaps of the sorted centre x above the largest ratio between consecutive sorted gaps (floored at 1 nm), if ≥ 10; else one stripe. Contour CSVs use the exact image centres (DesignX_Add − DesignX) when every file has them. The release stores the stripes in `ReleaseResult.stripe` (cache format 2: older saved analyses run again) and uses them for the scan direction, the measurement-order stitching and the Registration stripe selector. | User: an image dataset with several stripes showed one ('Stripes: all 1'); stripes can overlap by more or less than half a tile, so the old rule (jump > half the median tile width) merged stripes overlapping by more than half. User: within a stripe the metadata x are equal up to floating-point differences, not nm; chose the gap break over a small width fraction. On the 9 local contour-CSV folders, bounding-box centres split stripes with the gap rule (7–8 instead of 6 in 5 folders); image centres give 6 in all (old rule: 6). The same wrong stripes also fed the measurement-order pairs (stitching method 'measurement order first', slipped-pair check) and the scan direction of that dataset, and the research app's stripes; fixed with it. Without image centres a contour folder falls back to the bounding-box centres (may split stripes). Not yet confirmed on the user's image dataset. |
| D88 | 2026-10-16 | Release Registration tab: the arrow map and the dx, dy colour maps in **one figure side by side** (no Show switch), linked by centre and scale; arrows **coloured by their length dr** (jet, own colour bar) and the arrow map's profiles are the **mean dr** per row / column (dr also in the profiles table); **grid lines on every map, on at start**; a **Stripes** selector restricts the maps and profiles to the contacts of the ticked stripes (by x range; overlap contacts in both stripes; no refit). | User's refinements. Stripe membership from the stripes' x ranges because the merged contacts (and the saved result) no longer know their tiles; the cache format stays. Overlap contacts in both stripes and colour scales recomputed from the contacts shown were chosen here, not asked. dr profile = mean of |error| (not |mean error|); in grid mode an arrow's colour is the length of the cell's mean vector (= the arrow's length). Arrow map titled 'Δr: …' like the others: pyqtgraph sets a title's width as its column's minimum, so a longer title made that map wider (seen offscreen). Not yet seen on the real data. |
| D87 | 2026-10-16 | Tile-loading workers capped at **half the logical cores** (`analysis.worker_count`, at least 1), about one per physical core, instead of one per logical core (D85); the count is logged. | User: the release ran out of memory (numpy `ArrayMemoryError`, 21.2 MiB for a (68057, 41) float64 array: the edge-refinement profiles, ±5 px in 0.25 px steps) partway through loading 1494 tiles of 2048×2048 px, on a 32 GB PC with 8 cores / 16 logical (21 GB in use), a folder that loaded before D85. The kept images alone are ~6 GB as before; D85 added one whole-tile working set per worker, 16 at once. D85 measured the gain scaling with physical cores, so halving costs little speed. Per-worker peak memory not measured (user chose to go straight to the cap); a cap from free RAM would need it and a new dependency (`psutil`), not built. Not yet confirmed on the real data. |
| D86 | 2026-10-15 | `design_errors` chooses each tile's pairs at the **nominal placement** (its points before the corrections; `pair_points` for sets whose points already hold a placement, used by the research app for every set) and computes the errors at the given placement, as before. Shift scoring, clear winners and the neighbour choice (D48–D50) unchanged; `match_shift_nm` and `neighbour_jump_nm` are now nominal-placement shifts. `design_search_nm` stays (100 nm) with a new meaning: the largest offset of one image against its own `.oas` (stage error). | User: on a dataset with more tiles over a larger area (same FOV), tiles far from the centre had no arrows at all, ~50 nm errors next to them. Diagnosis (code, synthetic): the stitching moved far tiles more than 100 nm from their designs (an SEM scale error adds up across the overlaps; corrections average zero, so it grows from the centre), and the search after stitching, centred on zero, then found no shift (coarse pitch: no arrows) or one a row off (fine pitch: errors fold back). The corrections move a tile's pairs without changing them, so the pairs are chosen where the offset is only one image's stage error. User chose to keep the setting (not derive it from the pitch): future patterns may not be regular. Synthetic, 21×21 tiles, 0.5 % scale error: pitch 250 nm, tiles without arrows 153 → 0 of 441; pitch 130 nm (0.3 %), tiles whose shift jumps more than P/2 from a neighbour 16 → 0, errors keep the full ramp. Trade-off: an image whose stage error exceeds P/2 on a regular array now relies on the neighbour choice in the nominal frame. Not run on the real data. The cause of the scale error itself is open. |
| D85 | 2026-10-15 | Image tiles after the 3 sample tiles are loaded (`process_tile`) **in worker processes**, one per CPU core (`ProcessPoolExecutor`, default size), in `analysis.load_image_tiles`, so the research app and the release both gain; results and their order unchanged (test: parallel = one by one). Workers get plain values (paths, numbers), not the metadata records (their type is the remote's). | User: the release spends long on *N of M tiles loaded and detected*; core counts vary and are small. Tiles are independent; threads gave no gain in D73 (GIL), processes were noted as untried. Measured here on copies of a 1024×768 px sample tile (35 ms each, 6 physical / 12 logical cores): 1000 tiles 29.4 s serial → 9.1 s (×3.2) including ~1.5 s worker start-up, ×4 once started; returning the images costs little (7.2 s vs 6.5 s returning nothing); chunk size and one OpenCV thread per worker made no clear difference, so neither is set. The gain scales with physical cores; not measured on the real data. |
| D84 | 2026-10-15 | Several candidate metadata CSVs in a folder are **merged** instead of rejected: same header required, exact duplicate rows dropped, a row kept only if the file-name part of its *File name* (after the last `\` or `/`) equals a `.jpg` in the folder (case-insensitive); an image left with two different rows → `ValueError`. One CSV behaves as before. `find_metadata_csv` → `find_metadata_csvs` (list); contract test updated. Remote agent, task T005. | User: the release app failed on folders whose metadata is split into many CSVs (sometimes one per image); proposed merge, drop duplicates, keep rows whose File name contains a jpg name (paths differ). The file-name part is compared instead of a plain 'contains' so `CD1.jpg` cannot match `XCD1.jpg`; conflicting rows fail loudly rather than one being picked silently. |
| D83 | 2026-10-15 | Release: *Download and analyse* a remote folder into `<download folder>/<host>/<remote path>` (default download folder `Documents/MaskReg data`), only new or changed (size) files, `.part` then rename; analysed again only if a file was downloaded, else the saved analysis opens. Pre-set FTP settings in `maskreg.ini` (`MASKREG_CONFIG`, else `%LOCALAPPDATA%/MaskReg/`), auto-connect when a host is set. | User asked for downloading a folder to analyse, a proposal for where to keep the data, and pre-set server settings so users need not type them. A fixed place per remote folder lets the existing per-folder autosave (D77) be found again, so reopening is instant and nothing is fetched twice; the host level keeps servers apart. Size is the only change test (MLSD and LIST both give it; times differ in format and precision between servers). The password in the ini file is plain text: whoever can read the file can read it, the same exposure as plain FTP itself, which sends it unencrypted; the Windows credential store (`keyring`) would be a new dependency, not asked for. |
| D82 | 2026-10-15 | Release: an FTP browser (*File → Browse FTP server*): connect, list a remote folder's folders and files, walk up and down. Standard `ftplib`, plain FTP; `MLSD` with a `LIST` fallback (Unix and DOS formats); blocking calls with a 15 s timeout; password not stored. Tests against a local `pyftpdlib` server (dev extra). | User asked for FTP connection and a navigable remote folder view as a first step, more to follow. Plain FTP because that is what was asked (FTPS is `ftplib.FTP_TLS`, SFTP would need another library). Blocking instead of a background thread: listings are short, and the timeout bounds the wait; no new runtime dependency. |
| D81 | 2026-10-15 | Release Registration tab: grid-averaged values for colour maps and arrows (one shared *Values* control), grid lines on the colour maps, per-graph Copy / Save image (menus and right-click), row profiles show their y values. | User asked for grid-averaged arrows (offered in D80 because per-contact arrows are unreadable at full view), grid-averaged colour maps, grid lines and a copy button per graph. One control for both displays keeps them consistent. Menus instead of one button per graph: six graphs plus the colour bar would crowd the bar. Rendering a part of the view alone at 2× misplaced the axis texts (seen here), so a graph is cut out of the whole figure rendered at 2×. The row profiles' y values were hidden beside the map but are needed when a profile is copied alone. |
| D80 | 2026-10-15 | Release *Registration* tab (was *Registration, moving window*): arrow map as an alternative display; row profiles rotated beside each map, new column profiles below it, both following the map; the profile without affine removed; SEM image opacity slider on the Stitching tab. | User's refinements. Each colour map's profiles show its own component (a profile beside the dx map is the dx mean; the arrow map has both); this split was chosen here, not asked. The arrows are one per contact, not binned: at the full-field view of a large dataset they are too dense to read (the profiles carry the trend); zoom in, or raise the scale. |
| D79 | 2026-10-15 | Release: the scan direction of the first stripe is detected from the file numbers (`scan_direction`, `stripe_directions`) instead of the `first_stripe_upward` parameter; non-alternating stripes are logged. | User: did not know what the setting does; confirmed that the file numbers follow the measurement order. It only matters for the 'measurement order first' stitching and the slipped-pair check. The image file name is used, not `tile_id` (derived on the remote, format not fixed here). |
| D78 | 2026-10-15 | Release (MaskReg): default `max_match_nm` 100 nm (`release/analysis.DEFAULTS`; research app unchanged at 25 nm); the contact layer is detected in the `.oas` files (`detect_layer`: the only (layer, datatype) with shapes, over every file; several or none → error) and is no longer a parameter. | User: the layer differs between datasets and should be detected, with an error if there are several; asked for the 100 nm default. Note (told to the user): 100 nm is above half the contact pitch (~65 nm for a 130 nm pitch), against the S4 rule g < P/2: the gate only rejects pairs, so results are the same while real overlap offsets stay well below P/2 (synthetic: identical corrections and residuals at 25 and 100 nm), but an overlap offset by more than P/2 is now paired one row off instead of being reported as not matched, and the slipped-pair check (tolerance `max_match_nm`) misses slips smaller than 100 nm. Every file is read for the layer (not only the first), so a dataset with one deviating file stops before the run instead of flagging tiles. |
| D77 | 2026-10-15 | **Release app MaskReg** (`release/`): Stitching and Registration, moving window tabs only; error measured − design; only the uncorrected set (no in-image, drift or stripe drift corrections, no RANSAC); white presentation figures with Copy / Save image at 2×; parameters behind *Analysis → Parameters*; autosave to `MaskReg_analysis.npz` in the data folder (user cache folder if read-only), opened instead of a new run; *Restart analysis*; OpenGL drawing when it works, else CPU; a status bar with step, percentage and time. The Stitching tab always shows the points; the SEM images are loaded on request. | User: a separate minimal UI for real users, graphs usable in slides, results cached per folder, rerun on request, must run with and without a GPU (user: drawing, not computation), the old status bar showed only a moving blue bar. The error was already SEM − design, so nothing changed there. Name: the user found "AffineRANSAC" misleading for the release (no RANSAC in it) and asked for a new one. Results files hold no images (gigabytes), so the points view keeps a fresh and a reopened result the same (user's choice) and images are read only when asked, downsampled to a memory budget (~1,000 tiles of 2048² px would be 4 GB). The research app is untouched so the full analysis stays available. |
| D70 | 2026-10-13 | `RegistrationView` row profiles: **Copy rows** (clipboard, tab-separated) and **Export rows CSV...** (`row_table`): the lines whose Show box is checked, one y / dx / dy column trio per set and plot, blank below a set's last row; values from the plot items at export time. | User: needs to export the dX/dY line graphs, as CSV and to the clipboard. Wide columns because the sets (corrected subsets, external sites) have different rows; the Show bar picks what is exported; tabs paste into Excel as columns. |

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
   synthetic data? *Partly answered (2026-10-06):* the user asked for a full per-tile affine
   stitching to compare with translation stitching on real data (D51). Synthetic note: on tiles
   without their own distortion the per-tile affine fits tie noise from the edge strips and raises
   the 3σ (2 × 2 CSV tiles, 0.1 nm noise: 0.30 → 0.42 nm after RANSAC). On real data it failed
   (D53: a shared in-image distortion integrated into a tile-scale ramp); the user chose to try a
   shared intrafield map + per-tile translation (D53), with the robustness concern that mask errors
   repeating at the image spacing would be absorbed.
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
12. **Pairing:** *Answered:* by a **file-naming convention**: image `CD<number>`, design `SA…<number>.oas`
    with the same trailing number (user, 2026-10-15; D72).
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
