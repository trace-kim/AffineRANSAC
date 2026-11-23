# AffineRANSAC — Software Specification

> **Audience:** human developers and AI coding agents working on this repository.
> **Status:** living spec (first written 2026-09-30, before any code). The code is in
> `src/affine_ransac`; a change to it updates the section it touches. Why and when things changed
> is in `docs/DECISIONS.md`; what is built but not yet run on the real data is in `docs/OPEN.md`.
> **Rule for agents:** Treat this document as the source of truth for intent and architecture.
> If the code has to deviate from it, update this document in the same change and record the
> reason as a new row of the Decision Log (`docs/DECISIONS.md`; §12 here until 2026-11-16).
> Unresolved items live in §13 (Open Questions). Do not silently
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
| Per-tile affine + global affine; one in-image distortion shared by all images (polynomial formula D145; the node map of D53 removed, D146) | Other higher-order distortion models |
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
(`pipeline.process_tile`), never a constant. The CSV may be UTF-8 or cp949 (task T003, D68). A folder may hold **several** metadata CSVs (e.g. one per image): they are merged, exact duplicate rows dropped, and only rows whose *File name* (file-name part) names a `.jpg` in the folder kept (task T005, D84). A point measured **twice** (the SEM's pattern matching failed and it measured again; both images kept) has two rows with the same *Chip X/Y* and the same `Measure Point No`: the reader keeps the row whose image has the higher trailing number, the later measurement, with its own `.oas` (task T007, D94). `analysis.check_image_records` refuses records whose image centres are closer than 1 nm (`same_position_records`), before any tile is loaded. The reader (`io/metadata.py::read_tile_index`) and the
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
**Contrast check (D93, off by default since D103):** `process_tile(min_contrast)` (setting `min_contrast`, default 0 =
off) drops detected contacts
whose mean grey level differs from a 3 px ring around them by less than `min_contrast` noise σ
(`contact_contrast`; σ = 1.4826 × MAD of the image minus its blurred copy), before refinement. Thresholding always
cuts regions out of an image without pattern; they have about one σ of contrast, real contacts more, but on noisy
images only a few (synthetic, 20 grey levels of noise: 4.3–4.5 σ, contacts plain to see), so a fixed threshold can drop
every real contact (D103). Logged per run (`contrast_summary`): contacts dropped and tiles left without any contact;
a `!!` line when it removed every contact of a tile.
`method` (D39, D40; pipeline `detection`, notebook `DETECTION`): `"otsu"` (plain, default),
`"otsu3"` (three-class Otsu, `skimage.filters.threshold_multiotsu`: hole | background | bright
band; threshold between hole and background) or `"band"` (the bright band: pixels above the
background | band threshold; candidates are the non-band regions, 4-connected; the background
touches the border and is dropped, so the regions enclosed by a band remain; regions with solidity
< 0.9 are dropped, removing background pinched off between nearly touching bands). `"band"` does
not need the hole interior to be darker than the background. Wide bands break plain Otsu: it then
separates background from band. `tile_index.ipynb` shows the histogram, thresholds and the mask of
every method for one tile.

**Edge refinement (D22, D23, D122):** `features/edges.py::refine_edges(rule)`. Every detected contour
point is moved along its own **outward normal** (tangent from neighbours ±3 points) within a **fixed
±`search_px`** (default 5 px) to its edge on the lightly smoothed image (Gaussian, 1 px). Where the
edge is, is the setting `refinement` (images):
- **mid-level** (default since D122): each contact has two levels of its own, its interior (median of
  the smoothed image over the pixels inside its detected contour) and its band (median over its points
  of each profile's brightest value from the start outward), and its edge level half-way between them
  (the 50 % threshold edge of CD-SEM metrology). Each point goes where its profile rises through that
  level, the crossing nearest its start, sub-pixel by linear interpolation. A point without a crossing,
  or further from the median of its ±3 neighbours than 4 robust σ (of the contact's points) + 0.5 px,
  takes that median (`without_isolated`): a real deviation of the edge spans several neighbouring
  points and is kept. Bright contacts: the same on the negative image.
- **steepest gradient** (D22, D23; the only rule before D122): the steepest dark→bright rise
  (bright→dark for bright contacts), with a sub-pixel parabola fit. Differentiating amplifies the
  noise, and every point is decided alone, so on noisy, low-contrast images single points jump.
- **threshold** (D130): where the profile crosses the detection's threshold (`DetectedContacts.threshold`, the
  Otsu level), one level for the whole image: the detected outline placed sub-pixel. For images without a bright
  band (wafer ADI), where the mid-level rule's upper level, the brightest value outward, is a noise spike. One level
  fits only an even background: flatten it first (`background_px`) if the brightness varies. Points without a
  crossing, or isolated, are replaced as for the mid-level (`without_isolated`).
- **none** (D124): no refinement; the detected contours and centres are the measured ones (`process_tile`
  keeps `refined` = the detection, so the design tone, the stitching, the contact sizes and *Load contours*
  all use them). Until D124 this was a check box beside the rule, `stitch_refined` (unticked: the detected
  centres; its tone check still used the refined ones). Stored values in the old name are read with
  `analysis.current_names` (unticked → none), in saved runs and in remembered parameters.

After mid-level refinement the detection method no longer decides where the edges are, only which contacts
are found and how many false regions (synthetic, D124: from otsu, otsu3 and band outlines the refined centres
of the same contacts agree to 0.004–0.06 nm). With band detection the refined contour lies on average 0.2 px
inside the detected pixel outline (±0.3 px), so on the Stitching tab it covers it; from otsu / otsu3 outlines,
which sit lower on the edge, it lies 1.4–2.5 px outside.

**Background flattening (D130).** `background_px` (0 = off, the default; else ≥ 3): before detection,
`pipeline.find_contacts` replaces the image by `pipeline.prepared_image` = `features.contact.flatten_background`: the
background is a grey-level closing with a disk `background_px` across (an opening for bright contacts), which fills
every narrower contact with the brightness around it, smoothed with a Gaussian of σ = 2 × `background_px` (the closing
of a noisy image is noisy itself; σ = ½ × the disk added scatter at the contact scale, synthetic: 0.403 → 0.431 px at
noise 40) over the background continued past the image's edges with its own slope (odd reflection; a mirror bent a
ramp there and moved the contacts near the edges, those of the overlaps, by up to 0.1 px); the image minus it plus its
mean. Detection, the contrast check and refinement all see the flattened image. Synthetic, no band, dark disks 8 px
across, 512² px, a brightness ramp of 80 grey levels across the image: today's settings find 57 % of the contacts (one
threshold cannot fit; the rest merge with the darker background), flattened (disk 2.5 × the contact) 100 %, centre
scatter 0.127 px; without a ramp it changes nothing (6 px contacts, noise 8: 0.081 → 0.081 px). Cost: a closing and a
wide blur, 0.1 s for a 2048² tile with a 20 px disk, 0.3 s with 40 px. The window shows the flattened image (D129).

**Upscaling (D131).** `upscale` (1 = off, the default; 1 to `analysis.MAX_UPSCALE` = 8): after the flattening,
`pipeline.find_contacts` interpolates the image to `upscale` times as many pixels per side
(`features.contact.upscaled_image`, bicubic, `cv2.resize`) and detects and refines the contacts in it, so a contact a
few pixels across gets a threshold outline with sub-pixel steps instead of a staircase of whole pixels. The upscaled
image covers the same area: its pixel (x, y) lies at ((x + 0.5) / f − 0.5, (y + 0.5) / f − 0.5) of the image, and
`in_image_pixels` puts the contacts found back in the image's pixel frame (areas / f²). The settings in pixels
(`detection_blur_px`, `min_area_px` and `max_area_px` in px², `search_px`, `edge_step_px`, `edge_blur_px`) and
`normal_span` (contour points, f times denser upscaled) stay those of the image as loaded: they are multiplied by f
(areas by f²), so that turning `upscale` on finds the same contacts and needs no other setting changed. The contrast
check works on the image not upscaled (its own noise; the 3 px ring), with the contours mapped back. Synthetic, no band,
512² px, 3 images each (rms per axis, ×1 → ×2 → ×4): detected centres 4 px contacts noise 5: 0.122 → 0.084 → 0.078 px; 6
px, noise 8: 0.122 → 0.086 → 0.081; 6 px, noise 20: 0.224 → 0.208 → 0.207; refined centres (mid-level) unchanged on
clean images (0.081 → 0.081) and slightly worse on noisy ones (6 px noise 20: 0.222 → 0.230; 8 px noise 40: 0.442 →
0.471). Cost, a 2048² tile with 23,104 contacts 6 px across: 5.0 s at ×1, 6.0 s at ×2, 7.3 s at ×4 (refinement is most
of it); peak memory per process 0.5, 1.0 and 2.0 GB (1,600 contacts: 0.17 and 0.75 GB at ×1 and ×4), and a run has a
worker process per two CPUs. The window shows the upscaled image and its threshold mask, stretched over the image's
pixel frame (D129). On the user's real wafer ADI images upscaling gave much better contours, factors above 4 looking
good (user, D132); the synthetic numbers above did not predict that.

**Denoising (D132).** `denoise` (none, the default; median, non-local means, total variation): after the flattening and
before the upscaling, `features.contact.denoised_image` removes pixel noise, each method with its own option: **median**
(`median_px`, the window width, odd, ≥ 3; `cv2.medianBlur`: removes isolated bright or dark pixels and spots narrower
than half the window, keeps straight edges where they are), **non-local means** (`nlm_h`, OpenCV's strength h in grey
levels; patch 7 px and search window 21 px, OpenCV's recommended sizes, not settings; `cv2.fastNlMeansDenoising`) and
**total variation** (`tv_weight`, on the grey levels / 255; Chambolle's algorithm, `features.total_variation`:
the pixels of `skimage.restoration.denoise_tv_chambolle`, compiled with numba, about 7 times as fast, D159;
rounded back to grey levels: flat regions with sharp edges between them).
The compiled loops are warmed once per process, with concurrent callers waiting for that warm-up (D163).
Detection, the contrast check and refinement all see the denoised image; the option of a method not chosen changes
nothing (greyed in the window). Denoised before it is upscaled: bicubic interpolation spreads each noisy pixel over f ×
f pixels with smooth transitions, so a median would need an f times wider window, and non-local means, made for noise
independent from pixel to pixel, no longer sees it as noise; work on upscaling noisy images denoises first (it avoids
amplifying the noise; super-resolution first makes the noise "extremely complicated to remove"). Why these three: median
(with Gaussian) blur is the conventional denoising that CD-SEM contour work compares against; non-local means and BM3D
are the classical patch-based methods SEM denoisers are benchmarked with; total variation (anisotropic diffusion is its
relative) is the edge-preserving variational family used on microscopy images. Not built: BM3D (its Python package
`bm3d` is free for non-commercial use only), bilateral filtering (it keeps isolated outlier pixels: their neighbours
differ too much in grey level to be averaged with them), deep-learning denoisers (unsupervised U-Net, Noise2Void:
reported best for CD-SEM contours, but they are trained on the user's images and need a GPU stack), Anscombe variance
stabilisation (needs the detector's gain and offset). Cost per 2048² image: median 3 px < 0.01 s, 7 px 0.09 s; non-local
means 0.9 s; total variation 3.8 s. Not evaluated on synthetic images (the user: they do not represent the real images);
the user judges the methods on the real images in the diagnosis window.

**Equalization for the threshold (D167).** `equalize` (none, the default; global, adaptive), a detection setting
for every pattern: after the detection blur, `features.contact.contact_mask` equalizes the grey levels of the
(upscaled) image, and the detection method chooses and applies its threshold on them. Only the threshold mask sees
it: the contrast check and the edge refinement work on the image as before, so the edges are still measured on
grey levels that follow the signal, not on remapped ones. **global**: histogram equalization over the whole image
(`cv2.equalizeHist`, computed as its table, `equalization_lut`, tested equal to it). The table never decreases, so
the mask is still the blurred image cut at one grey level, and the threshold returned (used by refinement
`threshold` and named in the diagnosis line) is that grey level of the blurred image. Otsu on a flat histogram
divides the pixels into about equal shares (three for otsu3 and band), so global equalization moves the threshold
towards the median grey level; it cannot follow brightness that changes across the image. **adaptive**:
contrast-limited adaptive histogram equalization (CLAHE, Zuiderveld 1994; `cv2.createCLAHE`): the image is cut into
`clahe_tiles` × `clahe_tiles` tiles (8, OpenCV's default; a count per side, so the same when upscaled), each mapped
by its own histogram in which no grey level counts more than `clahe_clip` times the average (2.0, the value of
OpenCV's tutorial; OpenCV's own default, 40, hardly limits the noise of a tile without edges), the mappings blended
between neighbouring tiles. The grey level that divides the image then varies across it, and the threshold returned
is a grey level of the equalized image: `check_settings` refuses adaptive with refinement `threshold`, which needs one
grey level of the image. The other way to a threshold that fits varying brightness is to subtract the background
first (`background_px`, D130). Default none: results, `ANALYSIS` 9 and `FORMAT` 3 unchanged. Not evaluated on
synthetic images; the user judges it on the real images in the diagnosis window.

**Contour settings (D128).** Every option of the detection and the refinement is a setting
(`analysis.CONTOUR_SETTINGS`), its default the value the code had before: `background_px` (0 = off, D130),
`denoise` (none, D132) with `median_px` (3), `nlm_h` (10) and `tv_weight` (0.1), `upscale` (1 = off, D131),
`detection`, `dark_contacts` (on), `detection_blur_px` (1.5), `equalize` (none, D167) with `clahe_clip` (2.0) and
`clahe_tiles` (8), `min_area_px` (20),
`max_area_px` (0 = no limit), `min_solidity` (0.9, band only), `min_contrast`, `refinement`, `search_px`, `edge_step_px`
(0.25), `edge_blur_px` (1.0), `normal_span` (3). `find_contacts` takes them by name only (keyword arguments).
`pipeline.find_contacts(image, **analysis.contour_options(settings))` finds an image's contacts with them (detection,
contrast check, refinement); `process_tile`, hence every run and *Load contours*, and the release's contour diagnosis
(D129) call it, so all of them find the same contacts. Detection counts the regions it leaves out, each under the first
reason that applies (`DetectedContacts.left_out`: touching the image border, smaller than `min_area_px`, larger than
`max_area_px`, less solid than `min_solidity`). Not settings: the mid-level rule's own constants (±3 neighbours, 4
robust σ + 0.5 px), the contrast check's 3 px ring, the three classes of otsu3 and band.

**Lines (D153; step 1 of the line & space plan, `docs/notes/2026-11-08-line-space-plan.md`).** Setting `pattern`
(contacts, the default | lines | contours | horizontal L/S, D155 and D170 below; MaskReg only, `analysis.RELEASE_ONLY`, in the *Contour detection* parameters and the
contour diagnosis window's *Pattern* group, not one of `CONTOUR_SETTINGS`). With lines, `pipeline.find_lines` finds
the **horizontal lines** of one image (`features/lines.py`, `LineEdges`): the same pre-processing (flattening,
denoising, upscaling) and threshold mask as for contacts (`dark_contacts`: lines darker than the spaces; off:
brighter), and a **line is a region of the mask that runs from the image's left border to its right border**; the
other regions are left out (counted). Each line has an **upper edge** (the space above it, facing up) and a **lower
edge** (facing down), measured at **every column of the image** as y(x) (`LineEdges.upper`, `.lower`, (lines,
columns), from the top down); where the line touches the image's top or bottom border that edge is missing in that
column (NaN). Detected: the topmost and bottommost row of the region in each column. Refined (`refine_line_edges`):
each point moves up or down **its own column** by the contacts' edge rule (`refinement`; mid-level: half-way between
the line's interior, the median over its pixels, and its band, the median over both its edges' profiles of the
brightest value outward, on the negative image for bright lines; `without_isolated(closed=False)`: an edge is open,
its end points have neighbours on one side only), with `search_px`, `edge_step_px`, and `edge_blur_px` **across the
lines only** (a Gaussian along y): each column's edge comes from its own pixels, so the profile along the edge is not
smoothed and neighbouring columns do not share noise. Not used for lines (greyed out): the area and solidity limits,
the contrast check, `normal_span`. Per edge: polarity (upper or lower), height across (y in each column), extent
along (the columns where it is found), **profile along** (`edge_profiles`: the edge minus the straight line fitted
along it, the roughness with the noise). The window's line gives per image the **edge roughness** (`roughness_px`:
rms of all profiles) and the **noise per column** (`column_noise_px`: rms of the differences between neighbouring
columns / √2; an upper bound where the roughness varies within a few columns; it reads low on a denoised or upscaled
image, whose neighbouring columns share noise, and the line says so), in px: the window has no pixel size. Upscaled,
the edges are given at the image's own columns (`in_image_lines`: the upscaled column at f·x + (f − 1)/2, or the
mean of the two beside it for even f). **Only the contour diagnosis window uses lines so far:** a run with `pattern`
lines is refused (`release/analysis.analyse_folder`).
Synthetic (`tests/test_lines.py`, bright lines, edges at known sub-pixel heights): mid-level within 0.05 px in every
column, also upscaled ×2 and ×3; the threshold rule at the Otsu level (not half-way where the lines cover a third of
the image, up to 0.3 px off) and the steepest gradient on the steepest pixel step (up to 0.63 px off: down a column
the sampled profile is linear between pixels, so its gradient is flat within a pixel); two images of the same lines
with their own noise (12 grey levels, contrast 120) differ per column by 0.188 px of noise, and `column_noise_px` of
one of them gives 0.189 px.

**Temporary horizontal L/S (D170, D171).** In MaskReg image folders, **horizontal L/S** takes the brightest
class of the prepared SEM image as the white separators. It uses the upper threshold of three-class Otsu,
or the brighter level in a two-level image. Four-connected interior pixels keep diagonal separators intact.
Divide the field into ten equal widths along x. In each slice, close the upper and lower edges of each complete
interior at the two cuts and measure its area centroid. A region clipped at the top or bottom, or joined to
another interior at a cut, is excluded locally. Where refinement crosses the two edges or leaves the field,
use the detected pixel boundaries for that column. The area limits refer to native pixels over a full band
width, calculated from the mean thickness in the slice. *Contour diagnosis* gives the counts from left to right.
The detailed run log gives the SEM and design counts for each tile. *Load contours* reproduces the outlines
of the saved analysis, including the full-width outlines of analysis 9. Clip the drawn and clear OASIS regions at the
same ten cuts and take the centroids of both tones wherever a complete band crosses a slice.

**Contours of any pattern (D155).** `pattern` **contours** (MaskReg only, as lines): `pipeline.find_contours` finds
the contours of one image with **nothing assumed about the pattern**: no shape, size or direction, and a region need
not lie inside the image (`features/contours.py`, `Contours`). The same pre-processing and threshold mask as for
contacts (`dark_contacts`: the regions outlined are the darker side; off: the brighter). **Every region of the mask**
within the area limits (`min_area_px`, `max_area_px`; the others counted as left out) gives its **whole boundary**:
its outer one and that of each hole in it (`cv2.findContours`, `RETR_CCOMP`). The boundary points on the image's
**outermost rows and columns are not edges** (the region runs out of the image there) and are taken out
(`inside_image`): a boundary that never gets there is one **closed** contour; the others fall into **open** contours,
each ending at the image border (runs of fewer than 2 points are dropped). So a contact inside the image gives one
closed contour, a contact cut by the border the part of its edge that is in the image, a line through the image one
open contour per edge, whatever its direction, a ring two closed ones. Per contour: its points in order
(`Contours.points`), at each point the **direction its edge faces**, the unit normal out of its region
(`Contours.normals`; tangent across ±`normal_span` points as for contacts, stopping at an open contour's ends; a
hole's normals point into the hole), whether it is closed (`Contours.closed`) and its region (`Contours.region`: a
region with a hole, or cut by the border, has several). Refined (`refine_contours`): each point moves **along its
direction** by the contacts' edge rule (`refinement`, with `search_px`, `edge_step_px`, `edge_blur_px`, the
contacts' 2-D blur): mid-level with two levels per region, as per contact (interior: the median of the smoothed image
over the region's pixels, `features.region_medians.region_medians`; band: the median over the profiles of all its contours of the
brightest value outward); `without_isolated(closed=...)` per contour. Not used (greyed out): `min_solidity`, the
contrast check. With `detection` band the mask is everything but the band, so each band gives two contours, its
inner and its outer side. Upscaled, the contours are given in the image's own pixels (`in_image_contours`). The
window's contour layers draw them (closed ones closed, open ones as they are; no centres) and its line gives the
number of contours, how many are closed and how many open. A run with `pattern` contours stitches the images by
these contours and stops there (S5, *Stitching by contours*, D156); a run with lines is refused. Synthetic (`tests/test_contours.py`, bright patterns
rendered at known sub-pixel positions): vertical lines through the image, one open contour per edge within 0.05 px,
facing out of its line; a line 30° off the x axis, both edges within 0.1 px; a ring, outer and inner radius within
0.1 px, the inner contour facing the centre; contacts (`render_tile`): the closed contours are the contacts
`find_contacts` finds (their middles within 0.2 px of its centres), and the contacts it drops as cut give open ones.

**Region medians (D165).** Count labels and copy the pixels of requested regions into contiguous groups, then
select the middle value(s) with in-place partitioning. This replaces the global label/intensity sort used by
`scipy.ndimage.median`; background and discarded regions are not copied into the value buffer. The median uses
the same float32/float64 arithmetic, including even-sized groups. Tests compare medians exactly and the full
refined contours with total-variation denoising and 4× upscaling enabled. The preprocessing, TV solver and
upscaling settings are unchanged; no pixels are dropped from any requested region.

**Refinement memory (D166).** Convert to one owned float32 image, blur into that same buffer and reverse its sign
in place for bright regions. Do not retain separate conversion, blur and signed images. `sample_profiles`
allocates the final profile array once and fills it directly: each remap builds only its own coordinate maps,
with at most 30,000 rows and about 2²⁰ samples per map. The output profiles and smoothed pixel values are unchanged.
A memory failure during refinement names the working-image dimensions, preprocessing settings and failed timed
step when available. Denoising, upscaling, edge samples and precision are not reduced to avoid allocation errors.

No radius or centre is assumed, so either rule works for any shape (circles, ellipses, rectangles).
The refined centre is the area centroid of the refined contour. A point without a tangent (its ±3
neighbours coincide, at the tip of a 1 px wide spur) is not moved (D69). Known limit: at sharply curved
edges (curvature radius of a few px) image blur pulls the edge slightly inward. Why: on real data the
`.oas` contours sit on the white band, while Otsu contours sit just outside the dark area. Note: a
uniform radial shift of a contour does not move its centroid, so the edge rule matters for centres only
where the edge bias is not the same all around, and through the noise. Runs saved before D122 have no
`refinement` and are read as "steepest gradient" (`analysis.saved_settings`, `BEFORE`): a setting a
saved run lacks takes what runs did before it existed, not today's default.

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
exists; the line edges of D153 and the contours of any pattern of D155 (edge points with the direction they face) are
found per image; the contours stitch the images (S5, *Stitching by contours*, D156) and are not yet used by S6–S8.

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
  Done for the stitching image by image (D133), where the offsets between images exceed half the pitch.

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
other tile gets a **NaN** correction (`StitchResult.unplaced`), and `stitch_tiles` warns ("stitching incomplete" only when tiles are left without a position; overlaps not matched alone: "every tile stitched", D109). The
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

**Stitching image by image (D133).** The third `stitch_method`. Pairing every overlap at the nominal placement
measures its offset only modulo the pitch, so once neighbouring images lie more than half a pitch apart (a 50 nm
grid: 25 nm; user's data 2026-10-26) the overlaps carry whole-pitch errors of their own, contradict each other around
a group of tiles, and the solve splits the difference: images off by part of a pitch, errors far too large there,
while every overlap "matched" and the stitching reported OK. No gate helps: every contact's nearest neighbour lies
within P/√2 (the user's runs at `max_match_nm` 50, 100 and 300 nm were identical). Instead
`pipeline.placed_image_by_image(points, centers, fovs, search_nm, min_matched)` (`max_match_nm`) places the images
one at a time (D138): the image with the most overlap contacts stays where it is; then, again and again, one image is
placed against all its placed neighbours at once, one shift for all its overlaps. Its candidate shifts
(`matching.candidate_shifts` of its overlap contacts against theirs, any size up to `max_match_nm`) are each refined
until they settle (`fitted_placement`: the pairs within the gate of D137 moved by their robust fit and paired again)
and judged by their **agreement** (`seen_pairs`, as in the check below: of the contacts the images should see of each
other, the share that pair). A shift competes if it pairs at least `min_matched` contacts, `LINED_UP` (½) as many as
the shift pairing the most (so a sliver of an overlap whose few contacts happen to pair cannot) and `MIN_AGREEMENT` of
what it should; of those, the one agreeing best and those tied with it up to counting noise (`ties`: fewer than
`MIN_EXTRA` = 2 more contacts without a partner, counted on the best's contacts, plus three times the square root of
the best's own number). Of the waiting images, one whose options are one shift (clear) is placed first, taking it;
only when none is left, one with several (tied: nothing in its overlaps tells them apart), taking the smallest, which
puts it nearest its stage position (D48); either way the one sharing the most contacts with the placed images first.
So on a repeating pattern every whole-pitch shift ties and the smallest is taken, as before; where the pattern does not
repeat (e.g. missing contacts), a shift a pitch off pairs only part of the contacts and loses, however small; and an
image whose overlap happens to repeat (one complete row) waits for a neighbour that decides (the rule of
`consistent_choice`, D49). No tolerance near the noise decides: the gate is a quarter of the contact spacing and the fit
is robust. An image without options waits for more placed neighbours; one never placed is NaN (then paired at its
nominal placement, as unguided). Then `pipeline.stitch_image_by_image` (D139) pairs every overlap where these
corrections put its two images, within the same gate (`seen_pairs`, a quarter of the contact spacing), and uses it
only if it lines up there: at least `min_matched` pairs and `MIN_AGREEMENT` (half) of the contacts the two should see;
its shift is the robust fit of those pairs (`fit_overlap`), so the precision comes from the fit, not from a gate near
the noise. An overlap that does not line up is left out (not matched, its *show why* says how many of its contacts
paired) instead of pulling its tiles apart; the solve is the usual one (`solved_stitching`). `overlap_tolerance_nm`
(default 2 nm, D135) no longer pairs anything: it is the offset the check below allows. **The check after it
(D137):** `pipeline.overlap_checks` looks at every
overlap where the stitching placed its two images, whatever paired it. Of the contacts each image should see of the
other (those inside the box of all the other's contacts, so the rows an image loses at its own frame are not expected
of it), the share that pair within a quarter of the contact spacing (`PAIRING_GATE` × `contact_spacing`, the median
distance to the nearest contact: wide against the noise, well below half the pitch), counted on the image with fewer
(`agreement`, so contacts one image failed to detect do not count), and how far apart the two images lie (the length
of the robust fit of those pairs, `fit_overlap`). An image placed wrong by more than the gate shows in the agreement,
by less in the offset; by a whole pitch on a repeating pattern it lines up, which does not matter (D105). Where the
pattern does not fully repeat, a whole-pitch slip still pairs most contacts, so the agreement is also compared with
that of the other shifts of one image against the other (`placement_options` from where it is, up to `max_match_nm`;
`OverlapCheck.better`): one that agrees clearly better (`ties`, the placement's rule) means the overlap does not line up
(D140). An overlap lines up when at least half its contacts pair (`MIN_AGREEMENT`), no other shift agrees clearly
better and its images lie at most `overlap_tolerance_nm` apart; one that does not is logged with `!!` and graded an *Error* (`analysis.checked_overlaps`,
`ReleaseResult.slipped`), and the log gives the median and the worst of both numbers. Overlaps where either image has
fewer than `min_matched` such contacts (corners) are not checked. It replaced a comparison of the stitching with the
placement within `overlap_tolerance_nm`, which could never fire: every pair lies within the tolerance of the shift the
placement predicts, so their mean does too.
Why the agreement and not the mismatches of `scored_shifts` (D50) or the pairs (D133 at first) choose: an overlap one
or two rows thick often has no contact inside a box of contacts shrunk by the tolerance, which scores the true shift
worst; and the pairs alone let a shift that pairs most contacts win for being smaller (D138). The boxes of the
agreement are those of each image's own contacts, not shrunk. The design is still stitched over all overlaps, and the realignment (D105)
stitches the SEM over all overlaps whatever the method. On a repeating pattern the whole-pitch term of an image is not
known, which does not matter (the design is ideal, D105) as long as each tile's SEM–design pairs use the same term:
with this method `design_errors(follow_stitching=True)` (D134; research app, MaskReg pairs after the stitching,
D143) lets the tied tiles choose their candidate by the shift
after the corrections (candidate + SEM correction − design correction, at the tile's design centre), which agrees
between neighbours on a consistent stitching, instead of by the shift at the nominal placement, which differs between
neighbours by whole pitches here; the pairs are still chosen at the nominal placement (D86). *Known problem (diagnosed
2026-10-27; MaskReg: fixed by D143, which pairs after the stitching; the research app still pairs this way):* on a
fully repeating array the SEM–design pairs of many tiles do not follow the stitching,
so their errors are about a whole pitch. A tied tile may only choose among the candidates `near_ties` lets through
(within `min_extra` = 2 mismatches of the best, with no counting noise), and on a repeating array the candidates one
pitch apart differ in mismatches only by the missed and false detections inside each one's box. Synthetic, the user's
pitch (40 nm grid, 24 images of 1.44 µm, 3 % missed, 1 % false contacts, stage scatter 20 nm, `design_search_nm` 100,
`design_tolerance_nm` 10): the stitching right up to whole pitches (within 0.26 nm), yet 13–17 of 24 tiles paired a
row or more off (errors 40, 57, 89 nm); the candidate that follows the stitching was always among the 9, but pruned for
8–14 tiles; with every candidate allowed (one change) 4–8 tiles remain, chosen by the group's fewest absolute
mismatches, which favours the larger shifts (smaller boxes). D136's stitching gives the same (12–19 of 24), so it is not
from D137–D140. No missed or false contacts: 0–1 tiles. *Known problem 2 (diagnosed 2026-10-27, not fixed yet): on a
drifting mosaic whole regions are placed whole pitches off, and the check passes them.* User: tiles visibly stitched
wrong at 40 nm pitch, `max_match_nm` 50, with no Error on them; tiles 80 and 81 stitched (−14.2, −34.6) and (−17.9,
+9.9) nm, one pitch + 4.5 nm apart. Synthetic, the user's settings (84 images of 2.88 µm in a serpentine, 280 nm
overlaps, 40 nm grid with 2 % of the pattern's contacts missing, 2 % missed and 0.5 % false detections, drift 1.5 / 1.0
nm per image): 57 of 84 images whole pitches off the start image (D136: 59), 127 overlaps wrong, 6 flagged. Traced: the
right shift is among an image's options, often with the best agreement (0.980 against 0.962–0.971), but within the
counting noise of `ties`, so all tie and the smallest correction is taken (the stitching method fingerprint, D144,
decides these by the residuals where the mask's placement errors exceed the noise): the copy nearest the stage position in the
start image's frame, wrong by round(drift since the start / pitch) pitches. Images placed after it follow it, so whole
runs of images slip together and only their boundaries are wrong. One change at a time: without drift no slip; a
200 nm search, ties toward the placed neighbours' mean, or toward the neighbour nearest in measurement order (also
both with 200 nm) still 38–66 slipped (the first image of a stripe is placed against the stripe before, measured
minutes earlier). The evidence is there for a group, not for one overlap: a stripe moved one pitch at the right
placement leaves 137–152 extra unpaired contacts along its boundary against a counting noise of 40, but only 3–6 of its
61 boundary overlaps tell it on their own, which is why the check (per overlap) flags so few. Known limit: an offset
about as large as an overlap is wide cannot be found. Cost (D138): 8.3 s for 80 images of 2.88 µm (50 nm grid,
3,100 contacts each, 280 nm overlaps, `max_match_nm` 100 nm; 7.3 s before D138); an image is evaluated again each time
a neighbour is placed, and the pairs use trees (`seen_pairs`).

**Stitching by fingerprint (D144).** The fourth `stitch_method`, off by default. On a repeating array image by image
cannot tell an image's whole-pitch shifts apart (every one agrees as well) and takes the smallest, so whole regions can
slip (Known problem 2 above). But at the true shift each pair is one contact seen twice, which differ only by the
measurement noise; a whole pitch off each pair is two neighbouring contacts, which differ also by their own placement
errors on the mask: its fingerprint. `pipeline.placed_by_fingerprint` is the placement of image by image (the same
order: clear images first) with an image's options found as `placement_options(by_fit=True)`: every candidate
(`candidate_shifts(max_candidates=None)`, up to `max_match_nm`), no `LINED_UP` (the true shift can pair far fewer
contacts than one a pitch off, its overlap thinner) and no `ties` before the residuals (where a contact row lies along
an image frame, the true shift leaves a contact or two of the other image without a partner and a shift a pitch off
pairs them all: counted first, the true one was dropped). Of the options, those fitting as well as the one with the
smallest residual are kept (`fitting_best`; `fits_as_well`: the log of the ratio of their mean squared residuals within
three times its spread, √(1/n₁ + 1/n₂) for n₁, n₂ pairs) and, of those, the ones agreeing best (`agreeing_best`, the
rule of image by image), the smallest correction first (`fitted_options`): one where the residuals tell the shifts
apart; where they do not (placement errors not larger than the noise, or none), what image by image keeps, so it falls
back to image by image. The residual of an option is `capped_rms`: the rms length of all its pairs after the robust fit,
each counted at most at the fit's outlier limit (3 × the median length), per degree of freedom (n / (n − 1)). The rms of
the fit's inliers (`OverlapFit.rms`) is made small when the fit leaves out the largest residuals of a few pairs: two
options with the same spread were called clearly different 5–8 % of the time below 10 pairs (3σ: 0.3 %); the median of
all squared lengths was safe but needed about twice the pairs; the mean of all of them was ruined by one outlier (15–60
%). Capped: a wrong option (twice the mean squared residual) called clearly better than the true one ≤ 0.3 % of the
time, the true one found as often as with the inliers. The overlaps are then stitched as image by image
(`stitch_image_by_image`) and checked by `overlap_checks(by_fit=True)`: another shift of one image is clearly better
when it fits clearly better, or fits as well and agrees clearly better (`clearly_better`), as the method chooses
(counted, an exact placement with a contact row cut by a frame was an Error). The SEM–design pairs follow the stitching
(D134 research app, D143 MaskReg). The log says per stitching how each image was chosen: by its residual, by the
contacts paired, tied (smallest correction) or its one option, how clearly (second best residual / best, median) and
the residual taken (median: about 2 × the noise per axis; `analysis.fingerprint_text`). Synthetic (12–30 images of
1 µm, 120 nm overlaps, stage scatter 20 nm, `max_match_nm` 100): placement errors σ 1 nm with 0.2 nm noise, or 0.5–0.6
nm with 0.3 nm: every image within 0.2 nm of the truth (60 of 60 runs over a 50 nm square grid, a 40 nm staggered
array and a grid with 5 % of its contacts missing, 896 of 900 images chosen by their residual; ratio median 1.8–4.6),
where image by image is 54–90 nm off; 0.3 nm with 0.3 nm: mostly tied, whole pitches as image by image; no placement
errors: every image tied. Cost about that of image by image (0.8 s for 30 images). Needs placement errors clearly larger
than the noise and overlaps of some tens of contacts; whether a mask has them depends on the dataset (the log shows it).

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

**In-image (intrafield) distortion (D53, D145, D146).** `intrafield.py`: one distortion shared by all images, measured
against the design and removed from every SEM point **before** stitching, as one smooth formula over the image
(`estimate_formula`, below). Until D146 it could also be a map of values on nodes from edge to edge of the image, each
node's value a plane fitted to the residuals of the contacts nearest to it (D53); removed: on real data a corner node
whose contacts sat at two spots only took a tilt from noise and pushed every image's corner by tens of nm. Mask errors
at fixed places on the mask fall at a different place in every image and average out; **mask errors that repeat at
the image spacing (e.g. a writer field of that size) do not, and would be removed as SEM distortion** (user concern:
future datasets may have such fields). `map_shift` is bilinear between the nodes of the map the formula is kept as;
`correct_points(points, centre, map)` subtracts it. Checks: `overlap_slopes(points, stitch, centers)`, how B − A
changes across each vertical / horizontal overlap (ppm; the same contact in two images, so mask errors cancel and the
design is not used), which a correct formula brings to ~0; and the formula from the lower vs the upper half of the
field (notebooks). Synthetic: a shared bow of ~0.2 nm at the edge: overlap slopes −640 / +480 ppm → ~0. The release
removes it only on request (`in_image_correction`, D141).

**The formula (D145).** `intrafield.estimate_formula(design_local, sem_local, order=3)`: the shared distortion as one
smooth formula over the whole image, f(s) = Σ c·u^p v^q over every power p + q from 2 to `in_image_order`
(`formula_terms`; 3: x², xy, y², x³, x²y, xy², y³, 7 terms per direction, 14 numbers), (u, v) the SEM position
relative to the image centre divided by the largest |coordinate| of the contacts (each c is nm at the image edge).
Powers 0 and 1 are left out: each image's own affine takes them. Why this shape for an SEM: the image is drawn by a
beam swept by magnetic deflection coils through magnetic lenses, whose distortion (barrel, pincushion, spiral) is of
third order in electron optics, and e-beam writers, deflecting the same kind of beam, calibrate their deflection field
with polynomials up to the third order; a nonlinear scan along the line (S-shaped) is also cubic. Not covered: the
scan-start effect (after the flyback the start of every line is compressed, decaying over a short distance along one
edge only; a term of its own if it shows), and drift during a frame (different in every image; its linear part goes
into each image's affine). **Fit:** all images at once, each with its own affine B_k (6 numbers, free): in every image
s − f(s) = B_k(d), d the design position, s the SEM position (f is evaluated at s, where `correct_points` evaluates
it). The B_k are not solved for: in each image, what an affine of its design explains is removed from its SEM
positions and from every term, then one least-squares fit over all images gives the coefficients (the same solution
as with all B_k free). So partial images take part without biasing it and images with ≤ 3 contacts add nothing. One
rejection round (contacts beyond 5 median absolute deviations of the residual, per axis, left out; refit). **Guard:**
if the weakest combination of terms is below `DETERMINED` = 1 % of the strongest (singular values of the fit; full
images ≥ 5 % up to order 5, contacts in a few rows ~0.01 %), the contacts cannot determine the formula: None, no
correction, a `!!` log line (the lesson of the removed node map: never report a number the data does not determine).
**Kept as a map:** f without its affine part (left to the global affine) tabulated on `FORMULA_NODES` = 65 nodes per
axis over the contacts, an `IntrafieldMap`, so `map_shift`, `correct_points`, the run file (`intrafield/*`, `FORMAT`
unchanged; a run saved with a node map opens with it), *Load contours* and the views use it (bilinear between nodes
1/64 of the image apart: off by < 0.1 % of the third-order part). Synthetic: a radial cubic (2.7 nm at the corners),
each image with its own ±300 ppm affine, one half image: what no affine explains 0.19 nm → 0.02 nm (the noise) in
every image.

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

**Horizontal L/S centre stitching (D170, D171).** Find neighbours by their full tile rectangles. Intersect
the x slices in each overlap, comparing the widest common intervals first. Evaluate each centroid's height
at the midpoint of the common interval using the local slope of its filled contour. Pair mutually nearest
heights using the SEM or design match distance, using each centre at most once per tile pair. The usual minimum
pair count, robust overlap translation and weighted tile solve apply independently to each stitching. The x
correction is fixed at zero. On repeating rows this assumes the relative stage error is less than half the
spacing between adjacent measured centres, including space centres.

Pair stitched SEM and design heights within each x slice, following a common tile offset from neighbouring
tiles (D143). Use the tile's own design first, then neighbouring design samples at x positions in that slice.
Measure SEM height at the paired design centroid's x and merge repeated observations there. Use the usual
moving windows, optionally per stripe. Each window fits vertical displacement by least squares. A tilt term
requires an x span greater than half a slice of the narrowest field. A y scale term requires heights at least
the design matching tolerance apart in both sets after removing their tilt. Otherwise the corresponding
coefficient stays at its identity value. With per-stripe fitting, a stripe without measurements stays at its
stitched placement. The horizontal transform is the identity and its residual is unmeasured
(NaN in the run file). The Registration tab, profiles and command-line review report y. This temporary path
uses the paired design clips directly and omits the contact-specific realignment and in-image correction.

**Stitching by contours (D156; MaskReg, `pattern` contours).** `contour_stitching.py`. An edge point knows where it
is across its edge only, so two images are compared by how far their edges lie apart **across the edges**, edges
facing the same way only; nothing is assumed about the pattern (no direction, pitch or shape). Per image, its refined
contours (S2, D155) as edge points in nm, each with the way it faces (`edge_points`; points without an edge left out).
`placed_by_contours` places the images one at a time, as `placed_image_by_image` places contacts: the image with the
most edge points in its overlaps stays, then one image at a time against all its placed neighbours at once:
1. `mismatch`: for **every shift on the pixel grid** up to `max_match_nm` each way, the mean squared distance from its
   edge points to the neighbour's nearest edge facing the same way (8 sectors of 45°, its own or the one beside it),
   each distance at most the gate (`GATE_PX`, 3 px); only points the shift brings inside the neighbour's image count.
   Correlations by facing direction: the image of its points against the image of the distance to the
   neighbour's edges (chamfer matching; run sums or FFT, D164/D165).
2. `coarse_fit`: the shifts that fit as well as the best one on that grid (tried: at least `MIN_POINTS` 40 points and
   `MIN_SHARE` 0.25 of the most any shift compares; as good: the log of the ratio of the mean squares within three
   spreads, 2/n each, n one point per `SHARED_NOISE_PX` 4 px of edge, as neighbouring points share the edge blur's
   noise; the rounding to bins, bin²/12, added). The grid is blind to what is smaller than a pixel, so:
3. `exact_fits` compares those shifts without the grid, in bounded batches (`exact_fit` is the scalar reference):
   edge points that do not share noise (`thinned`: one per 4 px,
   at most 600), paired with the neighbours' points by position and facing (within the gate and about 30°), their
   distances across the edge after the small move across the edges that fits best; a point with no such edge counts
   the gate; points landing within two gates of the neighbour's border are not expected to pair. `as_good` again.
4. Of the shifts left, **the one nearest the stage position**, made exact by `lined_up`: least squares of the
   distances across the edges (beyond three robust spreads left out), moving only the directions the edges measure
   (`across_step`: the eigenvalues of Σ n nᵀ; a direction measured less than `WELL_MEASURED` 0.1 as well as the best
   one stays), until it moves less than 0.01 nm.

**CPU execution (D164/D165).** The capped distance field is filled at the cap, then only pixels strictly within
3 px of a reference edge are lowered, taking the minimum where these neighbourhoods meet. Duplicate reference
pixels are collapsed, as in the former binary raster. This gives the same float32 capped squared distances
without a whole-image distance transform. Each moving point adds its rectangle of allowed shifts to a small
difference grid; cumulative sums give exact covered-point counts without a full point image.

For each facing sector, group consecutive occupied pixels of equal point count into runs. Choose the raster
orientation with fewer runs. Where `runs × shifts <= 4 × distance-field pixels`, row prefix sums give the
correlation by interval subtraction; duplicate points keep their weight. Fragmented sectors use FFTs: only the
valid correlation window is needed, so FFT dimensions cover the larger image, not the sum of both images'
extents. Their spectra are summed before one inverse transform, then added to any directly summed sectors.
Every original point and search shift is retained; the choice of implementation does not assume a pattern or
edge direction. Compiled loops are serial and release the GIL, without another thread pool beside denoising.

Exact comparisons query the facing tree and solve the small normal equations in batches of at most 64 shifts
(at most 600 sampled points each), keeping every candidate and the same gate, border test, measured directions
and tie rule. A waiting candidate stores its nearest shift and number of fitting groups. Only the candidate
selected by the existing placement priority is refined; if refinement finds too few pairs, discard that choice
until another neighbour is placed and try the next candidate. Overlap samples and the tree are rebuilt for the
selected candidate rather than retained for all waiting tiles. Moved neighbours are reused while unplaced
neighbours need them; a placed tile's candidate surfaces are released. No GPU or new dependency is required.
FFT and reduction roundoff can differ; tests compare grid distances with direct sums and batched fits with the
scalar scores, including empty samples, straight/slanted/rough edges and mask-scale coordinates. These checks
establish the computation, not real-image stitching quality or a measured speedup on the remote PC. The later
requested comparison with TV denoising and 4× upscaling, including timings and output checks, is recorded in
`docs/notes/2026-11-17-contour-tv4-comparison.md`; the earlier estimate omitting preprocessing remains withdrawn.

So one rule covers the cases: **shapes** fix one position both ways; **straight lines** fix it across them, and
along them the image stays at its stage position, for lack of anything else (§13 question 14), unless the **edge
roughness** tells the shifts along the lines apart (then to the search grid, a pixel); on a **repeating pattern**
whose repeats the edges cannot tell apart (neither the roughness nor the differences between the lines seen twice
are above the noise), shifts apart fit equally: the nearest is taken, the smallest correction, and the image is
logged with !! (`ContourChoice.positions` > 1, `separate_groups`: groups of fitting shifts two gates or more apart).
This is the order of question 14 in one criterion: the roughness and the residuals of the lines seen twice are both
in the distance across the edges; tied → smallest correction, logged. Images with one group are placed first, so
that an undecided one does not decide for its neighbours. An image sharing fewer than `MIN_POINTS` edge points with
its placed neighbours is not placed (NaN, logged). `overlap_distances`: after the placement, per overlap the rms
distance across the edges between the two images' paired edges, the check that needs no trust in the placement
(logged: the median over the overlaps and the largest).
The run (`release/analysis.stitch_by_contours`): every image's contours in worker processes, the placement (one
process; the status bar counts the images placed; `lined_up` only on the image's edge points in its overlaps), the two
summary log lines and the extraction/stitching timings (D164/D165); **then it stops**: comparing contours with the design (S6) is not built. It ends as an `AnalysisFailed`
marked `stitched_only`, whose result (no contacts, no design; the corrections in `stitch`) the Stitching tab shows;
nothing is saved, and no `.oas` is read. The tab's status shows those log lines; *Load contours*
(`read_any_contours`) draws the detected and refined contours, open ones open, those of the tiles in view, at the
placement shown; no centres. Not used with contours: `stitch_method`, `min_matched`, the in-image correction. A run
with `pattern` lines (D153) is still refused. Synthetic (`tests/test_contour_stitching.py`, edge points of known
stage errors, noise 0.3 nm): circles, both ways within 0.2 nm; straight lines, across within 0.1 nm and along
unchanged; lines a pitch apart with a stage error above half a pitch, the nearest position, a whole pitch off,
flagged; lines with their own roughness of 1.5 nm, the true row and the position along within a pixel, not flagged.
`tests/test_release.py`: a folder of three images of rendered bright lines, across within 0.3 nm.

**Extraction scheduling (D166).** `contour_memory.contour_workers` reads image dimensions from their headers
using the existing Qt dependency and budgets for the largest upscaled image. Per active task the planning
allowance is `max(1 GiB, 32 × working pixels + 256 MiB)`; the floor covers numerical-library imports and thread
buffers. Active tasks use at most 60% of the smaller of free RAM and free commit capacity (`machine.memory_gb`),
limited further by the CPU/pool capacity and tile count, with at least one task. Unknown memory or an unreadable
header permits one task at a time; decoding still reports an invalid image normally. This is a conservative
planning estimate, not a bound on arbitrary contour geometry or a guarantee that one image fits.

An owned pool is created with that count, and its TV threads are divided among those workers. A borrowed pool
is left open and retains its thread configuration, but the same active-task limit applies. `bounded_results`
submits at most that many jobs and yields them in input order, dropping each consumed Future before yielding its
result and cancelling pending work on error/cancel. The run shifts owned worker coordinates in place, so it does
not retain both local and translated copies through stitching. *Load contours* for pattern contours uses the
same admission and bounded submission. The log names the worker limit, memory headroom and estimated allowance;
after extraction it gives the retained edge-array size. No algorithm thresholds or saved-run versions change.

The timing log (`StepLog`) lists overlap discovery/counting, grid search, sample/tree preparation, exact candidate
comparison, refinement and the final overlap check, in seconds. Counts give images, overlaps, edge points, grid
shifts per surface, surfaces actually computed, run-sum and FFT sectors, choices evaluated, exact shifts tested
and refinements actually attempted. The last line names the slowest stitching step. Extraction logs its elapsed
time and configured denoiser/upscaling, each processing step's time summed across workers (explicitly not run
elapsed time), and median/maximum TV rounds when used. These distinguish extraction costs from placement costs
without treating a stitching-only measurement as the full run or guessing from synthetic timings.

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

**MaskReg pairs after the stitching (D143):** image folders use `registration.design_errors_after_stitching` for the
errors at the stitched placement (the research app keeps `design_errors`, D86/D134; the in-image map, D141, still
pairs each tile on its own at the nominal placement, before the stitching). The tiles are paired one at a time where
the stitching put them, each against its own `.oas` contacts and its neighbours' (stitched; a neighbour's contact
within `group_radius_nm` of one already in is the same contact), so the strip of an image that the stitching moved
off its own file's field still pairs: such a pair has `design_index` −1 (no row in the tile's own file;
`merge_values` leaves it out of the design means) and `design_nominal_nm` = its stitched design − the tile's design
correction, so `merge_observations` groups it as before. A tile with paired neighbours takes the median SEM − design
offset of their pairs inside its own stitched image (the same contacts, seen by both), pairs its contacts within
`design_tolerance_nm` of it (mutual nearest, `matching.match_points_tree`) and again at the median of those pairs;
the waiting tile with the most such neighbour pairs goes next. The first tile of every group of neighbours (none
paired yet) is the one with the most contacts, matched on its own at the nominal placement (`match_with_shift`,
`design_search_nm`), then paired again where its corrections put it. A tile that, following, pairs fewer than half
(`MIN_FOLLOWED`) of its SEM contacts over the design (stitched more than `design_tolerance_nm` off the neighbours it
followed, or they off it; on a pattern that nearly repeats some still pair by chance) waits for other neighbours'
pairs, and is matched on its own if none come: a step at that seam, logged with `!!` (`offset_steps`: neighbouring
images whose offsets differ by more than `design_tolerance_nm`). The offset is followed from tile to tile, so a scale
added up over many tiles (D86's case) is followed too: no affine is needed before the pairing, and none is used. On a
repeating pattern every tile then takes the whole-pitch term of the stitching, which does not matter (the design is
ideal, D105), instead of its own, which can differ from its neighbours' by whole pitches. Contacts of an image moved
out beyond every design file are not paired. `match_shift_nm` holds each tile's offset after the corrections;
`ambiguous` and `match_mismatches` are None.

**Realignment (D91, D93, D104, D105, rare case, off by default):** if the SEM alignment failed, every image shows
the design some µm away from its own `.oas` (often more than one field of view), so no tile matches. With `realign`
on (`realign.realign_tiles`, called by `analysis.load_image_tiles`): the `.oas` files are read in normal tone without
matching (`process_tile(match_design=False)`; tone-reversed files are not handled); the SEM tiles are stitched on
their own (all overlaps); the `.oas` files are stitched on their own (`stitch_design`; a file that cannot be stitched
stays nominal) and their contacts put together; the one offset design − SEM is found over **every shift at which the
images still overlap the `.oas` files, of any size** (`find_design_offset`, D104: FFT correlations of point rasters in
cells of 2 × `design_tolerance_nm`; each shift scored by its mismatches as in D50, inside the tile boxes shrunk by the
median design contact spacing; only shifts comparing at least one image's worth of contacts, the median design
contacts per file). The **best fit** has the lowest mismatch fraction with 4 σ (`NOISE_SIGMA`) added to its count
(Poisson upper bound, right for small counts: a shift comparing few contacts cannot be it by chance). A shift **fits
as well** unless its fraction is higher by more than 4 standard errors of the difference (two fractions, pooled). On
a repeating pattern every whole-pitch shift of the true offset fits as well; the design is ideal, so they all give
the same errors and only where on the mask the errors lie is unknown (user, D105), which a badly aligned measurement
cannot tell anyway. So **of the shifts that fit as well, the one where the most SEM contacts have a design contact
within one cell is taken** (the most measured; equal: the smallest shift), and the log names how many others fit as
well (those comparing at least half as many contacts) and one of them. Where the pattern has features that do not
repeat (missing contacts, edges), only the true offset fits. Then, on its ±1-cell plateau, the shift with the most
contacts in the very same cell; refined to below a cell by the mean pair difference. The run stops (ValueError) only
when no shift compares one image's worth, or when at the offset taken more than 1 − `min_match_fraction` of the
contacts compared have no partner (the images do not show the pattern of the `.oas` files). The `.oas` files only hold
their planned fields (cut to the FOV here), so an offset across a single stripe larger than the stripe's width leaves
only whole-pitch equivalents to pair with. Cost (synthetic, 6 stripes × 250 images of 2.88 µm, 607k contacts per
set): 17 s, 4.0 GB of numpy arrays at the peak; one stripe of 120 images: 1.3 s. Every tile's centre then moves by
the offset and its design
becomes the merged design contacts (`merge_close`, `group_radius_nm`) inside its shifted field of view;
`TileResult.oas_center_nm` keeps where its own file sits (stitched), for drawing only (no design outlines or
contact sizes from it); a tile with no design contact is flagged. The run then continues as usual. A parameter in
the research app and in the release Parameters dialog (D92).

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
**The pairs are chosen at the nominal placement (D86;** MaskReg's image folders pair after the stitching instead,
D143, S6**)**, before the corrections, where a tile is
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
**Per stripe (D123, release only):** `moving_window_per_stripe(sem, design, stripe, groups, window_nm, step_nm)` runs
the same on each stripe's observations on its own (one observation per contact per image; `stripe`: each
observation's image's stripe; `groups`: `MergedErrors.members`). The stripes are measured at very different times
and drift differently, so every image is corrected by the windows of its own stripe; a contact's residual is the
mean of its observations' residuals (one in the overlap of two stripes: the mean of both stripes' corrections).
`MovingWindowResult.stripe` holds each window's stripe (empty for windows across all stripes; optional in run
files). Limit (synthetic): a y scale of its own per stripe is removed only in part (0.32 → 0.16 nm rms), because the
translation stitching turns the stripes' disagreement in their overlaps into per-tile shifts, which no affine per
stripe follows; an x scale per stripe is removed down to the noise (0.26 → 0.10 nm).

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
analyzer given to users; the research app (`app.py`) is unchanged. Two tabs (a third, *Contact size*, hidden unless turned on: see "Release: Contact size tab (D95)" below): **Stitching** (`release/views.
StitchingView`: tile outlines, overlaps used, design and measured centres at the nominal or stitched placement, or
*Stitched + affine* (D142: the measured side, SEM images, centres and contours, also moved by the moving-window affine
of each tile, `release/analysis.tile_affines`, as the registration errors are; the design stays stitched; not offered
for a failed run, which has no moving window); the layer *SEM–design pairs* (off at start, D142): an arrow from each
merged contact's design centre to its measured centre at the placement shown (stitched: `sem_nm`; with the affine:
`design_nm` + the moving window's residual, the registration error at true length; none at the nominal placement),
red where longer than half the contact spacing (paired with a neighbouring design contact);
the overlaps not matched (boxes filled in the colour of their stitching: measured orange, design blue) and the tiles not stitched (thick red outlines), each a layer of its own, off at start like the tile names (dark text on white labels, D108), D76, D107; the **status line** above the plot grades the stitching, worst first (D109): *Error* (red) when SEM overlaps look paired one contact row off (`ReleaseResult.slipped`, the slipped-pair check of D75, kept with the run; a run saved before has none, which the line says) or, stitched image by image, do not line up (D137), *Warning* (red) when tiles are not stitched (their contacts are not measured), else *Stitching OK* (green), amber *with overlaps not matched* when some overlaps were left out, saying how many of them are corner overlaps; then both stitchings' counts; the legend lists only the layers shown (a click on an entry unticks its layer); **a click** (D112) on a tile shows that tile alone (outlined black on white, named; its SEM image and design pattern in front of the others, so its own image shows in its overlaps) and, beside the plot, in words, what is wrong with it: each overlap not matched, in which stitching, with a *show why* link; the neighbours of its other overlaps; in an overlap the tile whose centre is nearer is taken. With *Overlaps not matched* shown, a click in such a box (the smallest holding the point), or *show why* (which also zooms to it), shows only the overlap's two tiles (solid and dashed outlines, both named) and each one's contacts in the box, where they were compared (nominal placement), filled for the first and hollow for the second (circles measured, squares design), named in the legend, and gives the verdict in words (pairs found and needed, each one's contacts in it); a click beside every tile clears the selection; **Load SEM images** reads the images in the background,
downsampled by a whole factor to stay under 4·10⁸ pixels in memory, `release/analysis.read_display_images`; an opacity slider, D80; **Load design** fills every `.oas` file's contacts in its stripe's colour, D93, in the tone the analysis read it: a tone-reversed file shows its holes, `release/analysis.drawn_outlines`, D106; **Load contours**, D113, detects and refines every SEM image again as the run did, `release/analysis.read_contours`, in worker processes, the centres the run used checked against its own (the run's in-image map, if any, removed from them first: D141), with `!!` where they differ, then four layers like the others, ticked once loaded: *Detected contours* and *Detected centres* (the detection method's, magenta) and *Refined contours* and *Refined centres* (after edge refinement, cyan; named after the run's edge rule, D122), every contour at full resolution; the contours are drawn for the contacts in view only, none when more than `MAX_CONTOURS` = 5000 are (a note says to zoom in); not saved with the run; the legend of the measured centres names the method the run used) and
**Registration** (`RegistrationView`, D80, D88: one figure, side by side and showing the same area (same centre and µm per pixel, `link_views`, copied after a zoom or pan with the mouse, D114): the **arrow map** (the error vectors, shaft + two head strokes, each coloured by its length dr = |error| in jet on its own colour bar, 32 steps; scale box, default so the 99th-percentile error is ~half a pitch long) and the **dx and dy colour maps** (shared colour bar); every map has its profiles beside it (`MarginalMap`): the mean per row on its right, rotated so the row's y runs down the side aligned with the map's y, and the mean per column (`row_means` over design x, same gap) below it aligned with its x, both following the map's zoom and pan; the means per column hidden unless *Mean per column* is ticked (D118: hidden, they are taken out of the layout, which kept their room otherwise, so that the maps get the height, and left out of the image menus); *Symbol size* (D121): the contacts on the dx and dy maps, *Auto* (a disk of 0.8 × the pitch, at least a pixel, growing as one zooms in, as before) or that many screen pixels across at every zoom (`draw_map(size_px)`), off in grid average; *Line width*: the lines of every profile, 2 px at start; the colour maps show their own component, the arrow map the mean dr; **View All fits the height** (D117, revising D114; `RegistrationView.fit`): at start, after a change of stripes or values, after *View All* in any map (`FitViewBox`; pyqtgraph's *A* button hidden; a profile's *View All* fits its error axis only) and after a resize unless the user zoomed or panned (then the area shown is kept), every map shows the full height of its `extent` (the contacts shown with half a pitch around them, or every cell of the grid), centred on it in x and as wide as the map is: only y is fitted (the x range asked for is one pitch wide, which the equal x and y scales widen to the map, so the fit holds through resizes), and the maps keep the width the layout's stretch factors give them (a run wider than that runs past the map's sides); titles *Δr (nm)*, *Δx (nm)*, *Δy (nm)*, the colour bars *|measured − design| (nm)* and *measured − design (nm)*; the labelled ticks of mask axes as far apart as their labels need (`MaskAxis.label_px`); **a range for each mean per row** (D119, revising D115; `RowRange`): boxes just below the plot, in a row of the figure left for them, over the figure (not in its images): *Auto* (fitted to its values; the boxes show its range) or from the first box to the second (Enter applies and unticks *Auto*), kept when the stripes change; a zoom or pan with the mouse there shows its range in the boxes, *View All* there ticks *Auto* again; dr, dx and dy each have their own; **the range of each mean per row, three ways** (D127, revising D126; `RangeLines`): one line just below its range boxes, also over the figure, of the profile as drawn (*Values*, *Stripes*): *all* (over all rows), *ramp* (its largest ramp: `registration.turning_points`, a zigzag, splits the profile where it turns by more than the *wiggle height*: a top counts once the values fall more than that below it, a bottom once they rise more than that above it; `largest_ramp`: the ramp whose values span the most; a sawtooth's drop is a ramp too) and *manual* (between two lines dragged on the graph; at start on the first profile's min and max, then kept), each its range (max − min, nm) beside a tick box with a swatch of its lines; its min and max in the tooltip. Ticked (off at start, D76; each graph its own), a kind draws its min and max on the graph as two vertical lines (`style.ALL_ROWS` grey dotted, `RAMP` purple dashed, `MANUAL` black, movable, its value at its top), outside the graph's *Auto* range (`ignoreBounds`); *all* and *ramp* circle the rows of their min and max, *ramp* shades its rows; what is drawn is in the graph's images. *Ramp wiggles up to* (second row of controls, one for all three): a height in nm, or *Auto*, `RangeLines.NOISE_TIMES` = 8 × each profile's noise (`registration.noise_sigma`: robust σ of the row-to-row steps / √2), named in the ramp's tooltip; Copy profiles / Export profiles CSV give rows and columns (dx, dy, dr); no profile without affine; D81: *Values* every contact or **grid average** (`registration.grid_means`: mean per square cell, cell size in µm, default ~2000 cells and at least two pitches; colour maps then one flat square per cell, arrows one per non-empty cell from its centre, coloured by the length of the cell's mean, scale reset to ~half a cell; the means per row and per column then average the contacts of each row and column of cells, one point at its centre: `registration.band_means`, bands one cell wide from the lowest contact as the grid's, dr the mean |error| of those contacts; *Copy profiles* heads them *grid row* / *grid column*, D125), **Grid lines** on every map (on at start, D88), and **Copy image / Save image** menus with the whole figure, *dx and dy: maps, profiles, colour bar* (D116: the rectangle of the figure holding those graphs as laid out, nothing of dr, `style.Area`) or any one graph (a map with its colour bar), also in each graph's right-click menu and in a **Copy button** over the top-left corner of every graph shown (D120, `CopyButton`: a copy icon, over the figure, not in its images); a graph is cut from the whole figure rendered at 2×; D88: **Stripes** (`StripeSelector`, a drop-down of tick boxes, all ticked at start, *All*; one stays ticked): the maps, profiles, colour scales (99th percentile of the contacts shown) and summary show only the contacts of the ticked stripes; a contact belongs to every stripe whose x range (`pipeline.stripe_ranges`: its tiles' boxes, stripes `ReleaseResult.stripe` from `tile_stripes`, computed once per run and saved, D89) holds it, so one in the overlap of two stripes is in both (`pipeline.contact_stripes`; outside every range: the nearest); the moving-window errors are not fitted again). The error is **measured (SEM) − design**
(`registration_error`; moving window `G_w(SEM) − design`). `release/analysis.analyse_folder` runs only what these
tabs need: tiles (`analysis.load_image_tiles` with per-tile progress, or the contour CSVs), the in-image map on
request (below), SEM stitching
(`stitch_method`, slipped-pair check), design stitching, the errors at the stitched ("mean") placement, merging,
the moving window (per stripe with `moving_window_per_stripe`, D123): no drift corrections, no RANSAC. A common shift of all contacts does not
change the moving-window residual, so the placement is not a parameter.

**Release: Map data (D172).** The Registration tab's *Map data* menu copies CSV or TSV to the clipboard and
exports either format to a file. The header is `x, y, dx, dy` (with the chosen delimiter). Each row gives a measured
point's stitched design position in mask µm (y up) and its moving-window `SEM − design` residual in nm. It uses
the full run at point resolution, including all stripes and points outside the viewport in either *Values* mode.
Horizontal L/S rows have a blank `dx` field because only the vertical error is measured.

**Release: frequency analysis of a profile (D173).** A small button over the top-right corner of every mean per row
and per column of the Registration tab (`views.SpectrumButton`, shown while its profile is) and *Frequency
analysis...* in the profile's right-click menu open the profile as shown (stripes ticked, every point or grid
average) in a window of its own (`release/spectrum_window.SpectrumWindow`); a window keeps the profile it was opened
with. It shows the profile along a horizontal axis (position µm, error nm) and its **amplitude spectrum**: for each
frequency (cycles per µm; the period in nm on the top axis) the amplitude in nm of the least-squares sine with a
free mean, scipy's Lomb–Scargle periodogram (`spectrum.amplitude_spectrum`, `normalize="amplitude"`,
`floating_mean`), because rows need not be evenly spaced (a band not found, mixed pitches). Frequencies run from one
over the profile's length (the resolution: closer peaks are not told apart) to the **limit**, one over twice the
median row spacing (a higher frequency shows folded below it), five steps per resolution step; scipy gets a few
frequencies at a time so that each table it builds stays near 16 MB (`CHUNK_CELLS`). For horizontal L/S the rows
are the centres of lines and spaces, half a pitch apart, so the limit is one over the pitch. **Markers**, off at
start, each a tick box with a swatch: the limit (named on the graph) and every multiple of one over the **image
step** up to it, where an error made inside every image would show (`views.image_step`: the median distance between
neighbouring tile centres, within a stripe along y, between stripes along x; none with one stripe). **A click on
the spectrum** takes its highest point within 8 screen pixels (at least one resolution step) and climbs to the top
of that peak, refined on a grid ten times finer (`spectrum.peak_near`); the peak is circled and its period, its
amplitude (all picked sines fitted together with the mean, `spectrum.fit_sines`) and the shorter periods it may have
folded from (k / spacing ± f, k = 1, 2: `spectrum.folded_from`) are written beside it. A click within the same
distance of a picked peak drops it. Copy / Save image: the window's figure or any of its graphs.

**Removal (D174).** Between the profile and the spectrum, *After removal*: the profile minus what is removed, its
spectrum a second line on the spectrum (*after removal*, shown and in the legend only while something is removed),
and the part removed drawn dashed over the profile (*removed*). Removed are the **slow part**, when *Remove slow
part* is ticked (off at start): a running median over a window in µm (`spectrum.running_median`, 20 median row
spacings at start), which follows ramps and keeps drops sharp, so it takes out a sawtooth; and the **sine of every
picked peak**, all picked sines fitted together with a mean on the profile minus the slow part (`fit_sines`). A new
pick is the peak nearest to the click in what is left, so a second sine of a beat can be picked once the first is
removed; a picked peak stays circled on the profile's spectrum. Untick the slow part to remove only the sines and
see the sawtooth. A running median shorter than a few periods of a sine takes part of it, and near a drop it lags by
up to half a window of the ramp (the tooltip says so).

**Options (D175).** *Spectrum*: Lomb–Scargle (at start) or *FFT, resampled evenly* (`spectrum.fft_spectrum`: the
profile resampled every median row spacing by straight lines, across gaps too, mean removed, Hann window, amplitude
2|X| / Σ window, frequencies from the first FFT bin above zero to the limit), for comparison; both lines and the
peaks follow the choice, and picked peaks are measured by least squares either way. *Remove slow part*: *running
median over* a window (at start) or *straight line per ramp* (`spectrum.line_per_ramp`): the profile split where it
turns by more than *wiggles up to* (`registration.turning_points`, as the tab's ramp range; at start half the
profile's range, not the tab's Auto, which a fast signal moving a lot from row to row raises above the sawtooth), a
least-squares line per ramp; a turn takes the longer ramp's line, and a point of a ramp shorter than a neighbour (a
drop, whose turn a wiggle can put a few rows from it) takes that neighbour's line where it passes within the wiggle
height (the nearer, if both do). Only the setting of the method chosen is enabled.

**In-image correction (D141):** with
`in_image_correction` (off by default; a parameter of both kinds of folder, release only: `RELEASE_ONLY`) the
distortion shared by all images (D53) is removed from every tile's SEM centres before the stitching
(`release/analysis.in_image_corrected`): every tile's contacts paired with its design at the nominal placement, the
formula from them (`analysis.in_image_map`, as the research app: order `in_image_order`, also a parameter, D145,
D146), `correct_points` about each image centre; then the run goes on with the corrected centres (`ReleaseResult.sem_points`
holds them: what was stitched). The log gives the formula's order, the images used and its
largest shift. Left out with a `!!` log line for contour CSVs without image centres (`DesignX`, `DesignY`), when no
image has 3 paired contacts (the run then fails as usual, D102) and when the contacts cannot determine the formula. The map is kept (`ReleaseResult.intrafield`, optional arrays
`intrafield/*` of the run file; a file without them: not corrected); the Registration tab's summary then reads
*measured (SEM, in-image distortion removed) − design*. Same limit as D53: mask errors that repeat at the image
spacing are removed with it. Parameters (`release/analysis.PARAMETERS`,
a subset of `Settings`) are hidden: *Analysis → Parameters* (the research app's `SettingsDialog`, with saved
parameters on top: D136, below); after a change
it offers a restart. *Analysis → Contour diagnosis* tries the contour settings on one image first (D129, below).
*Analysis → High-performance mode* (off by default, remembered; a warning of its costs when turned on, D151) keeps the
worker processes that load the tiles (`analysis.helper_pool`) from the first run on, for the next runs and *Load
contours*, until MaskReg closes (`closeEvent`) or the mode is turned off (`MaskRegWindow.stop_helpers`); it can be
changed only between runs. A run that finds the kept workers broken (`BrokenProcessPool`: one of them ended) fails,
the kept set is dropped with a `!!` line, and the next run starts new ones.
*Analysis → Restart analysis* (F5) analyses the open folder again. **A failed run (D102):** a
run that stops after the stitching because fewer than 3 contacts were measured raises `release/analysis.AnalysisFailed`,
which carries what was found (the `ReleaseResult` without contacts, `moving` None; never saved) and why each tile was
not measured (`DesignErrors.skipped`); the window then shows that result's Stitching tab alone (no Registration or
Contact size tab; status bar and title say the analysis failed and nothing was saved; the folder is not added to
*Open recent*), with a layer *Tiles not measured* (hatched red, off at start, D76), the count per reason in the tab's status line,
and on the cursor line the tiles under the cursor, each with its reason (`StitchingView.tiles_at`; any result); SEM
images and design load on request as for any result; change the parameters and restart. Failures before the stitching
(e.g. no tile matched its design, D73) still show only their message and the log. **The log** (*View → Log*, `release/log_view.LogView`, D148) styles each message by how it starts: `===` (a step begins) bold on a grey band, `!!` (a problem) and tracebacks dark red on a pale red band, `Warning:` amber; in every line the numbers standing on their own (with sign and unit; not those in names and paths) bold and the time grey; the text is as logged. *File → Open recent* (D97):
the last `RECENT` = 10 folders whose result was shown (saved or new), newest first (preference `recent_folders`, one
entry per folder, compared with `os.path.normcase`); a folder gone from the disk is greyed; *Clear list*. **Autosave**
(`release/cache.py`): after every run the result is written to `MaskReg_analysis.npz` in the data folder (plain
arrays, `allow_pickle=False`, its own `FORMAT`), or, if that cannot be written, under the user's cache folder;
opening a folder shows the saved result instead of a new run (another format → a new run). **Backups (D98):** a new
run moves the run it replaces into `MaskReg_backups/` next to it (`cache.back_up`), named
`<file name>_<YYYYMMDD-HHMMSS>.npz` after when that run was saved (a second already taken: the next free one), and
only the newest `KEEP_BACKUPS` = 5 are kept; saving the same run again (its contact sizes added) makes no backup
(`save_cache(backup=False)`). The file is written as `<name>.part` and renamed when complete, so a failed save leaves
the previous run in place. **Versions and run files (D99).** Three numbers, each with one rule:
- `affine_ransac.__version__` (MAJOR.MINOR.PATCH; `pyproject.toml` reads it), the version of MaskReg (and the
  package). Raised only when a version is released, as a git tag `vX.Y.Z` (README, *Releasing MaskReg*): MAJOR if
  `FORMAT` went up since the last release (saved runs of older versions are analysed again when opened), MINOR if
  `ANALYSIS` went up or something was added, PATCH for fixes that change neither. Between releases the code keeps the
  last released number; `CHANGELOG.md` collects the changes under *Unreleased*.
- `release/cache.FORMAT`: raised when a run file saved before can no longer be read.
- `release/analysis.ANALYSIS`: raised when the same folder and parameters would give other numbers.

Every run file holds the `version` and `analysis` of the MaskReg that made it (`ReleaseResult.version`, `.analysis`;
files saved before D99: "" and 0, shown as "MaskReg before 1.0.0"). **Checks on opening** (`cache.load_cache`,
`MaskRegWindow.show_opened`):

| Saved run | What happens |
|---|---|
| none | analysed |
| cannot be read (damaged) | logged `!!`, analysed again; the file becomes a backup (D98) |
| older `FORMAT` | logged `!!`, analysed again; the file becomes a backup |
| newer `FORMAT` | `NewerRunFile`: a message (update MaskReg); nothing analysed or replaced (*Restart analysis* still can, keeping it as a backup) |
| same `FORMAT`, lower `analysis` | shown; status bar `OUTDATED: made by MaskReg x`; the user is asked whether to analyse again (No: asked again next time) |
| same `FORMAT`, higher `analysis` | shown; status bar "made by MaskReg x, newer than this MaskReg"; no question |
| same `FORMAT` and `analysis` | shown |

**MaskReg for an AI agent (D154).** `release/agent.py`, a command line without a window, for an agent with a shell
on the data machine (opencode, DS Cowork):

```text
python -m affine_ransac.release.agent run <folder> [--set name=value ...] [--out file.npz]
python -m affine_ransac.release.agent review <file.npz>
```

`run` is MaskReg's own analysis (`analyse_folder`, both kinds of folder, so the same numbers and `ANALYSIS`) with
MaskReg's default parameters (`DEFAULTS`, not the ones the user saved in the Parameters dialog) except those given
with `--set`; only the names of the Parameters dialog are taken (`PARAMETERS[kind]`), `true` / `false` for the on-off
ones. The run is saved as a run file of its own (`cache.run_arrays`, `cache.write_run`: the same `FORMAT`), by
default `maskreg_runs/<folder name>_<YYYYMMDD-HHMMSS>.npz` under the current folder, **never as the folder's saved
run or its backups**: those stay the user's. Standard output is one JSON object, the review; the log and the
progress (a line per tenth) go to standard error. `review` prints the same for a run file saved before. The review
(`agent.review`; names per finding capped at `MAX_LISTED` = 20, the count always given; errors in nm):

| Key | What |
|---|---|
| `run_file`, `folder`, `kind`, `saved`, `maskreg_version`, `analysis`, `tiles`, `stripes` | which run |
| `parameters_changed_from_default` | the Parameters-dialog settings that differ from `DEFAULTS` |
| `stitching.verdict` | the Stitching tab's status lines (`views.stitching_verdict`, D109), as plain text |
| `stitching.measured`, `stitching.design` | overlaps used, not matched (names), skipped for too few contacts; tiles not stitched (names) |
| `stitching.overlaps_that_do_not_line_up` | `ReleaseResult.slipped` (D109, D137); null: not known |
| `contacts_measured`, `registration_error_after_moving_window_nm`, `moving_windows` | the merged contacts; count, mean and 3σ per axis and largest length of the moving-window residuals (`registration.error_summary`); the windows and the fewest contacts in one |
| `problems_logged` | the log's `!!` lines |

Exit code 1 with `{"error": ...}` when it could not run (not a folder, a parameter name or value not accepted, the
metadata reader missing, a run file of another format); a run that stopped after the stitching (`AnalysisFailed`)
prints the review of what was found with `failed` and `tiles_not_measured` (tile → reason), exit code 1, and saves
nothing, as in MaskReg.

**Guard tests** (`tests/test_release_versions.py`): `RESULTS` holds, per `ANALYSIS`, a fingerprint (contacts, residual
RMS and maximum, the first contact's residual, the summed stitching corrections) of the synthetic contour folder
(release defaults, and a 3 µm window so that the window moves) and of the synthetic image folder (tiles saved as PNG,
since JPEG encoders differ between OpenCV builds; Otsu); `LAYOUT` holds, per `FORMAT`, the arrays and info keys of a
run file. A change that moves either fails, saying which number to raise. Limit: synthetic folders only, so a change
that shows only on real data or with a setting off by default (e.g. `realign`) is judged by whoever makes it.
*Help → About* shows the three numbers; the log of every run names the version. **Packaging (D100):** MaskReg is
handed out as a program of its own, Python included: PyInstaller, one folder (`packaging/MaskReg.spec` →
`dist/MaskReg/MaskReg.exe`, no console window; entry `packaging/maskreg.py`), wrapped in an Inno Setup installer
(`packaging/MaskReg.iss` → `dist/MaskReg-setup-<version>.exe`; per user, no admin rights, into
`%LOCALAPPDATA%\Programs\MaskReg`, a Start menu entry, an optional desktop shortcut, an uninstall entry; a fixed
`AppId`, so a newer version installs over the older one, whose `_internal` folder is deleted first so no library of
it stays behind). `python packaging/build.py` makes both for `__version__`. Build **on the
data machine**: the image metadata reader (`io/metadata.py`, remote-only) is
bundled only where it exists, so the spec refuses to build without it (`MASKREG_TEST_BUILD=1` makes a test copy that
opens contour-CSV folders only). KLayout loads its format readers at run time from `klayout/db_plugins/`, which the
spec adds explicitly (without them `.oas` files fail with "Stream has unknown format"). `main()` calls
`multiprocessing.freeze_support()` (the tile-loading workers of a packaged MaskReg start through it) and opens the
folder given as its argument (`MaskReg.exe <folder>`). Test build on the development PC: 385 MB in 542 files (OpenCV
111 MB, PySide6 101 MB, scipy and its libraries 67 MB, KLayout 30 MB); the synthetic contour folder opened by
`MaskReg.exe` and the image folder analysed by a packaged `analyse_folder` with worker processes give the guard tests'
numbers; the first start of a new build took 12.9 s to the saved result, later starts 0.9 s (development Python 1.3 s).
The installer is 99 MB (LZMA2; build 132 s in all). Checked here: a silent install, the installed MaskReg giving the
same numbers, an upgrade 1.0.0 → 1.0.1 in place (one uninstall entry, a file only the old version had removed) and a
silent uninstall (folder, Start menu entry and uninstall entry gone). **Updates (D101,** `release/update.py`**):** the
installed MaskReg reads its update server from `update.ini` next to `MaskReg.exe` (`[update]` `host`, `port`, `folder`;
`build.py` copies it into the installer from `packaging/update.ini`, which is kept out of git, and refuses a release
build without it). At every start (and on *Help → Check for updates...*) it logs in anonymously in the background
(`UpdateCheck`, its own thread beside any analysis), lists the folder and takes the newest `MaskReg-setup-X.Y.Z.exe`
by its numbers (`newest_setup`; 1.10 after 1.9); if that is newer than `__version__` the user is asked, after a running
job if there is one (`offer_update`); No: asked again at the next start. Yes: the setup is downloaded into
`%TEMP%/MaskReg update/` (emptied first; a size different from the listing's fails), `run_setup` starts a hidden
PowerShell and MaskReg quits; the PowerShell waits until MaskReg's process is gone, runs the setup silently
(`/SILENT /SUPPRESSMSGBOXES /NORESTART`, log `<setup>.log`) and starts `MaskReg.exe` again: the new version, or the
old one if the installation failed. At start a failed check is only logged; asked for, it is shown. Not the installed
program (Python, or no `update.ini`): no checks, the menu entry greyed. Releasing = uploading the setup into that
folder. Checked here: the restart step with stand-in programs (the setup runs only after the process waited for has
quit; paths with a space and a `'`); end to end with real installers and a local anonymous FTP server: the installed
1.0.0 looked at the server at start, `run_setup` with its process, then 1.0.0 quitting, gave 1.0.1 installed 25 s
later and the new MaskReg started (and looking at the server). Found on the way: an installer started while MaskReg
still ran gave up on the files in use (exit code 5, rolled back), hence the wait. Plain FTP cannot prove where a
setup came from; signing it would need a code-signing certificate (not done). **Figures**
(`release/style.py`): white background, dark text, light grid, axes in mask µm (drawn relative to a whole-µm
origin, `MaskAxis`), no SI prefixes, line colours dx blue / dy orange (colour-blind-checked palette), error maps
jet (D35) with white where there is no contact; each figure has **Copy image** / **Save image...**, rendered
through the scene at 2× (works with an OpenGL viewport). **GPU / CPU:** drawing uses OpenGL if a context with a
real renderer can be made at start (`opengl_works`; not the Windows "GDI Generic" 1.1 fallback, with which the
plots stay black), else the CPU; *View → Draw with the GPU* overrides it and redraws at once. The computation is
CPU (numpy / scipy) either way. *View → Detailed tile-loading log* (off by default, remembered, D149) makes an image
run log its tile loading in detail (`analysis.load_image_tiles(detailed=True)`, below). **Defaults and the contact layer (D78):** the release's defaults are `release/analysis.DEFAULTS` (`max_match_nm` = 100 nm; the research app keeps 25 nm). The contact layer is not a parameter: `analysis.detect_layer(records)` reads every `.oas` of the folder and takes the one (layer, datatype) that holds shapes (`io/design.layers_with_shapes`: polygons, boxes, paths; empty layer definitions and texts do not count); more than one, or none, stops the run with a message naming the layers and a file of each. **Scan direction (D79):** `first_stripe_upward` is not a parameter either: the file numbers (image or CSV file name `CD<number>`) follow the measurement order (user), so `release/analysis.scan_direction` takes it from the first stripe (`pipeline.stripe_directions`: its last image in number order above its first → upward) and stores it in the result's settings; stripes that do not alternate are logged with `!!` (not a serpentine: the between-stripe pairs of `measurement_pairs` are then wrong); names without distinct numbers, or a first stripe of one image, fall back to the default (upward) with a `!!` line. **Status bar:** the step (e.g. "Loading tile 312 of 744"), a progress bar with its
percentage, the time so far and **Cancel** (D152: the job stops once the tiles or file in hand are done; nothing of it is shown or saved); when idle, what is shown and whether drawing is on the GPU or the CPU.
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
`maskreg.ini`, `[ftp]` keys `host`, `port`, `user`, `password`, `download_folder` (all optional), read from the folder
the user chose (*File → Folders...*), else the path in the `MASKREG_CONFIG` environment variable, else
`%LOCALAPPDATA%/MaskReg/maskreg.ini` (`GenericConfigLocation`; `config.config_path`);
pre-set values are used over remembered ones, and with a pre-set host the dialog connects when it opens. The
download folder is the one the user chose, else the file's `download_folder`, else `Documents/MaskReg data`
(`config.download_root`). A malformed file is reported and ignored. **Folders (D96):** *File → Folders...*
(`release/folders_dialog.FoldersDialog`) sets both folders (preferences `config_folder`, `download_folder`; empty = the
default, shown greyed in the field) and says whether `maskreg.ini` is found in the folder typed.

**Checks before the run (D73).** `analysis.check_settings(settings, kind)` (every setting of that kind
of folder in range: `POSITIVE` ones > 0, `min_matched` ≥ 1, `in_image_order` ≥ 2, `min_match_fraction`
in (0, 1], choices) runs first in both flows; for images `check_image_records(records, layer)` (images
listed and on disk, at least one image with an `.oas`, the `.oas` files on disk, the contact layer in the
first `.oas`), then `SAMPLE_TILES` = 3 tiles (first, middle, last with an `.oas`) are loaded and matched
to their design; if none matches, the run stops with their match fractions. The other tiles are then
loaded **in parallel**, in half as many worker processes as logical cores, about one per physical core
(`concurrent.futures.ProcessPoolExecutor`, `analysis.worker_count`, D85, D87; the count is logged),
in their order (`load_image_tiles(pool=...)`: in workers given and left running, MaskReg's high-performance mode,
D151; else in new ones, `analysis.helper_pool`, ended after the loading; every worker ends itself once the process
that started it has ended, `_worker_ready` → `_end_with_parent`); the result is the same as loading them one by one. Once loaded, one `Timing:` line is logged
(`analysis.timing_summary`, D90): the wall time, the median seconds per step of `process_tile` (read, detect,
refine, oas; then `choose_design_contacts`: contacts, shift, reversed; kept in `TileResult.seconds`),
the wall time the steps alone would take, and on how many tiles the reversed tone was tried. With `detailed` (D149)
the loading also logs, each line starting with its time since the loading began (`+12.3 s`, taken where the line is
made, so a stalled window still shows when it happened): each sample tile's step times; the logical CPUs, OpenCV
threads per process and `OPENBLAS_NUM_THREADS`; per worker (first tile back) when it was created and when it was
ready (`machine.process_created`, the pool's initializer `_worker_ready`); per tile its worker, when it started and
finished there, its step times, and when it came back; every `HEARTBEAT_S` (10 s) without the next tile, a status
line (tiles shown, tiles back, the machine's CPU in use since the previous line, RAM and committed memory,
`machine.MachineLoad`, Windows API, `""` elsewhere); at the end the longest wait with no tile coming back. The tiles
are submitted one by one (`pool.submit`, as `pool.map` did) and taken in their order; tiles not yet started are
cancelled after an error. After all tiles are loaded,
the run stops at once if no tile matched its design (before stitching). Each failure names its cause.

**Contact size (D71, D73, D74).** Measured **on request** after an image run, not during it: the
*Contact size* tab (`view_contact_size.ContactSizeTab`) has an **ROI width** (nm) and **Run measurement**,
which runs `analysis.measure_contact_sizes(result, roi_width_nm, progress)` in a background thread and
then shows the maps. It needs the run's tiles in memory (`AnalysisResult.images`, which keeps the
observations of every placement); a reopened result shows sizes saved with it, and cannot measure. Per tile
(`TileResult.contact_sizes(layer, roi, refined)`): area, xCD and yCD of the SEM contour (the one the tile is
measured with: refined, or detected with refinement none, D124; pixel → tile nm via `frames.py`) and of the `.oas` outline, read again from
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

**Release: Contact size tab (D95).** Hidden unless *View → Contact size tab* is ticked (remembered in the
preferences). While it is on, an image run keeps what measuring needs in memory
(`release/analysis.analyse_folder(keep_tiles=True)` → `ReleaseResult.tiles`, a `LoadedTiles`: the loaded tiles
with their images, their observations and the merged contacts; never saved); hiding the tab, opening a folder or
restarting the analysis lets them go first. *Run measurement* (ROI width, as in the research tab) runs
`release/analysis.measure_contact_sizes` → `analysis.merged_contact_sizes` (shared with the research app's
`measure_contact_sizes`) as a background job of the window (status bar, progress per tile), then saves the result
with its sizes (`ReleaseResult.contact_sizes`; optional `contact_sizes/*` arrays in the autosave, cache format
unchanged). A result without its tiles (reopened, or run with the tab off) shows its saved sizes and says to restart
the analysis to measure; contour CSVs: N/A. Maps (`release/views.ContactSizeView`, release style): one figure, Show
boxes Area / xCD / yCD (Area at start; a hidden map leaves no gap), Value SEM or SEM − design, a jet colour bar per
map from the 1st to the 99th percentile, white between the contacts, mask µm axes, Copy / Save image (whole figure
or one map with its bar); the maps show the same centre and scale (`link_views`, which ignores hidden plots), and a
map shown again takes the y range and centre of one shown before (not its area). The summary lines of the research
tab. No SEM images or design behind the maps.

**Release: Contour diagnosis (D129).** *Analysis → Contour diagnosis...* (Ctrl+D; `release/contour_dialog.ContourDialog`,
a window of its own, modal like *Parameters*). **Load image...** opens one SEM image (any file the loader reads; it
starts in the open folder). The preview shows the input to thresholding in the pixel frame (x, y px, y down; no grid
over the image). Preparation can flatten the background (D130), denoise (D132) and upscale (D131). The upscaled
image covers the same area so the axes and contours stay in pixels of the loaded image. **Original image** (a check
box at the start of the layers' row, off at start, D160) shows the image as loaded instead, over the same pixel
frame and under the same layers; the grey level under the cursor is then of that image. Beside it the contour
settings (D128) in
groups, *Pre-processing*, *Detection (thresholding)* and *Edge refinement*, named as in *Parameters*, what each does in
its tooltip; those that change nothing given the others are greyed (`min_solidity` unless band, the refinement's options
with refinement none). **Get contours** retrieves contours in a background thread. **Auto retrieve** (unchecked each
time the window opens) also retrieves them on image load and after a 150 ms pause in editing. Loading an image or
changing settings clears the overlays and discards any outdated result (D168). `contour_dialog.diagnose` calls the
run's own extraction functions and uses the same thresholding code for the mask. Layers (check boxes; the legend lists those shown, named after the
method and the rule): *Threshold mask* (the pixels the threshold takes as contact, translucent amber; off at start,
D76), *Detected contours* and *centres* (magenta), *Refined contours* and *centres* (cyan; not offered with refinement
none, D124), shown once contours are retrieved, as after *Load contours* (D113). One line in words: how many contacts were
found, the method and the threshold's grey level, and the regions left out per reason (`DetectedContacts.left_out`, then
the contrast check); invalid settings are named there instead, in red (`check_settings`), and cannot be applied.
The preview includes Gaussian smoothing at `detection_blur_px` (0 disables it, D169) and any `equalize` selected
(D167, in *Detection*; `clahe_clip` and `clahe_tiles` greyed unless adaptive). Image settings refresh it in manual
mode too (D168). The mask uses the same input, and edge refinement uses the prepared image's grey levels. The summary gives the global
threshold in grey levels before equalization and the adaptive threshold in grey levels of the equalized image.
Below the image, the cursor shows x, y and the grey level of the displayed pixel. *Copy image* / *Save image...* as every figure.
**Apply to parameters** keeps the settings for the next run, as *Parameters* does (`MaskRegWindow.apply_parameters`: the
remembered parameters, and a restart offered when they differ from those of the analysis shown); **Cancel** leaves them.
Enter in a setting closes nothing (no default button).

**Its log (D157).** Under the figure (a splitter, to resize or close it), a read-only text box: **every search
adds an entry, the newest last** (`contour_dialog.log_entry`): the time, the file and its size, the settings that
decide the time (pattern, detection, refinement, upscale, denoise, background_px, min_area_px); the seconds from the
start of the search until it was drawn; then **every step with its seconds and its share of that whole, the longest
marked**; what the steps do not account for as "not timed"; then the counts the times depend on. The steps are timed
where they run (`step_log.StepLog`, `timed`; `Diagnosis.log`): with `pattern` contours each step of
`find_contours`, `detect_contours`, `refine_contours` and `mid_levels` (flatten the background and denoise, each
only when it is on, total variation as two steps: loading numba and its compiled loops, first use in a process
only, and its rounds, with a line giving their count and the time of one, D159; upscale; threshold mask;
regions of the mask; going through the regions; tracing every kept region in its bounding box; cutting the boundaries
at the image border with the direction of every point; smoothing; sampling the image across the edge at every point;
the interior level of every region, which groups pixels and selects medians (D165); the band level of every contour, from all
edge points of its region; the edge position of every point, contour by contour; moving the points; back to the
image's pixels), with contacts the two times `find_contacts` gives (detect, refine), with lines one. Then, for every
pattern: the mask for the display, handing the image, mask and contours to the plot, and **drawing them on screen**
(the window repaints at once, so that this is timed; every zoom and pan draws again). Counts for contours: pixels
searched (upscaled), regions of the mask, kept and left out by size, the kept regions' bounding boxes as a multiple
of the image, the boundaries around holes (not left out by size), contours closed and open, edge points, and the
most contours in one region with that region's edge points. Timing changes nothing that is found
(`tests/test_contours.py`, `tests/test_contour_dialog.py`).

**Reusing preparation (D163).** The diagnosis window keeps one native-resolution flattened and denoised image
(`PreparedImageCache`). A later search reuses it if the loaded image, background size (and tone when flattening),
denoiser and its active strength are unchanged. Detection, refinement, pattern and upscaling settings do not
invalidate it. Loading an image replaces the cache, including while an older search finishes. A hit is logged as
*reuse flattened and denoised image*, with no denoising rounds attributed to that search. A run still prepares each
tile afresh. In contour refinement, the band median and resulting mid-level are computed once per region and shared
by all its contours; the pixels, profiles and arithmetic are unchanged. D165 replaces the whole-image sort for
interior medians with grouped selection (S2). The first-image solver and separate display upscale/mask remain as before.

**Release: Saved parameters (D136).** *Analysis → Parameters* is `release/parameters_dialog.ParametersDialog`: the
research app's `SettingsDialog` with one row on top: *Saved parameters* (a drop-down of the saved sets, by name in any
case), **Load**, **Save as...** and **Delete**, and a line saying what was done. **Sections (D147):** the parameters
are shown under headings that fold and unfold them, in the order the analysis runs (`parameters_dialog.SECTIONS`;
`SettingsDialog(sections=...)`, the research app keeps one list): *Stitching* (`stitch_method`,
`overlap_tolerance_nm`), *In-image correction* (`in_image_correction`) and *Registration* (`moving_window_um`,
`moving_step_um`, `moving_window_per_stripe`) open; *Contour detection* (`CONTOUR_SETTINGS`, image folders) and
*Advanced* (`max_match_nm`, `outlier_factor`, `min_matched`, `design_max_match_nm`, `design_search_nm`,
`design_tolerance_nm`, `min_match_fraction`, `realign`, `tile_margin_nm`, `in_image_order`, `group_radius_nm`,
`row_gap_nm`) folded, unless one of their parameters shown differs from its default (a changed value is never
hidden). Each heading counts its parameters; a section without one shown is left out. *Save as...* asks for a name (spaces
at the ends dropped; the chosen set's name proposed) and saves the parameters in the dialog, as they are; an existing
name is replaced only after asking. *Load* puts the chosen set in the dialog; **OK** then uses it like any change
(`apply_parameters`: remembered, and a restart offered when it differs from the analysis shown), **Cancel** does not.
*Delete* asks first. The dialog opens on the set the parameters in use come from, if they are one's, and says so. The
sets are kept in the preferences (`parameter_sets`: JSON, name → {parameter: value}), beside the parameters chosen
last (`settings`). A set holds every parameter a user can change (`USER_PARAMETERS`: those of both kinds of folder),
whichever kind the dialog shows, so a set saved with a contour CSV folder open keeps its image parameters. Loaded, a
set is read in today's names (`current_names`); a parameter it does not have (added to MaskReg after it was saved)
takes its default (`DEFAULTS`), as remembered parameters do; names unknown to MaskReg are ignored, and the settings
users cannot change keep their values.

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
├─ CHANGELOG.md              # MaskReg changes per version (D99)
├─ packaging/                # build.py: MaskReg.spec (PyInstaller, maskreg.py entry) + MaskReg.iss (Inno Setup installer), D100
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
│  ├─ contour_stitching.py   # S5 for pattern contours: images placed by their contours (D156)
│  ├─ analysis.py            # end-to-end flow of one folder (contour CSVs / images), D62
│  ├─ results_file.py        # save / load an AnalysisResult (.npz), D63
│  ├─ spectrum.py            # frequency analysis of a registration profile (Lomb–Scargle), D173
│  ├─ machine.py             # CPU, memory, process creation time (Windows API) for the detailed loading log, D149
│  ├─ app.py                 # standalone analyzer: python -m affine_ransac.app, D64
│  ├─ release/               # MaskReg, the minimal app for users: analysis, cache, style, views (+ contact size tab, D95), app (D77), agent (command line for an AI agent, D154), ftp + ftp_dialog + config (D82, D83), folders_dialog (D96), update (D101), contour_dialog (D129), parameters_dialog (D136), log_view (D148), spectrum_window (D173)
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

## 10. Validation Strategy

The real data is only on the user's machines, so the automated tests run on synthetic data. They
answer one question: does the code do what was intended (frames, signs, wiring, injected values
recovered)? Whether a method works on real SEM images, and how fast it is there, is judged only by
the user's run on the real data; methods are not ranked, tuned or rejected on synthetic images
(`CLAUDE.md`; D162). This section was written before real data was available and called itself
"synthetic-first".

The synthetic data generator and what the tests assert:

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

Moved to [`docs/DECISIONS.md`](DECISIONS.md) on 2026-11-16 (D162), rows unchanged: one row per decision (D1, D2, ...)
with its reason. This section number stays so that §13 and older mentions of "§12" still lead somewhere.

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
   *Why it matters for the stitching image by image (2026-10-27, after D140):* synthetic mosaics at the user's image
   size (2.88 µm, 280 nm overlaps, 50 nm grid, each image missing 3 % of its contacts and adding 1 % false ones):
   on a full grid no image is placed wrong (up to the harmless whole pitches) and about 1 overlap in 100 images is
   flagged without cause (a corner); where the pattern almost repeats (5 % of the contacts missing) images still slip
   by whole pitches (2 of 100 at 20 nm stage scatter, 15 to 28 of 100 at 30 to 40 nm), only partly flagged. There the
   overlaps of one image hold too little to tell a one-pitch slip from the truth against its own missed detections;
   the rules tried (McNemar's test on the contacts both shifts see, every lattice shift as a candidate, the surest image
   first, strict or wide ties) each traded these slips against images drifting from their stage positions or false
   alarms on a full grid, none better overall, so none was kept, and step 4b of the D137 plan (place again the images
   whose overlaps do not line up) was not built: it would act on exactly that test.
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
14. **Line & space datasets (next, 2026-11-08): the facts the method needs.** Implementation plan for a fresh session in
    `docs/notes/2026-11-08-line-space-plan.md`; proposal and principles in
    `docs/notes/2026-11-08-line-space-method.html` (edges with the direction they face; across the lines
    placed image by image, the row between neighbours decided from the overlap: the roughness correlation of
    the paired edges over the along-line lag, which also gives the position along the lines; with no clear
    peak, the residuals of the lines seen twice as in D144, the along position then from the stage; tied
    → smallest correction, logged (order corrected by the user, 2026-11-08; the page still shows the old
    order, the plan wins); a row offset of the whole mosaic against
    the design is harmless as in D105, a slip between neighbours is not, it averages different lines' errors in
    the overlap (user's review, 2026-11-08); along the lines from the stage, for lack of anything else). Not built. To ask before building: pitch, line and space
    widths (equal?), line tone (dark/bright) *(answered 2026-11-09: bright lines on dark spaces; whether lines
    and spaces are equally wide depends on the pattern; a line is 10 px or more wide; step 1 built, D153;
    2026-11-12: the real images' lines are vertical, the direction may be diagonal, whether a line runs through the
    image depends on the pattern, so D153 finds none; contours of any pattern built beside it, D155;
    2026-11-16: D155 outlines the real lines with a large enough `min_area_px`; what is measured and reported is the
    plan's step 3, settled when the plan was approved: per line and per image a centre error across the line and a
    width error, the terms the edges cannot see "not measured"; the stitching by contours is built, D156, and needed
    none of the facts below, which are asked only if a later step cannot be built without one)*; pixel size,
    image size, overlap widths within and between stripes; whether anything in the measured area breaks the repetition (line ends, width changes, marks,
    missing lines); whether the `.oas` clips hold rectangles cut at the FOV; the largest stage error and
    whether consecutive images drift by more than half a pitch; the wanted output per line (one across
    error per image, or a profile along the line).
