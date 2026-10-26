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
**total variation** (`tv_weight`, on the grey levels / 255; Chambolle's algorithm,
`skimage.restoration.denoise_tv_chambolle`, rounded back to grey levels: flat regions with sharp edges between them).
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

**Contour settings (D128).** Every option of the detection and the refinement is a setting
(`analysis.CONTOUR_SETTINGS`), its default the value the code had before: `background_px` (0 = off, D130),
`denoise` (none, D132) with `median_px` (3), `nlm_h` (10) and `tv_weight` (0.1), `upscale` (1 = off, D131),
`detection`, `dark_contacts` (on), `detection_blur_px` (1.5), `min_area_px` (20),
`max_area_px` (0 = no limit), `min_solidity` (0.9, band only), `min_contrast`, `refinement`, `search_px`, `edge_step_px`
(0.25), `edge_blur_px` (1.0), `normal_span` (3). `find_contacts` takes them by name only (keyword arguments).
`pipeline.find_contacts(image, **analysis.contour_options(settings))` finds an image's contacts with them (detection,
contrast check, refinement); `process_tile`, hence every run and *Load contours*, and the release's contour diagnosis
(D129) call it, so all of them find the same contacts. Detection counts the regions it leaves out, each under the first
reason that applies (`DetectedContacts.left_out`: touching the image border, smaller than `min_area_px`, larger than
`max_area_px`, less solid than `min_solidity`). Not settings: the mid-level rule's own constants (±3 neighbours, 4
robust σ + 0.5 px), the contrast check's 3 px ring, the three classes of otsu3 and band.

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
with this method `design_errors(follow_stitching=True)` (D134) lets the tied tiles choose their candidate by the shift
after the corrections (candidate + SEM correction − design correction, at the tile's design centre), which agrees
between neighbours on a consistent stitching, instead of by the shift at the nominal placement, which differs between
neighbours by whole pitches here; the pairs are still chosen at the nominal placement (D86). Known limit: an offset
about as large as an overlap is wide cannot be found. Cost (D138): 8.3 s for 80 images of 2.88 µm (50 nm grid,
3,100 contacts each, 280 nm overlaps, `max_match_nm` 100 nm; 7.3 s before D138); an image is evaluated again each time
a neighbour is placed, and the pairs use trees (`seen_pairs`).

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
StitchingView`: tile outlines, overlaps used, design and measured centres at the nominal or stitched placement;
the overlaps not matched (boxes filled in the colour of their stitching: measured orange, design blue) and the tiles not stitched (thick red outlines), each a layer of its own, off at start like the tile names (dark text on white labels, D108), D76, D107; the **status line** above the plot grades the stitching, worst first (D109): *Error* (red) when SEM overlaps look paired one contact row off (`ReleaseResult.slipped`, the slipped-pair check of D75, kept with the run; a run saved before has none, which the line says) or, stitched image by image, do not line up (D137), *Warning* (red) when tiles are not stitched (their contacts are not measured), else *Stitching OK* (green), amber *with overlaps not matched* when some overlaps were left out, saying how many of them are corner overlaps; then both stitchings' counts; the legend lists only the layers shown (a click on an entry unticks its layer); **a click** (D112) on a tile shows that tile alone (outlined black on white, named; its SEM image and design pattern in front of the others, so its own image shows in its overlaps) and, beside the plot, in words, what is wrong with it: each overlap not matched, in which stitching, with a *show why* link; the neighbours of its other overlaps; in an overlap the tile whose centre is nearer is taken. With *Overlaps not matched* shown, a click in such a box (the smallest holding the point), or *show why* (which also zooms to it), shows only the overlap's two tiles (solid and dashed outlines, both named) and each one's contacts in the box, where they were compared (nominal placement), filled for the first and hollow for the second (circles measured, squares design), named in the legend, and gives the verdict in words (pairs found and needed, each one's contacts in it); a click beside every tile clears the selection; **Load SEM images** reads the images in the background,
downsampled by a whole factor to stay under 4·10⁸ pixels in memory, `release/analysis.read_display_images`; an opacity slider, D80; **Load design** fills every `.oas` file's contacts in its stripe's colour, D93, in the tone the analysis read it: a tone-reversed file shows its holes, `release/analysis.drawn_outlines`, D106; **Load contours**, D113, detects and refines every SEM image again as the run did, `release/analysis.read_contours`, in worker processes, the centres the run used checked against its own, with `!!` where they differ, then four layers like the others, ticked once loaded: *Detected contours* and *Detected centres* (the detection method's, magenta) and *Refined contours* and *Refined centres* (after edge refinement, cyan; named after the run's edge rule, D122), every contour at full resolution; the contours are drawn for the contacts in view only, none when more than `MAX_CONTOURS` = 5000 are (a note says to zoom in); not saved with the run; the legend of the measured centres names the method the run used) and
**Registration** (`RegistrationView`, D80, D88: one figure, side by side and showing the same area (same centre and µm per pixel, `link_views`, copied after a zoom or pan with the mouse, D114): the **arrow map** (the error vectors, shaft + two head strokes, each coloured by its length dr = |error| in jet on its own colour bar, 32 steps; scale box, default so the 99th-percentile error is ~half a pitch long) and the **dx and dy colour maps** (shared colour bar); every map has its profiles beside it (`MarginalMap`): the mean per row on its right, rotated so the row's y runs down the side aligned with the map's y, and the mean per column (`row_means` over design x, same gap) below it aligned with its x, both following the map's zoom and pan; the means per column hidden unless *Mean per column* is ticked (D118: hidden, they are taken out of the layout, which kept their room otherwise, so that the maps get the height, and left out of the image menus); *Symbol size* (D121): the contacts on the dx and dy maps, *Auto* (a disk of 0.8 × the pitch, at least a pixel, growing as one zooms in, as before) or that many screen pixels across at every zoom (`draw_map(size_px)`), off in grid average; *Line width*: the lines of every profile, 2 px at start; the colour maps show their own component, the arrow map the mean dr; **View All fits the height** (D117, revising D114; `RegistrationView.fit`): at start, after a change of stripes or values, after *View All* in any map (`FitViewBox`; pyqtgraph's *A* button hidden; a profile's *View All* fits its error axis only) and after a resize unless the user zoomed or panned (then the area shown is kept), every map shows the full height of its `extent` (the contacts shown with half a pitch around them, or every cell of the grid), centred on it in x and as wide as the map is: only y is fitted (the x range asked for is one pitch wide, which the equal x and y scales widen to the map, so the fit holds through resizes), and the maps keep the width the layout's stretch factors give them (a run wider than that runs past the map's sides); titles *Δr (nm)*, *Δx (nm)*, *Δy (nm)*, the colour bars *|measured − design| (nm)* and *measured − design (nm)*; the labelled ticks of mask axes as far apart as their labels need (`MaskAxis.label_px`); **a range for each mean per row** (D119, revising D115; `RowRange`): boxes just below the plot, in a row of the figure left for them, over the figure (not in its images): *Auto* (fitted to its values; the boxes show its range) or from the first box to the second (Enter applies and unticks *Auto*), kept when the stripes change; a zoom or pan with the mouse there shows its range in the boxes, *View All* there ticks *Auto* again; dr, dx and dy each have their own; **the range of each mean per row, three ways** (D127, revising D126; `RangeLines`): one line just below its range boxes, also over the figure, of the profile as drawn (*Values*, *Stripes*): *all* (over all rows), *ramp* (its largest ramp: `registration.turning_points`, a zigzag, splits the profile where it turns by more than the *wiggle height*: a top counts once the values fall more than that below it, a bottom once they rise more than that above it; `largest_ramp`: the ramp whose values span the most; a sawtooth's drop is a ramp too) and *manual* (between two lines dragged on the graph; at start on the first profile's min and max, then kept), each its range (max − min, nm) beside a tick box with a swatch of its lines; its min and max in the tooltip. Ticked (off at start, D76; each graph its own), a kind draws its min and max on the graph as two vertical lines (`style.ALL_ROWS` grey dotted, `RAMP` purple dashed, `MANUAL` black, movable, its value at its top), outside the graph's *Auto* range (`ignoreBounds`); *all* and *ramp* circle the rows of their min and max, *ramp* shades its rows; what is drawn is in the graph's images. *Ramp wiggles up to* (second row of controls, one for all three): a height in nm, or *Auto*, `RangeLines.NOISE_TIMES` = 8 × each profile's noise (`registration.noise_sigma`: robust σ of the row-to-row steps / √2), named in the ramp's tooltip; Copy profiles / Export profiles CSV give rows and columns (dx, dy, dr); no profile without affine; D81: *Values* every contact or **grid average** (`registration.grid_means`: mean per square cell, cell size in µm, default ~2000 cells and at least two pitches; colour maps then one flat square per cell, arrows one per non-empty cell from its centre, coloured by the length of the cell's mean, scale reset to ~half a cell; the means per row and per column then average the contacts of each row and column of cells, one point at its centre: `registration.band_means`, bands one cell wide from the lowest contact as the grid's, dr the mean |error| of those contacts; *Copy profiles* heads them *grid row* / *grid column*, D125), **Grid lines** on every map (on at start, D88), and **Copy image / Save image** menus with the whole figure, *dx and dy: maps, profiles, colour bar* (D116: the rectangle of the figure holding those graphs as laid out, nothing of dr, `style.Area`) or any one graph (a map with its colour bar), also in each graph's right-click menu and in a **Copy button** over the top-left corner of every graph shown (D120, `CopyButton`: a copy icon, over the figure, not in its images); a graph is cut from the whole figure rendered at 2×; D88: **Stripes** (`StripeSelector`, a drop-down of tick boxes, all ticked at start, *All*; one stays ticked): the maps, profiles, colour scales (99th percentile of the contacts shown) and summary show only the contacts of the ticked stripes; a contact belongs to every stripe whose x range (`pipeline.stripe_ranges`: its tiles' boxes, stripes `ReleaseResult.stripe` from `tile_stripes`, computed once per run and saved, D89) holds it, so one in the overlap of two stripes is in both (`pipeline.contact_stripes`; outside every range: the nearest); the moving-window errors are not fitted again). The error is **measured (SEM) − design**
(`registration_error`; moving window `G_w(SEM) − design`). `release/analysis.analyse_folder` runs only what these
tabs need: tiles (`analysis.load_image_tiles` with per-tile progress, or the contour CSVs), SEM stitching
(`stitch_method`, slipped-pair check), design stitching, the errors at the stitched ("mean") placement, merging,
the moving window (per stripe with `moving_window_per_stripe`, D123): no in-image map, no drift corrections, no RANSAC. A common shift of all contacts does not
change the moving-window residual, so the placement is not a parameter. Parameters (`release/analysis.PARAMETERS`,
a subset of `Settings`) are hidden: *Analysis → Parameters* (the research app's `SettingsDialog`, with saved
parameters on top: D136, below); after a change
it offers a restart. *Analysis → Contour diagnosis* tries the contour settings on one image first (D129, below).
*Analysis → Restart analysis* (F5) analyses the open folder again. **A failed run (D102):** a
run that stops after the stitching because fewer than 3 contacts were measured raises `release/analysis.AnalysisFailed`,
which carries what was found (the `ReleaseResult` without contacts, `moving` None; never saved) and why each tile was
not measured (`DesignErrors.skipped`); the window then shows that result's Stitching tab alone (no Registration or
Contact size tab; status bar and title say the analysis failed and nothing was saved; the folder is not added to
*Open recent*), with a layer *Tiles not measured* (hatched red, off at start, D76), the count per reason in the tab's status line,
and on the cursor line the tiles under the cursor, each with its reason (`StitchingView.tiles_at`; any result); SEM
images and design load on request as for any result; change the parameters and restart. Failures before the stitching
(e.g. no tile matched its design, D73) still show only their message and the log. *File → Open recent* (D97):
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
`maskreg.ini`, `[ftp]` keys `host`, `port`, `user`, `password`, `download_folder` (all optional), read from the folder
the user chose (*File → Folders...*), else the path in the `MASKREG_CONFIG` environment variable, else
`%LOCALAPPDATA%/MaskReg/maskreg.ini` (`GenericConfigLocation`; `config.config_path`);
pre-set values are used over remembered ones, and with a pre-set host the dialog connects when it opens. The
download folder is the one the user chose, else the file's `download_folder`, else `Documents/MaskReg data`
(`config.download_root`). A malformed file is reported and ignored. **Folders (D96):** *File → Folders...*
(`release/folders_dialog.FoldersDialog`) sets both folders (preferences `config_folder`, `download_folder`; empty = the
default, shown greyed in the field) and says whether `maskreg.ini` is found in the folder typed.

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
starts in the open folder), shown in the pixel frame (x, y px, y down; no grid over the image) as the contacts are found
in it (`pipeline.prepared_image`: with `background_px`, the flattened image, D130; with `upscale`, the upscaled image,
stretched over the image's pixel frame so that the axes and contours stay in pixels of the image as loaded, D131;
denoised before that with `denoise`, D132; the
threshold mask and the grey level under the cursor are of that image too). Beside it the contour settings (D128) in
groups, *Pre-processing*, *Detection (thresholding)* and *Edge refinement*, named as in *Parameters*, what each does in
its tooltip; those that change nothing given the others are greyed (`min_solidity` unless band, the refinement's options
with refinement none). Every change finds the contacts again in a background thread, 150 ms after the typing pauses, for
the newest settings only (a search that finishes for older settings is not shown; the newest are searched then):
`contour_dialog.diagnose` calls `pipeline.find_contacts`, the run's own function, and `contact_mask` for the threshold
mask; about 0.2 s for a 2048² tile with 441 contacts. Layers (check boxes; the legend lists those shown, named after the
method and the rule): *Threshold mask* (the pixels the threshold takes as contact, translucent amber; off at start,
D76), *Detected contours* and *centres* (magenta), *Refined contours* and *centres* (cyan; not offered with refinement
none, D124), on once an image is loaded, as after *Load contours* (D113). One line in words: how many contacts were
found, the method and the threshold's grey level, and the regions left out per reason (`DetectedContacts.left_out`, then
the contrast check); invalid settings are named there instead, in red (`check_settings`), and cannot be applied. Below
the image: the cursor's x, y and the grey level of the pixel under it. *Copy image* / *Save image...* as every figure.
**Apply to parameters** keeps the settings for the next run, as *Parameters* does (`MaskRegWindow.apply_parameters`: the
remembered parameters, and a restart offered when they differ from those of the analysis shown); **Cancel** leaves them.
Enter in a setting closes nothing (no default button).

**Release: Saved parameters (D136).** *Analysis → Parameters* is `release/parameters_dialog.ParametersDialog`: the
research app's `SettingsDialog` with one row on top: *Saved parameters* (a drop-down of the saved sets, by name in any
case), **Load**, **Save as...** and **Delete**, and a line saying what was done. *Save as...* asks for a name (spaces
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
│  ├─ analysis.py            # end-to-end flow of one folder (contour CSVs / images), D62
│  ├─ results_file.py        # save / load an AnalysisResult (.npz), D63
│  ├─ app.py                 # standalone analyzer: python -m affine_ransac.app, D64
│  ├─ release/               # MaskReg, the minimal app for users: analysis, cache, style, views (+ contact size tab, D95), app (D77), ftp + ftp_dialog + config (D82, D83), folders_dialog (D96), update (D101), contour_dialog (D129), parameters_dialog (D136)
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
| D140 | 2026-10-26 | The check after the stitching image by image (D137) also asks whether **another shift of one image against the other agrees clearly better** than where they are (`OverlapCheck.better`: `placement_options` of tile j against tile i, searched from where tile j is, which `placement_options` now does for every placed tile; `ties`); if so the overlap does not line up, though most of its contacts pair. `overlap_checks` takes `search_nm` (`max_match_nm`); the log line counts these overlaps; the Error line names the reason. Results unchanged (`ANALYSIS` stays 7). | Step 4 of the D137 plan, first part. Found in step 2 (D138): with a fifth of the contacts missing, an image a pitch off still pairs 79–92 % in its overlaps, above the fixed half, so the check passed it. Synthetic (12 tiles, 50 nm grid, 20 % missing, tile 5 a pitch off): its four side overlaps flagged (each 81–94 % paired, the shift back clearly better); a corner the slip enlarges (80 %, too few contacts to tell) is not; the right placement and a slip on the full grid: none. Chosen here: the same counting-noise rule as the placement, not a higher fixed share, which would flag overlaps of images that miss contacts on real data. Cost 4.2 s for the 268 overlaps of 80 images of 2.88 µm. |
| D139 | 2026-10-26 | Stitching image by image: **the overlaps are paired within the wide gate and used only if they line up** (step 3 of the D137 plan): `pipeline.stitch_image_by_image` pairs every overlap where the placement puts its images, within a quarter of the contact spacing (`seen_pairs`), uses it if at least `min_matched` and half of the contacts the two should see pair, and fits its shift robustly; the solve is shared with `stitch_tiles` (`solved_stitching`). `overlap_tolerance_nm` now only sets the offset the check allows (help text says so). An overlap left out explains itself on the Stitching tab in those terms. `ANALYSIS` 7: runs stitched image by image give other numbers; the test folders those of 6. | Before, the overlaps were paired within `overlap_tolerance_nm` of the placement, so the one setting decided which true pairs were lost to the noise, and an overlap counted as matched with `min_matched` pairs of many. Synthetic control, same correct placement, 1 nm contact noise (12 tiles, 50 nm grid): pairing within 2 nm left 3 overlaps not matched, within 1 nm 6, within 0.5 nm 20 and 11 of 12 tiles not stitched (the user's 0.5 nm run: 39 tiles not measured); the new pairing none. The precision is the same (largest tile error 0.5 nm at 1 nm noise either way; 0.15 nm at 0.3 nm noise: it scales with the noise), so this is about what is lost and what may enter, not about precision. An overlap whose images the placement put 20 nm off is left out, not matched on a few contacts that happen to pair. Not changed: the D112 drawing of a not-matched overlap shows its contacts at the nominal placement, not where the placement compared them. |
| D138 | 2026-10-26 | Stitching image by image: **the placement chooses by agreement** (step 2 of the D137 plan). Each candidate shift is refined until it settles (pairs within a quarter of the contact spacing, robust fit, paired again; `fitted_placement`) and judged by the share of the contacts the images should see that pair (`seen_pairs`, as the check). It competes if it pairs `min_matched`, half as many as the most and half of what it should; of those the best and those tied with it up to counting noise (`ties`), the smallest of them. Images with one such shift are placed first; tied ones (nothing in their overlaps tells the shifts apart) only when none is left (as `consistent_choice`, D49). `overlap_tolerance_nm` no longer enters the placement (`placed_image_by_image` lost `tolerance_nm`); it still pairs the overlaps after it and sets the check. `ANALYSIS` 6: runs stitched image by image give other numbers; the release's test folders, stitched over all overlaps, the numbers of 5. | The user agreed the plan (D137) and said the mismatch in their badly stitched images is about the same all along an overlap (a shift per image is the right model). Measured on synthetic mosaics before the change (one change at a time): a 50 nm grid with a fifth of its contacts missing and one image 40 nm off its neighbours: that image placed one pitch off at `overlap_tolerance_nm` 10, 2 and 1 nm alike (the shift −10 nm pairs 80 % and is smaller), and the D137 check passes it (79–92 % agree); 80 images of 2.88 µm with 5 % of contacts missing, stage scatter 20 nm: 52 of 80 a pitch off at 2 nm; 1 nm contact noise, full grid: all 12 images placed at 2 nm but not all right, 4 at 1 nm, 1 at 0.5 nm (the user's 0.5 nm run lost 39 tiles). After: all exact; the 80 images all within 0.11 nm of the truth; with 1 nm noise all placed, right up to whole pitches. First try, rejected: ties within a fixed 0.05 of the best agreement (on 5 % missing contacts a pitch off loses about 0.05, so 53 of 80 slipped), and no floor on the pairs (a one-row corner overlap that happened to be complete chose for its later neighbours, three images a pitch off). Cost 8.3 s for the 80 images (7.3 s before), with tree pairing in `seen_pairs` (16 s with `match_points`' dense distances). Not done: the D137 check still passes a whole-pitch slip where 80 % of the contacts repeat (`MIN_AGREEMENT` ½); step 4 is to look at that. |
| D137 | 2026-10-26 | Stitching image by image: **a check that can fail.** After the stitching, every SEM overlap is checked where the stitching placed its images, whatever paired it (`pipeline.overlap_checks`, `contact_spacing`; `analysis.checked_overlaps`, used by both apps): the share of the contacts both images should see that pair within a quarter of the contact spacing, and how far apart the two images lie. Below half, or further apart than `overlap_tolerance_nm`: logged with `!!` and an *Error* on the Stitching tab (`ReleaseResult.slipped`, same array, so `FORMAT` is unchanged); the log line gives the median and worst of both. It replaces the comparison with the placement (D133). Results unchanged: `ANALYSIS` stays. | User, real data (50 nm pitch), image by image: at `overlap_tolerance_nm` 1 nm "no error in stitching", yet badly stitched images and registration errors up to 20 nm; 0.8 nm fewer of them; 0.5 nm "39 tiles not measured" and the run stopped (a moving-window position without contacts). The mismatch is about the same all along an overlap (user), so a shift per image is the right model. Review of the code: the old check paired within the tolerance and tested within the same tolerance, so it could never fire (proven: the mean of differences each within the tolerance is within it); the final stitching can then only reproduce the placement to within the tolerance; an overlap counts as matched with `min_matched` pairs out of many; the placement accepts a shift pairing half as many as the best and takes the smallest of those; and one tolerance sets the votes, both pairings and the check, so it alone separates right from wrong placements and no value is right everywhere. Which of these the user's bad area went through is not known. Plan agreed with the user: (1) this check; (2) the placement accepts only shifts that line up as well as the best (the smallest only among those); (3) the final stitching pairs within the wide gate, its robust fit sets the precision, and only overlaps that line up are used; (4) images whose overlaps do not line up are placed again, what still does not is left out and reported. Chosen here, not asked: the gate from the contacts (no new setting, any pitch); the agreement on the image with fewer contacts and the boxes of each image's contacts, not shrunk (D133's tries failed on boxes that emptied thin overlaps; synthetic two-row overlaps agree 100 % at the right placement); `overlap_tolerance_nm` as the offset allowed. Not drawn on the plot yet. Synthetic (12 tiles, 50 nm grid, noise 0.3 nm): right placement, every overlap ≥ 95 % paired, < 0.5 nm apart; an image 20 nm off: its overlaps < 10 % paired; 5 nm off: all pair, 5 nm apart; a whole pitch: lines up. |
| D136 | 2026-10-26 | Release: **saved parameters** in *Analysis → Parameters* (`release/parameters_dialog.py`): save the parameters in the dialog under a name, load a set into the dialog (OK uses it), delete one. Kept in the preferences (`parameter_sets`); a set holds every parameter a user can change, of both kinds of folder; one it lacks takes its default. Results and run files unchanged (`ANALYSIS`, `FORMAT` stay). | User: "I have quite a lot of parameter settings now and found that I need different settings for different datasets ... save and load parameter settings with my desired name." Chosen here, not asked: in the Parameters dialog (where they are edited) rather than a menu of its own; Load fills the dialog, so the set can be checked or changed before OK; kept in the user's preferences, as the parameters in use are (not files to share; can be added if wanted); both kinds of folder in every set, since the dialog shows one kind and passes the other through unchanged; a missing parameter takes its default (the rule for remembered parameters, not `BEFORE`, which is for saved runs), so a set always loads the same. |
| D135 | 2026-10-26 | Default `overlap_tolerance_nm` 2 nm (was 10 nm, D133). Only the stitching image by image uses it; runs keep the value they were made with, so `ANALYSIS` and `FORMAT` are unchanged. | User, real data (50 nm pitch): image by image works, but needed a much smaller tolerance than 10 nm; 2 nm used. |
| D134 | 2026-10-26 | With the stitching image by image, **the SEM–design pairs follow the stitched placement**: `registration.design_errors(follow_stitching=True)` (release and research app, that method only): tied tiles (a repeating pattern) choose among their candidates by the shift after the corrections, which must agree between neighbours, not by the shift at the nominal placement; clear winners unchanged; the neighbour jumps compare the shifts after the corrections. Other methods unchanged (`ANALYSIS`, `FORMAT` unchanged). | Step 3 of the plan agreed with the user for D133: placed image by image, a tile can sit a whole pitch from where its own pairing at the nominal placement puts it (the stage errors differ between neighbours by more than half a pitch, so their nominal shifts do too), which gave that tile's errors about one pitch. Synthetic (12 tiles, 50 nm grid, stage scatter 20 nm): spread of the errors over all tiles > 25 nm before, < 3 nm following the stitching. |
| D133 | 2026-10-26 | **Stitching image by image** (`stitch_method` "image by image", setting `overlap_tolerance_nm`, default 10 nm; default method unchanged, so `ANALYSIS` and `FORMAT` are unchanged): the images are placed one at a time, each matched against all its placed neighbours with one shift (`pipeline.placed_image_by_image`, `matching.scored_shifts_together`); of the shifts that line up the one nearest the stage position; every overlap is then paired at that placement within `overlap_tolerance_nm` and solved as before. `scored_shifts` is now the one-part case of `scored_shifts_together` (same numbers). The release log and status name the method's own check (slipped pairs against its placement). | User's new data (pitch 50 nm, 80 tiles): "Stitching OK. All 80 tiles stitched.", yet images clearly off in the Stitching tab and registration values far too large in places; identical results at `max_match_nm` 50, 100 and 300 nm. Diagnosis from the code: nearest-contact pairing at the nominal placement sees an overlap's offset only modulo the pitch; with any gate above P/√2 nothing is rejected, so the stitching cannot fail; the slipped-pair check (tolerance `max_match_nm`, at least one pitch here) cannot see a one-pitch slip; overlaps whose whole-pitch errors differ contradict each other and the solve spreads it. Why the offsets exceed 25 nm on these data is not known (stage, FOV, rotation, false contacts). The user asked for a solution, not only diagnostics; the measurement-order guide (D75) is a chain, one bad link carries into every tile after it. Synthetic (12 tiles, 50 nm grid, stage scatter 20 nm): all overlaps, worst overlap disagreement > 10 nm; image by image < 2 nm, every tile right up to whole pitches; scatter 4 nm: the same corrections as before within 0.2 nm. First tries, rejected: overlap contacts cut at the box (a whole-pitch shift scored best), then grown by the search (shifts into an image's interior scored best), both choosing by mismatches. |
| D132 | 2026-10-26 | **Denoising** before the contacts are found: `denoise` (none, the default, so `ANALYSIS` and `FORMAT` are unchanged; median, non-local means, total variation), each with its own option (`median_px`, `nlm_h`, `tv_weight`), after the flattening and before the upscaling (`features.contact.denoised_image`, `pipeline.prepared_image`); the diagnosis window shows the denoised image, greys the options of the methods not chosen and names the steps in its line. | User, on the real ADI images: interpolation gives much better results, upscaling above 4× looks good; with `detection_blur_px` 0 the threshold follows the boundaries best but noise makes false arms stick out of the holes (bright pixels inside the dark holes, dark spots in the bright background); a human can infer the boundary where a threshold of the pixels cannot. Asked for a denoising method taken from what others use, not tried on synthetic images ("they don't help at all and just waste time"; they and their evaluation are not ground truth), the order of denoising and upscaling left open. Searched: CD-SEM contour work compares against median / Gaussian blur and reports unsupervised deep-learning denoisers best; SEM denoisers are benchmarked against non-local means and BM3D; total variation and anisotropic diffusion are the edge-preserving classical family; upscaling work denoises first. Built the three classical methods that need no new dependency, so they can be compared on the real images; median first in the list as the direct answer to isolated outlier pixels. Order fixed (denoise, then upscale), from the sources and because interpolation spreads a noisy pixel; not asked. Not built: BM3D (licence: non-commercial use only), bilateral (keeps outliers), deep learning (training on the user's images, GPU), anything after the threshold (e.g. a morphological opening of the mask, which would also cut the arms). This also corrects D130 and D131: their synthetic tests said upscaling would not help, and the user's real images show it does. Sources: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11854/2600945/Unsupervised-machine-learning-based-SEM-image-denoising-for-robust-contour/10.1117/12.2600945.short, https://www.thermofisher.com/software-em-3d-vis/xtra-library/xtras/bse-sem-denoiser, https://arxiv.org/pdf/2505.08843, https://arxiv.org/pdf/1905.02538, https://patents.google.com/patent/EP3166070B1. |
| D131 | 2026-10-26 | **Upscaling** before the contacts are found: `upscale` (1 = off, the default, so `ANALYSIS` and `FORMAT` are unchanged; up to 8): after the flattening the image is interpolated (bicubic) to `upscale` times as many pixels per side, detection and refinement work on it, and the contacts come back in the image's own pixel frame (`features.contact.upscaled_image`, `in_image_pixels`). The other settings in pixels, and `normal_span`, stay pixels of the image as loaded (scaled by the factor inside); the contrast check stays on the image not upscaled. The diagnosis window shows the upscaled image and its mask over the same pixel frame; *Pre-processing* holds `background_px` and `upscale`. | User, after trying D130 on the real wafer ADI images: "your method doesn't improve at all. Implement upscaling first and then detect the contours." (They had suggested interpolation from the start, D130.) Settings kept in pixels of the image as loaded, so that a factor can be tried without changing any other setting, and the window's axes, cursor positions and every setting keep one meaning; chosen here, not asked. Bicubic (`cv2.INTER_CUBIC`), not asked: smooth, and its half-pixel-centre convention maps back exactly (checked: a spot's centroid lands on its pixel at ×2, ×3, ×4). Measured on synthetic no-band images (S2): the detected centres move closer to the truth (6 px contacts, noise 8: 0.122 → 0.081 px at ×4); the mid-level refined ones do not (0.081 → 0.081; noisy: slightly worse). Cost at ×4 on a dense 2048² tile: +2.4 s and up to 2 GB per worker process. Not built: other interpolations, refinement on the image not upscaled. |
| D130 | 2026-10-26 | Two contour options for images without a bright band (wafer ADI), both off by default (`ANALYSIS` and `FORMAT` unchanged): **background flattening** (`background_px`: a grey-level closing with a disk that wide, smoothed with σ = 2 × it over the background continued past the edges with its own slope, subtracted before every step; `pipeline.prepared_image`) and the refinement rule **threshold** (each contour point where the smoothed image crosses the detection's Otsu level). The diagnosis window shows the flattened image and has a *Pre-processing* group. `find_contacts` takes its options by name only. | User: new wafer ADI images have no white band and a pixel size large for the patterns; plain Otsu works best, but its and the refined contours do not locate the patterns; suggested interpolation and more smoothing. Measured first on synthetic no-band images (dark disks 4–16 px across, true centres known; rms per axis): ×4 bicubic upsampling and more Gaussian blur gave no gain over today's refinement, which already reads the image sub-pixel along each normal (6 px contacts, noise 8: 0.080 px today, 0.081 upsampled; more blur lost 18 % of the 6 px contacts at noise 20). What helped: flattening, where the brightness varies across the image (ramp of 80 grey levels: 57 % → 100 % of the contacts found), and the threshold rule on noisy images (0.442 → 0.403 px at noise 40, needing flattening under a ramp). Proposed with these numbers; the user did not fully agree but asked for this to test on the real images; their symptom on the real data is not known yet. σ and the odd reflection chosen on the same synthetic images (above, S2). Not built: denoising (non-local means helped noisy images but erased 4 px contacts), upsampling. |
| D129 | 2026-10-26 | Release: **contour diagnosis window** (*Analysis → Contour diagnosis...*, `release/contour_dialog.py`): load one SEM image, set the contour settings (D128) and see the detected and refined contours and centres (the threshold mask on request) found again at every change, in a background thread; one line in words with the count and the regions left out per reason; *Apply to parameters* keeps the settings for the next run (offering a restart, as *Parameters* does), *Cancel* does not. | User: "add a 'contour diagnosis' UI in the release app ... a popup window rather than a dedicated tab ... opened through one of the menu buttons": load individual SEM images, set any parameter of the thresholding and contour extraction, see the extracted contours and centres in real time, then apply them to the parameter settings or cancel; the next run uses them. The window calls the run's own function (`find_contacts`), so what it shows is what the run measures. Chosen here, not asked: the Analysis menu beside *Parameters*; the mask off at start (D76) and the contours on once an image is loaded (as *Load contours*, D113); the left-out regions counted in words, not drawn (most are normal: noise specks, contacts cut by the border, the band method's background); pixel axes, since every setting is in pixels and a loose image has no field of view; the grey level under the cursor, to compare with the threshold; a background search (~0.2 s on a 2048² tile) so that typing stays smooth. Checked on a synthetic 2048² tile (2.88 µm, 130 nm pitch, 441 contacts) with Windows fonts. |
| D128 | 2026-10-26 | **Every option of the contact detection and the edge refinement is a setting** (`analysis.CONTOUR_SETTINGS`): new `dark_contacts`, `detection_blur_px`, `min_area_px`, `max_area_px` (0 = no limit), `min_solidity` (band only), `edge_step_px`, `edge_blur_px` and `normal_span`, beside `detection`, `min_contrast`, `refinement` and `search_px`. Their defaults are the values the code had, so `ANALYSIS` and `FORMAT` stay (a run saved before takes the defaults, which is what it did). `pipeline.find_contacts` finds an image's contacts with them; `process_tile` calls it (its first ten parameters keep their places: the research notebook passes them by position), and `load_image_tiles` and `read_contours` now pass the options by name. Release *Parameters* lists them. `DetectedContacts.left_out` counts the regions detection leaves out, per reason. | User asked for a contour diagnosis window in the release app where "any parameter related to the thresholding and contour extracting methods" can be set and applied to the next run (D129); most of them were constants in the code. Taken: the parameters of `detect_contacts` and `refine_edges`; not the mid-level rule's own constants, the contrast ring or the three classes (chosen here, not asked). `min_solidity` stays band only, as it was: applied to otsu it would change those runs' results. One function for the run and the window, so the window shows what the run will measure. |
| D127 | 2026-10-26 | Release Registration tab, **revises D126**: the range of each mean per row is drawn where it is measured. Under the range boxes one line per graph, *all* / *ramp* / *manual* with their ranges (max − min) and a tick box each (`RangeLines`); ticked, a kind draws its min and max as two vertical lines on the graph in its colour (grey dotted, purple dashed, black), *all* and *ramp* with the rows of their min and max circled and the ramp's rows shaded; *manual* is two lines dragged on the graph, their values at their top. The min and max values are in each kind's tooltip. Off at start (D76). The table, the ramp's "at y … to … µm" text, the y interval and the Auto heights beside the wiggle control are gone. | User: "You have thrown away all UX. The output is not aligned with anything and it just bombards the UI with numbers. Moreover, the y interval setting is not what I wanted. I wanted to manually set the min/max lines by the user. Also, None of the all rows, largest ramp settings are displayed on the graph so I don't know where exactly the numbers are calculated from. That 'at y #### to #### um' text is absolutely horrible UX." Asked with sketches, the user chose all three kinds drawn at once, each in its own colour, with one line of ranges under the graph, and manual lines set by dragging only. Lines and circles show where each number comes from, so the positions need no text. Checked on screenshots of a synthetic field (3 stripes, 31 µm, 130 nm pitch; dy a 1 nm sawtooth per tile, dx a slow bump) at 1400 and 1920 px: the three lines of ranges (~350 px) do not overlap; on the sawtooth the largest ramp is one tooth's one-row drop (1.10 nm vs rises of ~1.0 nm + noise), shown by its two circles. |
| D126 | 2026-10-26 | Release Registration tab: **min, max and range (max − min) of each mean per row** (dr, dx, dy), printed in a table under its range boxes (`RowStats`, widgets over the figure like `RowRange`, so not in its images): over all rows; of the **largest ramp**, with where it is in y (`registration.turning_points`, a zigzag: the profile is split where it turns by more than the wiggle height, so wiggles up to it do not end a ramp; `largest_ramp`: the ramp whose values span the most); inside a **y interval** of each graph's own (boxes in mask µm, or a band dragged on the graph once *show* is ticked; off at start, D76). Wiggle height: one control, *Ramp wiggles up to* (nm or *Auto*), in the second row of controls; *Auto* = 8 × each profile's noise (`noise_sigma`: 1.4826 × MAD of the row-to-row steps / √2), the heights taken printed beside it. View only: `ANALYSIS` and `FORMAT` unchanged. | User: "an automatic 'range' calculation for mean per row graphs ... min-to-max, local min-to-max of a slope, or user manual setting of min/max etc. ... I want ALL types of min/max values printed in the UI." Asked (with sketches), the user chose: the slope's range = the height of one ramp (split at turning points, small wiggles ignored), the largest printed with where it is; the manual one = a y interval, typed or dragged; printed under each graph. The axis-range boxes of D119 stay as they were (they set the view, not a measurement). Wiggle multiple from synthetic profiles (300–10,000 rows, 30 trials each): at 5–6σ a slow 1 nm trend under noise is cut into short "ramps" (largest found 0.45 nm of 1 nm at 3000 rows, 5σ), because noise alone wiggles by about its expected range, ~6.5σ over 1000 rows; at 8σ it stays one ramp up to 10,000 rows, and every tooth of a 1 nm sawtooth is found while the noise is ≤ 0.1 nm (10σ). Any min-to-max of noisy values includes noise at both ends (a 1 nm sawtooth with 0.05 nm row noise reads ~1.2 nm). The wiggle height is one control rather than one per graph (chosen here, not asked): it sets what counts as a ramp, like *Line width* sets every line, while *Auto* still takes each profile's own noise. Layout: the first table, with the details beside the values, was ~520 px wide and the three overlapped at 1400 px; stacked (details on their own lines) it is ~300 px. |
| D125 | 2026-10-21 | Release Registration tab: in **grid average** the means per row and per column follow the grid: each point the mean of the contacts of one row (column) of cells, at its centre (`registration.band_means`: bands one cell wide from the lowest contact, the grid's rows and columns). Every contact counts once, so a band's dx and dy are its cells' means weighted by their contacts; dr stays the mean error length of the contacts (D88). *Every contact*: the rows and columns of contacts, as before. *Copy profiles* / *Export profiles CSV* give what is shown, headed *grid row* / *grid column*; the summary still counts the rows and columns of contacts. No smoothing control. | User: "The release app contains grid averaging functionality but this only applies to the maps. I want this to be applied to the mean per row graphs as well. Or alternatively, there could be some sort of smoothing functionality on the line graphs." The first: the *Values* control already chooses the averaging, so the profiles follow it and stay aligned with the map's rows of cells, and the cell size sets how much they are averaged (one control for every display, as in D81). Contacts weighted, not cells: a cell with few contacts at the field's edge would otherwise count as much as a full one. The means per column follow too (chosen here, not asked: *Values* then means the same for every graph; they are hidden at start). Synthetic contour folder (3 stripes × 12 images, FOV 2.88 µm, pitch 130 nm, 0.8 nm noise per contact, a dy wave of 0.4 nm amplitude and 8 µm period; rms of the dy mean per row about the wave, a fitted line removed): every contact 0.110 nm (242 rows); grid average 0.2 µm 0.097 nm (157 rows of cells), 0.5 µm 0.072 nm (63), 1 µm 0.063 nm (32). |
| D124 | 2026-10-19 | **One setting for the edge:** `refinement` = mid-level (default) / steepest gradient / **none** (no refinement: the detected contours and centres are measured; `process_tile` keeps `refined` = the detection, so the design tone, the stitching, the contact sizes and *Load contours* follow it). The check box `stitch_refined` is removed; stored values are read in today's names (`analysis.current_names`: unticked → none) in saved runs (`saved_settings`) and in the remembered parameters of both apps. With none the *Refined* contour layers are not offered. The help texts of `detection` (when to use each method) and `refinement` are rewritten; "Otsu centres" is gone. Release `ANALYSIS` 5: with refinement none the design tone is now chosen with the detected centres (the unticked check box used the refined ones there). | User: the Parameters window shows settings left over from the past: the 'edge refinement' check box still changes results, while the Stitching tab no longer seems to show a refined contour; what is refinement now, and what happened to otsu and otsu3? Diagnosed before any change (synthetic, 182 contacts per case): nothing was missing. With band detection the mid-level contour lies on average 0.2 px inside the detected pixel outline, ±0.3 px, and is drawn over it; the steepest gradient used to stand out by jumping ±0.4–1.2 px from point to point. From otsu / otsu3 outlines (lower on the edge) the refined contour lies 1.4–2.5 px outside. After mid-level refinement the refined centres from otsu, otsu3 and band agree to 0.004–0.06 nm: the detection decides which contacts are found, not where their edges are. The check box moved centres by 0.04–0.18 nm (error: detected 0.053 → refined 0.045 nm at 1.4 nm/px, 0.175 → 0.127 at 2.8 nm/px). The user chose one setting with three choices and kept all three detection methods, with clearer help. |
| D123 | 2026-10-19 | Release: **moving-window affine per stripe** (`moving_window_per_stripe` in *Parameters*, off by default, so runs give the numbers they gave): windows along y within each stripe; every image's contacts (its observations) corrected by its own stripe's windows; a contact seen by two stripes gets the mean of both corrections. `MovingWindowResult.stripe` is optional in run files (`results_file._get_fields`: a field with a default may be missing), so `FORMAT` stays 3. The Registration tab's summary says how the window was fitted. Not in the research app (`analysis.RELEASE_ONLY`). `ANALYSIS` unchanged. | User: the moving-window affine runs along y over all stripes, but tiles side by side in x belong to different stripes, measured far apart in time, so they drift differently; asked for the moving window per stripe as a release option. Asked how contacts in the overlap of two stripes are corrected, the user chose each image by its own stripe and the contact the mean (the alternative: the overlap split at its middle). Synthetic contour folder (2 stripes × 3 tiles, 0.1 nm noise; rms per axis, all stripes → per stripe): noise only 0.098 / 0.097 nm both ways; an x scale of ±300 ppm per stripe 0.258 → 0.098 nm (the noise); a y scale of ±300 ppm per stripe 0.324 → 0.157 nm (only in part: the translation stitching turns the stripes' disagreement in their overlaps into per-tile shifts). |
| D122 | 2026-10-19 | **Edge refinement at the mid-level** (`refine_edges(rule="mid-level")`, setting `refinement`, the default; the steepest gradient stays a choice): each contour point goes where the smoothed image rises through half-way between its contact's own interior and band levels; points without a crossing, or isolated far from their ±3 neighbours, take their neighbours' median. Release `ANALYSIS` 4. Saved runs without the setting are read as "steepest gradient" (`analysis.saved_settings`, `BEFORE`); remembered parameters take the new default. The *Refined contours / centres* layers name the rule; the notebooks have a `REFINEMENT` setting, and `tile_index.ipynb`'s contour check draws both rules. | User: band detection is the most robust so far but not good enough, and edge refinement adds error on low-contrast, low-resolution, noisy images (on their data: contours visibly wrong); asked for a method robust for all kinds of SEM images, proposed before it was built, and for it as the default. Measured on synthetic tiles (2 per case, ~380 contacts each, true centres known; bright band, beam blur, noise, JPEG; rms per axis after removing each image's common shift): the steepest gradient vs mid-level, noise 25 at 1.4 nm/px 0.184 vs 0.079 nm, 2.8 nm/px noise 15 0.142 vs 0.092, wide band 0.213 vs 0.079, hole as grey as the background 0.272 vs 0.118, one-sided band + scan smear + brightness ramp 0.155 vs 0.054, rounded rectangles 0.219 vs 0.118 (1.4–3.2× less scatter over 15 cases); the plain band centroid was never better than mid-level beyond the test's noise; contacts with a 4 nm edge bump kept their true (bumped) centroid (0.044 nm). Rejected after measuring: fitting the image's own mean contact to each contact (with one side of the band brighter, size variation became position error: 2.9 nm rms for 3 nm CD scatter, noise-free); the same with an affine warp per contact (7–30 % more precise at low resolution, but 30–100× slower in the prototype, one kind of contact per image, and its fit diverged once); the band's outer flank, more smoothing (1.5 px instead of 1: 8 % less error at 1.4 nm/px, 9–16 % more at 2.8 nm/px), per-ray profile matching. Synthetic only: whether it helps on the user's images shows on their data (contour layers, maps). Not done: finding contacts the band detection loses (16–60 % at low contrast with noise; correlation with the image's own mean contact found them all, but failed when the band detection produced junk regions under a brightness ramp); the user reports wrong contours, not missing contacts. |
| D121 | 2026-10-19 | Release Registration tab: **Symbol size** and **Line width** in the second row of controls. *Symbol size* (0–50 px, 0 shown as *Auto*): *Auto* draws each contact of the dx and dy maps as before (a disk of 0.8 × the pitch, at least a pixel, at most `MAX_DOT_PX`, growing as one zooms in); a size draws every contact that many screen pixels across at every zoom (`view_registration.draw_map(size_px)`: radius (size × device pixel ratio − 1) / 2 device pixels; `rasterize` takes a fractional radius, the research view's whole radii drawn as before); where disks overlap, the mean is drawn. Off in grid average (cells, no symbols). *Line width* (1–10 px, 2 at start): the lines of every mean per row and per column; not the arrows, nor the dashed zero lines. Not kept between runs. | User: "Add map symbol size and line plot thickness option." A size in screen pixels, as symbol sizes are in plotting programs; *Auto* keeps the maps as they were. |
| D120 | 2026-10-19 | Release Registration tab: **a Copy button over every graph** (`CopyButton`): over the top-left corner of each map, mean per row and mean per column (above its y axis, beside its title), shown while its graph is; it copies the graph to the clipboard at 2× (`style.render_graph`), a map with its colour bar, as the image menus' entries do. A copy icon (two sheets, drawn by `copy_icon`), named in its tooltip: the word *Copy* covered the start of a mean per row's title, whose y axis is ~35 px wide. Qt widgets over the figure, placed when their graph moves, so not in its images. | User: "Add copy image buttons for individual maps and plots." |
| D119 | 2026-10-19 | Release Registration tab, **revises D115**: every mean per row (dr, dx, dy) has **its own range**, in boxes just below it (`RowRange`: *from*, *to*, *Auto*): *Auto* at start (fitted to its values, the boxes showing its range); a value typed in (Enter, `keyboardTracking` off) unticks *Auto* and sets that plot alone, kept when the stripes change; a zoom or pan with the mouse there shows its range in its boxes; *View All* there ticks *Auto* again. The boxes are Qt widgets over the figure (children of its view), placed below their plot whenever it moves (`geometryChanged`), in a row of the figure's layout left for them (row 1; the means per column moved to row 2), so they stay just below it with the means per column shown or hidden, and are not in the figure's images. The shared range of D115 (a second row of controls, dx and dy only) is removed. | User: "Range selection must be done independently for all 3 plots. I never told you to group them and fit them in a single range." "Add individual range settings to individual mean per row plots. Place them suitably just below the plot." |
| D118 | 2026-10-19 | Release Registration tab: the **means per column are hidden** unless *Mean per column* (a check box in the second row of controls, off at start) is ticked (`show_columns`). Hidden, they are taken out of the figure's grid layout, and the maps get the height; ticked, they go back below their maps, aligned with them. While hidden they are not in the *Copy image* / *Save image* menus (nor in *dx and dy*); *Copy profiles* and *Export profiles CSV* still give the columns. | User: "Mean per column is not so necessary at most times. Hide it." ("at most times": so it can still be shown.) Hiding the plots alone left their room: pyqtgraph's grid layout kept a hidden plot's least height (78 px) and the row's share of the height (193 px at 1600 × 900); out of the layout, a map's height went from 499 to 698 px. |
| D117 | 2026-10-19 | Release Registration tab, **revises D114**: *View All* and the start fit **only y**, and the maps keep their width. `fit_layout` (each map shaped like its contacts, the room it did not need given to its profiles) is removed, with the colour bars' least height and the mean per column's title over its axis that served it. `fit`: the y range is the extent's; the x range asked for is one pitch wide at the extent's centre, which the equal x and y scales widen to the map's width, now and after any resize, so the fit holds without being done again. Kept from D114: *View All* fits (no longer zooms out), the *A* button hidden, a zoom kept on resize, `link_views` on the mouse only, short titles, `label_px`. | User: "The view all must only fit the y axis. You shrunk down x to fit the data, causing the map to be extremely narrow, and mean per row graph to be exceptionally big. This isn't what I wanted. Revert the x axis resizing. This actually is harmful when the user zooms in a point and the map is too small that the user cannot actually see whats going on near by." (D114 had made a 5 × 35 run's maps 81 px wide and its means per row 288 px.) The one-pitch x range: first tried, the x range of the map's width at that moment; when the width then changed, the equal scales kept that x and moved y, the mean per row's y labels and the width of its axis followed, and the map's width with them: on a synthetic wide run the maps swung between 398 and 420 px at every event. Not seen with the code before D114. A run wider than the map at that scale runs past its sides (asked: fit y only). |
| D116 | 2026-10-19 | Release Registration tab: the *Copy image* and *Save image* menus list first, after *Whole figure*, **dx and dy: maps, profiles, colour bar**: the rectangle of the figure that holds the dx and dy maps, their means per row and per column and their colour bar, as laid out, cut from the figure rendered at 2× (`style.Area`: graphs taken together as one rectangle for `render_graph`, which otherwise puts graphs side by side). Nothing of dr: its graphs are left of that rectangle. | User: "Copy image for dx and dy related graphs only excluding dr." One rectangle rather than the graphs side by side, so that each profile stays beside or below its map, aligned with it. |
| D115 | 2026-10-19 | Release Registration tab: **a range for the means per row of dx and dy** (`RegistrationView.set_row_range`), in a second row of controls: *Auto* (ticked at start: each fitted to its values; the two boxes, greyed, show the widest range of both) or, unticked, from the first box to the second (nm, applied on Enter: `keyboardTracking` off), the same for both and kept when the stripes change. A zoom or pan with the mouse in either gives its range to both and unticks *Auto* (`row_range_from`); *View All* there ticks it again. The mean dr per row keeps its own automatic range. | User: "A separate range selector(min,max) for the mean per row graphs. I can set them by right clicking and setting the range of x axis, but this also has an odd behavior(popup disappearing while typing in the numbers) and isn't very comfortable to use." dx and dy share one range, as they share a colour scale, so that the two compare at a glance and keep their scale from one stripe or run to the next; dr (a length, ≥ 0) is left out: a range for dx and dy (e.g. −3 to 3 nm) would leave half of its graph empty. The boxes follow the mouse so that they always say what the graphs show. |
| D114 | 2026-10-19 | Release Registration tab: **the maps show their contacts just**. `RegistrationView.fit`: every map shows its `extent` (the contacts shown with half a pitch around them, for their disks and arrows of ~half a pitch; grid average: every cell) at start, after a change of stripes or values, after *View All* in any map (`FitViewBox`: View All calls `fit`; in a profile it fits the error axis only, the other follows the map) and after a resize unless the user zoomed or panned (then the area is kept). `fit_layout`: each map shaped like its extent, as large as fits in the room the stretch factors give it; the room it does not need goes to the mean per row beside it (width) or the mean per column below (height); only maximum sizes are set. A map is never narrower than the title of its mean per column (now over that graph's axis too: ~81 px) nor shorter than the colour bars' labels (~200 px with the axes). Titles *Δr / Δx / Δy (nm)*; "measured − design" moved to the colour bars' labels. `link_views` copies only zooms and pans with the mouse. pyqtgraph's *A* button hidden on the maps and profiles. `MaskAxis` keeps its labelled ticks as far apart as their labels need (`label_px`: the widest label at the axis's ends + 12 px; a line of text + 8 px on a y axis). | User: "'View all' works, but doesn't actually fit properly to the ends and sometimes cuts off the map. I want the maps to be 'just' visible for all runs as the default, so that all data fits and no excess white space appears." Measured before, on synthetic runs of 7 × 25, 25 × 7, 12 × 14 and 5 × 35 images (2.6 µm apart, 130 nm pitch) in a 1600 × 900 window: at start the maps showed 2.6–9 × the contacts' height (the range was set before the window had its size, then passed between maps of other widths: 52 µm of white above and below 65 µm of contacts); *View All* in a dx or dy map fitted its own image, which is drawn for the view, so every View All zoomed out ~5 % more (margins 2.9 → 7.4 → 12.5 µm); pyqtgraph's *A* button there ended at 160 µm for 65 µm; and with equal x and y scales a map shaped otherwise than its contacts stayed 37–87 % white even when fitted. The figure's least width (the long map titles) was wider than a 1200 px window (1528 px for 1293, 12 × 14), so its right 235 px were cut off: probably the "cuts off" (not seen otherwise). Ranges copied between the maps while the window was resized (a map laid out later got the range of one laid out earlier) made the margins drift at every resize (12.5 → 18.6 µm), and cut contacts off by up to 0.09 µm once the maps were fitted without padding. After: all four fitted to within the half pitch kept around the contacts at 1200, 1600 and 1900 px, through View All and resizes, also with the GPU and in the MaskReg window (a resize while on the Stitching tab); the figure never wider than the window. Limits: the 5 × 35 maps at their 81 px least width (75 needed) keep 0.5 µm each side; runs more than ~2 × wider than tall keep white above and below; the mean per column of a wide run gets the height. Narrow maps made pyqtgraph's 2 µm tick labels run into each other (it draws two or more labelled ticks on any axis), hence `label_px` (first a fixed 50 px, which left a 149 px map one label where three fit). |
| D113 | 2026-10-19 | Release Stitching tab: **Load contours** (`release/analysis.read_contours` → `TileContours`): every SEM image detected and refined again with the run's settings (`process_tile` without its design, worker processes as for the tiles); the centres of the method the run used must equal its measured centres, else a `!!` log line names the tiles (an image changed since the run). Four layers, like the others and ticked once loaded: *Detected contours*, *Refined contours* (every contact's contour at full resolution, magenta and cyan, `style.DETECTED`, `style.REFINED`), *Detected centres*, *Refined centres* (+). Contours are drawn for the contacts in view (`update_contours`, after every zoom or pan), none when more than `MAX_CONTOURS` = 5000 are in view, with a note to zoom in. Kept when the tabs are rebuilt, not saved. The legend of the measured centres says which method the run used. | User: turning contour refinement on / off (`stitch_refined`) changes some results on their dataset; they want to compare the contours of the two methods. Asked, they chose: detected vs refined; "No just draw all contours and centers that can be checked/unchecked like any other value" (not a clicked tile, not a tab of their own); no field-wide view. Not saved because of their size: on a 2048 px synthetic image (2.88 µm, 441 contacts) the contours have 113–124 points per contact, both methods 0.9 MB per image as float32, about 160 MB for the user's 173 images, 1.4 GB for 1,500 (and a run file is kept with 5 backups); found again in about 0.1 s per image. Full resolution, not resampled: a refinement that fails at a few points of a contour (a likely reason for the results to change) must stay visible. The cap: more contacts than that are a few screen pixels each, and drawing them all is slow without a GPU. |
| D112 | 2026-10-19 | Release Stitching tab, **revises D110 and D111**. The layers *Measured / Design overlap contacts* are removed (and `pipeline.overlap_contacts`, `lower_left`). A click selects **one thing**: a tile (only its outline and name are drawn; its SEM image and design pattern come in front of the others; the panel says in words what is wrong with it, each overlap not matched with a *show why* link) or, with *Overlaps not matched* shown, a not-matched overlap's box (only its two tiles, solid and dashed, named; each one's contacts in the box where they were compared, filled and hollow, named in the legend; the verdict in words). *show why* selects that overlap and zooms to it. The status note says how many of the overlaps not matched are at corners. | User, on D110/D111: "When I click on an area, I get 9 overlaps activated. What am I supposed to look at? The reason for this functionality was to indicate 'which' tile I am on right now and 'how' it looks like, because overlap boxes are ambiguous." "I have circles and crosses and green dots all over the place, which don't provide intuitive meaning. Are crosses bad? Are circles bad? Do they need to overlap? I see a bunch of cross and circles in my clicked overlap area, and the popup box on the right doesn't show any issue at all, is this fine?" It was fine: in a used overlap some contacts are always unpaired, as each image leaves out those its own edge cuts, and D110 drew them red everywhere, with a code (ring = lower or left image) one had to learn. My mistakes: building for what was easy to draw rather than for the question asked, and checking only on the 2 × 2 test folder. Checked now on a synthetic mosaic like the user's (3 stripes × 6 images, FOV 2.88 µm, 280 nm overlaps, 130 nm pitch, stage errors ~8 nm) with `min_matched` 2: the only overlaps not matched are 4 of the 20 corners (measured); with 5 every corner is skipped and none is not matched; the inner tile clicked shows one outline, the corner shown through *show why* its 4 and 2 contacts, 1 in both. The contacts of a selected overlap are drawn at the nominal placement because the box is: drawn at the stitched one, a contact at the box's edge showed just outside it (~10 nm). |
| D111 | 2026-10-19 | Release Stitching tab: **a click selects a tile** (`StitchingView.select_at`: the tile whose box holds the point; in an overlap the one whose centre is nearer; beside every tile: none). It is outlined black on white (seen on any image), its overlaps filled by status (green used, orange / blue not matched in the measured / design stitching, grey dashed outline skipped), its neighbours outlined, all of them named, and a panel beside the plot (`tile_details`) gives its stripe, centre and stitched shifts, and per neighbour its direction and each stitching's verdict on their overlap (tie points and pairs; not matched: pairs among the contacts in it and `min_matched`; skipped: contacts in it). A click on a legend entry hides that layer and unticks its box (it hid the item and left the box ticked). | User: "The overlap, dashed boxes etc. are difficult to see 'which' tile/image they correspond to. Add an interactive function that highlights the clicked tile, corresponding tile number, boxes etc." The panel uses only what the run keeps (`StitchResult` pairs and rejected pairs), so runs saved before show it too. |
| D110 | 2026-10-19 | Release Stitching tab: layers **Measured overlap contacts** and **Design overlap contacts** (off at start): in every used or not-matched overlap, each image's contacts in its box (`pipeline.overlap_contacts`: per overlap, the points in the box and the tie contacts), green if used to stitch that overlap, else red: a ring from the lower or left image of the pair, a cross from the upper or right one (`pipeline.lower_left`). The legend lists only the layers shown. | User: "Dashed boxes don't show why it was not matched. Add a 'used/unused' contacts so that I can actually visually see what is going on"; and, with `min_matched` 2 on their data: did the overlaps not matched not find even 2 contacts to match? Yes: not matched means both images have at least `min_matched` contacts in the overlap box, but fewer pair up (mutual nearest within the gate). Found while building it, on the synthetic test folder (nothing wrong with it): with `min_matched` 2 one corner overlap of two `.oas` files is not matched. Each file has 2 contacts in the 120 × 120 nm corner but they share only one: each dropped the other's second contact because it touches its own frame (`inside_frame`; SEM detection drops border contacts the same way). With `min_matched` 5 the same corner is skipped (too few contacts). Whether the user's 115 measured and 132 design overlaps not matched are of this kind is not known; the layers show it. A first version marked a contact used if any overlap used it: the corner's contacts that only one file has were then green (used in the neighbouring strips), hiding the reason, hence the marks per overlap, with the image's side. Legend: with every layer listed it had 13 entries, and clicking an entry's eye hid the item without unticking its box. |
| D109 | 2026-10-19 | Release Stitching tab: the status line **grades** the stitching, worst first: *Error* when SEM overlaps look paired one contact row off (the D75 check, now kept in `ReleaseResult.slipped` and saved as the optional array `slipped`; `FORMAT` unchanged, older files load with None and the line says the check is in their log), *Warning* when tiles are not stitched (not measured), else *Stitching OK*, with a note when overlaps were not matched. The log's `!! … STITCHING INCOMPLETE` line and `stitch_tiles`' "stitching incomplete" warning now appear only when tiles are left without a position; overlaps not matched alone are listed plainly. | User: "'Stitching incomplete' is misleading. It doesn't show whether this is a warning or a severe failure that results in untrustable results." Their run: 173 of 173 tiles stitched in both stitchings, 115 measured and 132 design overlaps not matched, shown in red as *Stitching incomplete*. What each means, measured on a synthetic field (4 × 20 tiles, 130 nm pitch, a ±2 nm mask error, release flow): overlaps not matched with every tile stitched move the errors of the contacts shown by at most 0.04 nm for one overlap and RMS 0.07 nm, max 0.33 nm for 11 of 136; a tile not stitched leaves its contacts out (none is guessed, D27); one overlap paired one row off, which no box shows, moved 21 tiles by more than 5 nm, up to 34 nm. So only the last makes results untrustable, and it was only in the log. |
| D108 | 2026-10-19 | Release Stitching tab: tile names are dark text on a white label with a thin border (`views.name_label`). | User: "Gray tile numbers shown on gray SEM images is a terrible idea." The names were grey text (`style.MUTED`) without a background. |
| D107 | 2026-10-19 | Release Stitching tab: **overlaps not matched** are boxes **filled** in the colour of their stitching (measured orange, design blue, like their centres), **tiles not stitched** thick red outlines, tiles not measured (failed run) hatched red; *Overlaps not matched* and *Tiles not stitched* are separate check boxes (were one, *Stitching failures*), both off at start. These boxes are `pg.BarGraphItem`s, which QPainter draws on any viewport. The controls take two rows (layers; SEM images and design). | User: "your 'dashed box' display does NOT actually show dashed boxes. They also show a thick solid tile outline and I cannot distinguish between them except for the legend." Cause (pyqtgraph 0.14 source): on an OpenGL viewport `PlotCurveItem` draws its lines itself (`paintGL`, `glDrawArrays`) with the pen's colour and width only, never its style, so every dashed or dotted box came out solid; MaskReg draws on the GPU when it can (D80). Copy / Save image render through QPainter, so they showed the dashes the screen did not. A screenshot of this PC's GPU drawing could not be taken (the off-screen OpenGL view stayed blank), so the cause rests on the source; the new boxes do not use that path at all. Also: with both stitchings in one layer, a box did not say which one had failed (the user's run: 115 measured and 132 design overlaps not matched). Minimum width of the tab with Windows fonts: 1,390 px before, 966 px with two rows. |
| D106 | 2026-10-19 | *Load design* (release Stitching tab) draws each `.oas` in the **tone the analysis read**: a tone-reversed file shows its holes, the contacts (`release/analysis.drawn_outlines`). The tone is the one whose contacts are the tile's design centres (`same_contacts`: as many, each within 1 nm; normal tone through the analysis's frame filter, `pipeline.outlines_inside_frame`), so runs saved before need no new analysis. Realigned tiles stay in normal tone (the realignment reads every file so). | User: "I once saw a design that appeared to have just a large 'block' of blue and purple instead of the actual designs." Cause found in the code: `read_design_images` read every file in normal tone, and `contact_outlines` keeps only each merged shape's outer hull; a tone-reversed file (D17) draws the area around the holes as one shape whose hull is the whole field, so the whole field was filled. Blue and purple are the first two stripe colours. Reproduced: the test image folder with tone-reversed files is analysed exactly as before (196 contacts), but every design mask was 100 % filled (normal-tone files: 16–21 %); now the same masks as the normal-tone files. Not confirmed on the user's dataset: the fact that would confirm it is that its `.oas` files draw the area around the holes. |
| D105 | 2026-10-19 | Realignment on a **repeating pattern no longer stops**: of the offsets that fit as well (the true offset plus whole pitches), the one where the most SEM contacts have a design contact within one cell is taken, and the log says how many others fit. "Fits as well" is a two-fraction test against the best fit (pooled standard error, 4 σ); the best fit is ranked by a Poisson upper bound (4 σ). The run stops only if no shift compares one image's worth or if, at the offset taken, more than 1 − `min_match_fraction` of the contacts compared have no partner. `ANALYSIS` 3. | User: "Does it matter? The design is by definition ideal, and we're looking at errors against the design. Although periodic, the design has the ideal positions and the errors will stay the same. It's only the absolute position of where that error occurred, but this can't be known anyway because the SEM measurement itself was badly done." So stopping (D104) threw away measurable data; taking the offset that pairs the most contacts measures the most images. Found on the way (synthetic, user-sized single stripe, 0.5 % of contacts not detected): with the best fit's own fraction as the reference, a small overlap with by chance 0 unpaired of 2304 set it below the true rate, so the full-coverage shifts were excluded and only 1,755 SEM contacts were paired; and the pair count inside the shrunk boxes favoured a shift one pitch across the stripe (edge effects, 3 %) that pairs 5 % fewer contacts, hence the count of SEM contacts with a partner. A 5 σ allowance on each fraction separately was too loose on the 2 × 2 test field (a lattice shift with 12.8 % unpaired fitted as well as the true one with ~0 %); the two-fraction test separates them (z ≈ 5.5 there, ≈ 2.7 for the lucky small overlap above). Results (same stripes as D104): offsets found exactly wherever the pattern does not repeat; well aligned and periodic: (0, 0), 52,071 SEM contacts paired (D93 code: −166 µm); 5 µm off and periodic: the true offset plus 37 pitches along y, 52,051 paired; 4 µm across a single stripe (beyond the `.oas` files): a whole-pitch equivalent, 50,058 paired; 1.3 s each. 17 × 690 µm: 17 s, 4.0 GB. Not run on the real data. |
| D104 | 2026-10-19 | Realignment searches **offsets of any size** (every shift at which the images still overlap the `.oas` files; was: up to half the field in x and y), ranks shifts by the mismatch fraction with 3 σ of counting noise added (was: the plain fraction, among shifts matching ≥ `min_match_fraction` of the most matched), and **stops the run** when no offset is clearly the best (was: a `!!` line, then went on with the guess; the stop on a repeating pattern is replaced by D105). `ANALYSIS` 2. | User: a run with realign on moved every image −166.18 µm, which is the search limit (±1.4 µm in x, ±165.7 µm in y on their single stripe; found by the remote agent); realign exists for images several µm from their `.oas`, more than an image field (1.44 or 2.88 µm), so a limit on the shift has no place there. Two faults, both mine: (1) "half the field" is half the extent of all tiles: across a single stripe ±1.4 µm, half an image, which can never find such offsets (synthetic stripe, offset (600, −3300) nm: old code (100.6, −3200.3) nm, wrong by 5 and 1 pitches); (2) the scoring: on a repeating pattern every whole-pitch shift fits, the plain fraction of a shift that compares only part of the field is noisy, and the 50 % gate let shifts keeping half the field compete, so the lowest fraction was the luckiest far shift (user-sized stripe, true offset 0, 0.5 % of contacts not detected: (+0.95, +81.1) µm). The −166.18 µm lies beyond ±165.7 µm, at the search limit (the remote agent's figures; the refinement after the search moves the offset by at most 30 nm, so the reach was a little larger than the 165.7 quoted: inference): the limit is what produced the user's result. (I first wrote that it lay inside the range: wrong.) Synthetic, user-sized single stripe (120 images of 2.88 µm, pitch 136 nm, 2 % of contacts missing unless said): offsets 0, (0, −5), (1.8, −5) (beyond half the stripe), (0.3, −40) µm, with 3 blank images, and 0.5 % missing all found exactly; fully periodic (offset 0 or −5 µm) and images beyond the stripe (4 µm across) stop with the message; 1 s each. 17 × 690 µm field: 14 s, 3.4 GB. Rivals are only shifts comparing ≥ half as many contacts as the best: shifts with a tiny overlap fit a few contacts by chance and would block every decision (seen: 617 such on the 6 × 6 test field). Not run on the real data. |
| D103 | 2026-10-19 | The contrast check (D93) is **off by default** (`min_contrast` 0, was 5), in both apps; when on, a `!!` log line says when it removed every contact of a tile. | User: a dataset that used to work gave no measured (SEM) centre at all and "Only 0 measured contact(s)". Checked here: D102 does not change the SEM centres (the same synthetic folder before D93, before D102 and after: identical centres). The default of 5 was set on clean synthetic tiles only (D93). Synthetic tiles with added noise (JPEG): at 20 grey levels detection still finds all 56 contacts per tile, but they stand out by only 4.3–4.5 noise σ, so the check drops all of them; with `realign` off the run then stops at the sample check ("No design tone matched the SEM"), with it on (no sample check) it ends in "Only 0 measured contact(s)" with no SEM centre: the reported symptoms. With the check off every case works. **Not the cause on the user's dataset:** the user then ran it with `min_contrast` 0 (detection `band`): no change, still no SEM centre and no design centre shown (open, being diagnosed). The default stays off: a data-dropping step tuned on synthetic tiles only should not be on by default. The parameters a user saved keep their `min_contrast`: they set it to 0 themselves. `ANALYSIS` unchanged: the same parameters give the same numbers, and every run keeps the `min_contrast` it used. |
| D102 | 2026-10-19 | Release: a run that stops for too few measured contacts shows what it found on the **Stitching tab** (`AnalysisFailed` carries the partial `ReleaseResult` and the reason per tile not measured): layer *Tiles not measured* (off at start), the count per reason in the status line, the tiles under the cursor with their reason; SEM images and design on request; nothing saved, no other tab. | User: a remote run failed with "Only 0 measured contact(s): see the tiles not measured above" and showed no picture, so the SEM images and the design could not be inspected. At that point the tiles, their centres and both stitchings exist, and the Stitching tab needs nothing else, so the partial result is the usual `ReleaseResult` without contacts. The reasons travel with the exception, not in the result: a saved run could not hold them without a format change, and a failed run is never saved (the folder's last good run stays). Reproduced here: 4 synthetic tiles without overlaps, the first without an `.oas`; each stitching places only its first tile (the largest group is one tile), which is flagged, so 0 contacts. The cause on the user's data is not known. Only this failure is covered: those before the stitching (D73) still show only the message. Checked with Windows fonts offscreen: the failure layers off at start, the two-line status, the cursor line. |
| D101 | 2026-10-18 | **Automatic updates** (`release/update.py`): the installed MaskReg looks at every start, in the background, for a newer `MaskReg-setup-X.Y.Z.exe` in a read-only FTP folder (anonymous login; address in `update.ini`, built into the installer from `packaging/update.ini`, not in git); the user may postpone (asked again next start); Yes downloads it, quits MaskReg, and a hidden PowerShell runs it silently once MaskReg is gone and starts MaskReg again. *Help → Check for updates...*. The installer's own relaunch (`/RELAUNCH`) of the first draft is removed. | User: check at every start, users may postpone, the address built into the installer; a separate update FTP server whose login is not exposed (anonymous read-only, D100). The newest setup by its file name, so releasing is one upload, no index file to keep in step. Measured here: an installer started while MaskReg still runs (even for a second) fails on the files in use and rolls back, with no message in silent mode, so MaskReg must be gone first; waiting for its process in a separate step removes the race, and that step starts MaskReg again whatever the outcome, so a failed update leaves the user the old MaskReg instead of nothing. Download size compared with the listing (truncated downloads). The check's thread is its own class, not a `Job`: two `Job`s at once would swap `warnings.showwarning` in each other's way. |
| D100 | 2026-10-18 | **Packaging:** PyInstaller one-folder build (`packaging/MaskReg.spec`) in an Inno Setup per-user installer (`packaging/MaskReg.iss`; both made by `packaging/build.py`); updates (planned) by MaskReg running the next installer silently, fetched from a separate FTP server's read-only folder with anonymous login. Built on the data machine only (spec guard); KLayout's `db_plugins` bundled explicitly; `freeze_support()` and a folder argument in `main()`. `pyinstaller` in the dev extra. | User: a self-contained program, a real installer rather than a zip, versions managed on GitHub Enterprise, automatic updates; no official network share; a separate update FTP server is possible, but its login must not be exposed to users. PyInstaller one-folder rather than one-file (unpacks everything on every start), Nuitka (needs a C compiler; the heavy code is already compiled) or a bundled plain Python (no real exe); Inno Setup rather than MSI (IT deployment, not asked) or MSIX (needs a code-signing certificate, updates over web or file share only). Any login MaskReg uses can be extracted from the user's PC, and FTP sends it unencrypted, so reading releases needs none; only the user uploads, with their own account. Found by the test build: the reader of image metadata is missing from a build made without it (the exe then cannot open image folders), and the `.oas` reader plugins were not bundled (smoke test failed with "Stream has unknown format" until added). The 12.9 s first start is not explained by a measurement; a virus scan of the ~540 new files is the likely cause (inference). |
| D99 | 2026-10-16 | **Versions and run files:** one version number (`affine_ransac.__version__`, 1.0.0, raised at release as tag `vX.Y.Z`), `FORMAT` (file can be read) and `ANALYSIS` (same numbers) as separate integers, both saved with every run; checks on opening (older format or damaged → analysed again; newer format → refused, never replaced; older analysis → shown, flagged, user asked); guard tests that fail when results or the file layout change without their number; `CHANGELOG.md`. Failure message boxes show the message without the Python class name. | User: a management rule for updates, for how run files of earlier versions are handled and checked; versions to be managed on GitHub Enterprise (no release experience yet; auto-update wanted, mechanism still open). The version alone cannot tell readable from unreadable files or same from different numbers, and between releases the code keeps one version while both can change, so they get their own numbers, raised in the commit that changes them. Before D99 a newer-format file made opening run the analysis and overwrite it, which on a shared folder destroyed a newer MaskReg's run. Files without an analysis number count as outdated: their provenance is unknown (format 3 files came from code since D93, but T007's reader change can alter which images a folder uses). Checked here: the results guard fails for a window height (contour) and an edge search width (image) change, the layout guard for an extra key; a step change passed, because the synthetic field fits in one 40 µm window, hence the 3 µm case. The newer-format message box showed `affine_ransac.release.cache.NewerRunFile: ...`, hence the class name dropped. |
| D98 | 2026-10-16 | Release: a new run keeps the run it replaces in `MaskReg_backups/` (newest 5, named by when each was saved); saves are atomic (`.part`, then rename). | User: keep earlier runs as backups when a folder is analysed again; chose the last 5 (a run file is tens of MB). Writing straight onto the saved run (as before) left a damaged file after a failed or interrupted save; the rename also makes the backup step safe. Saving contact sizes into the same run makes no backup, or it would push real earlier runs out. Backups are files only: nothing in the app opens them (user chose recent folders without earlier runs, D97). |
| D97 | 2026-10-16 | Release: *File → Open recent*, the last 10 folders whose result was shown, newest first; opened like *Open folder* (saved analysis, else a run). | User asked for loading recent runs; chose a recent-folders menu over also picking among a folder's backed-up runs. A folder is added when its result is shown, not when opening starts, so a folder whose run failed is not listed. |
| D96 | 2026-10-16 | Release: *File → Folders...* sets the folder of `maskreg.ini` and the FTP download folder. Order: the user's choice, then (settings file) `MASKREG_CONFIG` or (downloads) the file's `download_folder`, then the defaults. | User: both folders must be settable by the user. The user's choice comes first, so a pre-set `download_folder` (D83) cannot silently override what the user picked; the pre-set value becomes the default shown in the field. The user chooses a folder (as asked), not the file itself. |
| D95 | 2026-10-16 | Release: **Contact size tab**, hidden unless turned on (*View → Contact size tab*, remembered). While it is on, an image run keeps its loaded tiles and how their contacts were merged in memory (`keep_tiles`, `ReleaseResult.tiles`; never saved); they are let go when the tab is hidden and before another run. The measurement is the research app's (`analysis.merged_contact_sizes`, now shared), run as a window job; the sizes are saved in the autosave as optional arrays (no format change) and shown again when the folder is reopened. Maps in the release style: Area / xCD / yCD boxes in one figure, SEM or SEM − design, own jet bar each, same centre and scale, Copy / Save image. Not ported: SEM images and design behind the maps (D74). `link_views` skips hidden plots. | User asked for the area / CD tab in the release as well, hidden by default and shown only when the user turns it on. Tiles kept only while the tab is on because they are large (D87: ~6 GB of images on a 1494-tile folder, plus the contours), so users who never turn it on pay nothing; let go before another run so that two folders' tiles are never in memory at once (D87 ran out of memory while loading). Measuring from the files instead (reading every image again) would also work on a reopened result, but costs as much as loading the tiles and needs the pairing saved: not built. Checked here offscreen: copying a map's displayed area to the maps of a new layout zoomed out a little more at every change, and a hidden map passed its stale range to the shown one, so a map shown again takes the y range and centre (same scale) and hidden plots are left out of the link. Remembering the choice was decided here, not asked. |
| D94 | 2026-10-16 | **One image per measure point:** the metadata reader keeps, per `Measure Point No`, only the row whose image has the highest trailing number (the later measurement, with its own `.oas`; warning naming what was dropped); remote agent, task T007. Analyzer side: `check_image_records` raises if two records are at the same position (`same_position_records`, < 1 nm); the contract test checks it on the real data. | User: when the SEM's pattern matching against the `.oas` fails (the realignment dataset, images µm away from their design), the tool measures the point again; both images stay in the folder, the CSV has two rows with the same Chip X/Y and `Measure Point No`, and the analyzer stitched both images and both designs (ill-posed). User answers: 'later' = the higher image number (it follows the measurement order, D79); the later image has its own `.oas`. Review: every consumer (both apps, the notebooks) gets the tiles only from `read_tile_index`, so the filtering is a metadata-reader change; the guard on this side makes such a folder fail at once with a message instead of a bad stitching, also for a folder whose CSV lacks the column. |
| D93 | 2026-10-16 | Realignment **on / off** (`realign`, replaces `realign_search_um`): shifts up to half the field; the `.oas` files are **stitched** before their contacts are merged; the gate is relative (≥ `min_match_fraction` of the most matched). **Contrast check** `min_contrast` (S2). Release Stitching tab: **Design pattern** on request (*Load design*, `read_design_images`: each `.oas`'s merged contacts filled into a mask, at most 1024 px per side within the image budget, drawn at `ReleaseResult.oas_centers` in its stripe's colour, `style.STRIPES`, above the SEM images; realigned files stay at their own positions in both placements); cache format 3 (`oas_paths`, `oas_centers`). | User, on the first realigned dataset: the design stitching looked wrong or overlapping, with no way to tell which centre came from which stripe; the first images show no pattern but give contours anyway; wants the design pattern drawn, a review for bugs, and an on/off switch without a largest offset. Found in the review: (1) the `.oas` files were merged without being stitched, so a contact drawn more than `group_radius_nm` apart in two files stayed two design contacts (synthetic, files up to 12 nm apart: closest design contacts in a tile 4.9 nm without the stitching, 100 nm = one pitch with it); (2) the gate counted every SEM contact, including the false ones of blank images and those outside every `.oas`, so the true shift could fail it (inference, not reproduced); (3) the runner-up exclusion (3 cells) could hide the lattice neighbour on a pitch under 4 cells. A realigned tile's design centres are cut from the same merged set, so neighbouring tiles show the same centres on top of each other: expected, which is why the pattern is drawn per file and stripe. Contrast on synthetic tiles (σ from the image): contacts 13–16, regions cut from white noise ≤ 1.0, from correlated noise ≤ 3.6 (correlation lowers the σ estimate); the default 5 is from these alone, not the real data. |
| D92 | 2026-10-16 | `realign_search_um` is a normal **release parameter** (Parameters dialog, image folders); the hidden `maskreg.ini` `[analysis]` section of D91 is removed. | User: wants to use it directly from the UI. |
| D91 | 2026-10-16 | **Realignment** for an image folder whose images all sit at one wrong offset from their design (`realign_search_um`, 0 = off; see S6): SEM stitched alone, the design of all `.oas` files together, one offset found over the whole field by FFT mismatch scoring, every tile moved by it and given the design in its shifted field. Normal tone only. Research app: a setting like the others; release: hidden, read only from `maskreg.ini` `[analysis]` (`config.HIDDEN`, `read_hidden_settings`; a value saved with the parameters is ignored, so removing it from the file turns it off). | User: a dataset was measured with a failed SEM alignment: the first image went several µm below design 1, so no image matches its own `.oas`; proposed reading and stitching everything, then finding the best alignment, and asked for it as is (rare case), hidden in the release and set by hand; tone-reversed files are unlikely there. Choices made here, not asked: one translation for all images (a single wrong alignment point); mismatch scoring (D50) rather than the most matches, which on a periodic array favours the shift with the most shared area; all shifts at once by FFT because a vote over µm ranges on a periodic array has a near-equal peak at every lattice vector; the per-tile matching after it is unchanged, so the stitching's accumulated error (D86) only affects the one offset, and the per-tile search (`design_search_nm`) removes what is left. A fully periodic field cannot tell the lattice shifts apart: logged with `!!`; equal scores take the most matched, then the smallest shift. Synthetic (6 × 6 tiles, pitch 100 nm, 15 % of contacts missing, offset (−250, −1730) nm): found within the stage errors; every inner design contact paired. Memory: one float32 raster per field of `rows × columns` cells (limit 10⁸). Not run on the real data. |
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
