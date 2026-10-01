# T001: Tile index reader (metadata CSV + file pairing) and data report

**Status: OPEN** (maintained by the local agent, from what the user reports back)
**For:** the remote agent, which has access to the real data
**From:** the local agent (no data access), 2026-10-01

**You create exactly these files and nothing else:**
- `src/affine_ransac/io/metadata.py` (§2)
- `tests/test_metadata.py` (§3)
- `docs/remote/T001-report.md` (§4–§5; `docs/remote/` is gitignored)

**Never commit, never push, never modify tracked files** (see `AGENTS.md`). The user pulls new
tasks with `git pull`, and that must keep working.

---

## 0. Read first

1. `AGENTS.md`: your standing rules.
2. `CLAUDE.md`: only "Development approach" and "Key rules" (small, simple, readable code; units).
3. `docs/SPEC.md` §3 (coordinate frames) and §4 (inputs).
4. This whole file before starting.

**Why this task is written so strictly:** the local agent cannot see the data, your code or
your files, ever. It builds the next stages against the interface in §2 **without seeing your
implementation**. Its only feedback is what the user forwards from your report. So follow the
interface **exactly** (a contract test checks it, §7), and make the report precise. When
something is unclear or differs from what this file assumes, don't guess silently: do the
sensible thing and **state it explicitly in the report**.

**Confidentiality:** never put data files (`.jpg`, `.oas`, `.csv`) or real rows, file names or
coordinate values into any repo file outside `docs/remote/`. In docs and tests, use **made-up values in exactly
the real format**: same column names, delimiter, number formatting, magnitudes, naming pattern.
Reports contain **aggregate statistics only** (counts, mean/std/min/max).

---

## 1. Known facts (from the user)

```
<DATA_DIR>/
    *.jpg              SEM images
    <something>.csv    one metadata CSV, one row per SEM image (exact name unknown)
    Contour/
        *.oas          one design file per SEM image
```

- An image and its `.oas` are paired by a **file-naming convention** (unknown; you determine it).
- Each `.oas` is in **local coordinates centred on (0, 0)**. The **global mask position of each
  `.oas` centre** is in the CSV.
- SEM **FOV = 2.88 µm**, **image size = 2048 × 2048 px**; both are in the metadata.
- Some `.oas` files are **tone reversed** (they draw the area around the holes). The reader
  `read_contacts_tone_reversed()` already exists; this task only asks you to report on it (§5f).

---

## 2. Build `src/affine_ransac/io/metadata.py`

Implement **exactly** this public interface (names, types, units):

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TileRecord:
    tile_id: str              # unique, stable id derived from the file names (document how)
    image_path: Path          # absolute path to the SEM .jpg
    oas_path: Path | None     # absolute path to the matching .oas in Contour/, None if missing
    center_x_nm: float        # global mask x of the .oas (0, 0) point, nm
    center_y_nm: float        # global mask y of the .oas (0, 0) point, nm, y UP
    fov_x_nm: float           # field of view, width, nm
    fov_y_nm: float           # field of view, height, nm (= width if only one FOV value exists)
    image_width_px: int       # from the metadata (not measured from the jpg)
    image_height_px: int
    stage_x_nm: float | None  # SEM stage position if the CSV has one, else None
    stage_y_nm: float | None
    extra: dict[str, str]     # every other CSV column of the row: raw string, original column name


def find_metadata_csv(data_dir: str | Path) -> Path:
    """Return the metadata CSV in data_dir. ValueError if none or more than one candidate."""


def read_tile_index(data_dir: str | Path) -> list[TileRecord]:
    """One TileRecord per CSV row, in CSV row order."""
```

Rules:

- **Units:** convert to **nm inside this module**. Callers never convert units. Document the
  source units in the report's "Format" section (§4).
- **Axis convention:** `center_x_nm, center_y_nm` must be in the **design frame**: the frame
  of the `.oas` files, y pointing **up**. If the CSV uses another convention (y down, swapped
  axes, different origin), convert here and document it. If you cannot determine the
  convention from the data, keep the values as they are and **say so explicitly** in the report.
- **Errors** (raise `ValueError` with a clear message naming the file/column/row):
  missing CSV or more than one candidate CSV; a required column missing; a CSV row whose image
  file does not exist. A **missing `.oas`** is *not* an error: set `oas_path=None` and list the
  count in the report. Images on disk that are not in the CSV are ignored; report the count.
- **Encoding:** open with `encoding="utf-8-sig"` (handles a BOM). If the real file uses another
  encoding (e.g. cp949), handle it and document it.
- **Dependencies:** standard library only (`csv`, `pathlib`, `dataclasses`). No pandas in
  production code. Do not add packages to `pyproject.toml`.
- **Paths:** `pathlib` only; must work on Windows with non-ASCII paths.
- **Simplicity:** plain functions plus the dataclass. No config options beyond this interface,
  no class hierarchies, no caching. Short functions with clear names, per `CLAUDE.md`.

## 3. Tests: `tests/test_metadata.py`

- Build a fake data directory in `tmp_path` that mimics the real one **exactly in format**:
  same CSV header, delimiter, number formatting and file-naming pattern, with made-up values,
  3–4 tiles. Dummy `.jpg`/`.oas` files can be empty; the reader must not open them.
- Test: exact nm values after unit conversion (and axis conversion, if any); image ↔ `.oas`
  pairing and `tile_id`; missing `.oas` → `None`; missing required column → `ValueError`;
  CSV row whose image is missing → `ValueError`; extra columns kept in `extra`; no CSV / two
  CSVs → `ValueError`.
- `pytest` must pass for the **whole** suite.

## 4. Format reference (report section "Format")

Part of `docs/remote/T001-report.md`. The local agent relies on it. Include:

1. CSV file name pattern, delimiter, encoding, header, typical number of rows.
2. A table of **every** column: exact name, meaning, unit, type, made-up example value in the
   real format, and the `TileRecord` field it maps to (or `extra`).
3. The `.jpg` and `.oas` file-naming conventions, how they pair, and how `tile_id` is derived,
   with made-up example names in the real pattern.
4. Coordinate conventions of the CSV: units, origin, axis directions, what "centre" refers to;
   whether a stage position exists and how it relates to the centre.
5. Anything surprising or inconsistent.

## 5. Data checks (report sections a–g)

Run these on the real data: all tiles if practical, otherwise at least 30 spread over the
dataset. Use the existing library functions as named. Put analysis scripts in `docs/remote/`
or outside the repo. In the report, give **aggregate numbers only**, in sections a–g with
tables.

**a. Counts:** CSV rows; `.jpg` on disk; `.oas` in `Contour/`; paired tiles; tiles missing `.oas`.

**b. Image size and data bar:** actual size from `load_sem_image(path).shape` vs. the metadata
size. Is there an info/data bar (extra rows beyond 2048, or a visible bar inside the image)?
If so: how many rows, at which edge, the same in every image? This sets `DATABAR_ROWS`.

**c. FOV and pixel size:** FOV values found (the same for all tiles?); pixel size = FOV / width.

**d. `.oas` content and extent:** layers/datatypes present (`list_layers`) and which one holds
the contacts; database unit; per file, the bounding box of all contact shapes. Report min/max
of the bbox edges compared with ±FOV/2. Does (0, 0) look like the image centre? Does the file
cover exactly the FOV?

**e. Orientation check (most important).** For each normal-tone tile (see f):

```
design_nm = read_contacts(layout, layer, datatype)[0]
image     = crop_databar(load_sem_image(jpg), DATABAR_ROWS)
found     = detect_contacts(image)                    # set dark_contacts=False if they are bright
sem_nm    = pixel_to_tile_nm(found.centers, image.shape, fov_x_nm / image.shape[1])
```

Apply each of these 8 transforms `M` to the SEM points (`p' = M @ p`, with `p = (x, y)`):

| name | M |
|---|---|
| identity | `[[1, 0], [0, 1]]` |
| flip_x | `[[-1, 0], [0, 1]]` |
| flip_y | `[[1, 0], [0, -1]]` |
| rot180 | `[[-1, 0], [0, -1]]` |
| rot90_ccw | `[[0, -1], [1, 0]]` |
| rot90_cw | `[[0, 1], [-1, 0]]` |
| transpose | `[[0, 1], [1, 0]]` |
| anti_transpose | `[[0, -1], [-1, 0]]` |

For each `M`: match every transformed SEM point to its nearest design point and compute the
**median distance (nm), with no translation** (both sets should already share the origin).
Then estimate a translation (median of `design − M·sem` over the matched pairs), remove it and
compute the median distance again.

Report: the winning `M` and in what fraction of tiles it wins; the median residual of the
winner vs. the runner-up (with and without translation); the typical translation (mean, std,
nm); the contact pitch (nm); the fraction of SEM contacts matched within half a pitch.

⚠ A regular contact array looks the same after many of these transforms, especially once a
translation is allowed. If the residuals of several `M` are about equal, the check is
**inconclusive** for that tile. Say so, and base the conclusion on tiles whose pattern is not
symmetric (array edges, missing contacts, irregular placement). Report how many tiles were
conclusive.

Also report detection statistics: contacts per tile (design vs. SEM), the range of Otsu
thresholds, whether contacts are dark or bright, and anything that looks wrong in the
`view_sem --detect` overlay.

**f. Tone reversal:** for each `.oas`, run `read_contacts` and
`read_contacts_tone_reversed(..., frame_nm=(fov, fov))`. How many files are tone reversed?
Which simple measurable property separates them (e.g. polygon count, drawn-area fraction of
the FOV, one big polygon)? **Suggest** a rule with numbers; don't implement it. Does
`read_contacts_tone_reversed` give contact counts and positions consistent with the SEM for
those tiles?

**g. Centre vs. stage:** if the CSV has a stage position, statistics of (stage − centre), in nm.

## 6. Do not

- Do not commit, push, or modify/delete any tracked file. That includes existing modules
  (`io/design.py`, `io/sem_image.py`, `features/contact.py`, `geometry/frames.py`, the viewers),
  `tests/test_metadata_contract.py`, `docs/SPEC.md`, `CLAUDE.md`, `AGENTS.md`, this file and the
  notebooks. If something is wrong or limiting for the real data, describe it in the report
  **with evidence**; the local agent will change it.
- Do not put data or real values in any file outside `docs/remote/`. Do not add dependencies.

## 7. Finish

1. Run the whole suite, `pytest`. Everything must pass.
2. Run the contract test against the real data:
   `AFFINE_RANSAC_DATA_DIR=<data folder> pytest tests/test_metadata_contract.py -v`
   (on Windows PowerShell: `$env:AFFINE_RANSAC_DATA_DIR="<data folder>"` first). All tests must
   pass, none skipped. If one fails and you believe the test itself is wrong, don't edit it:
   explain why in the report.
3. Check `git status`: it must list only your new files as untracked (`docs/remote/` is
   ignored, so it won't appear), and no modified tracked files.
4. Write `docs/remote/T001-report.md` in this order:
   - **Summary for the local agent** (at most ~40 lines; the user forwards this part): status
     (DONE / BLOCKED + reason); the CSV columns used for each `TileRecord` field with source
     units and any axis conversion; how `tile_id` and the image ↔ `.oas` pairing work; the
     pytest and contract-test results; and one line each for findings a–g (data-bar rows,
     contact layer, winning orientation and how conclusive it was, the tone-reversal rule).
     Mention anything the local agent must change.
   - **Format** (§4).
   - **Data checks a–g** (§5).
5. Show the user the summary section.
