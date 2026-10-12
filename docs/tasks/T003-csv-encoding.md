# T003: Metadata CSV that is not UTF-8; 1024 × 1024 px images

**Status: OPEN** (maintained by the local agent, from what the user reports back)
**For:** the remote agent, which has access to the real data
**From:** the local agent (no data access), 2026-10-12

**You modify exactly these files and nothing else:**
- `src/affine_ransac/io/metadata.py` (your file from T001/T002)
- `tests/test_metadata.py` (your file from T001/T002)
- `docs/remote/T003-report.md` (new; `docs/remote/` is gitignored)

**Never commit, never push, never modify tracked files** (see `AGENTS.md`).

This is a small fix. Do not change anything else in `metadata.py`: the `TileRecord` fields, the
`find_metadata_csv` / `read_tile_index` signatures, units, axis conversion and the T002 pairing
stay as they are.

---

## 1. What happened (from the user)

On a **new data folder**, the analyzer app (`python -m affine_ransac.app`, *Open folder*) fails
while reading the metadata CSV with a `'utf-8' codec can't decode ...` error
(`UnicodeDecodeError`). The older folders still work. The new folder's CSV is evidently not
UTF-8 (on Korean Windows it is most likely **cp949**, which the SEM software or Excel writes).

The new folder also has **1024 × 1024 px** images instead of 2048 × 2048 px, with the **same
FOV** (2.88 µm), so the pixel size is 2.8125 nm instead of 1.40625 nm.

## 2. Change the CSV reading in `metadata.py`

1. Find out the encoding of the new folder's CSV (e.g. try decoding its bytes; the T001 rule
   said: "If the real file uses another encoding (e.g. cp949), handle it and document it").
2. Read the CSV text like this, in one small function:
   - try `encoding="utf-8-sig"` first (as now; the older folders keep working);
   - on `UnicodeDecodeError`, try `cp949` (or the encoding you found in step 1, if different);
   - if both fail, raise `ValueError` naming the file and the encodings tried.
   - **No** `errors="replace"`/`"ignore"` and **no** `latin-1` fallback: they never fail, so a
     wrong encoding would silently garble non-ASCII file names or column names.
3. Every other step (header, columns, rows, pairing) uses the decoded text exactly as before.
4. `find_metadata_csv` must not open files with a fixed encoding either; if it reads a header
   to pick the CSV, it uses the same function.
5. Image size: check that nothing in `metadata.py` assumes 2048 px (a constant, a check, a
   default). `image_width_px` / `image_height_px` must come from the CSV as before. If you find
   such an assumption, remove it and say so in the report.

Standard library only; no new dependencies.

## 3. Tests: `tests/test_metadata.py`

Add, using your existing fake data folder (made-up values in the real format):
- the CSV written with `encoding="cp949"` and at least one non-ASCII character in a value
  (e.g. a Korean word in a text column that ends up in `extra`) → read correctly, value equal;
- the CSV written with `encoding="utf-8-sig"` → still read correctly;
- a file that is neither (e.g. bytes `b"\xff\xfe\xfa"` inside a row) → `ValueError`;
- a fake folder whose CSV says 1024 × 1024 px → `image_width_px == image_height_px == 1024`.

`pytest` must pass for the **whole** suite.

## 4. Checks on the real data

Run, for **the new folder and one older folder**:
1. `AFFINE_RANSAC_DATA_DIR=<folder> pytest tests/test_metadata_contract.py -v`
   (PowerShell: `$env:AFFINE_RANSAC_DATA_DIR="<folder>"` first). All pass, none skipped.
2. `read_tile_index(<folder>)`: number of records; the distinct `(image_width_px,
   image_height_px)` and `fov_x_nm` values; and for 5 tiles spread over the folder,
   `load_sem_image(image_path).shape` vs. the metadata size (must agree, `(rows, cols)` =
   `(height, width)`).
3. Open the new folder in the app (`python -m affine_ransac.app`, *Open folder*, default
   settings) and let the run finish. Report whether it finished, and the log lines that start
   with `!!`, as counts only (e.g. "3 tiles flagged"), no tile names.

## 5. Report: `docs/remote/T003-report.md`

**Summary** (at most ~20 lines, shown to the user):
- status (DONE / BLOCKED + reason);
- the encoding of the new CSV and how you determined it;
- the encodings now tried, in order;
- whether anything assumed 2048 px (and what you changed);
- pytest result, contract-test results for both folders;
- §4.2 numbers (counts and sizes only) and §4.3: did the app run finish, counts of `!!` lines.
