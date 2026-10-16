# T007: One image per measure point: keep the later image when a point was measured twice

**Status: OPEN** (maintained by the local agent, from what the user reports back)
**For:** the remote agent, which has access to the real data
**From:** the local agent (no data access), 2026-10-16

**You modify exactly these files and nothing else:**
- `src/affine_ransac/io/metadata.py` (your file from T001–T005)
- `tests/test_metadata.py` (your file from T001–T005)
- `docs/remote/T007-report.md` (new; `docs/remote/` is gitignored)

**Never commit, never push, never modify tracked files** (see `AGENTS.md`).

This is a small fix. Leave the rest of `metadata.py` alone: the `TileRecord` fields, the signatures of
`read_tile_index` and `find_metadata_csvs`, units, axis conversion, the T003 encoding handling, the T004
`.oas` pairing (`SA…<number>.oas` with the image's own trailing number) and the T005 merging of several
CSVs stay as they are. Only one filtering step is added.

---

## 1. What happened (from the user)

When the SEM's pattern matching against the `.oas` fails during a measurement, the tool measures the
**same point again** and saves a **second image**. Both images stay in the folder. The metadata then has
**two rows for the same point**: the same *Chip X/Y* position and the same value in the column
**`Measure Point No`**. The analyzer then loads both images (and both `.oas` files) as two tiles at one
position, which ruins the stitching of both the SEM and the design.

The user's rule: **for each `Measure Point No`, keep only the later image**. "Later" means the image whose
file name has the **higher trailing number** (`CD<number>`, compared as an integer, the same number the
T004 pairing uses). The image numbers follow the measurement order. The later image has **its own `.oas`**
(`SA…<its number>.oas`), paired as usual by T004.

## 2. The change in `read_tile_index(data_dir)`

Add one step after the rows are read and, for several CSVs, merged and filtered as in T005 (after T005
§3 step 4, before step 5); for one CSV, after it is read:

1. Group the rows by their `Measure Point No` value (the text, stripped of surrounding spaces). Rows
   whose `Measure Point No` is empty are not grouped: each is kept as before.
2. In each group with more than one row, keep only the row whose image (its *File name* column, file-name
   part as in T005) has the **highest trailing number**. If a row of such a group has no trailing number
   in its image name, or two rows of a group have the same number, raise `ValueError` naming the
   `Measure Point No` and the images.
3. If rows were dropped, call `warnings.warn` once, with the number of measure points measured more than
   once and, for each, the image kept and the image(s) dropped, e.g.
   `"2 measure point(s) measured more than once; kept the later image: point 17: CD000045 (dropped CD000044); ..."`.
   The app shows warnings in its log, so the user sees what was dropped.
4. If the CSV has **no** `Measure Point No` column, skip this step (no error, no warning).
5. Images on disk whose row was dropped are ignored, like images without a row (T001).
6. The records keep the order they had before (the dropped rows simply leave gaps).

Keep it short: one helper that takes the rows and returns the kept rows plus what was dropped. Standard
library only.

## 3. Tests: `tests/test_metadata.py`

Use made-up values in the real format:
- two rows with the same `Measure Point No` and the same Chip X/Y → one record, the image with the
  higher number, with its own `.oas`; a warning naming both images;
- the higher number written with different zero padding (e.g. `CD9` and `CD010`) → `CD010` kept;
- the later row coming **first** in the CSV → still the higher number kept;
- three rows for one point → only the highest kept;
- the duplicate rows split over **two CSVs** (T005 merging) → one record;
- different `Measure Point No` values → all kept, no warning;
- an empty `Measure Point No` in two rows → both kept;
- no `Measure Point No` column → all rows kept, no warning;
- two rows of one point with the same image number → `ValueError`.

`pytest` must pass for the **whole** suite, including `tests/test_metadata_contract.py`.

## 4. Checks on the real data

For **two folders**: the one with the repeated measurements and one without (any normal folder):
1. `AFFINE_RANSAC_DATA_DIR=<folder> pytest tests/test_metadata_contract.py -v`
   (PowerShell: `$env:AFFINE_RANSAC_DATA_DIR="<folder>"` first). All pass, none skipped. The new test
   `test_real_data_one_image_per_position` checks that no two records are at the same position.
2. Counts only (no file names or values): rows read; measure points with more than one row; rows dropped;
   records returned; `.jpg` in the folder.
3. Whether, in the folder with repeats, every dropped row had the **same Chip X/Y** as the row kept
   (yes / no, and how many not).
4. A **made-up** example of a `Measure Point No` value in the real format.
5. For the folder with repeats: open it in the app (`python -m affine_ransac.release.app`, *Open folder*,
   default settings) and let the run finish. Report whether it finished, and the number of log lines that
   start with `!!` (counts only, no tile names).

## 5. Report: `docs/remote/T007-report.md`

**Summary** (at most ~15 lines, shown to the user): status (DONE / BLOCKED + reason); what you changed;
the pytest result; the contract-test results for both folders; the §4 counts; the same-position answer;
the made-up `Measure Point No` example; whether the app run finished, and the `!!` count.
