# T005: Several metadata CSVs in one folder: merge them, keep the rows of the images present

**Status: OPEN** (maintained by the local agent, from what the user reports back)
**For:** the remote agent, which has access to the real data
**From:** the local agent (no data access), 2026-10-15

**You modify exactly these files and nothing else:**
- `src/affine_ransac/io/metadata.py` (your file from T001–T004)
- `tests/test_metadata.py` (your file from T001–T004)
- `docs/remote/T005-report.md` (new; `docs/remote/` is gitignored)

**Never commit, never push, never modify tracked files** (see `AGENTS.md`).

This is a small fix. Leave the rest of `metadata.py` alone: the `TileRecord` fields,
`read_tile_index`'s signature, units, axis conversion, the T003 encoding handling (`utf-8-sig`,
then `cp949`, per file) and the T004 `.oas` pairing stay as they are. Only how the CSV(s) are found
and read changes.

---

## 1. What happened (from the user)

The app fails on folders that contain **more than one candidate metadata CSV** (T001 made that a
`ValueError`). In some folders the metadata is split up, sometimes **one CSV per image**, so there
can be many CSV files. Together they hold the rows of the folder's images. Rows may repeat across
files, and a file may hold rows for images that are not in the folder.

## 2. Interface change

Replace `find_metadata_csv` with:

```python
def find_metadata_csvs(data_dir: str | Path) -> list[Path]:
    """Every candidate metadata CSV in data_dir, sorted by file name. ValueError if none."""
```

What counts as a candidate stays as in T001; say in the report what that rule is. Remove
`find_metadata_csv`; nothing else uses it (the local contract test now checks `find_metadata_csvs`).

## 3. `read_tile_index(data_dir)`

- **One candidate:** exactly as before, with the same errors (for example, a row whose image is
  missing → `ValueError`).
- **More than one candidate:**
  1. **Read** each file (encoding as in T003, per file). All files must have the **same header**
     (same column names, same order). Otherwise raise `ValueError` naming one file with the
     majority header and each file that differs.
  2. **Merge** the rows in file-name order, then row order.
  3. **Drop duplicates:** rows whose values are identical in every column; keep the first one.
  4. **Keep the valid rows:** a row is valid when its *File name* column (the column that holds
     the image's file name; use its real name) names a `.jpg` that exists in `data_dir`. The path in
     the column differs from the real one, so compare only the **file-name part**: the text after
     the last `\` or `/`, equal to the `.jpg`'s name (`Path.name`), **case-insensitive**. Don't use a
     plain "contains" test, because `CD1.jpg` is contained in `XCD1.jpg`. Drop invalid rows silently;
     the report gives their count.
  5. If an image still has **more than one row** (the rows differ in some column), raise
     `ValueError` naming the image, the files those rows came from, and the columns that differ.
  6. No valid rows at all → `ValueError` naming the folder and the number of candidate files.
  7. Build the `TileRecord`s from the remaining rows in that order, the same way as for one CSV.
     Images on disk without a row are still ignored (as in T001).
- Keep it short: one helper that reads one CSV into (header, rows), one that merges and filters.
  Use the standard library only.

## 4. Tests: `tests/test_metadata.py`

Use made-up values in the real format. Update the old "two CSVs → `ValueError`" test, then add:
- one CSV per image (3 files) → one record per image, the same values as with a single combined CSV;
- the same row in two files → one record;
- a row whose *File name* has a different folder part than the real path → kept;
- a row whose image is not in the folder → dropped (with several CSVs);
- `XCD1.jpg` named in a row while only `CD1.jpg` exists → dropped (no substring match);
- the name in a different letter case → kept;
- two **different** rows for the same image → `ValueError`;
- different headers → `ValueError`;
- several CSVs and no valid row → `ValueError`;
- `find_metadata_csvs`: sorted list; no CSV → `ValueError`.

`pytest` must pass for the **whole** suite, including `tests/test_metadata_contract.py`.

## 5. Checks on the real data

For **two folders**, the one with many CSVs that failed in the app and one with a single CSV:
1. `AFFINE_RANSAC_DATA_DIR=<folder> pytest tests/test_metadata_contract.py -v`
   (PowerShell: `$env:AFFINE_RANSAC_DATA_DIR="<folder>"` first). All pass, none skipped.
2. Counts only (no file names or values): candidate CSVs; rows read; rows after dropping duplicates;
   valid rows; `.jpg` in the folder; records returned; images without a row.
3. A **made-up** example of a *File name* value in the real format, and the `.jpg` name it matches.
4. For the many-CSV folder: open it in the app (`python -m affine_ransac.release.app`, *Open folder*,
   default settings) and let the run finish. Report whether it finished, and the number of log lines
   that start with `!!` (counts only, no tile names).

## 6. Report: `docs/remote/T005-report.md`

**Summary** (at most ~15 lines, shown to the user): status (DONE / BLOCKED + reason); what you
changed; the candidate-CSV rule; the pytest result; the contract-test results for both folders; the §5
counts; the made-up *File name* example; whether the app run finished, and the `!!` count.
