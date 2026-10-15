# T004: Pair each image with the `SA…` `.oas` that has the same trailing number

**Status: DONE (fixed by the user directly, 2026-10-15; nothing left for the remote agent)** (maintained by the local agent, from what the user reports back)
**For:** the remote agent, which has access to the real data
**From:** the local agent (no data access), 2026-10-15

**You modify exactly these files and nothing else:**
- `src/affine_ransac/io/metadata.py` (your file from T001–T003)
- `tests/test_metadata.py` (your file from T001–T003)
- `docs/remote/T004-report.md` (new; `docs/remote/` is gitignored)

**Never commit, never push, never modify tracked files** (see `AGENTS.md`).

This is a small fix. Do not change anything else in `metadata.py`: the `TileRecord` fields, the
`find_metadata_csv` / `read_tile_index` signatures, units, axis conversion, the T003 CSV encoding
handling and the `ContourCAD/` folder (T002) stay as they are. Only the image ↔ `.oas` pairing changes.

---

## 1. What happened (from the user)

The analyzer app finds no `.oas` for the images, so every tile is flagged and the run fails. In
**every** data folder the design files are named with the prefix **`SA`** and end in the **same
number** as the image's name, which has the prefix **`CD`**. The T002 rule ("the `.oas` file name
**contains** the image's key") cannot match, because the `.oas` names do not contain `CD`.

## 2. New pairing rule in `metadata.py`

Replace the T002 "contains" match by:

1. **Image number:** the run of digits at the **end** of the image's key (the T001 key, i.e. the
   image's name without extension, the text that starts with `CD`), read as an integer.
   E.g. made-up `CD000123` → 123. An image whose key does not end in digits: `oas_path = None`
   (listed in the report, not an error).
2. **Candidates:** the `.oas` files in `<DATA_DIR>/ContourCAD/` whose file name **starts with
   `SA`** (a module constant, e.g. `OAS_PREFIX = "SA"`, case-sensitive). Other `.oas` files are
   ignored.
3. **`.oas` number:** the run of digits at the end of the candidate's name without `.oas`, read as
   an integer. A candidate without trailing digits is ignored.
4. **Match:** image number == `.oas` number (as integers, so a different zero padding still
   matches; say in the report whether the padding is the same on the real data).
   - exactly one match → `oas_path` = that file (absolute path);
   - no match → `oas_path = None` (not an error, as before);
   - more than one match → `ValueError` naming the image and all matching files.
5. List the folder once and build a `{number: [paths]}` lookup; do not scan the folder per row.
   One short helper, e.g. `_oas_number(name: str) -> int | None`, plus the lookup. Standard
   library only (`re` is fine: `re.search(r"(\d+)$", stem)`).

Update the module docstring / comments that describe the pairing, and the T001 summary of the
pairing in your docstring if there is one.

## 3. Tests: `tests/test_metadata.py`

Update the fake data folder to the real pattern (made-up names): images `CD<number>.jpg`,
design files `SA<…><number>.oas` in `ContourCAD/` (copy the real format of what sits between
`SA` and the number, with made-up values). Tests:
- each image pairs with the `SA` file of the same number;
- the same number with a different zero padding (e.g. `CD000123` and `SA…123.oas`) → paired;
- an `.oas` with the same number but **not** starting with `SA` → ignored;
- no `SA` file with that number → `None`;
- two `SA` files with the image's number → `ValueError`;
- the number is the **trailing** one: `SA_7_000123.oas` pairs with `CD000123`, not with `CD000007`.

`pytest` must pass for the **whole** suite.

## 4. Checks on the real data

For **two different data folders** (one that failed in the app, one other):
1. `AFFINE_RANSAC_DATA_DIR=<folder> pytest tests/test_metadata_contract.py -v`
   (PowerShell: `$env:AFFINE_RANSAC_DATA_DIR="<folder>"` first). All pass, none skipped.
2. `read_tile_index(<folder>)`, counts only (no file names): CSV rows; `.oas` files in
   `ContourCAD/`; of these, how many start with `SA`; paired tiles; tiles with `oas_path=None`;
   `SA` files not paired with any image; images matching more than one file (must be 0).
3. A made-up example of one image name and its `.oas` name in the real pattern, and whether the
   zero padding of the numbers is the same.
4. Open the failing folder in the app (`python -m affine_ransac.app`, *Open folder*, default
   settings) and let the run finish. Report whether it finished, and the log lines that start
   with `!!`, as counts only (e.g. "3 tiles flagged"), no tile names.

## 5. Report: `docs/remote/T004-report.md`

**Summary** (at most ~15 lines, shown to the user): status (DONE / BLOCKED + reason); what you
changed; pytest result; contract-test results for both folders; the §4 counts; the made-up name
example and the padding answer; whether the app run finished and the counts of `!!` lines.
