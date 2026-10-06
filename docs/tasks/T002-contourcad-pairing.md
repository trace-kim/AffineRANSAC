# T002: Design folder is `ContourCAD/`; pair `.oas` files by "name contains"

**Status: OPEN** (maintained by the local agent, from what the user reports back)
**For:** the remote agent, which has access to the real data
**From:** the local agent (no data access), 2026-10-06

**You modify exactly these files and nothing else:**
- `src/affine_ransac/io/metadata.py` (your file from T001)
- `tests/test_metadata.py` (your file from T001)
- `docs/remote/T002-report.md` (new; `docs/remote/` is gitignored)

**Never commit, never push, never modify tracked files** (see `AGENTS.md`).

This is a small fix. Do not change anything else in `metadata.py`: the `TileRecord` fields,
`find_metadata_csv`, `read_tile_index` signatures, units and axis conversion stay as they are.

---

## 1. What changed in the data (from the user)

1. The design folder is **`<DATA_DIR>/ContourCAD/`**, not `<DATA_DIR>/Contour/`.
2. The `.oas` file names now carry an **extra prefix** in front of the name you matched on in
   T001. So the exact-name lookup no longer finds them.

## 2. Change the pairing in `metadata.py`

- Look for `.oas` files in `<DATA_DIR>/ContourCAD/` only.
- Keep the T001 key you derive for each image (the text you used to build the expected `.oas`
  name, without the `.oas` extension). An `.oas` file **matches** an image when its file name
  **contains** that key.
- Exactly one match → `oas_path` = that file (absolute path).
- No match → `oas_path = None` (not an error, as before).
- More than one match → `ValueError` naming the image and all matching files. Watch for keys
  that are contained in longer keys (e.g. `tile_1` in `tile_10`); if that happens on the real
  data, say so in the report and do **not** silently pick one.
- Missing `ContourCAD/` folder: every `oas_path` is `None` (as for a missing `Contour/` before).
- Keep it simple: one short helper, e.g. `_find_oas(oas_files, key) -> Path | None`; list the
  folder once, not once per row.

## 3. Tests: `tests/test_metadata.py`

Update the fake data directory to the new layout: `ContourCAD/` and `.oas` names with a made-up
prefix in the real format. Add/adjust tests for:
- pairing by "contains" with the prefix present;
- no match → `None`;
- two files containing the key → `ValueError`;
- an `.oas` in a folder named `Contour/` is **not** found.

`pytest` must pass for the **whole** suite, including `tests/test_metadata_contract.py` with
`AFFINE_RANSAC_DATA_DIR` set to the real data folder (it now checks the parent folder is
`ContourCAD`).

## 4. Checks on the real data

Run `read_tile_index` on the real data folder and report (counts only, no file names or values):
- CSV rows; `.oas` files in `ContourCAD/`; paired tiles; tiles with no `.oas`;
  `.oas` files not paired with any image;
- whether any image matched more than one `.oas` (should be 0; if not, the count and the
  made-up pattern of the collision);
- the prefix pattern in a made-up example in the real format (e.g. `PREFIX_<key>.oas`), and
  whether it is the same for all files.

## 5. Report: `docs/remote/T002-report.md`

A short summary (≤ 15 lines): what you changed, test results (`pytest` pass/fail counts), the
§4 counts, and anything that differed from this file. Show the user the summary.
