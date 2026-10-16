# T006: Time each step of loading one image tile

**Status: OPEN** (maintained by the local agent, from what the user reports back)
**For:** the remote agent, which has access to the real data
**From:** the local agent (no data access), 2026-10-16

**You create exactly these files and nothing else:**
- `docs/remote/T006_time_tiles.py` (new; `docs/remote/` is gitignored)
- `docs/remote/T006-report.md` (new)

**Never commit, never push, never modify tracked files** (see `AGENTS.md`). This task only measures
things. Don't change any code in `src/` or `tests/`.

---

## 1. Why

In the release app (`python -m affine_ransac.release.app`), loading the image tiles takes most of the
run time. Every tile goes through `affine_ransac.pipeline.process_tile` in a worker process. We need to
know how long each step inside it takes, and whether the edge refinement is worth skipping.

## 2. The script: `docs/remote/T006_time_tiles.py`

Run as `python docs/remote/T006_time_tiles.py <image folder> [n_tiles=5]`. Single process, no workers.
Use `time.perf_counter()`. Keep it short and plain.

1. `records = affine_ransac.io.metadata.read_tile_index(folder)`;
   `layer = affine_ransac.analysis.detect_layer(records)` (a `(layer, datatype)` tuple).
2. Take the settings from `affine_ransac.release.analysis.DEFAULTS` (a `Settings`): `detection`,
   `search_px`, `design_search_nm`, `min_match_fraction`, `design_tolerance_nm`.
3. Pick `n_tiles` records that have an `.oas` (`r.oas_path is not None`), spread evenly over the list.
   Run one more such tile first, untimed, as a warm-up.
4. For each picked tile, time these steps separately. They are the steps of `process_tile` in
   `src/affine_ransac/pipeline.py`; copy its calls exactly:
   | step | call |
   |---|---|
   | `read` | `image = load_sem_image(r.image_path)` |
   | `detect` | `otsu = detect_contacts(image, method=detection)` |
   | `refine` | `refined = refine_edges(image, otsu, search_px=search_px)` |
   | `oas` | `layout = load_layout(r.oas_path)` |
   | `match_refined` | `pixel_to_tile_nm(refined.centers, ...)`, then `choose_design_contacts(layout, ...)`, as in `process_tile` |
   | `match_otsu` | the same, with `otsu.centers` instead of `refined.centers` |
   | `pickle` | `pickle.dumps(process_tile(...))` with the same arguments the app uses (see `load_image_tiles` in `src/affine_ransac/analysis.py`); time only the `dumps` and record its size in MB |
   `match_otsu` is not part of `process_tile` today. It tells us what matching would cost, and whether
   it still succeeds, if the refinement were skipped. For both matches, record `design.ok` and
   `design.match_fraction`.
5. Then time the whole-folder load the app does, once (wall clock):
   `affine_ransac.analysis.load_image_tiles(records, replace(DEFAULTS, layer=layer[0], datatype=layer[1]))`.
   Record the number of tiles, `affine_ransac.analysis.worker_count()` and `os.cpu_count()`.
6. Print a table: per step, the median and max seconds over the tiles, and each step's share of the
   per-tile total (`read + detect + refine + oas + match_refined + pickle`).

## 3. Checks on the real data

Run the script on **one image folder**, preferably the one the user ran in the app when it felt slow.
Ask the user which one. Use `n_tiles=5`.

## 4. Report: `docs/remote/T006-report.md`

Give numbers and counts only: no file names, tile ids or coordinates.

**Summary** (at most ~15 lines, shown to the user):
- status (DONE / BLOCKED + reason);
- the step table from §2.6 (median / max seconds, % of the per-tile total);
- the image size in pixels (width × height), the number of contacts per tile (median), and the pickled
  `TileResult` size in MB (median);
- the whole-folder load: wall time in seconds, number of tiles, `worker_count()`, `os.cpu_count()`;
  also wall time ÷ number of tiles, and the per-tile total ÷ `worker_count()`, so they can be compared;
- the matching: in how many of the tiles `design.ok` is True with the refined centres and with the Otsu
  centres, and the median `match_fraction` of each.
