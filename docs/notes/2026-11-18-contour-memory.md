# Contour-refinement memory, 2026-11-18

The reported failure was an allocation during contour refinement. The reported two-dimensional array size
is consistent with a full working-image buffer. Its dtype, the affected PC and its free memory were not
provided, so the precise allocation and remote peak are not established.

## What the code established and what changed

- Refinement kept the float32 conversion while making a blur result and then a signed copy. At 8192×8192,
  each float32 image takes 256 MiB. Smoothing and sign reversal now reuse one owned buffer; the caller's image
  is unchanged. This applies to both contour diagnosis and contour extraction for stitching.
- Sampling made whole-contour x/y maps and a list of remap outputs before concatenating them. It now fills
  one final output array using maps bounded by 30,000 rows and about 2²⁰ samples. No samples are omitted.
- The run chose worker count only from CPUs. The new planner reads actual image dimensions from headers and
  limits concurrency from RAM and commit headroom. It also applies to an existing high-performance-mode pool.
  The estimate is `max(1 GiB, 32 × upscaled pixels + 256 MiB)` per task, using at most 60% of current headroom.
  It is a planning allowance, not a guarantee for arbitrary contour geometry. Denoising and upscaling stay configured.
- Every tile was submitted immediately, and consumed futures stayed referenced throughout stitching. They
  retained local coordinates after the parent made translated copies. Submission is now bounded, consumed
  futures are released, and owned coordinates are translated in place. Pending tasks are cancelled promptly
  on failure or user cancellation; an external pool remains open.
- The run logs its admission limit and retained edge-array bytes. Refinement allocation errors name the
  working-image size, preprocessing settings and failed timed step when available.

## Full-size check

The same synthetic 2048×2048 image was extracted in separate fresh processes before and after the change,
with total-variation denoising (weight 0.1, 44 rounds) and 4× upscaling to 8192×8192. Baseline: `14c06bc`.
CPU only, same Python environment; Numba six threads. OpenBLAS was left at its normal default, rather than
using the one-thread setting from the previous speed benchmark. The probe reads Windows process high-water
memory counters via the locally available psutil package; production uses the existing Windows memory API
and gains no dependency.

| Process peak | Before, MiB | After, MiB |
|---|---:|---:|
| Resident working set | 1295.73 | 1020.03 |
| Committed memory | 1943.70 | 1672.51 |

The extracted 98,280 point coordinates and normals have identical SHA-256 hashes. This checks the computation
at the working-image size, not the remote data's quality or memory needs. Other shapes/edge densities can have
different profile and contour storage costs; other applications can consume memory after admission planning.

The normal production run path was also checked on eight of the previously generated images, with the same
TV and 4× settings and **no manually supplied pool**. It automatically selected two active workers from a
six-worker CPU limit at 8.27 GiB headroom, using a 2.25 GiB planning allowance. All eight placement coordinates
matched the recorded D165 results exactly. The retained edge arrays totalled 0.022 GiB.

Regression tests cover in-place smoothing for both tones with/without blur, exact profile equality across chunk
boundaries, RAM versus commit limits, largest-image sizing, unknown headers, real Unicode image paths,
bounded pending work, release of consumed futures, cancellation with a retained exception traceback, and
continued use of borrowed pools. The full suite passed: **566 passed, 1 remote-only test skipped**, with five
existing synthetic-fixture warnings. The tightened profile-map size assertion also passed separately.

Local artifacts are ignored by git: `build/contour-memory-before.json`, `build/contour-memory-after.json` and
`build/contour-memory-run-check.json`; full-suite output is `build/contour-memory-tests.log`. The probes are `build/contour_memory_probe.py` and
`build/contour_memory_run_check.py`. They use only the earlier generated synthetic input. The production
algorithm, numerical precision, `ANALYSIS` 9 and `FORMAT` 3 remain unchanged.

**Assumed, not known:** how much memory the user's other processes and real contours require, and whether
this change removes the reported remote failure. The fixes target costs established by reading the code;
the user's rerun is needed to confirm that particular failure is gone.
