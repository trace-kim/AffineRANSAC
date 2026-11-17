# Contour CPU speed and output comparison, 2026-11-17

The user asked for the speed of the latest changes and whether they change the results. This renews the request
for measurement after the earlier cancelled benchmark. These measurements include total-variation denoising
and 4× upscaling. They do not reuse the withdrawn no-denoising/no-upscaling estimate.

## Workload and controls

- 32 synthetic 2048×2048 grayscale images, six horizontal bright-line/dark-space periods, 50% duty cycle.
- Camera centres move upward 683 pixels (about two-thirds overlap); independent vertical stage errors within
  ±3 pixels, with zero true horizontal error. Gaussian edge blur 1.2 pixels and independent detector noise
  of 6 grey levels. Seed 20261116, 1 nm/pixel. No real images or metadata.
- Production `pattern=contours`, bright regions, Otsu, mid-level refinement, search ±100 nm,
  **`denoise="total variation"`, `tv_weight=0.1`, `upscale=4`**. All other options use MaskReg defaults.
  All images used 44 TV rounds in both revisions. Upscaling produces 8192×8192 working images.
- CPU only: Windows 11, Ryzen 5 7500F, 6 cores/12 logical CPUs, 32 GiB installed RAM. Python 3.13.3,
  NumPy 2.5.3, SciPy 1.18.1, OpenCV 5.0.0. CUDA hidden and OpenCL disabled.
- **Two extraction workers for both versions**, six Numba threads per worker, OpenBLAS threads limited to one.
  About 7 GiB of RAM was free before the run, so the worker limit avoids comparing under heavy paging pressure.
  This is not a measurement of the default six-worker pool or the remote PC.
- Previous version: `f6e4b96` (D164). Latest production version: `0fdac5f` (D165). Separate source checkouts,
  the same interpreter and benchmark driver. The recorded module locations and revisions verify which ran.
- One complete extraction per revision; two stitching runs for each prefix of 8, 16 and 32 images. Stitching
  numbers below are medians of those two runs. Image generation and saving benchmark artifacts are excluded.
  Extraction includes image reading, denoising, upscaling, contour detection/refinement, worker communication,
  startup and shutdown. Placement plus the final overlap check constitute stitching.

## Measured time

| Work on 32 images | Previous, seconds | Latest, seconds | Speedup |
|---|---:|---:|---:|
| Extraction, including TV and 4× upscaling | 161.41 | 27.13 | 5.95× |
| Placement | 35.60 | 17.31 | 2.06× |
| Final overlap check | 2.09 | 2.08 | 1.00× |
| Stitching, including overlap check | 37.69 | 19.39 | 1.94× |
| Extraction + stitching | **199.10** | **46.52** | **4.28×** |

The combined number is the measured extraction elapsed time plus the median measured stitching time; it is not
a fresh GUI-run stopwatch measurement. It excludes metadata discovery, GUI drawing and design comparison
(the contours method still stops after stitching). First-use compilation is included: the previous checkout
spent 13.87 worker-seconds loading/compiling TV, versus 1.23 in the latest checkout. This startup difference is
small compared with the removed median-sorting cost; the extrapolation below uses later extraction throughput.

| Images | Previous stitching, seconds | Latest stitching, seconds |
|---:|---:|---:|
| 8 | 7.83 | 4.02 |
| 16 | 17.69 | 8.94 |
| 32 | 37.69 | 19.39 |

For 32 images the two stitching runs were 37.655–37.717 seconds before and 19.353–19.428 seconds after. The
first 8-image run includes first-use startup. These are observations, not a statistical confidence interval.

The interior-median operation fell from **265.96 to 9.10 seconds summed across extraction workers**, about 29×.
These are worker-step times, not elapsed run times. D165 removes the global label/intensity sort of the upscaled
image while retaining every requested region pixel. TV's solver is unchanged; its rounds took 14.84 and 17.43
summed worker-seconds respectively, so the extraction improvement is not a denoising-speed claim.

The grid search fell from 21.91 to 4.37 elapsed seconds, about 5×. The latest stitcher used run sums for 85
sectors and FFTs for 37; it refined 31 selected candidates instead of refining all 61 evaluated choices.
Remaining placement costs were exact candidate comparisons 5.62 seconds, refinement 5.40 seconds and
sample/tree preparation 1.79 seconds. The whole stitching stage therefore improved less than the grid alone.

The original stitcher, before either CPU optimization (`dab7548`), was also run twice on these same 32 images'
saved edge arrays. Placement plus the overlap check took 62.177 and 62.417 seconds (median **62.30 seconds**),
versus 19.39 seconds now: **3.21× faster stitching across both optimization rounds**. Its placement coordinates
and complete choice diagnostics also matched the latest version exactly. Its extraction was not rerun: that
code is identical to D164's extraction, checked with the source diff. Do not multiply the earlier undenoised
benchmark's 1.91× factor by this benchmark's factors; they use different preprocessing.

## Did results change?

No differences were observed between D164 and D165 on this workload:

- The extracted coordinates and normals of every image have identical SHA-256 hashes (3,071,250 edge points).
- Every placement coordinate is exactly equal, for all 8/16/32-image prefixes and both repeats: maximum
  difference **0 nm**.
- Placement order, pair counts, RMS diagnostics, ambiguity counts and measured directions are also exactly equal.
- All 32 images were placed. Against the synthetic truth, maximum vertical error was 0.011577 nm and maximum
  horizontal correction was 0.000212 nm in both versions.

The computational algorithms changed: local bounded distances, interval sums, grouped median selection and
deferred refinement. The mathematical search, scoring, gates, sampling, fitting and placement rules did not.
This comparison establishes identical outputs for these inputs; it does not prove bitwise identity for every
input, because FFT/reduction roundoff can differ. It does not establish real-image registration quality or fix
the previously recorded along-line drift on a different, undenoised synthetic workload.

## Approximation for 1,200 images

For this same synthetic workload, CPU and **two-worker limit**, the latest code extrapolates to approximately
**28 minutes**: 15.2 minutes extraction and 12.7 minutes stitching. The preceding version extrapolates to
approximately 119 minutes under these same conditions.

- Extraction throughput comes from images 17–32, after worker startup:
  `(26.9366644 - 14.7689675) / 16 = 0.7604811` elapsed seconds per image.
- A 32-image column has 61 overlaps and a 1,200-image column has 2,397:
  `19.39058375 × 2397 / 61 = 761.95` seconds of stitching.
- Combined: `(0.7604811 × 1200 + 761.95) / 60 = 27.91` minutes.

This is an extrapolation from 32 images, not a 1,200-image measurement or a prediction for the remote machine.
Different TV convergence, contours, pixel size/search range, overlap layout, memory pressure or worker count
can change it. In particular, the full-image sort's removal benefits upscaled mid-level refinement; it does not
give that extraction gain when the sort was not being performed.

## Reproduction and artifacts

The benchmark driver records the imported library's revision, per-worker step times, edge hashes, complete
placements and fitting diagnostics. Both commands used the current driver; `PYTHONPATH` selected the library:

```powershell
git worktree add --detach build/contour-comparison-before f6e4b96
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONPATH="$PWD/build/contour-comparison-before/src"
python benchmarks/contour_stitching.py --counts 8 16 32 --repeats 2 --denoise 'total variation' --upscale 4 --tv-weight 0.1 --workers 2 --out build/contour-tv4-before
$env:PYTHONPATH="$PWD/src"
python benchmarks/contour_stitching.py --counts 8 16 32 --repeats 2 --denoise 'total variation' --upscale 4 --tv-weight 0.1 --workers 2 --out build/contour-tv4-after
```

Local artifacts (ignored by git): each output folder's `results.json`, `edges.npz` and generated images;
`build/contour-tv4-comparison.json` holds the comparison. No production algorithm was changed during this
measurement. D165's previously reported correctness suite remains 553 passed and one remote-only skip.
`build/contour-tv4-original.json` holds the two original-stitcher measurements; the original source is a detached
checkout under `build/contour-comparison-original`. The original extraction was not needed for this additional
stitching-only comparison because D164 and D165 produced identical edge arrays.

**Assumed, not known:** the selected TV strength, 4× upscaling and synthetic geometry represent the intended
comparison. They are not asserted to be the user's remote settings, and no remote speed or quality is measured.
