# CPU contour-stitching benchmark, 2026-11-16

**Withdrawn as a whole-run estimate.** The user correctly rejected this benchmark's omission of required
total-variation denoising and upscaling. Its former 24-minute estimate does not answer the intended workload.
The user then cancelled the proposed CPU rerun; no replacement benchmark is authorized or reported here.
The measurements below are retained only as historical evidence for the explicitly undenoised, unscaled
synthetic input and the earlier implementation. They must not be used to forecast the user's remote run.

## Acquisition and machine

- 2048 x 2048 grayscale images, six horizontal bright-line/dark-space periods, 50% duty cycle.
- Each camera centre moves upward 683 pixels, leaving 66.65% overlap. A tile overlaps its two predecessors;
  there are `2*N - 3` overlapping pairs for this single column of images.
- Synthetic pixel scale: 1 nm/pixel. Nominal centres are separated by 683 nm; independent known vertical stage
  errors are uniform in +/-3 pixels. There is no true horizontal stage error.
- Intensities 40 and 190, Gaussian edge blur sigma 1.2 pixels, independent Gaussian detector noise sigma 6 grey
  levels. Fixed seed 20261116. Images are rendered from a shared world pattern with subpixel positions; each
  acquisition has independent noise. No real data or metadata are used.
- Windows 11, AMD Ryzen 5 7500F (6 cores / 12 logical CPUs), 32 GiB installed RAM. Python 3.13.3, NumPy 2.5.3,
  SciPy 1.18.1, OpenCV 5.0.0. CUDA is hidden and OpenCL is disabled; all array operations run on CPU.
- MaskReg's normal six-worker process pool extracts contours. Stitching uses its normal single process;
  the 128-image placement consumed 118.17 CPU seconds in 121.70 wall seconds.
- Production revision: `cfbe4373549d343cf5a3dbe58680fc48b30d44a3`.

Settings: `pattern=contours`, `dark_contacts=False`, `detection=otsu`, `refinement=mid-level`,
`max_match_nm=100` (a +/-100-pixel search), `denoise=none`, `background_px=0`, `upscale=1`,
`detection_blur_px=1.5`, `edge_blur_px=1`, `edge_step_px=0.25`, `normal_span=3`, `min_area_px=20`,
`max_area_px=0`. Otsu is chosen for the synthetic two-tone bands. These are benchmark settings, not presumed
settings of the user's remote run. Denoising or upscaling costs are not included in the estimate.

## Measurements

Every row stitches a prefix of the same independently generated and extracted image sequence. Image generation
and extraction are outside the stitching timer. The last column includes the production overlap check.

| Images | Placement, s | Overlap check, s | Stitching total, s | Placed |
|---:|---:|---:|---:|---:|
| 8 | 6.130 | 0.095 | 6.224 | 8 |
| 16 | 13.912 | 0.227 | 14.139 | 16 |
| 32 | 29.962 | 0.556 | 30.518 | 32 |
| 64 | 61.707 | 1.128 | 62.835 | 64 |
| 128 | 121.704 | 1.972 | 123.676 | 128 |

An earlier 8-image pilot took 7.568 seconds, illustrating warm-up/background-load variability. The table has
one timed run per size; it is not a statistical confidence interval.

Generating and encoding 128 lossless PNGs took **24.37 seconds**, excluded from the processing estimate because
real acquisitions already exist. Reading those PNGs, extracting contours with the production `_edge_tile`
worker, returning their edge arrays and shutting down the pool took **26.98 seconds**. In the same extraction
run, the first 32, 64 and 128 results were available at 8.554, 14.724 and 26.839 seconds; these are throughput
milestones from one shared pool, not three separate extraction runs. Each image yielded 22,506 to 24,552
finite edge points (median 24,552); the total was 3,122,579 points.

The 128-image placement spent:

| Phase | Seconds | Share of placement |
|---|---:|---:|
| Grid search against placed neighbours | 95.049 | 78.1% |
| Exact candidate comparisons | 16.618 | 13.7% |
| Final shift refinements | 8.583 | 7.1% |
| Prepare samples and facing trees | 1.259 | 1.0% |
| Find overlaps and count edges | 0.060 | <0.1% |

It computed 253 overlap search surfaces, evaluated 253 choices, and compared 45,179 shifts exactly. Each search
surface had 40,401 grid shifts. Thus the remaining dominant cost for this workload is the grid search.

## Withdrawn estimate for 1,200 images

This acquisition geometry has 2,397 overlapping pairs at 1,200 images. The larger measured runs cost
0.5003, 0.5027 and 0.4888 seconds per overlapping pair (32, 64 and 128 images).

- Scaling the 128-image stitching time by overlap count gives
  `123.6759635 * 2397 / 253 = 1171.74 s`.
- Overlap enumeration alone was also measured directly on 1,200 synthetic tile boxes: 1.241 seconds. Adding
  this twice is a small conservative allowance for the quadratic enumeration in placement and checking, giving
  **1,174 seconds, or 19.6 minutes**. Some enumeration time is already included in the scaled sample.
- An independent straight-line fit to the 32/64/128-image stitching times predicts **1,161 seconds**, so the
  two estimates agree to about 1%. This agreement does not bound real-data uncertainty.
- Fitting the extraction throughput milestones gives `2.496 + 0.19030*N` seconds: **231 seconds, or 3.8 minutes**,
  at 1,200 images with the same six-worker pool.
- Combined: **1,405 seconds, or 23.4 minutes**. Round to **about 24 minutes** for this synthetic workload on this
  machine. Image generation, metadata reading, GUI drawing, denoising, upscaling and later design comparison
  are excluded. The current contours method stops after stitching and does not perform design comparison.

More cores alone should not be assumed to divide the stitching time: this run's placement was essentially one
CPU core's work. Different pixel scale/search radius, contour geometry, noise, overlap layout, preprocessing,
CPU or storage can change these timings. No real-image timing has been measured here.

## Before versus after the CPU optimization

The same 16 extracted images were placed with the previous implementation (`dab7548`) and the optimized one
(`cfbe437`), back to back on CPU. Image extraction and overlap checks are excluded from this comparison:

| Implementation | Placement, s |
|---|---:|
| Before | 25.989 |
| After | 13.617 |

This is **1.91x faster for this synthetic sample**. All tile corrections, placement order, pair counts,
ambiguity counts and measured-direction counts agreed; the maximum correction difference was exactly zero.
It is not a speedup claim for the remote data.

## Accuracy finding

All 128 images were placed. Relative to the fixed anchor, the known vertical corrections were recovered with
0.0389 nm RMS error and **0.0714 nm maximum error** (at the synthetic 1 nm/pixel scale). The median overlap
edge RMS was 0.0658 nm; the worst overlap was 0.1560 nm.

However, perfectly horizontal lines contain no geometric horizontal-position information. The noisy synthetic
images still produced **horizontal corrections up to 86 pixels**, despite zero true horizontal error and all
127 non-anchor choices reporting one measured direction. Forty-eight choices were marked ambiguous. This
is a limitation found by the test, not evidence of correct 2D registration. The 16-image before/after comparison
returned the same 42-pixel maximum horizontal shift in both versions, so that example predates the optimization.
No behavior was adjusted to improve the benchmark result.

## Reproduction and artifacts

Historical command, at revision `f6e4b96` (not a request to rerun):

```powershell
python benchmarks/contour_stitching.py --counts 8 16 32 64 128 --repeats 1 --out build/contour-benchmark
```

The script uses the production extraction, placement and overlap-check functions. It writes generated PNGs,
the extracted edge arrays (`edges.npz`), and full measurements/settings (`results.json`) under the ignored
`build/contour-benchmark/` directory. On the machine where this report was made, `comparison.json` also holds
the additional before/after and 1,200-box measurements. The generated images and arrays are not committed.
The 33 contour-stitching unit tests passed after the benchmark.

The current runner requires explicit `--denoise` and `--upscale` arguments and accepts `--tv-weight`, so it cannot
silently repeat this preprocessing omission. D165 changes the implementation again; no timing for it is reported.

**Assumed, not known:** the same workload and processor for the 1,200-image estimate; its transfer to real SEM
data, different settings or the remote CPU is not established.
