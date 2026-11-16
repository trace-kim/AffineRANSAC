"""Chambolle's total variation denoising, compiled with numba (docs/SPEC.md D159): the pixels of
skimage.restoration.denoise_tv_chambolle(image, weight), about 7 times as fast on a 2048 x 2048 image.

scikit-image runs each round as about twenty whole-image NumPy operations, each reading and writing the full image
again; here one round is two passes over the image, each spread over the processor's threads. Every pixel is
computed with the same operations in the same order, so the result is the same number for number. The two sums of the
stop test are added row by row here, in another order than NumPy's, so the round the loop stops at could differ only
if the cost's change came within rounding (about 1e-12 of it) of the threshold.

Imported only when total variation is used (features.contact.denoised_image): numba takes about 0.5 s to import.
The loops are compiled at their first use on a PC (about 1 s) and kept in numba's cache for later processes."""

import threading

import numba
import numpy as np

STEP = 0.25  # scikit-image's step: 1 / (2 x the number of dimensions)
_one_search_at_a_time = threading.Lock()  # numba's simplest thread pool ends the program if two threads enter it
_compile_lock = threading.Lock()
_loops_ready = False


@numba.njit(parallel=True, cache=True)
def _flat_image(image, p, flat, row_sums):
    """flat = image minus the divergence of p; row_sums[i]: the sum over row i of (flat - image) squared."""
    rows, cols = image.shape
    for i in numba.prange(rows):
        total = 0.0
        for j in range(cols):
            removed = -(p[0, i, j] + p[1, i, j])
            if i > 0:
                removed += p[0, i - 1, j]
            if j > 0:
                removed += p[1, i, j - 1]
            flat[i, j] = image[i, j] + removed
            total += removed * removed
        row_sums[i] = total


@numba.njit(parallel=True, cache=True)
def _step_p(flat, p, weight, row_sums):
    """p takes a step against the gradient of flat (rows, columns; 0 past the last row and column) and is divided
    by 1 + STEP / weight x the gradient's length; row_sums[i]: the sum over row i of those lengths."""
    rows, cols = flat.shape
    scale = STEP / weight
    for i in numba.prange(rows):
        total = 0.0
        for j in range(cols):
            down = flat[i + 1, j] - flat[i, j] if i < rows - 1 else 0.0
            right = flat[i, j + 1] - flat[i, j] if j < cols - 1 else 0.0
            length = np.sqrt(down * down + right * right)
            total += length
            divisor = length * scale + 1.0
            p[0, i, j] = (p[0, i, j] - STEP * down) / divisor
            p[1, i, j] = (p[1, i, j] - STEP * right) / divisor
        row_sums[i] = total


def compile_loops():
    """Warm the two loops once per process; concurrent callers wait until both are ready."""
    global _loops_ready
    # A separate lock: total_variation itself takes _one_search_at_a_time. Mark ready only after success,
    # so a failed first compilation can be retried by the next call.
    with _compile_lock:
        if not _loops_ready:
            total_variation(np.zeros((2, 2)), 0.1, max_rounds=1)
            _loops_ready = True


def total_variation(image: np.ndarray, weight: float, eps: float = 2.0e-4,
                    max_rounds: int = 200) -> tuple[np.ndarray, int]:
    """(the 2-D float64 image closest to `image` with the least total variation, the rounds it took), as
    skimage.restoration.denoise_tv_chambolle(image, weight, eps, max_rounds). Ends when the cost (what was removed,
    squared, plus weight x the gradient lengths; mean per pixel) changes by less than eps x the first round's cost."""
    image = np.ascontiguousarray(image, dtype=np.float64)
    p = np.zeros((2,) + image.shape)
    flat = np.empty_like(image)
    removed_sums, length_sums = np.empty(image.shape[0]), np.empty(image.shape[0])
    with _one_search_at_a_time:
        for done in range(max_rounds):
            _flat_image(image, p, flat, removed_sums)
            _step_p(flat, p, float(weight), length_sums)
            cost = (removed_sums.sum() + weight * length_sums.sum()) / float(image.size)
            if done == 0:
                first = previous = cost
            elif abs(previous - cost) < eps * first:
                break
            else:
                previous = cost
    return flat, done + 1
