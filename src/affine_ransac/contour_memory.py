"""Bound contour extraction's simultaneous work and release consumed worker results (D166).

The memory allowance is a scheduling estimate, not a bound on arbitrary contour geometry. It includes headroom
for upscaled pixels, region statistics and profiles; the unreserved memory accommodates accumulated edge points
and other programs. Image headers provide the largest size without decoding every image a second time.
"""

from collections import deque

from affine_ransac.analysis import worker_count
from affine_ransac.machine import GB, memory_gb


def image_pixels(path) -> int | None:
    """Native pixel count from the header; Qt is already a dependency, imported only by the submitting process."""
    from PySide6.QtGui import QImageReader
    size = QImageReader(str(path)).size()
    return size.width() * size.height() if size.isValid() else None


def contour_workers(paths, upscale: int, record=lambda text: None, *, pool=None) -> int:
    """Limit active tasks, including on a borrowed pool; unknown size/memory permits one at a time."""
    capacity = min(worker_count(), getattr(pool, "_max_workers", worker_count()))
    largest, unknown = 0, False
    for path in paths:
        pixels = image_pixels(path)
        if pixels is None:
            unknown = True
        else:
            largest = max(largest, pixels * upscale ** 2)
    # 32 bytes per working pixel + 256 MiB, with a 1 GiB floor for numerical-library imports/thread buffers.
    # Retain 40% of current headroom outside active tasks.
    # Contours/profiles depend on the image, so this is deliberately an allowance, not a peak-memory claim.
    allowance = max(GB, largest * 32 + 256 * 2 ** 20)
    memory = memory_gb()
    available = None if memory is None else max(0.0, min(memory[1] - memory[0], memory[3] - memory[2]))
    workers = 1 if unknown or available is None else max(1, min(capacity, int(0.6 * available * GB / allowance)))
    workers = min(workers, max(1, len(paths)))
    free = "unknown" if available is None else f"{available:.2f} GiB"
    record(f"Contour memory: {workers} active worker(s), CPU/pool limit {capacity}; "
           f"available RAM/commit headroom {free}; estimated allowance {allowance / GB:.2f} GiB per worker"
           + ("; unknown image size, processing one at a time" if unknown else ""))
    return workers


def bounded_results(pool, function, arguments, limit: int):
    """Yield results in input order with at most limit submitted tasks; cancel pending work on close/error.

    Consume a future before submitting its replacement, so completed arrays cannot accumulate behind a slow
    earlier tile or remain attached to every future through stitching. The caller closes this iterator on cancel.
    """
    pending, arguments = deque(), iter(arguments)
    try:
        for _ in range(limit):
            item = next(arguments, None)
            if item is None:
                break
            pending.append(pool.submit(function, *item))
        while pending:
            future = pending.popleft()
            result = future.result()
            del future
            yield result
            del result
            item = next(arguments, None)
            if item is not None:
                pending.append(pool.submit(function, *item))
    finally:
        for future in pending:
            future.cancel()
        pending.clear()
