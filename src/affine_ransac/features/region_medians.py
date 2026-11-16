"""Exact medians of selected contour regions, without sorting the entire image (S2, D165).

Count labels, copy the wanted pixels into contiguous groups once, and partition each group at its middle.
Background and discarded regions never enter the value buffer. Connected-component labels are nonnegative
integers bounded by the pixel count. The image and label arrays are never changed.
"""

import numba
import numpy as np


@numba.njit(cache=True, nogil=True)
def _label_counts(labels):
    # Unlike NumPy's bincount, this does not first copy an int32 label image into a full int64 image.
    counts = np.zeros(labels.max() + 1, dtype=np.int64)
    for label in labels:
        counts[label] += 1
    return counts


@numba.njit(cache=True, nogil=True)
def _collect_values(image, labels, positions, values):
    for k in range(len(labels)):
        label = labels[k]
        where = positions[label]
        if where >= 0:
            values[where] = image[k]
            positions[label] = where + 1


def region_medians(image: np.ndarray, labels: np.ndarray, numbers: np.ndarray) -> np.ndarray:
    """Medians for the requested present region numbers, with ndimage.median's float32/float64 arithmetic."""
    selected, inverse = np.unique(numbers, return_inverse=True)
    if not len(selected):
        return np.empty(0, dtype=image.dtype)
    flat_labels = np.ascontiguousarray(labels).ravel()
    counts = _label_counts(flat_labels)
    sizes = counts[selected]
    if np.any(sizes == 0):
        raise ValueError("Each requested contour region must contain pixels")
    starts = np.r_[0, np.cumsum(sizes)]
    positions = np.full(len(counts), -1, dtype=np.int64)
    positions[selected] = starts[:-1]
    values = np.empty(starts[-1], dtype=image.dtype)
    _collect_values(np.ascontiguousarray(image).ravel(), flat_labels, positions, values)
    lower, upper = (sizes - 1) // 2, sizes // 2
    for k in range(len(selected)):
        values[starts[k]:starts[k + 1]].partition((lower[k], upper[k]))
    # Vector arithmetic matches SciPy's rounding for even AND odd-sized float32 groups.
    medians = (values[starts[:-1] + lower] + values[starts[:-1] + upper]) / 2.0
    return medians[inverse]
